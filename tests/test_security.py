"""Phase 0 security regression tests.

Guards the Phase 0 fixes:
  (0) tests themselves never load the owner's real credentials;
  (1) UI policy  -- no secret, or anything derived from one, reaches any Streamlit element (including
      aliases, columns/containers, getattr, and session_state);
  (2) secret variables are only ever assigned from the secret loaders;
  (3) key-flow policy -- a credential reaches an external call ONLY at the three vetted sites (the two
      Polygon helpers and the Telegram POST), whatever HTTP client / print / logging / file API is used;
      the Polygon host allow-list resists URL-parser differentials;
  (4) Telegram alert control is fail-closed to the owner's local session (unit, structural and
      behavioural -- the real module is re-executed under controlled conditions);
  (5) the local launcher binds to loopback.

(1) and (3) share a small interprocedural taint engine (aliases, tuples, containers, function params and
returns, closures, lambdas, for/with targets, and source calls like _secret() / st.secrets[...] /
os.environ). Every detector is mutation-tested (the *_catches_* tests), so a weakened check fails loudly.
All HTTP is mocked, keys are fake, and the conftest isolates the real secrets file.
"""
import ast
import importlib.util
import itertools
import os
import sys

import pytest
import requests
from urllib3.util import parse_url

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCANNER = os.path.join(ROOT, "alpha_scanner.py")
BACKTEST = os.path.join(ROOT, "alpha_backtest.py")
SCAN_ALERT = os.path.join(ROOT, "scan_alert.py")
PREMARKET = os.path.join(ROOT, "premarket_report.py")
CONTRACTION = os.path.join(ROOT, "contraction_backtest.py")
REPLAY = os.path.join(ROOT, "live_rules_replay.py")
SECRET_NAMES = {"POLYGON_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"}
SECRET_LOADERS = {"_secret", "_polygon_key"}
SECRET_ENV_VARS = ("POLYGON_KEY", "POLYGON_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                   "TELEGRAM_ALERTS_ENABLED")
FAKE_KEY = "FAKEKEY_for_tests_0123456789abcd"

# The ONLY places a credential may leave the process: (enclosing function, callee).
VETTED_SINKS = {
    SCANNER: {("_polygon_get", "requests.get"), ("send_telegram_alert", "requests.post")},
    BACKTEST: {("_polygon_request", "requests.get")},
    SCAN_ALERT: set(), PREMARKET: set(), CONTRACTION: set(),
    REPLAY: {("main", "OptionData")},   # key handed to the options-data client object
}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# =================================================================== static-analysis helpers
def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _expr_root(e):
    """Leftmost Name of a call/attribute/subscript chain: st.sidebar.text_input(...) -> 'st'."""
    while True:
        if isinstance(e, ast.Call):
            e = e.func
        elif isinstance(e, (ast.Attribute, ast.Subscript)):
            e = e.value
        else:
            break
    return e.id if isinstance(e, ast.Name) else None


def _callee_name(f):
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _docstring_ids(tree):
    ids = set()
    for n in ast.walk(tree):
        if (isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)):
            ids.add(id(n.body[0].value))
    return ids


def _scope_nodes(body):
    """Nodes of one scope: stops at nested def/class (their own scopes); lambdas stay in scope."""
    out, stack = [], list(body)
    while stack:
        n = stack.pop()
        out.append(n)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(n))
    return out


# ------------------------------------------------------------------------------ taint engine
SANITIZERS = {"bool", "len", "isinstance", "callable", "_alerts_control_mode", "_session_is_local",
              "_truthy_flag", "_is_polygon_url"}           # return only yes/no or a mode, never the value
TRANSFORMER_FUNCS = {"str", "repr", "int", "float", "bytes", "format", "sorted", "list", "tuple", "dict", "set"}
TRANSFORMER_METHODS = {"strip", "lstrip", "rstrip", "lower", "upper", "replace", "split", "rsplit", "join",
                       "encode", "decode", "format", "casefold", "title", "partition", "rpartition", "zfill",
                       "quote", "quote_plus"}              # carry taint to their result; not sinks themselves
SOURCE_ROOTS = {"st.secrets", "os.environ"}


def _is_source(e, seeds):
    if isinstance(e, ast.Call):
        f = e.func
        if _callee_name(f) in SECRET_LOADERS:
            return True
        if _dotted(f) in {"os.getenv", "os.environ.get", "os.environ.setdefault", "os.environ.pop",
                          "st.secrets.get", "st.secrets.to_dict"}:
            return True
        return (isinstance(f, ast.Name) and f.id == "getattr" and len(e.args) >= 2
                and isinstance(e.args[1], ast.Constant) and e.args[1].value in seeds)
    if isinstance(e, ast.Subscript) and isinstance(e.ctx, ast.Load):
        if _dotted(e.value) in SOURCE_ROOTS:
            return True
        return (isinstance(e.value, ast.Call) and isinstance(e.value.func, ast.Name)
                and e.value.func.id in {"globals", "vars", "locals"}
                and isinstance(e.slice, ast.Constant) and e.slice.value in seeds)
    if isinstance(e, ast.Attribute) and isinstance(e.ctx, ast.Load):
        return e.attr in seeds or _dotted(e) in SOURCE_ROOTS
    return False


def _is_transformer(f):
    return ((isinstance(f, ast.Name) and f.id in TRANSFORMER_FUNCS)
            or (isinstance(f, ast.Attribute) and f.attr in TRANSFORMER_METHODS))


class _Ctx:
    def __init__(self, seeds, local_funcs, ret_tainted, fn, vetted):
        self.seeds, self.local_funcs, self.ret_tainted, self.fn, self.vetted = seeds, local_funcs, ret_tainted, fn, vetted


def _tainted(e, T, ctx):
    if not isinstance(e, ast.AST):
        return False
    if isinstance(e, ast.Compare) or (isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not)):
        return False
    if isinstance(e, ast.Name):
        return isinstance(e.ctx, ast.Load) and e.id in T
    if _is_source(e, ctx.seeds):
        return True
    if isinstance(e, ast.Call):
        f = e.func
        if isinstance(f, ast.Name) and f.id in SANITIZERS:
            return False
        if isinstance(f, ast.Name) and f.id in ctx.local_funcs:
            return f.id in ctx.ret_tainted
        if (ctx.fn, _dotted(f)) in ctx.vetted:
            return False          # a vetted request's RESPONSE is not the credential
    return any(_tainted(c, T, ctx) for c in ast.iter_child_nodes(e))


def _target_names(t, exclude=frozenset()):
    """Names bound (or containers written) by an assignment target. obj.attr = v does NOT taint obj."""
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        return [x for el in t.elts for x in _target_names(el, exclude)]
    if isinstance(t, ast.Starred):
        return _target_names(t.value, exclude)
    if isinstance(t, ast.Subscript):          # d[k] = secret -> d is tainted
        r = _expr_root(t.value)
        return [r] if r and r not in exclude else []
    return []


def _propagate(nodes, T, ctx, exclude=frozenset()):
    T, changed = set(T), True
    while changed:
        changed = False
        for n in nodes:
            new = []
            if isinstance(n, ast.Assign) and _tainted(n.value, T, ctx):
                for t in n.targets:
                    new += _target_names(t, exclude)
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)) and n.value is not None \
                    and _tainted(n.value, T, ctx):
                new += _target_names(n.target, exclude)
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)) and _tainted(n.iter, T, ctx):
                new += _target_names(n.target, exclude)
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for it in n.items:
                    if it.optional_vars is not None and _tainted(it.context_expr, T, ctx):
                        new += _target_names(it.optional_vars, exclude)
            for nm in new:
                if nm not in T:
                    T.add(nm)
                    changed = True
    return T


def _params(f):
    a = f.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    return names | ({a.vararg.arg} if a.vararg else set()) | ({a.kwarg.arg} if a.kwarg else set())


def _analyze(tree, seeds, vetted=frozenset(), exclude=frozenset()):
    """Interprocedural fixpoint. Returns [(nodes, tainted_names, ctx)] for the module and every function."""
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    local = {f.name for f in fns}
    parent = {}
    for f in fns:                                  # ast.walk is outer-first -> the last write is innermost
        for c in ast.walk(f):
            if c is not f and isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent[id(c)] = f
    by_name = {}
    for f in fns:
        by_name.setdefault(f.name, []).append(f)
    param_T = {f.name: set() for f in fns}
    ret_T = set()
    mod_nodes = _scope_nodes(tree.body)
    mod_T = set(seeds)
    scopes = {}
    for _ in range(60):
        snap = (frozenset(mod_T), frozenset(ret_T), tuple(sorted((k, frozenset(v)) for k, v in param_T.items())))
        ctx0 = _Ctx(seeds, local, ret_T, None, vetted)
        mod_T = _propagate(mod_nodes, mod_T, ctx0, exclude)
        scopes[None] = (mod_nodes, mod_T, ctx0)
        for f in fns:
            nodes = _scope_nodes(f.body)
            declared_global = {nm for n in nodes if isinstance(n, ast.Global) for nm in n.names}
            assigned = set()
            for n in nodes:
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        assigned.update(_target_names(t))
                elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
                    assigned.update(_target_names(n.target))
            enclosing = scopes[id(parent[id(f)])][1] if id(f) in parent and id(parent[id(f)]) in scopes else mod_T
            base = (enclosing - _params(f) - (assigned - declared_global)) | param_T[f.name]
            ctx = _Ctx(seeds, local, ret_T, f.name, vetted)
            T = _propagate(nodes, base, ctx, exclude)
            scopes[id(f)] = (nodes, T, ctx)
            if any(isinstance(n, ast.Return) and n.value is not None and _tainted(n.value, T, ctx) for n in nodes):
                ret_T.add(f.name)
            for n in nodes:                        # `global X; X = secret` taints the module-level X
                if isinstance(n, ast.Assign) and _tainted(n.value, T, ctx):
                    for t in n.targets:
                        mod_T.update(nm for nm in _target_names(t) if nm in declared_global)
        for nodes, T, ctx in list(scopes.values()):  # call sites -> parameter taint
            for c in nodes:
                if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in by_name):
                    continue
                for f in by_name[c.func.id]:
                    pos = [x.arg for x in f.args.posonlyargs + f.args.args]
                    for i, a in enumerate(c.args):
                        inner = a.value if isinstance(a, ast.Starred) else a
                        if not _tainted(inner, T, ctx):
                            continue
                        if isinstance(a, ast.Starred):
                            param_T[f.name] |= set(pos) | ({f.args.vararg.arg} if f.args.vararg else set())
                        elif i < len(pos):
                            param_T[f.name].add(pos[i])
                        elif f.args.vararg:
                            param_T[f.name].add(f.args.vararg.arg)
                    for kw in c.keywords:
                        if _tainted(kw.value, T, ctx):
                            param_T[f.name] |= ({kw.arg} if kw.arg else (set(pos) | {x.arg for x in f.args.kwonlyargs}))
        if snap == (frozenset(mod_T), frozenset(ret_T), tuple(sorted((k, frozenset(v)) for k, v in param_T.items()))):
            break
    return list(scopes.values())


def _ui_handles(tree):
    """'st', streamlit import aliases, and anything derived from them (sb = st.sidebar, c1, c2 =
    st.columns(2), `with st.expander() as e`, for-targets over st.tabs(...), widget return values)."""
    H = {"st"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "streamlit" or a.name.startswith("streamlit."):
                    H.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module and (n.module == "streamlit" or n.module.startswith("streamlit.")):
            H.update(a.asname or a.name for a in n.names)
    changed = True
    while changed:
        changed = False
        for n in ast.walk(tree):
            new = []
            if isinstance(n, ast.Assign) and _expr_root(n.value) in H:
                for t in n.targets:
                    new += [x for x in _target_names(t) if not isinstance(t, ast.Subscript)]
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for it in n.items:
                    if it.optional_vars is not None and _expr_root(it.context_expr) in H:
                        new += _target_names(it.optional_vars)
            elif isinstance(n, (ast.For, ast.AsyncFor)) and _expr_root(n.iter) in H:
                new += _target_names(n.target)
            for nm in new:
                if nm not in H:
                    H.add(nm)
                    changed = True
    return H


def _args(call):
    return list(call.args) + [k.value for k in call.keywords]


# ================================================= (0) tests never load the real credentials
def test_tests_never_load_real_secrets(asc, ab):
    assert not os.path.exists(os.path.join(ROOT, ".streamlit", "secrets.toml")), (
        "a real secrets.toml sits inside the repo; tests would load it. Keep it in ~/.streamlit/ instead.")
    assert "alphascanner-tests-home-" in os.path.expanduser("~"), "conftest home isolation is not active"
    assert asc.POLYGON_KEY == asc.TELEGRAM_BOT_TOKEN == asc.TELEGRAM_CHAT_ID == ""
    assert not ab._polygon_key()
    # ...which also proves the fail-closed module defaults with no secrets configured:
    assert asc.TELEGRAM_CONFIGURED is False and asc.ALERTS_ENABLED_DEFAULT is False
    assert asc._alert_mode == "unconfigured" and asc.enable_alerts is False


# ======================================================== (1) UI policy: nothing secret reaches UI
def ui_offenders(src, vetted=frozenset()):
    tree = ast.parse(src)
    H = _ui_handles(tree)
    out = []
    for nodes, T, ctx in _analyze(tree, SECRET_NAMES, vetted, exclude=frozenset(H)):
        for n in nodes:
            if isinstance(n, ast.Call):
                f = n.func
                is_ui = _expr_root(f) in H
                if (isinstance(f, ast.Call) and isinstance(f.func, ast.Name) and f.func.id == "getattr"
                        and f.args and _expr_root(f.args[0]) in H):
                    is_ui = True           # getattr(st, "text_input")(...)
                if isinstance(f, ast.Name) and f.id == "setattr" and n.args and _expr_root(n.args[0]) in H:
                    is_ui = True           # setattr(st.session_state, "k", secret)
                if is_ui and any(_tainted(a, T, ctx) for a in _args(n)):
                    out.append((n.lineno, "ui-call", ast.unparse(f)[:60]))
            elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and n.value is not None \
                    and _tainted(n.value, T, ctx):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, (ast.Subscript, ast.Attribute)) and _expr_root(t) in H \
                            and "session_state" in ast.unparse(t):
                        out.append((n.lineno, "session_state-write", ast.unparse(t)[:60]))
    return out


def test_no_secret_or_derivative_reaches_the_ui():
    """The original leak was st.text_input("Bot Token", value=TELEGRAM_BOT_TOKEN, type="password")."""
    offenders = ui_offenders(_read(SCANNER), VETTED_SINKS[SCANNER])
    assert not offenders, "secret (or a value derived from one) reaches a Streamlit element: %s" % offenders


@pytest.mark.parametrize("snippet", [
    'st.text_input("Bot Token", value=TELEGRAM_BOT_TOKEN, type="password")',   # the original leak
    'tok = TELEGRAM_BOT_TOKEN\nst.text_input("x", value=tok)',                   # alias
    'tok = TELEGRAM_BOT_TOKEN or ""\nst.write(tok)',                              # BoolOp returns the value
    'a = POLYGON_KEY\nb = a.strip()\nst.code(b)',                                 # two-hop alias
    'a, b = TELEGRAM_BOT_TOKEN, 1\nst.write(a)',                                  # tuple unpack
    'd = {}\nd["t"] = TELEGRAM_BOT_TOKEN\nst.json(d)',                            # container write
    'st.text_input("t", value=_secret("TELEGRAM_BOT_TOKEN"))',                   # direct loader call
    'st.text_input("t", value=st.secrets["TELEGRAM_BOT_TOKEN"])',                # st.secrets[...]
    'st.write(st.secrets)',                                                       # the whole secrets store
    'import os\nst.write(os.environ.get("POLYGON_KEY"))',                        # env
    'st.write(getattr(mod, "POLYGON_KEY"))',                                      # getattr by name
    'st.write(globals()["POLYGON_KEY"])',                                         # globals()
    'st.write(asc.TELEGRAM_BOT_TOKEN)',                                           # another module's global
    'c1, c2 = st.columns(2)\nc1.write(TELEGRAM_CHAT_ID)',                        # column handle
    'sb = st.sidebar\nsb.text_input("x", value=TELEGRAM_BOT_TOKEN)',             # sidebar alias
    'import streamlit as stx\nstx.write(POLYGON_KEY)',                            # import alias
    'from streamlit import sidebar\nsidebar.write(POLYGON_KEY)',                  # from-import
    'with st.expander("x") as e:\n    e.write(POLYGON_KEY)',                     # with-as handle
    'getattr(st, "text_input")("x", value=TELEGRAM_BOT_TOKEN)',                  # getattr on st
    'st.session_state["tg"] = TELEGRAM_BOT_TOKEN\nst.text_input("T", key="tg", type="password")',
    'st.session_state.tg = TELEGRAM_BOT_TOKEN',                                   # attribute form
    'setattr(st.session_state, "tg", TELEGRAM_BOT_TOKEN)',                       # setattr form
    'def show(v):\n    st.code(v)\nshow(TELEGRAM_BOT_TOKEN)',                     # helper parameter
    'def tok():\n    return TELEGRAM_BOT_TOKEN\nst.write(tok())',                  # helper return
    'for t in [TELEGRAM_BOT_TOKEN]:\n    st.write(t)',                            # for-loop target
    'def outer():\n    x = TELEGRAM_BOT_TOKEN\n    def inner():\n        st.write(x)\n    inner()',  # closure
    'st.button("x", on_click=lambda: st.write(TELEGRAM_BOT_TOKEN))',             # lambda body
    'st.markdown(f"<b>{TELEGRAM_CHAT_ID}</b>", unsafe_allow_html=True)',          # f-string
])
def test_ui_policy_catches_mutations(snippet):
    assert ui_offenders(snippet), "UI policy missed: " + snippet


@pytest.mark.parametrize("snippet", [
    'ok = bool(TELEGRAM_BOT_TOKEN)\nst.caption(str(ok))',        # yes/no only
    'ok = TELEGRAM_BOT_TOKEN != ""\nst.caption(str(ok))',        # comparison
    'st.caption("TELEGRAM_BOT_TOKEN missing")',                   # the NAME inside a string
    'st.write(len(POLYGON_KEY))',                                  # length only
    'if not POLYGON_KEY:\n    st.error("no key")',                # truthiness gate
    'flag = _truthy_flag(_secret("TELEGRAM_ALERTS_ENABLED"))\nst.write(flag)',
])
def test_ui_policy_allows_non_revealing_uses(snippet):
    assert not ui_offenders(snippet), "UI policy false positive: " + snippet


def test_no_ui_credential_inputs_or_public_test_send():
    src = _read(SCANNER)
    assert '"Bot Token"' not in src and '"Chat ID"' not in src, "a credential input is back in the UI"
    assert "Test Alert" not in src, "a public test-send control is back"


# ============================================ (2) secret variables only come from the loaders
def _mentions_secret_target(t):
    for x in ast.walk(t):
        if isinstance(x, ast.Name) and x.id in SECRET_NAMES and isinstance(x.ctx, ast.Store):
            return True
        if isinstance(x, ast.Attribute) and x.attr in SECRET_NAMES and isinstance(x.ctx, ast.Store):
            return True
        if (isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Constant) and x.slice.value in SECRET_NAMES
                and ((isinstance(x.value, ast.Call) and _callee_name(x.value.func) in {"globals", "vars", "locals"})
                     or (isinstance(x.value, ast.Attribute) and x.value.attr == "__dict__"))):
            return True
    return False


def _is_loader_call(v):
    return isinstance(v, ast.Call) and _callee_name(v.func) in SECRET_LOADERS


def secret_override_offenders(src):
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign):
            simple_ok = (len(n.targets) == 1 and isinstance(n.targets[0], (ast.Name, ast.Attribute))
                         and _is_loader_call(n.value))
            if any(_mentions_secret_target(t) for t in n.targets) and not simple_ok:
                out.append((n.lineno, ast.unparse(n)[:70]))
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.For, ast.AsyncFor)) \
                and _mentions_secret_target(n.target):
            out.append((n.lineno, type(n).__name__))
        elif isinstance(n, (ast.With, ast.AsyncWith)) and any(
                it.optional_vars is not None and _mentions_secret_target(it.optional_vars) for it in n.items):
            out.append((n.lineno, "with-as"))
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr"
              and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and n.args[1].value in SECRET_NAMES):
            out.append((n.lineno, "setattr"))
    return out


@pytest.mark.parametrize("path", [SCANNER, BACKTEST, SCAN_ALERT, PREMARKET, CONTRACTION, REPLAY])
def test_secret_variables_only_assigned_from_secret_loaders(path):
    """Catches the old `TELEGRAM_BOT_TOKEN = bot_token` (a widget overriding a secret) in any form."""
    bad = secret_override_offenders(_read(path))
    assert not bad, "secret variable written from a non-loader source in %s: %s" % (os.path.basename(path), bad)


@pytest.mark.parametrize("snippet,flag", [
    ('TELEGRAM_BOT_TOKEN = bot_token', True),                              # the original override
    ('TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID = bot_token, chat_id', True),   # tuple
    ('(TELEGRAM_BOT_TOKEN := bot_token)', True),                           # walrus
    ('globals()["TELEGRAM_BOT_TOKEN"] = bot_token', True),                 # globals()
    ('setattr(mod, "POLYGON_KEY", k)', True),                              # setattr
    ('mod.__dict__["POLYGON_KEY"] = k', True),                             # __dict__
    ('for POLYGON_KEY in keys:\n    pass', True),                          # for target
    ('TELEGRAM_BOT_TOKEN += "x"', True),                                   # augassign
    ('TELEGRAM_BOT_TOKEN = _secret("TELEGRAM_BOT_TOKEN")', False),         # the allowed form
    ('asc.POLYGON_KEY = ab._polygon_key()', False),                        # allowed (scan_alert)
])
def test_secret_override_detector_catches_mutations(snippet, flag):
    assert bool(secret_override_offenders(snippet)) is flag


# ======================================== (3) key-flow policy + Polygon transport hardening
def key_flow_offenders(src, vetted):
    """A credential-derived value passed to ANY external call (HTTP client of any kind, print, logging,
    file/JSON writes, exception constructors, ...) outside the vetted sites. Transformers (str, .strip,
    ...) only carry taint forward; sanitizers (bool, len, comparisons) end it."""
    tree = ast.parse(src)
    out = []
    for nodes, T, ctx in _analyze(tree, SECRET_NAMES, vetted):
        for n in nodes:
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if (isinstance(f, ast.Name) and (f.id in SANITIZERS or f.id in ctx.local_funcs)) \
                    or _is_transformer(f) or _is_source(n, ctx.seeds):
                continue
            if any(_tainted(a, T, ctx) for a in _args(n)) and (ctx.fn, _dotted(f)) not in vetted:
                out.append((n.lineno, ctx.fn or "<module>", ast.unparse(f)[:60]))
    return out


@pytest.mark.parametrize("path", sorted(VETTED_SINKS))
def test_credentials_only_leave_through_vetted_sites(path):
    bad = key_flow_offenders(_read(path), VETTED_SINKS[path])
    assert not bad, "credential reaches an unvetted external call in %s: %s" % (os.path.basename(path), bad)


@pytest.mark.parametrize("snippet", [
    'import requests\ndef fetch(u):\n    return requests.get(u, headers={"Authorization": "Bearer " + POLYGON_KEY})',
    'import requests\nrequests.Session().get(nxt, headers={"Authorization": "Bearer " + POLYGON_KEY})',
    'import requests as _rq\n_rq.get("https://api.polygon.io/x", params={"apiKey": _polygon_key()})',
    'import urllib.request\nurllib.request.urlopen("https://x/?k=" + POLYGON_KEY)',
    'from requests import get\nget(u, params={"apiKey": ab._polygon_key()})',
    'print(POLYGON_KEY)',
    'import logging\nlogging.info("key=%s", _polygon_key())',
    'import json\njson.dump({"k": POLYGON_KEY}, open("f", "w"))',
    'raise RuntimeError("bad key " + POLYGON_KEY)',
    'def h(k):\n    print(k)\nh(TELEGRAM_BOT_TOKEN)',
    'st.write(os.environ["TELEGRAM_BOT_TOKEN"])',
    'import requests\nrequests.post(u, json={"chat_id": TELEGRAM_CHAT_ID})',   # a second Telegram sender
])
def test_key_flow_policy_catches_mutations(snippet):
    assert key_flow_offenders(snippet, VETTED_SINKS[SCANNER]), "key-flow policy missed: " + snippet


@pytest.mark.parametrize("snippet", [
    'import requests\ndef _polygon_get(u):\n    return requests.get(u, headers={"Authorization": "Bearer " + POLYGON_KEY})',
    'key = _polygon_key()\nif not key:\n    raise RuntimeError("no key")',
    'ok = bool(POLYGON_KEY)\nprint(ok)',
    'import requests\ndef send_telegram_alert(m):\n    u = "https://api.telegram.org/bot%s/x" % TELEGRAM_BOT_TOKEN\n'
    '    r = requests.post(u, json={"chat_id": TELEGRAM_CHAT_ID, "text": m})\n    return r.status_code == 200',
])
def test_key_flow_policy_allows_vetted_and_non_revealing_uses(snippet):
    assert not key_flow_offenders(snippet, VETTED_SINKS[SCANNER]), "key-flow false positive: " + snippet


def apikey_offenders(src):
    """Ways the key could be put in a URL/query: dict key, subscript, keyword, a bare "apiKey" constant
    (list-of-tuples params, setdefault), or a non-docstring string containing 'apiKey='."""
    tree = ast.parse(src)
    docs = _docstring_ids(tree)
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == "apiKey":
            hits.append((n.lineno, "subscript"))
        elif isinstance(n, ast.keyword) and n.arg == "apiKey":
            hits.append((n.value.lineno, "keyword"))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            low = n.value.strip().lower()
            if low == "apikey" or "apikey=" in low:
                hits.append((n.lineno, "constant"))
    return hits


@pytest.mark.parametrize("path", sorted(VETTED_SINKS))
def test_no_apikey_query_parameter_anywhere(path):
    hits = apikey_offenders(_read(path))
    assert not hits, "Polygon key placed in a URL/query in %s: %s" % (os.path.basename(path), hits)


@pytest.mark.parametrize("snippet", [
    'requests.get(u, params={"apiKey": key})',
    'params["apiKey"] = key',
    'requests.get(u, params=dict(apiKey=key))',
    'requests.get(u, params=[("apiKey", key)])',
    'params.setdefault("apiKey", key)',
    'u = f"https://api.polygon.io/v2/x?apiKey={key}"',
    'u = "https://api.polygon.io/v2/x?adjusted=true&apiKey=" + key',
])
def test_apikey_detector_catches_mutations(snippet):
    assert apikey_offenders(snippet), "detector missed: " + snippet


def auth_literal_sites(src):
    tree = ast.parse(src)
    docs = _docstring_ids(tree)
    owner = {}
    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for n in ast.walk(fn):
            owner[id(n)] = fn.name
    return {owner.get(id(n), "<module>") for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
            and ("bearer" in n.value.lower() or "authorization" in n.value.lower())}


@pytest.mark.parametrize("path,allowed", [(SCANNER, {"_polygon_get"}), (BACKTEST, {"_polygon_request"}),
                                          (SCAN_ALERT, set()), (PREMARKET, set()), (CONTRACTION, set()), (REPLAY, set())])
def test_auth_headers_are_only_built_inside_the_helpers(path, allowed):
    sites = auth_literal_sites(_read(path))
    assert sites <= allowed, "Authorization/Bearer header built outside the helpers in %s: %s" % (
        os.path.basename(path), sites - allowed)


class _FakeResp:
    status_code = 200

    def json(self):
        return {}


def _capture_get(monkeypatch, module):
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None, **kw):
        calls.append({"url": url, "params": params, "headers": headers or {}})
        return _FakeResp()

    monkeypatch.setattr(module.requests, "get", fake_get)
    return calls


def test_scanner_polygon_get_uses_bearer_header_not_url(asc, monkeypatch):
    calls = _capture_get(monkeypatch, asc)
    monkeypatch.setattr(asc, "POLYGON_KEY", FAKE_KEY)
    asc._polygon_get("https://api.polygon.io/v2/aggs/ticker/SPY/prev", params={"adjusted": "true"})
    (c,) = calls
    assert c["headers"].get("Authorization") == "Bearer " + FAKE_KEY
    assert "apiKey" not in (c["params"] or {})
    assert FAKE_KEY not in c["url"]


def test_backtest_polygon_request_uses_bearer_header_not_url(ab, monkeypatch):
    calls = _capture_get(monkeypatch, ab)
    ab._polygon_request("https://api.polygon.io/v3/reference/options/contracts", FAKE_KEY, params={"limit": 1})
    (c,) = calls
    assert c["headers"].get("Authorization") == "Bearer " + FAKE_KEY
    assert "apiKey" not in (c["params"] or {})
    assert FAKE_KEY not in c["url"]


ACCEPTED_URLS = [
    "https://api.polygon.io/v3/snapshot/options/SPY?cursor=YXA9MTAwJmFzPSZsaW1pdD0yNTA%3D",
    "https://api.polygon.io:443/v3/reference/options/contracts?limit=5",
    "https://API.POLYGON.IO/v2/aggs/ticker/SPY/prev",
]
HOSTILE_URLS = [
    "http://api.polygon.io/v3/snapshot/options/SPY",               # plain http
    "https://evil.example.com/v3/snapshot/options/SPY",           # other host
    "https://api.polygon.io.evil.example.com/v3/x",               # suffix trick
    "https://evil.example.com/?next=https://api.polygon.io/v3/x",  # polygon only in the query
    "https://evil.com\\@api.polygon.io/v3/x",   # PARSER DIFFERENTIAL: urllib->polygon, requests->evil.com
    "https://api.polygon.io\\@evil.com/v3/x",   # reverse differential
    "https://evil.com@api.polygon.io/v3/x",     # userinfo
    "https://api.polygon.io@evil.com/v3/x",     # userinfo pointing elsewhere
    "https://api.polygon.io%2F@evil.com/",      # encoded slash + userinfo
    "https://api.polygon.io\t/x",               # control char: parsers disagree
    " https://api.polygon.io/v3/x",             # leading whitespace
    "https://api.polygon.io./v3/x",             # trailing-dot FQDN
    "https://api.polygon.io:8443/v3/x",         # non-443 port
    "https://api\u3002polygon\u3002io/v3/x",    # unicode ideographic full stops (IDNA tricks)
    "ftp://api.polygon.io/x",
    "",
]


@pytest.mark.parametrize("url", ACCEPTED_URLS)
def test_polygon_urls_are_accepted(asc, ab, url):
    assert asc._is_polygon_url(url) and ab._is_polygon_url(url)


@pytest.mark.parametrize("url", HOSTILE_URLS)
def test_scanner_refuses_to_send_key_off_polygon(asc, monkeypatch, url):
    calls = _capture_get(monkeypatch, asc)
    monkeypatch.setattr(asc, "POLYGON_KEY", FAKE_KEY)
    with pytest.raises(ValueError) as ei:
        asc._polygon_get(url)
    assert not calls, "a request was sent to a non-Polygon URL"
    assert FAKE_KEY not in str(ei.value) and (not url or url not in str(ei.value))


@pytest.mark.parametrize("url", HOSTILE_URLS)
def test_backtest_refuses_to_send_key_off_polygon(ab, monkeypatch, url):
    calls = _capture_get(monkeypatch, ab)
    with pytest.raises(ValueError):
        ab._polygon_request(url, FAKE_KEY)
    assert not calls


def _fuzz_urls():
    hosts = ["api.polygon.io", "evil.com", "api.polygon.io.evil.com", "API.POLYGON.IO", "api.polygon.io."]
    seps = ["", "\\", "@", "\\@", "%40", "#", "?", "/", ":443", ":8443", "\t", " ", "%2F", "\u3002", ".", ";"]
    tails = ["", "api.polygon.io", "evil.com"]
    return sorted({"%s%s%s%s/v3/x" % (sch, h, s, t) for sch in ("https://", "http://", "HTTPS://")
                   for h in hosts for s in seps for t in tails})


def test_every_accepted_url_really_connects_to_polygon(asc):
    """Property: for EVERY fuzzed URL the allow-list accepts, the host requests actually connects to
    (urllib3, after requests' own preparation) is api.polygon.io over https."""
    accepted = 0
    for u in _fuzz_urls():
        if asc._is_polygon_url(u):
            accepted += 1
            prepared = parse_url(requests.Request("GET", u).prepare().url)
            assert prepared.scheme == "https" and (prepared.host or "").lower() == "api.polygon.io", u
    assert accepted > 0


def test_scanner_and_backtest_host_checks_agree(asc, ab):
    for u in _fuzz_urls() + HOSTILE_URLS + ACCEPTED_URLS:
        assert asc._is_polygon_url(u) == ab._is_polygon_url(u), u


# ================================= (4) alert control: owner's local session only, fail-closed
class _OptStub:
    def __init__(self, value=None, raises=False):
        self.value, self.raises = value, raises

    def __call__(self, key):
        if self.raises:
            raise RuntimeError("no runtime")
        assert key == "server.address"
        return self.value


LOCAL_PATH = r"C:\Users\someone\AlphaScanner\alpha_scanner.py"
CLOUD_PATH = "/mount/src/tradehub/alpha_scanner.py"


@pytest.mark.parametrize("path,address,expected", [
    (LOCAL_PATH, "localhost", True),
    (LOCAL_PATH, "127.0.0.1", True),
    (LOCAL_PATH, "::1", True),
    (LOCAL_PATH, "LocalHost", True),
    (LOCAL_PATH, "", False),            # default: listens on ALL interfaces -> reachable -> public
    (LOCAL_PATH, None, False),
    (LOCAL_PATH, "0.0.0.0", False),
    (LOCAL_PATH, "192.168.1.131", False),
    (CLOUD_PATH, "localhost", False),   # Community Cloud is never local, even if bound to loopback
    (CLOUD_PATH, "", False),
])
def test_session_is_local_matrix(asc, monkeypatch, path, address, expected):
    monkeypatch.setattr(asc, "__file__", path)
    monkeypatch.setattr(asc.st, "get_option", _OptStub(address), raising=False)
    assert asc._session_is_local() is expected


def test_session_is_local_fails_closed_when_option_unavailable(asc, monkeypatch):
    monkeypatch.setattr(asc, "__file__", LOCAL_PATH)
    monkeypatch.setattr(asc.st, "get_option", _OptStub(raises=True), raising=False)
    assert asc._session_is_local() is False


@pytest.mark.parametrize("configured,is_local,mode", [
    (False, False, "unconfigured"),
    (False, True, "unconfigured"),    # no secrets -> never controllable, even locally
    (True, False, "owner-locked"),    # public session -> no control
    (True, True, "local-toggle"),     # the ONLY case with a toggle
])
def test_alerts_control_mode_truth_table(asc, configured, is_local, mode):
    assert asc._alerts_control_mode(configured, is_local) == mode


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("false", False), ("False", False), (False, False), ("0", False),
    ("no", False), ("off", False), ("ture", False), ("enabled", False),
    (True, True), ("true", True), (" TRUE ", True), ("1", True), ("yes", True), ("on", True), ("On", True),
])
def test_alerts_flag_parser_fails_closed(asc, value, expected):
    assert asc._truthy_flag(value) is expected


GUARD = ast.dump(ast.parse('_alert_mode == "local-toggle"', mode="eval").body)


def toggle_guard_problems(src):
    tree = ast.parse(src)
    probs = []
    keys = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "enable_alerts_sidebar"]
    if len(keys) != 1:
        probs.append("expected exactly one alerts widget key, found %d" % len(keys))
    guard_if = None
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and any(isinstance(c, ast.Constant) and c.value == "enable_alerts_sidebar"
                                         for s in n.body for c in ast.walk(s)):
            guard_if = n                     # walk is outer-first: the last match is the innermost
    if guard_if is None or ast.dump(guard_if.test) != GUARD:
        probs.append("alerts widget is not guarded by exactly `_alert_mode == \"local-toggle\"`")
    inside = {id(x) for s in (guard_if.body if guard_if else []) for x in ast.walk(s)}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and id(n) not in inside:
            tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(x == "enable_alerts" for t in tgts for x in _target_names(t)):
                v = n.value
                if not ((isinstance(v, ast.Name) and v.id == "ALERTS_ENABLED_DEFAULT")
                        or (isinstance(v, ast.Constant) and v.value is False)):
                    probs.append("line %d: enable_alerts set from `%s` outside the local-toggle branch"
                                 % (n.lineno, ast.unparse(v)[:50]))
    return probs


def test_alerts_toggle_only_in_the_local_toggle_branch():
    assert toggle_guard_problems(_read(SCANNER)) == []


@pytest.mark.parametrize("snippet", [
    'if _alert_mode != "local-toggle":\n    enable_alerts = st.checkbox("x", key="enable_alerts_sidebar")',  # inverted
    'if _alert_mode in ("local-toggle", "owner-locked"):\n    enable_alerts = st.checkbox("x", key="enable_alerts_sidebar")',
    'if _alert_mode == "local-toggle":\n    enable_alerts = st.checkbox("x", key="enable_alerts_sidebar")\n'
    'else:\n    enable_alerts = st.toggle("y")',                     # second, differently keyed widget
    'if _alert_mode == "local-toggle":\n    enable_alerts = st.checkbox("x", key="enable_alerts_sidebar")\n'
    'else:\n    enable_alerts = _prefs.get("enable_alerts")',       # shared prefs file in a public session
    'enable_alerts = st.checkbox("x", key="enable_alerts_sidebar")',  # unguarded
])
def test_toggle_guard_detector_catches_mutations(snippet):
    assert toggle_guard_problems(snippet), "toggle-guard detector missed: " + snippet


def test_toggle_guard_detector_accepts_the_intended_shape():
    ok = ('if _alert_mode == "local-toggle":\n    enable_alerts = st.checkbox("x", key="enable_alerts_sidebar")\n'
          'elif _alert_mode == "owner-locked":\n    enable_alerts = ALERTS_ENABLED_DEFAULT\n'
          'else:\n    enable_alerts = False')
    assert toggle_guard_problems(ok) == []


WIDGET_CALLS = {"button", "form_submit_button", "checkbox", "toggle", "radio", "selectbox", "multiselect",
                "text_input", "text_area", "number_input", "slider", "select_slider", "date_input",
                "time_input", "file_uploader", "color_picker", "pills", "segmented_control", "chat_input",
                "data_editor", "download_button", "link_button", "feedback"}


def unguarded_sends(src):
    """Every send_telegram_alert(...) must sit in the BODY of an `if enable_alerts and ...` and must not be
    triggered by an inline widget (a button etc.) in any enclosing condition."""
    tree = ast.parse(src)
    parents = {id(c): p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and _callee_name(n.func) == "send_telegram_alert"):
            continue
        gated = ui_triggered = in_own_def = False
        child, p = n, parents.get(id(n))
        while p is not None:
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) and p.name == "send_telegram_alert":
                in_own_def = True
            if isinstance(p, (ast.If, ast.While, ast.IfExp)):
                if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in WIDGET_CALLS
                       for c in ast.walk(p.test)):
                    ui_triggered = True
                in_body = any(child is s for s in (p.body if isinstance(p.body, list) else [p.body]))
                if (in_body and isinstance(p.test, ast.BoolOp) and isinstance(p.test.op, ast.And)
                        and any(isinstance(v, ast.Name) and v.id == "enable_alerts" for v in p.test.values)):
                    gated = True
            child, p = p, parents.get(id(p))
        if not in_own_def and (not gated or ui_triggered):
            out.append((n.lineno, "gated" if gated else "UNGATED", "widget-triggered" if ui_triggered else ""))
    return out


def test_every_telegram_send_is_gated_by_enable_alerts():
    assert unguarded_sends(_read(SCANNER)) == []


@pytest.mark.parametrize("snippet", [
    'send_telegram_alert("x")',                                                       # ungated
    'if st.sidebar.button("Test Telegram"):\n    send_telegram_alert("x")',            # renamed test button
    'if st.button("Test") and enable_alerts:\n    send_telegram_alert("x")',          # widget-triggered
    'if not enable_alerts:\n    pass\nelse:\n    send_telegram_alert("x")',           # else-branch
    'if enable_alerts or True:\n    send_telegram_alert("x")',                        # Or, not And
    'if TELEGRAM_BOT_TOKEN:\n    send_telegram_alert("x")',                            # credentials-only gate
])
def test_send_gate_detector_catches_mutations(snippet):
    assert unguarded_sends(snippet), "send-gate detector missed: " + snippet


def test_send_gate_detector_accepts_the_intended_shape():
    ok = ('if enable_alerts and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:\n'
          '    if send_telegram_alert(msg):\n        pass')
    assert unguarded_sends(ok) == []


def test_alert_mode_is_wired_to_secrets_and_session():
    tree = ast.parse(_read(SCANNER))
    want = ast.dump(ast.parse("_alerts_control_mode(TELEGRAM_CONFIGURED, _session_is_local())", mode="eval").body)
    assigns = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
               and any(x == "_alert_mode" for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
                       for x in _target_names(t))]
    assert len(assigns) == 1 and ast.dump(assigns[0].value) == want, \
        "_alert_mode must be set exactly once, from _alerts_control_mode(TELEGRAM_CONFIGURED, _session_is_local())"


_REEXEC = itertools.count()


@pytest.mark.parametrize("env,address,mode,alerts_on", [
    ({}, "localhost", "unconfigured", False),
    ({"TELEGRAM_BOT_TOKEN": "111:FAKE", "TELEGRAM_CHAT_ID": "42"}, "localhost", "local-toggle", False),
    ({"TELEGRAM_BOT_TOKEN": "111:FAKE", "TELEGRAM_CHAT_ID": "42"}, "", "owner-locked", False),
    ({"TELEGRAM_BOT_TOKEN": "111:FAKE", "TELEGRAM_CHAT_ID": "42", "TELEGRAM_ALERTS_ENABLED": "true"},
     "0.0.0.0", "owner-locked", True),
    ({"TELEGRAM_BOT_TOKEN": "111:FAKE", "TELEGRAM_CHAT_ID": "42", "TELEGRAM_ALERTS_ENABLED": "ture"},
     "", "owner-locked", False),
    ({"TELEGRAM_BOT_TOKEN": "111:FAKE"}, "localhost", "unconfigured", False),   # half-configured
])
def test_real_module_wiring_behaviour(ab, monkeypatch, tmp_path, env, address, mode, alerts_on):
    """Re-execute the REAL alpha_scanner.py (offline stubs; auto-scan skipped) with controlled secrets and
    server address, and check the module-level alert wiring end to end. Any network call fails the test."""
    st_stub = sys.modules["streamlit"]
    monkeypatch.setattr(st_stub, "get_option", _OptStub(address), raising=False)
    for k in SECRET_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    network = []

    def _no_network(*a, **k):
        network.append(a[:1])
        raise AssertionError("network call while importing the scanner")

    monkeypatch.setattr(requests, "get", _no_network)
    monkeypatch.setattr(requests, "post", _no_network)
    monkeypatch.chdir(tmp_path)
    name = "alpha_scanner_reexec_%d" % next(_REEXEC)
    spec = importlib.util.spec_from_file_location(name, SCANNER)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    assert mod._alert_mode == mode
    if mode != "local-toggle":
        assert mod.enable_alerts is alerts_on
    assert not network


# ======================================================== (5) local launcher binds to loopback only
def test_tradehub_launcher_binds_loopback():
    bat = open(os.path.join(ROOT, "tradehub.bat"), encoding="utf-8", errors="replace").read()
    assert "--server.address localhost" in bat
