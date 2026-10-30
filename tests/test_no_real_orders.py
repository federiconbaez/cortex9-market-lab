"""Safety tests: this project must never be able to place a real order."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from conftest import FakeExchange
from cortex9_market_lab.data.exchange import (
    PUBLIC_METHODS,
    ForbiddenExchangeCallError,
    MarketDataClient,
    PublicOnlyExchange,
    _public,
    default_factory,
)
from cortex9_market_lab.data.feed import PollingFeed

SRC = Path(__file__).resolve().parents[1] / "src" / "cortex9_market_lab"

# ccxt methods that trade, move funds or need credentials (snake_case and camelCase)
FORBIDDEN = re.compile(
    r"create_?\w*order|cancel_?\w*order|edit_?order|fetch_?balance|fetch_?my_?trades|"
    r"fetch_?(open|closed)?_?orders?|withdraw|transfer|set_?leverage|"
    r"private\w*|api_?key|secret|password",
    re.IGNORECASE,
)

# credential attribute names that the guard in default_factory is allowed to mention
CREDENTIAL_GUARD = {"apiKey", "secret", "password", "privateKey"}

# the only functions allowed to look up attributes by name
GETATTR_ALLOWED = {
    "default_factory",  # getattr(ccxt, exchange_id) and the credential guard
    "_public",  # the whitelist path
    "PublicOnlyExchange.__getattribute__",  # runtime guard forwarding whitelisted names
}
DYNAMIC_LOOKUPS = {"getattr", "__getattribute__", "attrgetter", "methodcaller", "vars"}


def _fold(node: ast.AST) -> str | None:
    """Evaluate string constants and '+' concatenations / f-strings of constants."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left), _fold(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.FormattedValue):
        return _fold(node.value)
    if isinstance(node, ast.JoinedStr):
        parts = [_fold(v) for v in node.values]
        if all(p is not None for p in parts):
            return "".join(p for p in parts if p is not None)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and node.args
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ):
        sep = _fold(node.func.value)
        parts = [_fold(e) for e in node.args[0].elts]
        if sep is not None and all(p is not None for p in parts):
            return sep.join(p for p in parts if p is not None)
    return None


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def scan_forbidden(source: str) -> set[str]:
    """Identifiers, attribute names and (folded) string values matching FORBIDDEN."""
    tree = ast.parse(source)
    docs = _docstring_ids(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Attribute):
            candidates.append(node.attr)
        elif isinstance(node, ast.Name):
            candidates.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            candidates.append(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            candidates.append(node.arg)
        elif isinstance(node, (ast.Constant, ast.BinOp, ast.JoinedStr, ast.Call)):
            if id(node) in docs:
                continue
            folded = _fold(node)
            if folded is not None:
                candidates.append(folded)
        found.update(c for c in candidates if FORBIDDEN.search(c))
    return found


def scan_dynamic_lookups(source: str) -> set[str]:
    """Qualified names of functions that call getattr-like helpers."""
    tree = ast.parse(source)
    hits: set[str] = set()

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, [*scope, child.name])
                continue
            if isinstance(child, ast.Call):
                fn = child.func
                name = (
                    fn.id
                    if isinstance(fn, ast.Name)
                    else (fn.attr if isinstance(fn, ast.Attribute) else None)
                )
                if name in DYNAMIC_LOOKUPS:
                    hits.add(".".join(scope) or "<module>")
            visit(child, scope)

    visit(tree, [])
    return hits


INTROSPECTION_CALLS = {"eval", "exec", "compile", "getattr_static", "__import__"}
INTROSPECTION_ATTRS = {"__dict__", "__get__"}
WRAPPER_CLASS = "PublicOnlyExchange"
WRAPPER_SLOT = "_inner"

# legitimate uses in the code base, listed explicitly as (file, scope, token)
INTROSPECTION_ALLOWED = {
    ("logging.py", "<module>", "__dict__"),  # reserved LogRecord fields
    ("logging.py", "JsonFormatter.format", "__dict__"),  # extra fields of a LogRecord
}


def scan_introspection(source: str) -> set[tuple[str, str]]:
    """(scope, token) pairs for eval/exec/compile/getattr_static/__import__ calls,
    ``__dict__`` / ``__get__`` access, and any reference to the wrapper's private slot
    outside ``PublicOnlyExchange``."""
    tree = ast.parse(source)
    docs = _docstring_ids(tree)
    hits: set[tuple[str, str]] = set()

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, [*scope, child.name])
                continue
            where = ".".join(scope) or "<module>"
            if isinstance(child, ast.Call):
                fn = child.func
                name = (
                    fn.id
                    if isinstance(fn, ast.Name)
                    else (fn.attr if isinstance(fn, ast.Attribute) else None)
                )
                if name in INTROSPECTION_CALLS:
                    hits.add((where, name))
            if isinstance(child, ast.Attribute) and child.attr in INTROSPECTION_ATTRS:
                hits.add((where, child.attr))
            inside_wrapper = bool(scope) and scope[0] == WRAPPER_CLASS
            if not inside_wrapper and id(child) not in docs:
                if (isinstance(child, ast.Attribute) and child.attr == WRAPPER_SLOT) or (
                    isinstance(child, ast.Name) and child.id == WRAPPER_SLOT
                ):
                    hits.add((where, WRAPPER_SLOT))
                elif isinstance(child, (ast.Constant, ast.BinOp, ast.JoinedStr)):
                    folded = _fold(child)
                    if folded is not None and WRAPPER_SLOT in folded:
                        hits.add((where, WRAPPER_SLOT))
            visit(child, scope)

    visit(tree, [])
    return hits


def _sources() -> list[Path]:
    files = sorted(SRC.rglob("*.py"))
    assert files, "no sources found"
    return files


# -- tests on the real source tree ------------------------------------------------------
def test_public_whitelist_is_market_data_only():
    assert frozenset({"load_markets", "fetch_ohlcv", "fetch_ticker"}) == PUBLIC_METHODS


def test_source_has_no_order_or_credential_code_path():
    offenders = {}
    for path in _sources():
        hits = scan_forbidden(path.read_text(encoding="utf-8"))
        if path.name == "exchange.py":
            hits -= CREDENTIAL_GUARD
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


def test_dynamic_attribute_lookup_only_in_whitelist_path():
    seen: set[str] = set()
    for path in _sources():
        hits = scan_dynamic_lookups(path.read_text(encoding="utf-8"))
        if path.name != "exchange.py":
            assert hits == set(), f"{path.name}: dynamic lookup in {hits}"
        seen |= hits
    assert seen <= GETATTR_ALLOWED, seen - GETATTR_ALLOWED


def test_no_introspection_outside_explicit_whitelist():
    found = set()
    for path in _sources():
        for scope, token in scan_introspection(path.read_text(encoding="utf-8")):
            found.add((path.name, scope, token))
    assert found == INTROSPECTION_ALLOWED, found ^ INTROSPECTION_ALLOWED


# -- mutation tests: the scanners must catch evasions ---------------------------------
MUTANTS = {
    "camel_create": "def f(ex):\n    return ex.createOrder('BTC/USDT', 'market', 'buy', 1)\n",
    "camel_balance": "def f(ex):\n    return ex.fetchBalance()\n",
    "concat_getattr": "def f(ex):\n    return getattr(ex, 'create' + '_order')('X', 'm', 'b', 1)\n",
    "fstring_getattr": "def f(ex, k='order'):\n    return getattr(ex, f'cancel_{\"order\"}')(1)\n",
    "join_getattr": "def f(ex):\n    return getattr(ex, ''.join(['fetch', 'Balance']))()\n",
    "cancel_all": "def f(ex):\n    ex.cancel_all_orders()\n",
    "edit": "def f(ex):\n    ex.editOrder(1)\n",
    "my_trades": "def f(ex):\n    ex.fetchMyTrades()\n",
    "open_orders": "def f(ex):\n    ex.fetch_open_orders()\n",
    "withdraw": "def f(ex):\n    ex.withdraw('USDT', 1, 'addr')\n",
    "transfer": "def f(ex):\n    ex.transfer('USDT', 1, 'spot', 'funding')\n",
    "leverage": "def f(ex):\n    ex.setLeverage(5)\n",
    "api_key_kw": "import ccxt\nex = ccxt.okx({'apiKey': 'k'})\n",
    "private_endpoint": "def f(ex):\n    ex.privatePostTradeOrder({})\n",
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_scanner_flags_mutants(name, tmp_path):
    mod = tmp_path / f"mutant_{name}.py"
    mod.write_text(MUTANTS[name], encoding="utf-8")
    assert scan_forbidden(mod.read_text(encoding="utf-8")), f"scanner missed {name}"


def test_dynamic_lookup_scanner_flags_getattr_outside_whitelist(tmp_path):
    mod = tmp_path / "mutant_getattr.py"
    mod.write_text(
        "class Sneaky:\n"
        "    def go(self, ex, n):\n"
        "        return getattr(ex, n)()\n"
        "def other(ex):\n"
        "    import operator\n"
        "    return operator.methodcaller('x')(ex)\n",
        encoding="utf-8",
    )
    assert scan_dynamic_lookups(mod.read_text(encoding="utf-8")) == {"Sneaky.go", "other"}


INTROSPECTION_MUTANTS = {
    "eval": ("def f(ex):\n    return eval('ex.create_' + 'order')\n", "eval"),
    "exec": ("def f(ex):\n    exec('ex.x()')\n", "exec"),
    "compile": ("def f():\n    return compile('1', 'm', 'eval')\n", "compile"),
    "getattr_static": (
        "import inspect\ndef f(w):\n    return inspect.getattr_static(w, 'x')\n",
        "getattr_static",
    ),
    "dunder_import": ("def f():\n    return __import__('os')\n", "__import__"),
    "dict_access": ("def f(w):\n    return w.__dict__\n", "__dict__"),
    "descriptor_get": (
        "def f(w, t):\n    return t.__dict__['x'].__get__(w, t)\n",
        "__get__",
    ),
    "inner_attr": ("def f(w):\n    return w._inner\n", "_inner"),
    "inner_string": (
        "def f(w):\n    return object.__getattribute__(w, '_in' + 'ner')\n",
        "_inner",
    ),
    "inner_name": ("_inner = None\n", "_inner"),
}


@pytest.mark.parametrize("name", sorted(INTROSPECTION_MUTANTS))
def test_introspection_scanner_flags_mutants(name, tmp_path):
    code, token = INTROSPECTION_MUTANTS[name]
    mod = tmp_path / f"mutant_{name}.py"
    mod.write_text(code, encoding="utf-8")
    tokens = {t for _, t in scan_introspection(mod.read_text(encoding="utf-8"))}
    assert token in tokens, f"scanner missed {name}: {tokens}"


def test_inner_slot_allowed_only_inside_wrapper_class():
    inside = (
        "class PublicOnlyExchange:\n"
        "    def __init__(self, i):\n"
        "        object.__setattr__(self, '_inner', i)\n"
    )
    outside = "class Other:\n    def g(self, w):\n        return w._inner\n"
    assert scan_introspection(inside) == set()
    assert scan_introspection(outside) == {("Other.g", "_inner")}


def test_scanner_ignores_docstrings_but_not_code():
    src = '"""We never call create_order here."""\nx = 1\n'
    assert scan_forbidden(src) == set()
    assert scan_forbidden(src + "y = 'create_order'\n") == {"create_order"}


# -- runtime guards ----------------------------------------------------------------------
class StrictExchange(FakeExchange):
    """Fails the test on any method that is not a public market-data method."""

    def __getattribute__(self, name):
        if (
            not name.startswith("_")
            and callable(object.__getattribute__(self, name))
            and name not in PUBLIC_METHODS
        ):
            raise AssertionError(f"client used non-public method {name}")
        return object.__getattribute__(self, name)

    def create_order(self, *a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("create_order called")


def test_client_and_feed_only_touch_public_methods(frames):
    fx = StrictExchange(frames)
    client = MarketDataClient(["okx"], factory=lambda _: fx, sleep=lambda s: None)
    client.connect()
    client.fetch_ohlcv_range("AAA/USDT", "1h", 0, 10**13)
    client.fetch_last_price("AAA/USDT")
    feed = PollingFeed(client, ["AAA/USDT"], "1h", window=20)
    feed.poll_bars(frames["AAA/USDT"].index[-1])
    assert {c[0] for c in fx.calls} <= PUBLIC_METHODS


def test_client_exposes_no_public_exchange_handle(frames):
    client = MarketDataClient(["okx"], factory=lambda _: FakeExchange(frames))
    client.connect()
    public_attrs = {a for a in vars(client) if not a.startswith("_")}
    assert public_attrs == {"exchanges", "policy", "metrics", "active_exchange"}
    assert not hasattr(client, "client")
    assert not hasattr(client, "exchange")


@pytest.mark.parametrize(
    "name",
    [
        "create_order",
        "createOrder",
        "cancel_order",
        "fetchBalance",
        "withdraw",
        "apiKey",
        "_inner",
        "__dict__",
    ],
)
def test_wrapper_refuses_non_whitelisted_attributes(frames, name):
    wrapped = PublicOnlyExchange(FakeExchange(frames))
    with pytest.raises(ForbiddenExchangeCallError):
        getattr(wrapped, name)
    with pytest.raises(ForbiddenExchangeCallError):
        wrapped.anything = 1
    assert wrapped.fetch_ticker("AAA/USDT")["last"] > 0


def test_non_whitelisted_method_is_refused(frames):
    fx = FakeExchange(frames)
    for name in ("create_order", "createOrder", "cancel_order", "fetch_balance", "withdraw"):
        with pytest.raises(ForbiddenExchangeCallError):
            _public(fx, name)
    client = MarketDataClient(["okx"], factory=lambda _: fx, sleep=lambda s: None)
    with pytest.raises(ForbiddenExchangeCallError):
        client._call("x", "create_order", "AAA/USDT", "market", "buy", 1.0)


def test_default_factory_builds_unauthenticated_client():
    ex = default_factory("okx")  # constructing a ccxt instance does no network I/O
    for attr in ("apiKey", "secret", "password"):
        assert not ex.__dict__.get(attr)
    with pytest.raises(Exception):  # noqa: B017 - ccxt raises AuthenticationError
        ex.check_required_credentials()
