"""Generate ONE merged, self-contained module from a pipeline's module closure.

Steps per group:
  1. parse every module in the closure
  2. harvest top-level defs/assigns; record re-export aliases
  3. rewrite cross-module references so they resolve inside the single file:
       - `from ssq_x import NAME`            -> drop; NAME is now local
       - `from ssq_x import NAME as _alias`  -> drop; mark _alias -> NAME
       - `import ssq_x as _x`                -> drop; mark _x -> module stem
       - `_x.NAME` / `_alias.NAME`           -> `NAME`
  4. drop pure compatibility wrappers that only delegate to a concrete impl
  5. detect collisions and report/stub them for manual reconciliation
"""

from __future__ import annotations

import ast
import collections
import copy
import importlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repository root
PY = ROOT / "py"
sys.path.insert(0, str(PY))
LOCAL = {p.stem for p in PY.glob("ssq_*.py")}

ENTRIES = {
    "data_prep": "ssq_data_processor",
    "analysis": "ssq_analyzer",
    "bonus_check": "ssq_bonus_calculation",
}

# Modules whose same-named definitions are compatibility wrappers that bind DI
# dependencies for a concrete implementation living in another module. When a
# name collides, the concrete module wins; alias rewriting then redirects the
# wiring to the concrete callable.
DELEGATOR_MODULES = {
    "ssq_analyzer",
    "ssq_backtesting",
    "ssq_candidates",
    "ssq_console",
    "ssq_data_processor",
    "ssq_bonus_calculation",
    "ssq_ranking",
    "ssq_reporting",
    "ssq_rule_registry",
    "ssq_selection",
    "ssq_training",
    "ssq_rules",
    "ssq_workflow",
}


def local_deps(mod: str) -> set[str]:
    """Local modules imported by `mod` (static, ignores TYPE_CHECKING guards)."""
    path = PY / f"{mod}.py"
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    deps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in LOCAL:
                    deps.add(root)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            root = (node.module or "").split(".")[0]
            if root in LOCAL:
                deps.add(root)
    return deps


def closure(entry):
    """Static transitive closure over local module imports.

    Runtime `sys.modules` is NOT the right notion here: some modules (e.g.
    ssq_domain) are only pulled in under `if TYPE_CHECKING`, yet their
    constants are read at import time by other modules, so they must be merged
    too. Walk the static import graph instead.
    """
    seen: set[str] = set()
    stack = [entry]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(d for d in local_deps(cur) if d not in seen)
    return sorted(seen)


class ModuleInfo:
    def __init__(self, name):
        self.name = name
        self.path = PY / f"{name}.py"
        self.src = self.path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.src)
        self.defs = {}          # name -> node
        self.aliases = {}       # local alias -> concrete name
        self.imported_names = set()  # local names bound by `from local_mod import`
        self.modrefs = {}       # alias -> target module stem
        self.plain_imports = []  # (module, [names]) from `from mod import n1,n2`
        self.star = False
        self.other = []         # non-import, non-def statements
        self._scan()

    def _scan(self):
        # Walk the WHOLE tree, not just module level: imports nested inside
        # `if TYPE_CHECKING:` / `try:` blocks are still real imports that must be
        # hoisted, and missing them produced NameError at runtime.
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                non_local = [a for a in node.names if a.name not in LOCAL]
                for a in node.names:
                    if a.name in LOCAL:
                        self.modrefs[a.asname or a.name] = a.name
                if non_local:
                    hoisted = ast.Import(names=non_local)
                    self.plain_imports.append(ast.copy_location(hoisted, node))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                mod = node.module or ""
                if mod in LOCAL:
                    for a in node.names:
                        if a.name == "*":
                            self.star = True
                            continue
                        local = a.asname or a.name
                        self.aliases[local] = a.name
                        # remember that this local name came from an import, so
                        # `name.attr` may be a module-style attribute access.
                        # Without this, `PRIZE_RULES.get(...)` where PRIZE_RULES
                        # was imported from another module got mis-rewritten.
                        self.imported_names.add(local)
                else:
                    self.plain_imports.append(node)

        # top-level definitions and alias assignments (needed for emission)
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.defs[node.name] = node
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Name
            ):
                tgt = node.targets[0].id
                v = node.value
                if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name):
                    owner = v.value.id
                    if owner in self.modrefs or (
                        owner in self.aliases and owner not in self.imported_names
                    ):
                        self.aliases[tgt] = v.attr
                    else:
                        self.defs[tgt] = node
                else:
                    self.defs[tgt] = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                self.defs[node.target.id] = node
            else:
                self.other.append(node)


class Rewriter(ast.NodeTransformer):
    """`_x.NAME` -> `NAME`; alias -> concrete name.

    Note: public-name -> private-impl renaming is deliberately NOT done here.
    Only the folded wrapper's own body may call the private impl; wiring such as
    `prepare_history=prepare_history` in default_analysis_dependencies must keep
    referring to the public wrapper.
    """

    def __init__(self, info: ModuleInfo, known: set[str], renames: dict | None = None):
        self.info = info
        self.known = known  # every name available in the merged file
        self.renames = renames or {}

    def _sub(self, ident: str) -> str:
        return self.renames.get(ident, ident)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            owner = node.value.id
            # Case 1: `import ssq_x as _x` -> `_x.NAME` is always an attribute
            # of that module, so hoist it to a bare NAME.
            # Case 2: `from ssq_x import NAME as _n` -> `_n.attr` is only a
            # module attribute access if `_n` really is a module; an imported
            # *object* such as PRIZE_RULES must keep its own `.get(...)`.
            is_modref = owner in self.info.modrefs
            is_module_alias = (
                owner in self.info.aliases and owner not in self.info.imported_names
            )
            if is_modref or is_module_alias:
                return ast.copy_location(
                    ast.Name(id=self._sub(node.attr), ctx=node.ctx), node
                )
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id in self.info.aliases:
                concrete = self.info.aliases[node.id]
                return ast.copy_location(ast.Name(id=concrete, ctx=node.ctx), node)
            if node.id in self.renames:
                return ast.copy_location(
                    ast.Name(id=self.renames[node.id], ctx=node.ctx), node
                )
        return node


def unparse(node):
    return ast.unparse(node)


def param_names(fn) -> list[str]:
    a = fn.args
    return [p.arg for p in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)]


def required_params(fn) -> set[str]:
    a = fn.args
    names = [p.arg for p in list(a.posonlyargs) + list(a.args)]
    n_defaults = len(a.defaults)
    required = set(names[: len(names) - n_defaults]) if n_defaults else set(names)
    for p, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is None:
            required.add(p.arg)
    return required


def apply_param_defaults(fn, defaults: dict):
    """Deprecated: superseded by delegator folding (see `plan_folding`).

    Changing an existing required parameter's default in place is not possible
    without either making it keyword-only (which breaks positional call sites)
    or filling earlier parameters with placeholder defaults. Kept as a guard so
    the config cannot silently reintroduce that mistake.
    """
    raise SystemExit(
        "apply_param_defaults is disabled; use _overrides.json + delegator folding"
    )



def callable_with(delegator, concrete) -> bool:
    """Can `concrete` be called using only the delegator's parameter names?

    Merged callers use the *delegator's* (narrower) signature, so the concrete
    function must accept exactly those names to serve them directly.
    """
    dnames = set(param_names(delegator))
    cnames = set(param_names(concrete))
    return required_params(concrete) <= dnames and dnames <= cnames


def plan_folding(delegator, concrete, concrete_module, delegator_info=None) -> dict | None:
    """Work out the merged public signature for a delegator/concrete pair.

    The old compatibility wrapper existed only to supply values the concrete
    function required. Instead of keeping a wrapper (which risks dropping the
    caller's own arguments), we emit the CONCRETE body with a merged signature:

      * parameters the delegator declares keep their requirement status, so both
        `f(data, dir, deps)` and `f(data)` style call sites work;
      * parameters the delegator fills in from its call become defaults taken
        from that call expression;
      * remaining parameters keep the concrete function's own defaults.

    Returns {"defaults": {param: source_expression}} or None when the concrete
    signature already serves every caller.
    """
    if not isinstance(delegator, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    if not isinstance(concrete, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    def resolves_to_concrete(n) -> bool:
        if not isinstance(n, ast.Call):
            return False
        f = n.func
        if isinstance(f, ast.Name):
            if f.id == concrete.name:
                return True
            return bool(delegator_info) and delegator_info.aliases.get(f.id) == concrete.name
        if isinstance(f, ast.Attribute):
            return f.attr == concrete.name
        return False

    calls = [n for n in ast.walk(delegator) if resolves_to_concrete(n)]
    if not calls:
        return None
    if callable_with(delegator, concrete):
        return None  # concrete alone serves every caller

    call = calls[0]
    cnames = param_names(concrete)
    dnames = set(param_names(delegator))

    # values the delegator passes into the concrete call
    supplied: dict[str, ast.expr] = {}
    for i, arg in enumerate(call.args):
        if i < len(cnames):
            supplied[cnames[i]] = arg
    for kw in call.keywords:
        if kw.arg:
            supplied[kw.arg] = kw.value

    defaults: dict[str, str] = {}
    for pname in cnames:
        if pname in dnames:
            continue  # the caller supplies this one
        if pname not in supplied:
            continue
        expr = supplied[pname]
        # The delegator may simply forward its own parameter of the same name
        # (`dependencies=dependencies`). That is not a default value; recording
        # it would emit the self-referential `dependencies=dependencies`.
        if isinstance(expr, ast.Name) and expr.id == pname:
            continue
        # Keep a neutral default on the signature; the delegator's wrapped value
        # is applied inside the body so it resolves at CALL time. This preserves
        # the old wrapper's patchability (tests patch module attributes such as
        # ssq_bonus_workflow.validate_draw_dates_not_future).
        defaults[pname] = "None"
    if not defaults:
        return None
    injected = {
        pname: ast.unparse(supplied[pname])
        for pname in defaults
        if pname in supplied
    }
    return {"defaults": defaults, "injected": injected}


def merge_signature(concrete, delegator, injected: dict[str, str]):
    """Build the merged signature for a folded function.

    Structure is taken from the delegator (the wrapper callers were written
    against) so its requirement/default layout is preserved exactly. Any
    parameter the concrete implementation additionally needs is appended as
    keyword-only, defaulting to the expression the wrapper used to pass, with a
    neutral `None` default on the signature plus a call-time assignment in the
    body (see `apply_call_time_defaults`) so patching still works.
    """
    fn = copy.deepcopy(delegator)
    fn.name = concrete.name
    fn.body = copy.deepcopy(concrete.body)

    # carry over annotations/returns from the implementation where available
    fn.returns = copy.deepcopy(concrete.returns)
    fn.decorator_list = copy.deepcopy(concrete.decorator_list) or copy.deepcopy(
        delegator.decorator_list
    )

    existing = set(param_names(fn))
    for pname in injected:
        if pname in existing:
            continue
        fn.args.kwonlyargs.append(ast.arg(arg=pname))
        fn.args.kw_defaults.append(ast.Constant(value=None))

    if len(fn.args.kw_defaults) != len(fn.args.kwonlyargs):
        raise SystemExit(f"kwonly/default arity mismatch merging {fn.name}")
    return fn


def check_undefined(rewritten: dict, import_lines: list[str]) -> list[str]:
    """Report module-level names that are read but neither defined nor imported.

    Catches the class of merge defect that otherwise only shows up as a
    NameError deep inside a run.
    """
    builtins_ok = set(dir(__builtins__)) if not isinstance(__builtins__, dict) else set(__builtins__)
    # `__file__` / `__name__` / `__doc__` are supplied by the interpreter
    builtins_ok |= {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__"}
    defined = set(rewritten)
    imported: set[str] = set()
    for line in import_lines:
        if line.startswith("import "):
            for part in line[len("import "):].split(","):
                part = part.strip()
                if " as " in part:
                    imported.add(part.split(" as ")[1].strip())
                else:
                    imported.add(part.split(".")[0].strip())
        elif line.startswith("from "):
            _, rest = line.split(" import ", 1)
            for part in rest.split(","):
                part = part.strip()
                imported.add(part.split(" as ")[-1].strip())

    missing: dict[str, list[str]] = {}
    for name, node in rewritten.items():
        for ref in global_refs(node):
            if ref in defined or ref in imported or ref in builtins_ok:
                continue
            missing.setdefault(ref, []).append(name)
    return [f"{ref} (read by {', '.join(sorted(set(users)))})" for ref, users in sorted(missing.items())]


def apply_call_time_defaults(fn, injected: dict[str, str]):
    """Insert `param = param if param is not None else <delegator expression>`
    at the top of the folded function body.

    The compatibility wrapper used to evaluate these expressions on every call.
    Freezing them as definition-time parameter defaults would break code that
    patches the module attribute (a widely used seam in this codebase's tests),
    so the lookup must stay at call time.
    """
    prelude: list[ast.stmt] = []
    for pname, expr in injected.items():
        test = ast.Compare(
            left=ast.Name(id=pname, ctx=ast.Load()),
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        )
        assign = ast.Assign(
            targets=[ast.Name(id=pname, ctx=ast.Store())],
            value=ast.IfExp(
                test=test,
                body=ast.Name(id=pname, ctx=ast.Load()),
                orelse=ast.parse(expr, mode="eval").body,
            ),
        )
        prelude.append(assign)
    if not prelude:
        return fn
    doc: list[ast.stmt] = []
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        doc = [body.pop(0)]
    fn.body = doc + prelude + body
    return fn


def parse_signature(spec: str) -> ast.arguments:
    """Parse a parameter list spec such as `a, b=CONST, c=None`."""
    stub = ast.parse(f"def _f({spec}): pass")
    return stub.body[0].args


def merge_signature_explicit(concrete, spec: str):
    """Apply an explicitly declared public signature to the concrete function.

    The merge cannot infer the correct public signature in every case (a
    compatibility wrapper sometimes narrowed the signature, sometimes supplied
    constants, sometimes had genuine defaults). Rather than guess, the few
    functions needing this are declared in `_signatures.json`; the
    implementation body is kept unchanged.
    """
    fn = copy.deepcopy(concrete)
    fn.args = parse_signature(spec)
    ast.fix_missing_locations(fn)
    return fn


def apply_call_defaults(fn, call_defaults: dict, sig_index: dict):
    """Fill in arguments at a call site.

    Some delegators forwarded a *late-bound* module value into the concrete
    function (`score_combination(..., RED_RULES)`). Inlining that as a callee
    default would freeze it at definition time, so the argument is supplied
    explicitly here instead. A parameter already filled positionally is left
    alone, otherwise the call would pass it twice.
    """
    if not call_defaults:
        return fn
    for call in ast.walk(fn):
        if not isinstance(call, ast.Call):
            continue
        f = call.func
        callee = (
            f.id
            if isinstance(f, ast.Name)
            else (f.attr if isinstance(f, ast.Attribute) else None)
        )
        wanted = call_defaults.get(callee)
        if not wanted:
            continue
        order = sig_index.get(callee)
        filled = {kw.arg for kw in call.keywords if kw.arg}
        if order:
            filled |= set(order[: len(call.args)])
        for pname, expr in wanted.items():
            if pname in filled:
                continue
            call.keywords.append(
                ast.keyword(arg=pname, value=ast.parse(expr, mode="eval").body)
            )
    return fn


def arg_param_names(a: ast.arguments) -> list[str]:
    return [p.arg for p in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)]


def global_refs(node) -> set[str]:
    """Names this definition reads from the module namespace.

    Only *definition-time* and *call-time* reads matter for ordering. The tricky
    part is that a function's default expressions are evaluated at definition
    time in the enclosing scope, so `def f(p=CONST)` reads CONST from module
    scope even though `p` is a parameter. Parameter names are therefore treated
    as bindings (they shadow inside the body) while everything appearing in the
    signature outside of the parameter names themselves is still a read.
    """
    binds: set[str] = set()
    loads: set[str] = set()

    def is_local_scope(n) -> bool:
        return isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))

    def collect_targets(target) -> None:
        for t in ast.walk(target):
            if isinstance(t, ast.Name):
                binds.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for e in t.elts:
                    collect_targets(e)
            elif isinstance(t, ast.Starred):
                collect_targets(t.value)

    # parameter names bind inside their own function body
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
                binds.add(arg.arg)
            if a.vararg:
                binds.add(a.vararg.arg)
            if a.kwarg:
                binds.add(a.kwarg.arg)
        # nested function/class definitions bind their own name locally
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            binds.add(n.name)
        if isinstance(n, ast.ExceptHandler) and n.name:
            binds.add(n.name)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                binds.add((al.asname or al.name).split(".")[0])
        # binding targets
        if isinstance(n, ast.Assign):
            for t in n.targets:
                collect_targets(t)
        elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
            collect_targets(n.target)
        elif isinstance(n, (ast.withitem,)):
            if n.optional_vars is not None:
                collect_targets(n.optional_vars)

    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            loads.add(n.id)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            loads.update(n.names)

    return loads - binds


def topo_order(rewritten: dict, forced_after: dict | None = None) -> list[str]:
    """Order definitions so that a name is emitted after the names it reads.
    Handles cycles (mutually recursive functions) by grouping SCCs."""
    names = list(rewritten)
    edges = {
        n: {d for d in global_refs(rewritten[n]) if d in rewritten and d != n}
        for n in names
    }
    # a folded wrapper must be emitted after the implementation it calls
    for impl_name in (forced_after or {}):
        public = impl_name[len("_impl_"):] if impl_name.startswith("_impl_") else None
        if public and public in edges:
            edges[public].add(impl_name)
        if impl_name in edges:
            edges[impl_name].discard(impl_name)

    # Tarjan SCC (iterative)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []
    counter = 0
    for root in names:
        if root in index:
            continue
        work = [(root, iter(sorted(edges[root])))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(sorted(edges[nxt]))))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == node:
                        break
                sccs.append(comp)

    # Kahn's algorithm over the condensation, iterating in sorted order for
    # determinism. Emitting a whole SCC at once breaks dependency cycles.
    comp_of = {n: i for i, c in enumerate(sccs) for n in c}
    comps = [sorted(c) for c in sccs]
    body = {i: set(c) for i, c in enumerate(comps)}
    deps_of = {
        i: {comp_of[d] for n in comps[i] for d in edges[n] if comp_of[d] != i}
        for i in body
    }
    dependents: dict[int, set[int]] = {i: set() for i in body}
    for i, ds in deps_of.items():
        for d in ds:
            dependents[d].add(i)

    remaining = {i: len(ds) for i, ds in deps_of.items()}
    ready = sorted(i for i, n in remaining.items() if n == 0)
    out: list[str] = []
    while ready:
        i = ready.pop(0)
        out.extend(comps[i])
        for j in sorted(dependents[i]):
            remaining[j] -= 1
            if remaining[j] == 0:
                ready.append(j)
        ready.sort()
    if len(out) != len(names):
        raise SystemExit("topological sort failed: unresolved cycle ordering")
    return out


def build_group(group, entry, overrides, out_path, defaults_config=None, sig_config=None):
    mods = closure(entry)
    infos = [ModuleInfo(m) for m in mods]

    # every concrete top-level name available in the merged file
    all_defs = collections.defaultdict(list)
    for info in infos:
        for name, node in info.defs.items():
            all_defs[name].append((info.name, node))
    known = set(all_defs)

    collisions = {}
    for name, hits in all_defs.items():
        if len(hits) > 1:
            dumps = {ast.dump(n) for _, n in hits}
            collisions[name] = {
                "modules": [m for m, _ in hits],
                "identical": len(dumps) == 1,
            }

    unresolved = {}
    chosen = {}
    dropped: set[str] = set()
    for name, hits in all_defs.items():
        mods_ = [m for m, _ in hits]
        if len(hits) == 1:
            chosen[name] = hits[0]
            continue
        if collisions[name]["identical"] and name not in overrides:
            # identical bodies: either copy is fine, prefer the non-delegator
            concrete = [h for h in hits if h[0] not in DELEGATOR_MODULES]
            chosen[name] = (concrete or hits)[0]
            continue
        # manual override: pick the source module ("" = drop the name entirely)
        pick = overrides.get(name)
        if pick is None:
            unresolved[name] = mods_
            chosen[name] = hits[0]
            continue
        if pick == "":
            dropped.add(name)
            continue
        match = [h for h in hits if h[0] == pick]
        if not match:
            raise SystemExit(f"override for {name} names module {pick!r} not in {mods_}")
        chosen[name] = match[0]

    # emit
    from_map: dict[str, dict] = {}   # module -> {name: asname}
    plain_modules: set[str] = set()  # `import x` / `import x as y`
    plain_aliases: set[tuple[str, str | None]] = set()
    for info in infos:
        for node in info.plain_imports:
            if isinstance(node, ast.Import):
                for a in node.names:
                    plain_aliases.add((a.name, a.asname))
            else:
                bucket = from_map.setdefault(node.module, {})
                for a in node.names:
                    bucket[a.name] = a.asname

    import_lines = []
    for name, asname in sorted(plain_aliases):
        import_lines.append(f"import {name}" + (f" as {asname}" if asname else ""))
    for mod in sorted(from_map):
        names = from_map[mod]
        rendered = ", ".join(
            n if names[n] in (None, n) else f"{n} as {names[n]}" for n in sorted(names)
        )
        import_lines.append(f"from {mod} import {rendered}")

    seen_modules = set()
    def emit_def(info: ModuleInfo, name: str) -> str:
        node = info.defs[name]
        rewriter = Rewriter(info, known)
        new = rewriter.visit(ast.parse(ast.unparse(node)).body[0])
        ast.fix_missing_locations(new)
        # annotate origin
        origin = f"# --- from {info.name}.py ---"
        return origin + "\n" + unparse(new)

    # Rewrite each chosen definition ONCE, up front. Ordering must be computed
    # from the rewritten AST: a definition's real module-level reads only appear
    # after `_mod.NAME` -> `NAME` and alias -> concrete substitutions.
    rewritten: dict[str, ast.stmt] = {}
    group_signatures = (sig_config or {}).get(group, {})
    keep_delegators = ((sig_config or {}).get(group) or {}).get("keep_delegator", {})

    # parameter order for functions whose signature is declared explicitly, used
    # to avoid passing a call argument twice (positionally and by keyword)
    sig_index = {}
    for _name, _spec in group_signatures.items():
        if "signature" not in _spec:
            continue
        _a = parse_signature(_spec["signature"])
        # positional-only parameters that a positional call would fill, in order
        sig_index[_name] = [
            p.arg for p in list(_a.posonlyargs) + list(_a.args)
        ]

    for name, (info_name, node) in chosen.items():
        info = next(i for i in infos if i.name == info_name)
        rewriter = Rewriter(info, known)
        new = rewriter.visit(ast.parse(ast.unparse(node)).body[0])
        spec = group_signatures.get(name)
        if spec and isinstance(new, (ast.FunctionDef, ast.AsyncFunctionDef)):
            new = merge_signature_explicit(new, spec["signature"])
            new = apply_call_time_defaults(new, spec.get("inject", {}))
            new = apply_call_defaults(new, spec.get("call_defaults", {}), sig_index)
        ast.fix_missing_locations(new)
        rewritten[name] = new

    # Kept delegators: the concrete implementation is renamed to a private name
    # and the delegator is published under the public name. The delegator's own
    # call is redirected to the private implementation, while its DI wiring keeps
    # referring to module globals so `patch.object(module, 'build_red_pool')`
    # style seams still work.
    for name, concrete_mod in keep_delegators.items():
        hits = all_defs.get(name)
        if not hits:
            continue
        concrete_hit = next((h for h in hits if h[0] == concrete_mod), None)
        delegator_hit = next((h for h in hits if h[0] != concrete_mod), None)
        if concrete_hit is None or delegator_hit is None:
            continue
        # the delegator replaces whatever the collision rule had chosen
        chosen.pop(name, None)
        rewritten.pop(name, None)
        impl_name = f"_impl_{name}"

        # Within implementation bodies, references to a kept-delegator name must
        # resolve to its private implementation. The concrete predicates take
        # `rule_definitions` as their third positional parameter, while the DI
        # wiring calls them with the rejection set there, so wiring has to reach
        # the implementation rather than the public delegator.
        body_renames = {_n: f"_impl_{_n}" for _n in keep_delegators}

        cinfo = next(i for i in infos if i.name == concrete_hit[0])
        impl = Rewriter(cinfo, known, body_renames).visit(
            ast.parse(ast.unparse(concrete_hit[1])).body[0]
        )
        impl.name = impl_name
        ast.fix_missing_locations(impl)
        rewritten[impl_name] = impl

        dinfo = next(i for i in infos if i.name == delegator_hit[0])
        deleg = Rewriter(dinfo, known).visit(
            ast.parse(ast.unparse(delegator_hit[1])).body[0]
        )
        for call in ast.walk(deleg):
            if isinstance(call, ast.Call):
                f = call.func
                if isinstance(f, ast.Name) and f.id == name:
                    call.func = ast.Name(id=impl_name, ctx=ast.Load())
                elif isinstance(f, ast.Attribute) and f.attr == name:
                    call.func = ast.Name(id=impl_name, ctx=ast.Load())
        ast.fix_missing_locations(deleg)
        rewritten[name] = deleg

    order = topo_order(rewritten, forced_after={})

    parts = [f'"""Merged module: {group} (generated from {len(mods)} modules)."""', ""]
    parts.extend(import_lines)
    parts.append("")
    origin = {}
    for name, (info_name, _) in chosen.items():
        origin[name] = info_name

    for name in order:
        parts.append(f"# --- from {origin.get(name, 'entry')}.py ---")
        parts.append(unparse(rewritten[name]))
        parts.append("")
        parts.append("")

    # Preserve the entry point's `if __name__ == '__main__': main()` guard,
    # otherwise `python <merged>.py` defines everything and silently exits.
    entry_info = next(i for i in infos if i.name == entry)
    for node in entry_info.other:
        if isinstance(node, ast.If):
            parts.append(f"# --- entry point: {entry}.py ---")
            parts.append(unparse(node))
            parts.append("")
            break

    out_path.write_text("\n".join(parts), encoding="utf-8")

    return {
        "group": group,
        "modules": mods,
        "module_count": len(mods),
        "defs": len(all_defs),
        "unresolved": unresolved,
        "collision_count": len(collisions),
        "identical": sum(1 for v in collisions.values() if v["identical"]),
        "undefined": check_undefined(rewritten, import_lines),
    }


if __name__ == "__main__":
    # utf-8-sig tolerates a BOM if a config file was saved by an editor that adds one
    overrides_path = HERE / "overrides.json"
    overrides = (
        json.loads(overrides_path.read_text(encoding="utf-8-sig"))
        if overrides_path.exists()
        else {}
    )
    defaults_path = HERE / "defaults.json"
    defaults_config = (
        json.loads(defaults_path.read_text(encoding="utf-8-sig"))
        if defaults_path.exists()
        else {}
    )
    sig_path = HERE / "signatures.json"
    sig_config = (
        json.loads(sig_path.read_text(encoding="utf-8-sig")) if sig_path.exists() else {}
    )
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = []
    for group, entry in ENTRIES.items():
        if only and group != only:
            continue
        out = PY / f"_merged_{group}.py"
        res = build_group(group, entry, overrides.get(group, {}), out, defaults_config, sig_config)
        results.append(res)
        print(f"### {group}: {res['module_count']} modules, {res['defs']} defs, "
              f"collisions={res['collision_count']} identical={res['identical']}, "
              f"UNRESOLVED={len(res['unresolved'])}")
        for name, ms in sorted(res["unresolved"].items()):
            print(f"    !! {name:34s} {ms}")
        if res["undefined"]:
            print(f"    !! UNDEFINED NAMES ({len(res['undefined'])}):")
            for u in res["undefined"]:
                print(f"        {u}")
        print(f"    -> {out}  ({len(out.read_text(encoding='utf-8').splitlines())} lines)")

