import ast
import functools
import importlib.util
import os
import warnings
import pkgutil

def append_xla_flags(flags: str) -> None:
    os.environ['XLA_FLAGS'] = f"{os.environ.get('XLA_FLAGS', '')} {flags}".strip()

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
if os.environ.get('BINDCRAFT_WORKER_CPUS') and hasattr(os, 'sched_setaffinity'):
    os.sched_setaffinity(0, {int(cpu) for cpu in os.environ['BINDCRAFT_WORKER_CPUS'].split(',') if cpu.strip()})
if os.path.exists('/usr/lib/wsl/lib/libcuda.so.1'):
    os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'cuda_async')
OPERATOR_COMPILATION_CACHE = os.environ.get('JAX_COMPILATION_CACHE_DIR')
# WHERE COMPILED GRAPHS LIVE.  This has to be set before jax is imported --
# jax reads it into its config at import and a later assignment does nothing,
# which is why use_campaign_compile_cache's per-campaign directory never takes
# effect and those directories stay empty.
#
# It is worth getting right: a warm cache takes a design step's first-call
# wait from 22.1 s to 7.6 s at 224 tokens, its XLA compile from 16.8 s to
# 2.2 s.  A shared cache is also better than a per-campaign one, because the
# second campaign at a length someone has already designed at starts warm.
#
# Prefer a persistent location over /tmp, which a reboot wipes.
_cache_home = os.environ.get('XDG_CACHE_HOME') or os.path.join(
    os.path.expanduser('~'), '.cache')
_default_cache = os.path.join(_cache_home, 'bindcraft', 'xla')
try:
    os.makedirs(_default_cache, exist_ok=True)
except OSError:          # a read-only home, a container: /tmp still works
    _default_cache = os.path.join(os.environ.get('TMPDIR') or '/tmp',
                                  'bindcraft_xla_cache')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', _default_cache)
os.environ.setdefault('JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES', 'none')
if 'triton_gemm' not in os.environ.get('XLA_FLAGS', ''):
    append_xla_flags('--xla_gpu_enable_triton_gemm=false')
# TWO IDENTICAL RUNS HAVE TO AGREE.  XLA benchmarks several kernels for each
# fusion while it compiles and keeps the fastest; which one wins depends on
# what else the card was doing at the time, so a SECOND PROCESS picks
# differently and every number downstream moves.  Measured on
# model_1_multimer_v3, two chains of 64, same states, same key, dropout off:
# predicted coordinates 1.4 A apart, pLDDT differing in the third decimal,
# and a design GRADIENT at cos 0.963 -- 48 blocks of backward amplify it.
# That is the "1 in 6 design steps lands elsewhere" effect, and it is not
# one in six.
#
# Turning the autotuner off makes two processes bit-identical on every
# field, and it is not a trade: a design step measured 1099 ms against
# 1090 at 192 residues (+0.8%), 127.1 ms against 126.8 at 64, and COMPILE
# is 2.2 s faster.  On this model and these cards it finds nothing better
# than the default kernel and charges compile time to look.
#
# BC2_AUTOTUNE=1 puts it back.
if os.environ.get('BC2_AUTOTUNE', '') != '1' and \
        'autotune_level' not in os.environ.get('XLA_FLAGS', ''):
    append_xla_flags('--xla_gpu_autotune_level=0')
if os.environ.get('BC2_XLA_EXTRA', '').strip():
    append_xla_flags(os.environ['BC2_XLA_EXTRA'].strip())
warnings.filterwarnings('ignore', message=r"Error reading persistent compilation cache entry for 'jit_(_where|dot_product_attention)", category=UserWarning)

COMMAND_ENTRY = 'main'
DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
DISPATCHER_MODULE = 'cli'

@functools.cache
def package_modules() -> tuple[str, ...]:
    return tuple(sorted(name for _, name, is_package in pkgutil.iter_modules(__path__) if not is_package and (not name.startswith('_'))))

def module_source(module_name: str) -> str:
    spec = importlib.util.find_spec(f'{__name__}.{module_name}')
    read_source = getattr(spec.loader, 'get_source', None) if spec else None
    return (read_source(spec.name) or '') if read_source else ''

def assigned_names(node: ast.stmt) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return [target.id for target in targets if isinstance(target, ast.Name)]

def defined_names(source: str) -> tuple[str, ...]:
    body = ast.parse(source).body
    declared = [node.value for node in body if '__all__' in assigned_names(node)]
    if declared:
        return tuple(ast.literal_eval(declared[-1]))
    defined = [name for node in body for name in ([node.name] if isinstance(node, DEFINITION_NODES) else assigned_names(node))]
    return tuple(name for name in dict.fromkeys(defined) if not name.startswith('_'))

@functools.cache
def module_definitions() -> dict[str, tuple[str, ...]]:
    definitions = {}
    for module_name in package_modules():
        try:
            definitions[module_name] = defined_names(module_source(module_name))
        except (OSError, SyntaxError, ValueError):
            definitions[module_name] = ()
    return definitions

@functools.cache
def public_api() -> dict[str, str]:
    api: dict[str, str] = {}
    for module_name, names in module_definitions().items():
        for name in names if module_name != DISPATCHER_MODULE else ():
            if name != COMMAND_ENTRY:
                api.setdefault(name, module_name)
    return api

@functools.cache
def shadowed_names() -> dict[str, tuple[str, ...]]:
    contributing = {module: names for module, names in module_definitions().items() if module != DISPATCHER_MODULE}
    defining = {name: tuple(module for module, names in contributing.items() if name in names) for name in public_api()}
    return {name: modules for name, modules in defining.items() if len(modules) > 1}

@functools.cache
def command_modules() -> tuple[str, ...]:
    return tuple(name for name, names in module_definitions().items() if COMMAND_ENTRY in names and name != DISPATCHER_MODULE)

def reload_api() -> None:
    for reading in (package_modules, module_definitions, public_api, shadowed_names, command_modules):
        reading.cache_clear()

def __getattr__(name: str):
    if name == '__all__':
        return tuple(sorted(public_api()))
    if name in public_api():
        return getattr(importlib.import_module(f'{__name__}.{public_api()[name]}'), name)
    if name in package_modules():
        return importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

def __dir__() -> list[str]:
    return sorted({*globals(), *public_api(), *package_modules()})
