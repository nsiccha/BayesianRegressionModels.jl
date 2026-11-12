# test/jax_module_driver.jl — run an RKBRMI's `jax_source` module in a real
# Python/JAX process and compare it with the native RKPPL queries of the same
# binding (brm-use references/reactivekernels.md, "A Python/JAX module").
#
# One Python process per module evaluates every dataset and point given:
# `log_density`, `log_prior`, `log_likelihood`, `log_jacobian` eagerly, under
# `jax.jit` with the prepared data closed over (the documented sampler form)
# and under `jax.jit` with the prepared data traced as arguments (`jit_args`,
# one executable for new data of the same shapes; failures are recorded, not
# raised), `jax.grad` of `log_density` in both jit forms, and `constrain`. It also
# records the interpreter, jax/jaxlib/numpy versions and devices, and whether
# importing the module alone enables float64.
#
# `RK_JAX_PYTHON` names a Python with `jax`. Without it the Python checks are
# `@test_skip`, unless `BRM_JAX_REQUIRE=1`, which makes a missing Python a
# failure: a skipped Python check is never evidence that a module runs.
using JSON

const BRM_JAX_PYTHON = get(ENV, "RK_JAX_PYTHON", "")
const BRM_JAX_REQUIRE = get(ENV, "BRM_JAX_REQUIRE", "") == "1"
brm_jax_available() = !isempty(BRM_JAX_PYTHON) && isfile(BRM_JAX_PYTHON)

function brm_jax_require_python()
    brm_jax_available() && return true
    if BRM_JAX_REQUIRE
        @test brm_jax_available()   # RK_JAX_PYTHON names no Python with jax
    else
        @test_skip "RK_JAX_PYTHON names no Python with jax"
    end
    false
end

# Python literals of `rk_bound_data` columns, as `prepare_data` takes them
# (rkppl-use §14): numeric arrays as float64/int64/bool NumPy arrays in Julia's
# column-major layout, a ragged column as a list of arrays, labels as strings,
# `missing` as `None`.
_py(x::Bool) = x ? "True" : "False"
_py(x::Integer) = string(x)
_py(x::AbstractFloat) = isnan(x) ? "float('nan')" :
    isinf(x) ? (x > 0 ? "float('inf')" : "float('-inf')") : repr(Float64(x))
_py(x::AbstractString) = repr(String(x))
_py(::Missing) = "None"
_py(::Nothing) = "None"
_py(x::Symbol) = repr(String(x))
_py(x) = repr(string(x))
_py_dtype(::Type{Bool}) = "bool"
_py_dtype(::Type{<:Integer}) = "np.int64"
_py_dtype(::Type{<:AbstractFloat}) = "np.float64"
function _py(x::AbstractArray{T}) where {T<:Number}
    flat = join(map(_py, vec(collect(x))), ", ")
    shape = join(size(x), ", ") * (ndims(x) == 1 ? "," : "")
    "np.asarray([$flat], dtype=$(_py_dtype(T))).reshape(($shape), order='F')"
end
_py(x::AbstractVector{<:AbstractVector{<:Number}}) = "[" * join(map(_py, x), ", ") * "]"
_py(x::AbstractArray) = ndims(x) == 1 ? "[" * join(map(_py, x), ", ") * "]" :
    error("no Python literal for a $(ndims(x))-dimensional $(typeof(x)) column")
_py(x::Tuple) = "(" * join(map(_py, x), ", ") * (length(x) == 1 ? ",)" : ")")

const _BRM_JAX_DRIVER = raw"""
import importlib.util, json, math, platform, sys
import numpy as np
import jax
x64_before = bool(jax.config.jax_enable_x64)
spec = importlib.util.spec_from_file_location("brm_jax_module", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
x64_module = bool(jax.config.jax_enable_x64)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlib
exec(open(sys.argv[2]).read())  # DATASETS, POINTS

def enc(x):
    if isinstance(x, dict):
        return {str(k): enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    a = np.asarray(x)
    flat = [(v if math.isfinite(v) else repr(v)) for v in a.astype(np.float64).flatten(order="F").tolist()]
    return {"shape": list(a.shape), "data": flat, "dtype": str(a.dtype)}

def untranslated(u):
    pairs = u.items() if isinstance(u, dict) else ((x, "") for x in u)
    return {str(k): str(v) for k, v in pairs}

FUNCS = ["log_density", "log_prior", "log_likelihood", "log_jacobian"]
# `jit_args`: one executable per function, the prepared data traced as
# arguments (new data of the same shapes reuse it). Its failure is recorded,
# not raised, so the closure form below is still measured.
jit_args = {f: jax.jit(getattr(m, f)) for f in FUNCS + ["constrain"]}
grad_args = jax.jit(jax.grad(m.log_density))
out = {
    "versions": {"python": platform.python_version(), "jax": jax.__version__,
                 "jaxlib": jaxlib.__version__, "numpy": np.__version__,
                 "devices": [str(d) for d in jax.devices()]},
    "x64_before_import": x64_before, "x64_after_import": x64_module,
    # `DIMENSION` exists when no parameter size depends on the data; from
    # RKPPL 0.1.134 `m.dimension(prepared)` gives it for every binding.
    "module": {"DIMENSION": int(m.DIMENSION) if hasattr(m, "DIMENSION") else None,
               "DATA": list(m.DATA), "PREPARED": list(m.PREPARED),
               "INPUTS": [str(i) for i in m.INPUTS],
               "UNTRANSLATED": untranslated(getattr(m, "UNTRANSLATED", {}))},
    "datasets": [],
}

def attempt(thunk):
    try:
        return {"ok": True, "value": enc(thunk())}
    except Exception as e:
        return {"ok": False, "error": (type(e).__name__ + ": " + str(e))[:600]}

# `prepare_data` takes the bound columns by name. A name that is no Python
# identifier (a dotted submodel binding) has no keyword spelling, so then the
# call is positional, in `m.DATA`'s order.
def prepare(data):
    if all(k.isidentifier() for k in data):
        return m.prepare_data(**data), "by_name"
    return m.prepare_data(*[data[k] for k in m.DATA]), "positional"

for data, points in zip(DATASETS, POINTS):
    prepared, called = prepare(data)
    # `jit`: the documented sampler form, the prepared data closed over
    # (brm-use: `potential = lambda u: -m.log_density(u, **prepared)`).
    jit = {f: jax.jit(lambda u, f=f: getattr(m, f)(u, **prepared)) for f in FUNCS + ["constrain"]}
    grad = jax.jit(jax.grad(lambda u: m.log_density(u, **prepared)))
    r = {"prepared": {k: list(np.shape(v)) for k, v in prepared.items()}, "called": called,
         "dimension": int(m.dimension(prepared)) if hasattr(m, "dimension") else None, "points": []}
    for u in points:
        u = jnp.asarray(u, dtype=jnp.float64)
        p = {}
        for f in FUNCS:
            p[f] = {"eager": enc(getattr(m, f)(u, **prepared)), "jit": enc(jit[f](u)),
                    "jit_args": attempt(lambda: jit_args[f](u, **prepared))}
        p["grad"] = enc(grad(u))
        p["grad_eager"] = enc(jax.grad(m.log_density)(u, **prepared))
        p["grad_args"] = attempt(lambda: grad_args(u, **prepared))
        p["constrain_eager"] = enc(m.constrain(u, **prepared))
        p["constrain_jit"] = enc(jit["constrain"](u))
        r["points"].append(p)
    out["datasets"].append(r)
print("BRM_JAX_RESULT " + json.dumps(out))
"""

_jax_dec(v::Real) = Float64(v)
_jax_dec(v::AbstractString) = v == "nan" ? NaN : v == "inf" ? Inf : v == "-inf" ? -Inf :
    error("unexpected value $v")
function _jax_decode(x::AbstractDict)
    if haskey(x, "shape") && haskey(x, "data")
        data = Float64[_jax_dec(v) for v in x["data"]]
        shape = Tuple(Int.(x["shape"]))
        return isempty(shape) ? only(data) : reshape(data, shape)
    end
    Dict(String(k) => _jax_decode(v) for (k, v) in x)
end
_jax_decode(x::AbstractVector) = [_jax_decode(v) for v in x]
_jax_decode(x) = x

"""
    brm_jax_run(src, datasets, points) -> NamedTuple

Run module `src` once in Python over `datasets` (each a `NamedTuple` of the
raw columns `prepare_data` takes, by name) with `points[k]` the unconstrained
points of dataset `k`. Returns the decoded results (see the driver above).
"""
function brm_jax_run(src, datasets, points)
    # `BRM_JAX_WORKDIR` keeps each run's module, inputs and stderr for reading.
    dir = haskey(ENV, "BRM_JAX_WORKDIR") ?
        mktempdir(mkpath(ENV["BRM_JAX_WORKDIR"]); cleanup=false) : mktempdir()
    module_path = joinpath(dir, "brm_jax_module.py")
    write(module_path, src)
    lines = ["DATASETS = ["]
    for d in datasets
        # A dict literal: a bound name may hold a dot (`cdar_mu_day.cell`).
        push!(lines, "  {" * join(("$(repr(String(k))): $(_py(v))" for (k, v) in pairs(d)), ", ") * "},")
    end
    push!(lines, "]", "POINTS = [")
    for us in points
        push!(lines, "  [" * join(("[" * join(map(_py, u), ", ") * "]" for u in us), ", ") * "],")
    end
    push!(lines, "]")
    args_path = joinpath(dir, "brm_jax_inputs.py")
    write(args_path, join(lines, "\n") * "\n")
    driver = joinpath(dir, "brm_jax_driver.py")
    write(driver, _BRM_JAX_DRIVER)
    errors = joinpath(dir, "stderr.txt")
    output = try
        read(pipeline(`$BRM_JAX_PYTHON $driver $module_path $args_path`; stderr=errors), String)
    catch e
        e isa ProcessFailedException || rethrow()
        error("the emitted module failed in Python (module at ", module_path, "):\n",
            join(last(readlines(errors), 40), "\n"))
    end
    line = only(l for l in split(output, '\n') if startswith(l, "BRM_JAX_RESULT "))
    raw = JSON.parse(line[length("BRM_JAX_RESULT ")+1:end])
    (; versions=raw["versions"], x64_before_import=raw["x64_before_import"],
        x64_after_import=raw["x64_after_import"], module_constants=raw["module"],
        datasets=_jax_decode(raw["datasets"]), dir)
end

# The public module-input accessor supplies exactly `m.DATA` by name.
brm_jax_inputs(rk, src) = rk_bound_data(rk, src)

const _BRM_JAX_NUTS_DRIVER = raw"""
import importlib.util, json, math, platform, sys, time
import numpy as np
import jax
spec = importlib.util.spec_from_file_location("brm_jax_module", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
exec(open(sys.argv[2]).read())  # DATA, INIT, SETTINGS
prepared = (m.prepare_data(**DATA) if all(k.isidentifier() for k in DATA)
            else m.prepare_data(*[DATA[k] for k in m.DATA]))
potential = jax.jit(lambda u: -m.log_density(u, **prepared))
kernel = NUTS(potential_fn=potential, target_accept_prob=SETTINGS["target_accept"])
mcmc = MCMC(kernel, num_warmup=SETTINGS["warmup"], num_samples=SETTINGS["samples"],
            num_chains=SETTINGS["chains"], chain_method="sequential", progress_bar=False)
start = time.time()
mcmc.run(jax.random.PRNGKey(SETTINGS["seed"]), init_params=jnp.asarray(INIT, dtype=jnp.float64),
         extra_fields=("diverging",))
elapsed = time.time() - start
u = np.asarray(mcmc.get_samples(group_by_chain=True))     # chains x draws x d
div = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
out = {"versions": {"python": platform.python_version(), "jax": jax.__version__,
                    "numpyro": numpyro.__version__, "numpy": np.__version__,
                    "devices": [str(d) for d in jax.devices()]},
       "elapsed_seconds": elapsed, "shape": list(u.shape),
       "draws": u.flatten(order="F").tolist(), "divergences": int(div.sum())}
print("BRM_JAX_RESULT " + json.dumps(out))
"""

"""
    brm_jax_nuts(src, data, init; chains, warmup, samples, seed, target_accept = 0.8)

Sample the module's `log_density` with NumPyro's NUTS (`potential_fn`, the
prepared data closed over) in Python, from `init[k]` for chain `k`. Returns
the unconstrained draws as a `d × draws × chains` array, the divergence count,
the versions and the sampling time.
"""
function brm_jax_nuts(src, data, init; chains, warmup, samples, seed, target_accept=0.8)
    dir = haskey(ENV, "BRM_JAX_WORKDIR") ?
        mktempdir(mkpath(ENV["BRM_JAX_WORKDIR"]); cleanup=false) : mktempdir()
    module_path = joinpath(dir, "brm_jax_module.py")
    write(module_path, src)
    args_path = joinpath(dir, "brm_jax_inputs.py")
    write(args_path, "DATA = {" * join(("$(repr(String(k))): $(_py(v))" for (k, v) in pairs(data)), ", ") * "}\n" *
        "INIT = [" * join(("[" * join(map(_py, u), ", ") * "]" for u in init), ", ") * "]\n" *
        "SETTINGS = dict(chains=$chains, warmup=$warmup, samples=$samples, seed=$seed, target_accept=$target_accept)\n")
    driver = joinpath(dir, "brm_jax_nuts.py")
    write(driver, _BRM_JAX_NUTS_DRIVER)
    errors = joinpath(dir, "stderr.txt")
    output = try
        read(pipeline(`$BRM_JAX_PYTHON $driver $module_path $args_path`; stderr=errors), String)
    catch e
        e isa ProcessFailedException || rethrow()
        error("NUTS on the emitted module failed in Python (module at ", module_path, "):\n",
            join(last(readlines(errors), 40), "\n"))
    end
    line = only(l for l in split(output, '\n') if startswith(l, "BRM_JAX_RESULT "))
    raw = JSON.parse(line[length("BRM_JAX_RESULT ")+1:end])
    C, N, d = Int.(raw["shape"])
    u = reshape(Float64.(raw["draws"]), C, N, d)          # column-major of the C×N×d array
    (; draws=permutedims(u, (3, 2, 1)), divergences=raw["divergences"],
        versions=raw["versions"], elapsed=raw["elapsed_seconds"], dir)
end
