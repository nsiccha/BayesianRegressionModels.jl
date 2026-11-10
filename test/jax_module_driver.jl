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
    "module": {"DIMENSION": int(m.DIMENSION), "DATA": list(m.DATA),
               "PREPARED": list(m.PREPARED), "INPUTS": [str(i) for i in m.INPUTS]},
    "datasets": [],
}

def attempt(thunk):
    try:
        return {"ok": True, "value": enc(thunk())}
    except Exception as e:
        return {"ok": False, "error": (type(e).__name__ + ": " + str(e))[:600]}

for data, points in zip(DATASETS, POINTS):
    prepared = m.prepare_data(**data)
    # `jit`: the documented sampler form, the prepared data closed over
    # (brm-use: `potential = lambda u: -m.log_density(u, **prepared)`).
    jit = {f: jax.jit(lambda u, f=f: getattr(m, f)(u, **prepared)) for f in FUNCS + ["constrain"]}
    grad = jax.jit(jax.grad(lambda u: m.log_density(u, **prepared)))
    r = {"prepared": {k: list(np.shape(v)) for k, v in prepared.items()}, "points": []}
    for u in points:
        u = jnp.asarray(u, dtype=jnp.float64)
        p = {}
        for f in FUNCS:
            p[f] = {"eager": enc(getattr(m, f)(u, **prepared)), "jit": enc(jit[f](u)),
                    "jit_args": attempt(lambda: jit_args[f](u, **prepared))}
        p["grad"] = enc(grad(u))
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
        push!(lines, "  dict(" * join(("$k=$(_py(v))" for (k, v) in pairs(d)), ", ") * "),")
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

# The raw columns `prepare_data` takes, by the names the module lists
# (`m.DATA`): a data-free replay binds more than it takes.
function brm_jax_inputs(rk, src)
    bound = rk_bound_data(rk)
    names = [Symbol(first(i)) for i in only(f for f in src.functions if f.name == "prepare_data").inputs]
    NamedTuple{Tuple(names)}(Tuple(bound[n] for n in names))
end
