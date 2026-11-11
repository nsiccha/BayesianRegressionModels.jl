# research/jax/reproduce.jl — the worked example of docs/src/jax.md: a
# hierarchical regression on synthetic data, exported as a Python/JAX module
# (`jax_source(rk.model, rk_bound_plan(rk))`), sampled natively with WarmupHMC
# and in Python with NumPyro NUTS on the module's `log_density`, with a
# value/gradient parity check at posterior draws.
#
# Run from the repository root in the pinned test environment
# (`julia --project=test test/setup_env.jl` once):
#
#   RK_JAX_PYTHON=<python with jax and numpyro> \
#     julia --startup-file=no --threads=1 --project=test research/jax/reproduce.jl
#
# Writes research/jax/model.py (the emitted module), research/jax/results/
# (inputs.json, points.json, python.json and the TSV tables the page reads).
#
# The docs build evaluates everything ABOVE `varying_slopes_model` (the data and
# the model, which need only BayesianRegressionModels, Distributions and Random)
# through `Main.BRMDocsComparisons.evaluate_source_prelude(...; before=:varying_slopes_model)`
# and shows the model through `source_function(...)`. Everything below the model
# runs only as a script, so the docs environment never needs ReactiveKernels.
using BayesianRegressionModels, Distributions
using Random, LinearAlgebra

"""
Synthetic data for a varying-intercepts, varying-slopes regression: `groups`
groups with 6 to 20 rows each, one standard-normal covariate `x`, and each group's
intercept and slope drawn correlated around a population line. Returns the
columns the model reads (`data`), the generating values (`truth`) and each
group's generating deviations (`effects`).
"""
function varying_slopes_data(; groups = 24, rng = Xoshiro(2026))
    truth = (; intercept = 1.5, slope = -0.8, sd_intercept = 0.6, sd_slope = 0.3,
        cor = 0.4, sigma = 0.5)
    scale = [truth.sd_intercept 0.0; 0.0 truth.sd_slope]
    L = scale * cholesky([1.0 truth.cor; truth.cor 1.0]).L
    labels = ["g" * lpad(k, 2, '0') for k in 1:groups]
    deviations = [L * randn(rng, 2) for _ in 1:groups]
    g = String[]; x = Float64[]; y = Float64[]
    for k in 1:groups
        for _ in 1:rand(rng, 6:20)
            xi = randn(rng)
            mu = truth.intercept + deviations[k][1] + (truth.slope + deviations[k][2]) * xi
            push!(g, labels[k]); push!(x, xi); push!(y, mu + truth.sigma * randn(rng))
        end
    end
    effects = [(; g = labels[k], intercept = deviations[k][1], slope = deviations[k][2]) for k in 1:groups]
    (; data = (; g, x, y), truth, effects)
end

"""
The model: a population line, each group's own intercept and slope around it
(one correlated block), and Normal noise.
"""
function varying_slopes_model(data = varying_slopes_data().data)
    (@brm begin
        "Residual scale of the response."
        sigma ~ Exponential(1)
        "Expected response: a population line, and each group's own intercept and slope, correlated."
        mu ~ 1 + x + (1 + x | g)
        effect(mu, Intercept) ~ Normal(0, 2)
        effect(mu, x) ~ Normal(0, 1)
        sd(mu, g) ~ Exponential(1)
        "Measured response."
        y ~ Normal(mu, sigma)
    end)(data)
end

# ---------------------------------------------------------------------------
# Everything below runs only as a script (`julia --project=test research/jax/reproduce.jl`).
using ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, LogDensityProblems
using WarmupHMC, MCMCDiagnosticTools
using JSON, Statistics, SHA, TreeArrays, Treebars
include("presentation.jl")
using .JAXExampleResults: POPULATION

const HERE = @__DIR__
const RESULTS = joinpath(HERE, "results")
const PYTHON = get(ENV, "RK_JAX_PYTHON", "")

# JSON encoding of a bound column as `prepare_data` takes it: numbers as typed
# arrays in column-major order with their shape, labels as strings, `missing`
# as null (the shape of test/jax_module_driver.jl's Python literals).
jsonable(x::AbstractArray{<:AbstractString}) = Dict("dtype" => "str", "shape" => collect(size(x)), "data" => vec(String.(x)))
jsonable(x::AbstractArray{Bool}) = Dict("dtype" => "bool", "shape" => collect(size(x)), "data" => vec(x))
jsonable(x::AbstractArray{<:Integer}) = Dict("dtype" => "int64", "shape" => collect(size(x)), "data" => vec(x))
jsonable(x::AbstractArray{<:Real}) = Dict("dtype" => "float64", "shape" => collect(size(x)), "data" => [isfinite(v) ? Float64(v) : string(v) for v in vec(x)])
jsonable(x::AbstractArray) = Dict("dtype" => "object", "shape" => collect(size(x)), "data" => [ismissing(v) ? nothing : v for v in vec(x)])
jsonable(x::Real) = x
jsonable(x::AbstractString) = String(x)

# The raw columns `prepare_data` takes, by the names the module lists.
function module_inputs(rk, src)
    bound = rk_bound_data(rk)
    names = [Symbol(first(i)) for i in only(f for f in src.functions if f.name == "prepare_data").inputs]
    Dict(string(n) => jsonable(bound[n]) for n in names)
end

flat_names(nt::NamedTuple, prefix = "") = reduce(vcat,
    [v isa NamedTuple ? flat_names(v, prefix * string(k) * ".") : [(prefix * string(k), v)] for (k, v) in pairs(nt)];
    init = Tuple{String,Any}[])

# Scalar constrained quantities per draw from `draws[:, j, c]`:
# `names`, `values[name][draw, chain]`. Names are the module's own scope path
# (`constrain` nests by scope, as the module's `constrain` does).
function constrained_scalars(layout, draws)
    d, N, C = size(draws)
    first_draw = flat_names(ReactiveKernelsPPL.constrain(layout, draws[:, 1, 1]))
    names = String[]
    for (name, v) in first_draw
        for i in eachindex(v)
            push!(names, length(v) == 1 && !(v isa AbstractArray) ? name : "$name[$i]")
        end
    end
    values = Dict(n => zeros(N, C) for n in names)
    for c in 1:C, j in 1:N
        k = 0
        for (_, v) in flat_names(ReactiveKernelsPPL.constrain(layout, draws[:, j, c]))
            for x in (v isa AbstractArray ? vec(v) : (v,))
                k += 1
                values[names[k]][j, c] = x
            end
        end
    end
    names, values
end

function summarize(x::AbstractMatrix)   # draws × chains
    tree = TreeData(x, :draw, :chain)
    qs = quantile(tree, (q025=0.025, q05=0.05, q25=0.25, q50=0.5,
        q75=0.75, q95=0.95, q975=0.975); dims=(:draw, :chain))
    chains = TreeActualArray(TreeData(reshape(x, size(x, 1), size(x, 2), 1),
        :draw, :chain, :quantity))
    bulk = only(MCMCDiagnosticTools.ess(chains; kind = :bulk))
    tail = only(MCMCDiagnosticTools.ess(chains; kind = :tail))
    (; (k => parent(getproperty(qs, k)) for k in (:q025, :q05, :q25, :q50, :q75, :q95, :q975))...,
       mean = mean(tree), sd = std(tree),
       ess_bulk = bulk, ess_tail = tail, rhat = only(MCMCDiagnosticTools.rhat(chains)),
       mcse = only(MCMCDiagnosticTools.mcse(chains; kind = mean)))
end

function write_tsv(path, rows)
    isempty(rows) && error("no rows for $path")
    header = collect(keys(first(rows)))
    open(path, "w") do io
        println(io, join(header, '\t'))
        for r in rows
            println(io, join((string(r[k]) for k in header), '\t'))
        end
    end
    println("TABLE ", relpath(path, HERE))
end

function main(; chains = 4, n_draws = 1000, warmup = 1000, seed = 7,
              reuse_draws = false, progress = nothing)
    isfile(PYTHON) || error("RK_JAX_PYTHON names no Python with jax and numpyro: ", repr(PYTHON))
    mkpath(RESULTS)
    synthetic = varying_slopes_data()
    data, truth = synthetic.data, synthetic.truth
    write_tsv(joinpath(RESULTS, "data.tsv"), [(; g = data.g[i], x = data.x[i], y = data.y[i]) for i in eachindex(data.y)])
    write_tsv(joinpath(RESULTS, "generating_values.tsv"), [(; quantity = string(k), value = v) for (k, v) in pairs(truth)])
    write_tsv(joinpath(RESULTS, "generating_effects.tsv"), synthetic.effects)

    # 1. The module.
    rk = RKBRMI(varying_slopes_model(data))
    plan = rk_bound_plan(rk)
    src = jax_source(rk.model, plan; values = (:mu,))
    write(joinpath(HERE, "model.py"), src)
    d = rk.model.layout.total
    inputs = module_inputs(rk, src)
    open(io -> JSON.print(io, inputs), joinpath(RESULTS, "inputs.json"), "w")
    signature = bytes2hex(sha256(src.text * read(joinpath(RESULTS, "inputs.json"), String)))

    # 2. The native fit (WarmupHMC's quorum sampler on the Enzyme-differentiated density).
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    cache = joinpath(RESULTS, "native.json")
    if !reuse_draws
        Threads.nthreads() == 1 || error("Run this RK/Enzyme example with --threads=1; concurrent gradients are unsupported.")
        native_seconds = @elapsed fit = completion_warmup_mcmc([Xoshiro(100 + c) for c in 1:chains], problem;
            min_completed = div(3chains, 4), n_draws, monitor_ess = true, progress)
        native = cat((r.posterior_position[:, end-n_draws+1:end] for r in fit.results)...; dims = 3)
        # Report divergences across the completed native chains.
        native_divergences = sum(r.n_divergent_samples for r in fit.results)
        saved = (; signature, shape=size(native), draws=vec(native), seconds=native_seconds,
            divergences=native_divergences, completed=fit.completion.n_completed,
            chains_requested=chains, seed=100, samples=n_draws)
        open(io -> JSON.print(io, saved), cache, "w")
    end
    saved = JSON.parsefile(cache)
    saved["signature"] == signature || error("Saved native draws belong to another module or dataset.")
    saved["samples"] == n_draws && saved["chains_requested"] == chains || error("Saved run settings differ.")
    native = reshape(Float64.(saved["draws"]), Int.(saved["shape"])...)
    native_seconds, native_divergences = saved["seconds"], saved["divergences"]

    # 3. Parity points: posterior draws of the native fit, and the native
    # density and Enzyme gradient at each.
    points = [native[:, j, 1] for j in round.(Int, range(1, n_draws; length = 8))]
    open(io -> JSON.print(io, points), joinpath(RESULTS, "points.json"), "w")
    native_values = map(points) do u
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        (value, copy(gradient))
    end

    # 4. Python: parity values and NumPyro NUTS (research/jax/sample_numpyro.py).
    python_out = joinpath(RESULTS, "python.json")
    if !reuse_draws || !isfile(python_out)
        run(`$PYTHON $(joinpath(HERE, "sample_numpyro.py")) $(joinpath(HERE, "model.py"))
        $(joinpath(RESULTS, "inputs.json")) $(joinpath(RESULTS, "points.json")) $python_out
        --chains $chains --warmup $warmup --samples $n_draws --seed $seed`)
    end
    py = JSON.parsefile(python_out)
    py["signature"] == signature || error("Saved Python draws belong to another module or dataset.")
    settings = py["settings"]
    all(settings[k] == v for (k,v) in (("chains",chains), ("warmup",warmup),
        ("samples",n_draws), ("seed",seed), ("target_accept",0.8))) ||
        error("Saved Python run settings differ.")
    C, N, dd = Int.(py["shape"])
    dd == d || error("the module's dimension $dd differs from the native layout's $d")
    python = permutedims(reshape(Float64.(py["draws"]), C, N, d), (3, 2, 1))   # d × N × C

    # 5. Parity table.
    parity = [(; point = j,
        log_density_native = native_values[j][1],
        log_density_jax = py["parity"][j]["log_density_jit"],
        log_density_abs_diff = abs(native_values[j][1] - py["parity"][j]["log_density_jit"]),
        log_density_rel_diff = abs(native_values[j][1] - py["parity"][j]["log_density_jit"]) / abs(native_values[j][1]),
        gradient_max_abs_diff = maximum(abs.(native_values[j][2] .- Float64.(py["parity"][j]["gradient"]))),
        gradient_max_rel_diff = maximum(abs.(native_values[j][2] .- Float64.(py["parity"][j]["gradient"])) ./ max.(1, abs.(native_values[j][2]))))
        for j in eachindex(points)]
    write_tsv(joinpath(RESULTS, "parity.tsv"), parity)
    all(r -> r.log_density_rel_diff < 1e-10 && r.gradient_max_rel_diff < 1e-8, parity) ||
        error("Native/JAX value or gradient parity failed; see parity.tsv.")

    # 6. Posterior tables: every scalar constrained quantity, both methods.
    names, nv = constrained_scalars(rk.model.layout, native)
    _, pv = constrained_scalars(rk.model.layout, python)
    rows = []
    for n in names
        (iszero(std(nv[n])) || iszero(std(pv[n]))) && continue   # a correlation factor's unit entry
        a, b = summarize(nv[n]), summarize(pv[n])
        z = abs(a.mean - b.mean) / sqrt(a.mcse^2 + b.mcse^2)
        for (method, s) in (("native", a), ("jax", b))
            push!(rows, (; quantity = n, method, s..., z))
        end
    end
    write_tsv(joinpath(RESULTS, "posterior.tsv"), rows)
    # Unconstrained coordinates: the distributional check of test/rk_jax_inference.jl.
    zrows = [(; coordinate = i,
        z = (a = summarize(native[i, :, :]); b = summarize(python[i, :, :]);
             abs(a.mean - b.mean) / sqrt(a.mcse^2 + b.mcse^2))) for i in 1:d]
    write_tsv(joinpath(RESULTS, "unconstrained_z.tsv"), zrows)

    # 7. Keep draws for the plotting package's density estimator. These six
    # scope paths are the parameters of this particular generated module.
    population = first.(POPULATION)
    all(n -> n in names, population) || error("Inspect the module's changed parameter layout: ", names)
    write_tsv(joinpath(RESULTS, "population_draws.tsv"),
        [(; quantity=n, method, draw=j, chain=c, value=v[n][j,c])
         for n in population for (method,v) in (("native",nv),("jax",pv))
         for c in axes(v[n],2) for j in axes(v[n],1)])

    # Transform this model's standardized effects to deviations in response
    # units, using the scoped parameters documented in the emitted module.
    labels = sort(unique(data.g))
    bound = rk_bound_data(rk)
    all(labels[bound[:g_level][i]] == data.g[i] for i in eachindex(data.g)) ||
        error("Group labels and emitted level codes differ.")
    groups = []
    for (method, draws) in (("native", native), ("jax", python))
        effects = zeros(size(draws, 2), size(draws, 3), length(labels), 2)
        for c in axes(draws, 3), j in axes(draws, 2)
            block = ReactiveKernelsPPL.constrain(rk.model.layout, draws[:,j,c]).b_mu__g_g
            effects[j,c,:,:] = block.z * transpose(block.tau .* block.L)
        end
        for k in eachindex(labels), (margin, quantity) in enumerate(("intercept", "slope"))
            push!(groups, (; group=labels[k], quantity, method,
                summarize(view(effects, :, :, k, margin))...))
        end
    end
    write_tsv(joinpath(RESULTS, "group_effects.tsv"), groups)

    # 8. Run record.
    record = [(; key = "rows", value = length(data.y)), (; key = "groups", value = length(unique(data.g))),
        (; key = "dimension", value = d), (; key = "chains", value = chains), (; key = "draws_per_chain", value = n_draws),
        (; key = "native_seconds", value = native_seconds), (; key = "native_completed_chains", value = saved["completed"]),
        (; key = "native_divergences", value = native_divergences),
        (; key = "python_sampling_seconds", value = py["elapsed_seconds"]), (; key = "python_divergences", value = py["divergences"]),
        (; key = "python_first_log_density_seconds", value = py["first_call_seconds"]),
        (; key = "python_version", value = py["versions"]["python"]), (; key = "jax_version", value = py["versions"]["jax"]),
        (; key = "numpyro_version", value = py["versions"]["numpyro"]), (; key = "numpy_version", value = py["versions"]["numpy"]),
        (; key = "julia_version", value = string(VERSION)),
        (; key = "BayesianRegressionModels", value = string(pkgversion(BayesianRegressionModels))),
        (; key = "ReactiveKernels", value = string(pkgversion(ReactiveKernels))),
        (; key = "ReactiveKernelsPPL", value = string(pkgversion(ReactiveKernelsPPL))),
        (; key = "WarmupHMC", value = string(pkgversion(WarmupHMC))),
        (; key = "untranslated", value = join(sort(collect(keys(py["untranslated"]))), ", ")),
        (; key = "data_inputs", value = join(py["data_names"], ", ")),
        (; key = "prepared_inputs", value = join(py["prepared_names"], ", "))]
    write_tsv(joinpath(RESULTS, "run.tsv"), record)
end

if abspath(PROGRAM_FILE) == @__FILE__
    with_progress(:state; description="BRM native and JAX example") do progress
        snapshot_progress(progress, get(ENV, "KB_PROGRESS_FILE", joinpath(HERE, "fit.progress"))) do
            main(; reuse_draws="--reuse-draws" in ARGS, progress)
        end
    end
end
