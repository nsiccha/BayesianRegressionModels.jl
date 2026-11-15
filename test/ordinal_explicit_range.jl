# test/ordinal_explicit_range.jl — coarsened evidence over an explicit-threshold
# `Ordinal` (snag ordinal-explicit-149b4a5f):
#
#     y_min ~ coarsened(Ordinal(structure, link, eta, thresholds[, d]); upper = y_max)
#
# at model scope and in a `@plate` cell, where each examiner's thresholds may be
# computed in the cell. Both columns hold category codes 1:K, K =
# length(thresholds) + 1, as the explicit form's response does; equal codes are
# an exact observation.
#
# Oracle: the priors written out with Distributions.jl plus, per row, the log
# of the executable `Ordinal`'s summed `probs` over the row's code range, at the
# same constrained point. Every prior here is on an unbounded support, so the
# unconstrained point is the constrained one and no Jacobian enters. Code 1 is
# never observed, so a backend that renumbered the codes by the observed levels
# would disagree with the oracle.
#
# Run: BRM_JAX_REQUIRE=1 julia --startup-file=no --project=test test/ordinal_explicit_range.jl [filter...]
#      (RK_JAX_PYTHON names a Python with jax for the JAX module parity.)
#      The stopping-ratio cell's JAX gradient needs ReactiveKernels 0.1.263.
using Test, Random, LinearAlgebra
using BayesianRegressionModels, Distributions, LogDensityProblems
using StanBlocks
const BridgeStan = StanBlocks.BridgeStan
# `import`, not `using`: an older ReactiveKernelsPPL exported its own link tags.
import ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPL: jax_source
using ReactiveKernels: JAXSource
import Enzyme
using DifferentiationInterface: AutoEnzyme

include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "jax_module_driver.jl"))

const N_EX, PER = 3, 8
function range_data(seed)
    rng = Random.Xoshiro(seed)
    n = N_EX * PER
    examiner = repeat(1:N_EX, inner = PER)
    # Codes 2:5; code 1 is never observed and 5 only as an upper end. Roughly
    # half the rows are exact (lo == hi), the others ranges of two or three.
    lo = rand(rng, 2:4, n)
    hi = min.(lo .+ rand(rng, [0, 0, 0, 1, 1, 2], n), 5)
    hi[1], hi[PER + 1] = 5, lo[PER + 1]   # a top range and an exact row
    (; y_min = lo, y_max = hi, x = randn(rng, n), eta_data = 0.3 .* randn(rng, n),
       examiner, exids = collect(1:N_EX), zero_gap = [zeros(3) for _ in 1:N_EX],
       zero_stage = [zeros(4) for _ in 1:N_EX])
end
const DATA = range_data(20261115)
@assert !(1 in DATA.y_min) && any(DATA.y_min .< DATA.y_max) && any(DATA.y_min .== DATA.y_max)

range_logmass(d, lo, hi) = log(sum(probs(d)[lo:hi]))
function range_loglik(laws)
    lp = 0.0
    for (i, d) in enumerate(laws)
        lp += range_logmass(d, DATA.y_min[i], DATA.y_max[i])
    end
    lp
end

pooled_cuts(m, log_gap, u) = (s = cumsum(vcat(0.0, exp.(log_gap .+ u))); m .+ s .- mean(s))

# The reporter's model: each examiner's thresholds partially pooled around
# population log-gaps and centred on a shared mean, ordered by positive gaps.
const POOLED = quote
    m ~ Normal(0.0, 1.5)
    log_gap[1:3] .~ Normal.(0.0, 1.0)
    rows = groups(examiner => exids)
    @plate for e in eachindex(exids)
        u ~ Normal(zero_gap[e], 0.5)
        s = cumsum(vcat(0.0, exp.(log_gap .+ u)))
        cuts = m .+ s .- mean(s)
        y_min[rows[e]] ~ coarsened(Ordinal(Cumulative(), LogitLink(), eta_data[rows[e]], cuts);
            upper = y_max[rows[e]])
    end
end

function pooled_oracle(m, log_gap, U)   # U: 3 × N_EX, one column per examiner
    lp = logpdf(Normal(0, 1.5), m) + sum(logpdf.(Normal(0, 1), log_gap)) +
         sum(logpdf.(Normal(0, 0.5), U))
    lp + range_loglik(Ordinal(Cumulative(), LogitLink(), DATA.eta_data[i],
        pooled_cuts(m, log_gap, U[:, DATA.examiner[i]])) for i in eachindex(DATA.y_min))
end

# Per-examiner stopping-ratio stage intercepts, and a discrimination keyword.
const STAGES = quote
    rows = groups(examiner => exids)
    @plate for e in eachindex(exids)
        t ~ Normal(zero_stage[e], 1.0)
        y_min[rows[e]] ~ coarsened(Ordinal(StoppingRatio(), CloglogLink(), eta_data[rows[e]], t;
            discrimination = 0.7); upper = y_max[rows[e]])
    end
end

function stages_oracle(T)   # T: 4 × N_EX
    sum(logpdf.(Normal(0, 1), T)) + range_loglik(Ordinal(StoppingRatio(), CloglogLink(),
        DATA.eta_data[i], T[:, DATA.examiner[i]]; discrimination = 0.7)
        for i in eachindex(DATA.y_min))
end

# A declared model vector read in the cell, with a positional discrimination.
const SHARED_IN_CELL = quote
    c[1:4] .~ Normal.(0.0, 1.0)
    rows = groups(examiner => exids)
    @plate for e in eachindex(exids)
        y_min[rows[e]] ~ coarsened(Ordinal(Cumulative(), ProbitLink(), eta_data[rows[e]], c, 1.5);
            upper = y_max[rows[e]])
    end
end

const MODEL_SCOPE = quote
    c[1:4] .~ Normal.(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    eta = b * x
    y_min ~ coarsened(Ordinal(Cumulative(), LogitLink(), eta, c); upper = y_max)
end

const MODEL_SCOPE_STAGES = quote
    t[1:4] .~ Normal.(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    eta = b * x
    y_min ~ coarsened(Ordinal(StoppingRatio(), ProbitLink(), eta, t; discrimination = 0.7);
        upper = y_max)
end

function shared_oracle(c; b = nothing, structure = Cumulative(), link = LogitLink(),
                       disc = 1.0)
    structure isa Cumulative && !issorted(c; lt = <=) && return -Inf
    lp = sum(logpdf.(Normal(0, 1), c)) + (b === nothing ? 0.0 : logpdf(Normal(0, 1), b))
    lp + range_loglik(Ordinal(structure, link,
        b === nothing ? DATA.eta_data[i] : b * DATA.x[i], c; discrimination = disc)
        for i in eachindex(DATA.y_min))
end

model(source, data = DATA) = brm(source; mod = @__MODULE__)(data)
cache_dir = mkpath(joinpath(tempdir(), "brm-ordinal-explicit-range"))

function stan_problem(brmi, tag)
    sb = SBBRMI(brmi; mod = @__MODULE__)
    code = stan_code(sb)
    checked = StanBlocks.stanc_check(code; warn_pedantic = false)
    checked.ok || @error "stanc rejected $tag" output = checked.output
    @test checked.ok
    d = brm_descriptor(sb)
    problem = brm_execute(d, :instantiate; path = joinpath(cache_dir, string(d.id) * ".stan"))
    (; sb, code, d, problem, names = BridgeStan.param_names(problem.model))
end

stan_lp(p, q) = BridgeStan.log_density(p.problem.model, q; propto = false, jacobian = true)
function stan_vector(p, q, name)
    values = BridgeStan.param_constrain(p.problem.model, q)
    picked = [v for (n, v) in zip(p.names, values) if occursin(Regex("(^|_)$(name)\\.\\d+\$"), n)]
    isempty(picked) && error("no coordinates named $name in $(p.names)")
    picked
end
stan_scalar(p, q, name) = only(v for (n, v) in
    zip(p.names, BridgeStan.param_constrain(p.problem.model, q)) if n == name)

# The unconstrained RKBRMI point holding the given named values (every prior
# here is unbounded, so a coordinate is its value).
function rk_point(rk, dim, named::NamedTuple)
    u = zeros(dim)
    for j in 1:dim
        restored = rk_restore_draws(rk, reshape(Float64.(1:dim .== j), :, 1))
        for (key, value) in pairs(named)
            i = findfirst(!iszero, vec(getproperty(restored, key)))
            i === nothing || (u[j] = value isa AbstractVector ? value[i] : value)
        end
    end
    u
end
# The named values an unconstrained RKBRMI point holds; a cell latent restores
# one array per cell, here one column per cell.
function rk_named(rk, u, key)
    value = getproperty(rk_restore_draws(rk, reshape(u, :, 1)), key)
    value isa AbstractVector{<:AbstractArray} ? reduce(hcat, vec.(value)) : vec(value)
end

points(dim, k; scale = 0.4) = [scale .* randn(Random.Xoshiro(10 + j), dim) for j in 1:k]

function central_differences(problem, u; h = 1e-6)
    [(LogDensityProblems.logdensity(problem, u .+ h .* (eachindex(u) .== j)) -
      LogDensityProblems.logdensity(problem, u .- h .* (eachindex(u) .== j))) / 2h
     for j in eachindex(u)]
end

function check_jax(rk, problem, us)
    src = jax_source(rk.model, rk_bound_plan(rk))
    @test src isa JAXSource
    brm_jax_require_python() || return
    run = brm_jax_run(src, [brm_jax_inputs(rk, src)], [us])
    for (j, u) in enumerate(us)
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
        res = run.datasets[1]["points"][j]
        @test res["log_density"]["eager"] ≈ lp rtol = 1e-12
        @test res["log_density"]["jit"] ≈ lp rtol = 1e-12
        @test vec(collect(res["grad"])) ≈ g rtol = 1e-10 atol = 1e-10
    end
end

@stestset "SBBRMI: model scope" begin
    p = stan_problem(model(MODEL_SCOPE), "model scope")
    @test occursin("brm_ordinal_explicit_range_lpmf", p.code)
    @test occursin("brm_ordinal_explicit_range_lpmfs", p.code)     # pointwise
    @test occursin("brm_ordinal_explicit_range_int_rng", p.code)   # predictive
    @test p.names == ["c.1", "c.2", "c.3", "c.4", "b"]
    for point in ([-1.0, -0.2, 0.4, 1.3, 0.6], [-0.8, 0.1, 0.2, 0.9, -0.3])
        @test stan_lp(p, point) ≈ shared_oracle(point[1:4]; b = point[5]) rtol = 1e-10
    end
    # A cumulative vector that is not strictly increasing: log density -Inf.
    @test stan_lp(p, [0.5, -0.2, 0.4, 1.3, 0.6]) == -Inf

    s = stan_problem(model(MODEL_SCOPE_STAGES), "model scope, stopping ratio")
    for point in ([0.5, -0.2, 0.4, 1.3, 0.6], [-0.8, 0.1, 0.2, 0.9, -0.3])
        @test stan_lp(s, point) ≈ shared_oracle(point[1:4]; b = point[5],
            structure = StoppingRatio(), link = ProbitLink(), disc = 0.7) rtol = 1e-10
    end
end

@stestset "SBBRMI: equal codes are the exact explicit Ordinal" begin
    exact = merge(DATA, (; y_max = copy(DATA.y_min)))
    p = stan_problem(model(MODEL_SCOPE, exact), "exact range")
    plain = stan_problem(model(quote
        c[1:4] .~ Normal.(0.0, 1.0)
        b ~ Normal(0.0, 1.0)
        eta = b * x
        y_min ~ Ordinal(Cumulative(), LogitLink(), eta, c)
    end, exact), "exact plain")
    for q in ([-1.0, -0.2, 0.4, 1.3, 0.6], [-0.8, 0.1, 0.2, 0.9, -0.3])
        @test stan_lp(p, q) ≈ stan_lp(plain, q) rtol = 1e-12
    end
end

@stestset "SBBRMI: one examiner's thresholds per cell" begin
    p = stan_problem(model(POOLED), "pooled")
    @test occursin("brm_ordinal_explicit_range_lpmf", p.code)
    for q in points(length(p.names), 4)
        U = reshape(stan_vector(p, q, "u"), 3, N_EX)
        @test stan_lp(p, q) ≈ pooled_oracle(stan_scalar(p, q, "m"),
            stan_vector(p, q, "log_gap"), U) rtol = 1e-10
    end
    s = stan_problem(model(STAGES), "stages")
    for q in points(length(s.names), 3)
        @test stan_lp(s, q) ≈ stages_oracle(reshape(stan_vector(s, q, "t"), 4, N_EX)) rtol = 1e-10
    end
    c = stan_problem(model(SHARED_IN_CELL), "shared in cell")
    increasing = [-1.0, -0.2, 0.4, 1.3]
    @test stan_lp(c, increasing) ≈ shared_oracle(increasing; link = ProbitLink(),
        disc = 1.5) rtol = 1e-10
    @test stan_lp(c, [0.5, -0.2, 0.4, 1.3]) == -Inf
end

@stestset "SBBRMI: predictive draws are single categories" begin
    p = stan_problem(model(MODEL_SCOPE), "model scope draws")
    draws = brm_predictive_draws(p.d, repeat([-1.0 -0.2 0.4 1.3 0.6], 500); problem = p.problem,
        seed = 20261115)
    @test size(draws.y_min) == (500, length(DATA.x))
    @test all(in(1:5), draws.y_min)
    # Each row's draws against the uncoarsened law's category probabilities.
    c, b = [-1.0, -0.2, 0.4, 1.3], 0.6
    for i in (1, 7, 13)
        expected = probs(Ordinal(Cumulative(), LogitLink(), b * DATA.x[i], c))
        observed = [count(==(k), draws.y_min[:, i]) for k in 1:5] ./ 500
        @test maximum(abs.(observed .- expected)) < 0.08
    end
end

@stestset "RKBRMI: model scope" begin
    for (source, kw) in ((MODEL_SCOPE, (;)),
            (MODEL_SCOPE_STAGES, (; structure = StoppingRatio(), link = ProbitLink(), disc = 0.7)))
        brmi = model(source)
        p = stan_problem(brmi, "model scope")
        rk = RKBRMI(brmi)
        problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
        @test LogDensityProblems.dimension(problem) == 5
        key = haskey(kw, :structure) ? :t : :c
        named = [([-1.0, -0.2, 0.4, 1.3], 0.6), ([-0.8, 0.1, 0.2, 0.9], -0.3),
                 ([0.5, -0.2, 0.4, 1.3], 0.6)]
        us = [rk_point(rk, 5, NamedTuple{(key, :b)}((c, b))) for (c, b) in named]
        @test [rk_named(rk, u, key) for u in us] == first.(named)
        for ((c, b), u) in zip(named, us)
            lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
            expected = shared_oracle(c; b, kw...)
            if isfinite(expected)
                @test lp ≈ expected rtol = 1e-10
                @test all(isfinite, g)
                @test g ≈ central_differences(problem, u) rtol = 1e-5 atol = 1e-6
                @test lp ≈ stan_lp(p, vcat(c, b)) rtol = 1e-10
            else
                @test lp == -Inf
                @test stan_lp(p, vcat(c, b)) == -Inf
            end
        end
        @testset "the JAX module" begin
            check_jax(rk, problem, us[1:2])
        end
    end
end

@stestset "RKBRMI: an explicit Ordinal response never renumbers its codes" begin
    # Code 1 unobserved: the response binds the codes 2:5 as they stand.
    brmi = model(quote
        c[1:4] .~ Normal.(0.0, 1.0)
        b ~ Normal(0.0, 1.0)
        eta = b * x
        y_min ~ Ordinal(Cumulative(), LogitLink(), eta, c)
    end)
    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    c, b = [-1.0, -0.2, 0.4, 1.3], 0.6
    lp = LogDensityProblems.logdensity(problem, rk_point(rk, 5, (; c, b)))
    expected = sum(logpdf.(Normal(0, 1), c)) + logpdf(Normal(0, 1), b) +
        sum(logpdf(Ordinal(Cumulative(), LogitLink(), b * DATA.x[i], c), DATA.y_min[i])
            for i in eachindex(DATA.y_min))
    @test lp ≈ expected rtol = 1e-10
end

@stestset "RKBRMI: a shared declared vector read in a cell" begin
    rk = RKBRMI(model(SHARED_IN_CELL))
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    named = ([-1.0, -0.2, 0.4, 1.3], [-0.6, 0.0, 0.7, 0.8], [0.5, -0.2, 0.4, 1.3])
    us = [rk_point(rk, 4, (; c)) for c in named]
    for (k, (c, u)) in enumerate(zip(named, us))
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
        if k < 3
            @test lp ≈ shared_oracle(c; link = ProbitLink(), disc = 1.5) rtol = 1e-10
            @test g ≈ central_differences(problem, u) rtol = 1e-5 atol = 1e-6
        else
            @test lp == -Inf
        end
    end
    @testset "the JAX module" begin
        check_jax(rk, problem, us[1:2])
    end
end

@stestset "RKBRMI: one examiner's thresholds per cell" begin
    rk = RKBRMI(model(POOLED))
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    dim = LogDensityProblems.dimension(problem)
    @test dim == 1 + 3 + 3 * N_EX
    us = points(dim, 3)
    for u in us
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
        U = reshape(rk_named(rk, u, :u), 3, N_EX)
        @test lp ≈ pooled_oracle(only(rk_named(rk, u, :m)), rk_named(rk, u, :log_gap), U) rtol = 1e-10
        @test g ≈ central_differences(problem, u) rtol = 1e-5 atol = 1e-6
    end
    @testset "the JAX module" begin
        check_jax(rk, problem, us[1:2])
    end

    rk = RKBRMI(model(STAGES))
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    us = points(LogDensityProblems.dimension(problem), 2)
    for u in us
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test lp ≈ stages_oracle(reshape(rk_named(rk, u, :t), 4, N_EX)) rtol = 1e-10
        @test g ≈ central_differences(problem, u) rtol = 1e-5 atol = 1e-6
    end
    # A stopping-ratio range over thresholds computed in a cell: its survival
    # loop runs to the vector's padded length, a `fori_loop` `jax.grad`
    # reverses from RK 0.1.263 (ReactiveKernels snag `jax-source-of-an-6bfaf24a`).
    @testset "the JAX module" begin
        check_jax(rk, problem, us)
    end
end

@stestset "the reporter's model builds whole: strictness, discrimination, pooled gaps" begin
    full = quote
        m ~ Normal(0.0, 1.5)
        log_gap[1:3] .~ Normal.(0.0, 1.0)
        tau_gap ~ Exponential(1.0)
        eta ~ 0 + x + (1 | examiner)
        log_disc ~ Normal(0.0, 0.3)
        rows = groups(examiner => exids)
        @plate for e in eachindex(exids)
            u ~ Normal(zero_gap[e], tau_gap)
            s = cumsum(vcat(0.0, exp.(log_gap .+ u)))
            cuts = m .+ s .- mean(s)
            y_min[rows[e]] ~ coarsened(Ordinal(Cumulative(), LogitLink(), eta[rows[e]], cuts,
                exp(log_disc)); upper = y_max[rows[e]])
        end
    end
    brmi = model(full)
    p = stan_problem(brmi, "full")
    @test all(isfinite, [stan_lp(p, q) for q in points(length(p.names), 3; scale = 0.2)])
    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    @test LogDensityProblems.dimension(problem) == length(p.names)
    for u in points(LogDensityProblems.dimension(problem), 2; scale = 0.2)
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test isfinite(lp) && all(isfinite, g)
    end
end

@stestset "refusals name the code contract" begin
    builds = (b -> SBBRMI(b; mod = @__MODULE__), RKBRMI)
    # An upper code below its row's lower code, at model scope: refused at build.
    swapped = merge(DATA, (; y_max = vcat(DATA.y_min[1] - 1, DATA.y_max[2:end])))
    for build in builds
        err = try build(model(MODEL_SCOPE, swapped)); nothing catch e; sprint(showerror, e) end
        @test err !== nothing && occursin("row 1: upper category code", err)
    end
    # A label 0 rather than a code, in either column.
    for bad in (merge(DATA, (; y_min = vcat(0, DATA.y_min[2:end]))),
                merge(DATA, (; y_max = vcat(0, DATA.y_max[2:end])))), build in builds
        err = try build(model(MODEL_SCOPE, bad)); nothing catch e; sprint(showerror, e) end
        @test err !== nothing && occursin("category codes 1:(length(thresholds) + 1)", err)
    end
    # In a cell: RKBRMI refuses the swapped row when it builds; SBBRMI's Stan
    # program refuses it, and a code beyond K, in its data check.
    err = try RKBRMI(model(POOLED, swapped)); nothing catch e; sprint(showerror, e) end
    @test err !== nothing && occursin("row 1: upper category code", err)
    stan_error(p) = try stan_lp(p, zeros(length(p.names))); nothing catch e; sprint(showerror, e) end
    err = stan_error(stan_problem(model(POOLED, swapped), "swapped cell"))
    @test err !== nothing && occursin("upper category code is below its row's lower code", err)
    beyond = merge(DATA, (; y_max = vcat(6, DATA.y_max[2:end])))
    err = stan_error(stan_problem(model(MODEL_SCOPE, beyond), "beyond K"))
    @test err !== nothing && occursin("exceeds length(thresholds) + 1", err)
    # A cell gives its thresholds explicitly.
    implicit = quote
        rows = groups(examiner => exids)
        @plate for e in eachindex(exids)
            y_min[rows[e]] ~ coarsened(Ordinal(Cumulative(), ProbitLink(), eta_data[rows[e]]);
                upper = y_max[rows[e]])
        end
    end
    for build in builds
        err = try build(model(implicit)); nothing catch e; sprint(showerror, e) end
        @test err !== nothing && occursin("takes its thresholds explicitly", err)
    end
end
