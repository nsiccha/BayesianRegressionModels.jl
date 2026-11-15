# test/ordinal_group_thresholds.jl — `Ordinal(structure, link, eta, thresholds)`
# with thresholds the model gives (snag ordinal-varying-36cc2a82): one threshold
# vector per group computed in a `@plate` cell, such as each examiner's ordered,
# partially pooled and centred thresholds, and a declared vector at model scope.
#
# Every density is checked against an independent oracle: the priors written
# out with Distributions.jl plus `logpdf` of the executable `Ordinal` at the
# same constrained point. Every prior here is on an unbounded support, so the
# unconstrained point is the constrained one and no Jacobian enters.
#
# The reporter's model, with tau_gap, an examiner shift of eta and a
# discrimination, adds a bounded scale and BRM's default priors; its oracle
# writes them out with their log-scale Jacobian.
#
# RKBRMI's per-group (cell-computed) thresholds need ReactiveKernelsPPL 0.1.201
# (ReactiveKernels snag `rkppl-sampled-pe-bc377786`).
#
# Run: julia --project=test test/ordinal_group_thresholds.jl
#      (RK_JAX_PYTHON names a Python with jax for the JAX module parity.)
using Test, Random, LinearAlgebra
using BayesianRegressionModels, Distributions, LogDensityProblems
using StanBlocks
const BridgeStan = StanBlocks.BridgeStan
# `import`, not `using`: ReactiveKernelsPPL also exports `ProbitLink`,
# `LogitLink` and `CloglogLink` (BRM snag `brm-ordinal-link-ca8cd39c`).
import ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels
using ReactiveKernelsPPL: jax_source
using ReactiveKernels: JAXSource
import Enzyme
using DifferentiationInterface: AutoEnzyme

include("jax_module_driver.jl")

const N_EX, PER = 3, 8
function group_data(seed)
    rng = Random.Xoshiro(seed)
    examiner = repeat(1:N_EX, inner = PER)
    # Category 5 is never observed; the thresholds still define it.
    y = rand(rng, 1:4, N_EX * PER)
    (; y, x = randn(rng, N_EX * PER), eta_data = 0.3 .* randn(rng, N_EX * PER),
       examiner, exids = collect(1:N_EX), zero_gap = [zeros(3) for _ in 1:N_EX],
       zero_stage = [zeros(4) for _ in 1:N_EX])
end
const DATA = group_data(20261115)

# The reporter's decomposition: population log-gaps, each examiner's
# deviations from them, thresholds centred on a shared mean `m` (so they do not
# duplicate a location on the latent scale), ordered by positive gaps.
pooled_cuts(m, log_gap, u) = (s = cumsum(vcat(0.0, exp.(log_gap .+ u))); m .+ s .- mean(s))

const POOLED = quote
    m ~ Normal(0.0, 1.5)
    log_gap[1:3] .~ Normal.(0.0, 1.0)
    rows = groups(examiner => exids)
    @plate for e in eachindex(exids)
        u ~ Normal(zero_gap[e], 0.5)
        s = cumsum(vcat(0.0, exp.(log_gap .+ u)))
        cuts = m .+ s .- mean(s)
        y[rows[e]] ~ Ordinal(Cumulative(), ProbitLink(), eta_data[rows[e]], cuts)
    end
end

function pooled_oracle(m, log_gap, U)   # U: 3 × N_EX, one column per examiner
    lp = logpdf(Normal(0, 1.5), m) + sum(logpdf.(Normal(0, 1), log_gap)) +
         sum(logpdf.(Normal(0, 0.5), U))
    for i in eachindex(DATA.y)
        cuts = pooled_cuts(m, log_gap, U[:, DATA.examiner[i]])
        lp += logpdf(Ordinal(Cumulative(), ProbitLink(), DATA.eta_data[i], cuts), DATA.y[i])
    end
    lp
end

# The reporter's whole model (brm-use, "Thresholds the model gives").
const FULL = quote
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
        y[rows[e]] ~ Ordinal(Cumulative(), ProbitLink(), eta[rows[e]], cuts,
            exp(log_disc))
    end
end

# At unconstrained values: `tau_gap` enters through its logarithm (Jacobian
# `log(tau_gap)`). BRM's default priors are standard normal on the coefficient
# of `x`, on the log scale of the examiner effects and on their non-centred
# deviations `xi` (effect `exp(log_scale) * xi`).
function full_oracle(m, log_gap, log_tau_gap, beta, log_scale, xi, log_disc, U)
    tau_gap = exp(log_tau_gap)
    lp = logpdf(Normal(0, 1.5), m) + sum(logpdf.(Normal(0, 1), log_gap)) +
         logpdf(Exponential(1.0), tau_gap) + log_tau_gap +
         logpdf(Normal(0, 1), beta) + logpdf(Normal(0, 1), log_scale) +
         sum(logpdf.(Normal(0, 1), xi)) + logpdf(Normal(0, 0.3), log_disc) +
         sum(logpdf.(Normal(0, tau_gap), U))
    for i in eachindex(DATA.y)
        e = DATA.examiner[i]
        eta = beta * DATA.x[i] + exp(log_scale) * xi[e]
        d = Ordinal(Cumulative(), ProbitLink(), eta, pooled_cuts(m, log_gap, U[:, e]);
            discrimination = exp(log_disc))
        lp += logpdf(d, DATA.y[i])
    end
    lp
end

const STAGES = quote
    rows = groups(examiner => exids)
    @plate for e in eachindex(exids)
        t ~ Normal(zero_stage[e], 1.0)
        y[rows[e]] ~ Ordinal(StoppingRatio(), CloglogLink(), eta_data[rows[e]], t;
            discrimination = 0.7)
    end
end

function stages_oracle(T)   # T: 4 × N_EX
    lp = sum(logpdf.(Normal(0, 1), T))
    for i in eachindex(DATA.y)
        d = Ordinal(StoppingRatio(), CloglogLink(), DATA.eta_data[i], T[:, DATA.examiner[i]];
            discrimination = 0.7)
        lp += logpdf(d, DATA.y[i])
    end
    lp
end

# A cell latent read whole as one examiner's cumulative cutpoints: nothing
# orders it, so a point where one examiner's vector is not strictly increasing
# has log density -Inf.
const RAW_CELL = quote
    rows = groups(examiner => exids)
    @plate for e in eachindex(exids)
        t ~ Normal(zero_stage[e], 1.0)
        y[rows[e]] ~ Ordinal(Cumulative(), LogitLink(), eta_data[rows[e]], t)
    end
end

function raw_cell_oracle(T)   # T: 4 × N_EX
    all(e -> issorted(T[:, e]; lt = <=), 1:N_EX) || return -Inf
    lp = sum(logpdf.(Normal(0, 1), T))
    for i in eachindex(DATA.y)
        d = Ordinal(Cumulative(), LogitLink(), DATA.eta_data[i], T[:, DATA.examiner[i]])
        lp += logpdf(d, DATA.y[i])
    end
    lp
end

const SHARED_IN_CELL = quote
    c[1:4] .~ Normal.(0.0, 1.0)
    rows = groups(examiner => exids)
    @plate for e in eachindex(exids)
        y[rows[e]] ~ Ordinal(Cumulative(), LogitLink(), eta_data[rows[e]], c;
            discrimination = 1.5)
    end
end

const MODEL_SCOPE = quote
    c[1:4] .~ Normal.(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    eta = b * x
    y ~ Ordinal(Cumulative(), ProbitLink(), eta, c)
end

function shared_oracle(c; b = nothing, link = LogitLink(), disc = 1.5)
    issorted(c; lt = <=) || return -Inf
    lp = sum(logpdf.(Normal(0, 1), c)) + (b === nothing ? 0.0 : logpdf(Normal(0, 1), b))
    for i in eachindex(DATA.y)
        eta = b === nothing ? DATA.eta_data[i] : b * DATA.x[i]
        lp += logpdf(Ordinal(Cumulative(), link, eta, c; discrimination = disc), DATA.y[i])
    end
    lp
end

model(source, data = DATA) = brm(source; mod = @__MODULE__)(data)
cache_dir = mkpath(joinpath(tempdir(), "brm-ordinal-group-thresholds"))

function stan_problem(brmi, tag)
    sb = SBBRMI(brmi; mod = @__MODULE__)
    code = stan_code(sb)
    checked = StanBlocks.stanc_check(code; warn_pedantic = false)
    checked.ok || @error "stanc rejected $tag" output = checked.output
    @test checked.ok
    d = brm_descriptor(sb)
    problem = brm_execute(d, :instantiate; path = joinpath(cache_dir, string(d.id) * ".stan"))
    (; sb, code, problem, names = BridgeStan.param_names(problem.model))
end

stan_lp(p, q) = BridgeStan.log_density(p.problem.model, q; propto = false, jacobian = true)
# Coordinates named `<name>.<k>` (or a cell latent `<cell>_<name>.<k>`), in order.
function stan_vector(p, q, name)
    values = BridgeStan.param_constrain(p.problem.model, q)
    picked = [v for (n, v) in zip(p.names, values) if occursin(Regex("(^|_)$(name)\\.\\d+\$"), n)]
    isempty(picked) && error("no coordinates named $name in $(p.names)")
    picked
end
stan_scalar(p, q, name) = only(v for (n, v) in
    zip(p.names, BridgeStan.param_constrain(p.problem.model, q)) if n == name)

# The restored values of an RKBRMI point by flat name: `b.z` for a nested
# field, and a cell latent's per-group blocks concatenated in group order.
rk_named(rk, u) = _rk_flat(rk_restore_draws(rk, reshape(u, :, 1)), "")
_rk_flat(x::NamedTuple, prefix) = mapfoldl(merge, keys(x); init = Dict{String,Vector{Float64}}()) do k
    _rk_flat(getproperty(x, k), string(prefix, k, "."))
end
_rk_flat(x::AbstractVector{<:AbstractArray}, prefix) = Dict(String(chop(prefix)) => reduce(vcat, vec.(x)))
_rk_flat(x::AbstractArray, prefix) = Dict(String(chop(prefix)) => vec(Float64.(x)))

# The RKBRMI point holding the given UNCONSTRAINED values, by flat name (a
# bounded value as its unconstrained coordinate, `log(tau)` for a scale):
# moving one coordinate from zero says which named entry it is.
function rk_point(rk, dim, named)
    base = rk_named(rk, zeros(dim))
    u = zeros(dim)
    for j in 1:dim
        moved = rk_named(rk, Float64.(1:dim .== j))
        hits = [(string(key), findfirst(moved[string(key)] .!= base[string(key)]))
                for (key, _) in pairs(named)]
        filter!(h -> h[2] !== nothing, hits)
        length(hits) == 1 || error("coordinate $j moves $(first.(hits)), not one named entry")
        key, i = only(hits)
        value = named[Symbol(key)]
        u[j] = value isa AbstractVector ? value[i] : value
    end
    u
end

points(dim, k; scale = 0.4) = [scale .* randn(Random.Xoshiro(10 + j), dim) for j in 1:k]

@testset "SBBRMI: one examiner's thresholds per cell" begin
    @testset "centred, partially pooled cumulative probit" begin
        p = stan_problem(model(POOLED), "pooled")
        @test occursin("brm_ordinal_explicit", p.code)
        @test occursin("brm_ordinal_explicit_lpmf", p.code)
        @test occursin("brm_ordinal_explicit_int_rng", p.code)   # predictive draws
        for q in points(length(p.names), 4)
            U = reshape(stan_vector(p, q, "u"), 3, N_EX)
            @test stan_lp(p, q) ≈ pooled_oracle(stan_scalar(p, q, "m"),
                stan_vector(p, q, "log_gap"), U) rtol = 1e-10
        end
    end
    @testset "per-examiner stopping-ratio thresholds and a discrimination keyword" begin
        p = stan_problem(model(STAGES), "stages")
        for q in points(length(p.names), 3)
            @test stan_lp(p, q) ≈ stages_oracle(reshape(stan_vector(p, q, "t"), 4, N_EX)) rtol = 1e-10
        end
    end
    @testset "a shared declared vector read in the cell" begin
        p = stan_problem(model(SHARED_IN_CELL), "shared in cell")
        increasing, decreasing = [-1.0, -0.2, 0.4, 1.3], [0.5, -0.2, 0.4, 1.3]
        @test stan_lp(p, increasing) ≈ shared_oracle(increasing) rtol = 1e-10
        # A cumulative vector that is not strictly increasing: log density -Inf.
        @test stan_lp(p, decreasing) == -Inf
        @test stan_lp(p, [-1.0, -0.2, -0.2, 1.3]) == -Inf
    end
end

@testset "SBBRMI and RKBRMI: explicit thresholds at model scope" begin
    brmi = model(MODEL_SCOPE)
    p = stan_problem(brmi, "model scope")
    @test p.names == ["c.1", "c.2", "c.3", "c.4", "b"]
    point = [-1.0, -0.2, 0.4, 1.3, 0.6]
    @test stan_lp(p, point) ≈ shared_oracle(point[1:4]; b = point[5],
        link = ProbitLink(), disc = 1.0) rtol = 1e-10
    @test stan_lp(p, [0.5, -0.2, 0.4, 1.3, 0.6]) == -Inf

    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    @test LogDensityProblems.dimension(problem) == 5
    # RKBRMI orders its coordinates differently from SBBRMI (`b` first), so each
    # point is placed by name: (c, b) on both backends.
    named = [([-1.0, -0.2, 0.4, 1.3], 0.6), ([-0.8, 0.1, 0.2, 0.9], -0.3),
             ([0.5, -0.2, 0.4, 1.3], 0.6)]
    us = [rk_point(rk, 5, (; c, b)) for (c, b) in named]
    @test [vec(rk_restore_draws(rk, reshape(u, :, 1)).c) for u in us] == first.(named)
    for ((c, b), u) in zip(named, us)
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
        expected = shared_oracle(c; b, link = ProbitLink(), disc = 1.0)
        if isfinite(expected)
            @test lp ≈ expected rtol = 1e-10
            # Native Enzyme gradient against central differences of the density.
            h = 1e-6
            fd = [(LogDensityProblems.logdensity(problem, u .+ h .* (1:5 .== j)) -
                   LogDensityProblems.logdensity(problem, u .- h .* (1:5 .== j))) / 2h for j in 1:5]
            @test all(isfinite, g)
            @test g ≈ fd rtol = 1e-5 atol = 1e-6
            # The same physical point on SBBRMI, in its own coordinate order.
            @test lp ≈ stan_lp(p, vcat(c, b)) rtol = 1e-10
        else
            @test lp == -Inf
            @test stan_lp(p, vcat(c, b)) == -Inf
        end
    end
    finite = us[1:2]
    @test all(u -> isfinite(LogDensityProblems.logdensity(problem, u)), finite)

    @testset "the JAX module of the RKBRMI" begin
        src = jax_source(rk.model, rk_bound_plan(rk))
        @test src isa JAXSource
        if brm_jax_require_python()
            run = brm_jax_run(src, [brm_jax_inputs(rk, src)], [finite])
            for (j, u) in enumerate(finite)
                lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
                res = run.datasets[1]["points"][j]
                @test res["log_density"]["eager"] ≈ lp rtol = 1e-12
                @test res["log_density"]["jit"] ≈ lp rtol = 1e-12
                @test vec(collect(res["grad"])) ≈ g rtol = 1e-10 atol = 1e-10
            end
        end
    end
end

# Thresholds ordered by construction, at model scope: prior draws are valid.
# (A model-scope value applies per row, so it is written without dots.)
const MODEL_SCOPE_GAPS = quote
    c1 ~ Normal(-1.0, 1.0)
    log_gap[1:3] .~ Normal.(0.0, 1.0)
    cuts = cumsum(vcat(c1, exp(log_gap)))
    b ~ Normal(0.0, 1.0)
    eta = b * x
    y ~ Ordinal(Cumulative(), ProbitLink(), eta, cuts)
end

@testset "an omitted response: prior draws (SBBRMI)" begin
    # Explicit thresholds read no response values, so the model lowers without
    # them, to BRM's prior program (generated quantities, no sampled
    # coordinates).
    d = brm_descriptor(SBBRMI(model(MODEL_SCOPE_GAPS, Base.structdiff(DATA, (; y = nothing)));
        mod = @__MODULE__))
    problem = brm_execute(d, :instantiate; path = joinpath(cache_dir, string(d.id) * ".stan"))
    @test LogDensityProblems.dimension(problem) == 0
    n_draws = 4000
    prior = brm_predictive_draws(d, zeros(n_draws, 0); problem, seed = 20261115)
    @test size(prior.y) == (n_draws, length(DATA.x))
    @test all(in(1:5), prior.y)
    # The marginal category frequencies against an independent Monte Carlo of
    # the same prior through the executable `Ordinal`.
    rng = Random.Xoshiro(7)
    reference = zeros(5)
    for _ in 1:n_draws
        c1, log_gap, b = rand(rng, Normal(-1, 1)), randn(rng, 3), randn(rng)
        cuts = cumsum(vcat(c1, exp.(log_gap)))
        for x in DATA.x
            reference[rand(rng, Ordinal(Cumulative(), ProbitLink(), b * x, cuts))] += 1
        end
    end
    reference ./= sum(reference)
    observed = [count(==(k), prior.y) for k in 1:5] ./ length(prior.y)
    # Each frequency is a mean of ~10^5 draws, correlated within a prior draw;
    # 0.02 is several standard errors.
    @test maximum(abs.(observed .- reference)) < 0.02
    @test all(>(0), observed)   # every category, the never-observed 5 included

    # A prior that does not order the vector draws thresholds outside the
    # cumulative law's domain; that draw is refused, not mis-drawn.
    d_bad = brm_descriptor(SBBRMI(model(MODEL_SCOPE, Base.structdiff(DATA, (; y = nothing)));
        mod = @__MODULE__))
    problem_bad = brm_execute(d_bad, :instantiate;
        path = joinpath(cache_dir, string(d_bad.id) * ".stan"))
    err = try
        brm_predictive_draws(d_bad, zeros(200, 0); problem = problem_bad, seed = 1)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing && occursin("not strictly increasing", err)
end

@testset "RKBRMI: a shared declared vector read in a cell" begin
    rk = RKBRMI(model(SHARED_IN_CELL))
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    named = ([-1.0, -0.2, 0.4, 1.3], [-0.6, 0.0, 0.7, 0.8], [0.5, -0.2, 0.4, 1.3])
    for (k, c) in enumerate(named)
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, rk_point(rk, 4, (; c)))
        if k < 3
            @test lp ≈ shared_oracle(c) rtol = 1e-10
            @test all(isfinite, g)
        else
            @test lp == -Inf
        end
    end
end

# RKBRMI at SBBRMI's points, placed by name: the density against the oracle and
# against SBBRMI at the same point, the native Enzyme Reverse gradient against
# central differences, and the JAX module's density and `jax.grad`.
# `named_and_oracle(p, q)` reads SBBRMI's point `q` and returns the RKBRMI
# names' unconstrained values and the oracle's density.
function check_rk_cells(named_and_oracle, source, tag; npoints = 3, scale = 0.4)
    brmi = model(source)
    p = stan_problem(brmi, tag)
    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
    dim = LogDensityProblems.dimension(problem)
    @test dim == length(p.names)
    us = Vector{Float64}[]
    for q in points(dim, npoints; scale)
        named, expected = named_and_oracle(p, q)
        u = rk_point(rk, dim, named)
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test isfinite(expected)
        @test lp ≈ expected rtol = 1e-10
        @test lp ≈ stan_lp(p, q) rtol = 1e-10
        h = 1e-6
        fd = [(LogDensityProblems.logdensity(problem, u .+ h .* (1:dim .== j)) -
               LogDensityProblems.logdensity(problem, u .- h .* (1:dim .== j))) / 2h for j in 1:dim]
        @test all(isfinite, g)
        @test g ≈ fd rtol = 1e-5 atol = 1e-6
        push!(us, u)
    end
    @testset "the JAX module of the RKBRMI" begin
        src = jax_source(rk.model, rk_bound_plan(rk))
        @test src isa JAXSource
        if brm_jax_require_python()
            run = brm_jax_run(src, [brm_jax_inputs(rk, src)], [us])
            for (j, u) in enumerate(us)
                lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
                res = run.datasets[1]["points"][j]
                @test res["log_density"]["eager"] ≈ lp rtol = 1e-12
                @test res["log_density"]["jit"] ≈ lp rtol = 1e-12
                @test vec(collect(res["grad"])) ≈ g rtol = 1e-10 atol = 1e-10
            end
        end
    end
    (; p, rk, problem)
end

@testset "RKBRMI: one examiner's thresholds per cell" begin
    @testset "centred, partially pooled cumulative probit" begin
        check_rk_cells(POOLED, "pooled"; npoints = 4) do p, q
            m, log_gap = stan_scalar(p, q, "m"), stan_vector(p, q, "log_gap")
            U = reshape(stan_vector(p, q, "u"), 3, N_EX)
            (; m, log_gap, u = vec(U)), pooled_oracle(m, log_gap, U)
        end
    end
    @testset "per-examiner stopping-ratio thresholds and a discrimination keyword" begin
        check_rk_cells(STAGES, "stages") do p, q
            T = reshape(stan_vector(p, q, "t"), 4, N_EX)
            (; t = vec(T)), stages_oracle(T)
        end
    end
    @testset "the reporter's whole model: strictness, discrimination, pooled gaps" begin
        check_rk_cells(FULL, "full"; scale = 0.3) do p, q
            m, log_gap = stan_scalar(p, q, "m"), stan_vector(p, q, "log_gap")
            log_tau_gap = log(stan_scalar(p, q, "tau_gap"))
            beta = stan_scalar(p, q, "pop_eta_beta_pop.1")
            log_scale = stan_scalar(p, q, "r_eta_examiner_log_scale")
            xi = stan_vector(p, q, "r_eta_examiner_xi")
            log_disc = stan_scalar(p, q, "log_disc")
            U = reshape(stan_vector(p, q, "u"), 3, N_EX)
            named = (; m, log_gap, tau_gap = log_tau_gap, log_disc, u = vec(U),
                var"pop_eta.beta_pop" = beta, var"b_examiner.tau" = log_scale,
                var"b_examiner.z" = xi)
            named, full_oracle(m, log_gap, log_tau_gap, beta, log_scale, xi, log_disc, U)
        end
    end
    @testset "a cell latent as cumulative cutpoints: -Inf where one is not increasing" begin
        brmi = model(RAW_CELL)
        p = stan_problem(brmi, "raw cell")
        @test length(p.names) == 4N_EX
        rk = RKBRMI(brmi)
        problem = rk_logdensity_problem(rk; ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
        ordered = [-1.2 -0.9 -1.5; -0.3 0.1 -0.4; 0.5 0.6 0.2; 1.4 1.1 0.9]
        tie = copy(ordered); tie[3, 2] = tie[2, 2]          # examiner 2: a tie
        swapped = copy(ordered); swapped[[1, 4], 3] = swapped[[4, 1], 3]
        for (T, finite) in ((ordered, true), (ordered .+ 0.3, true), (tie, false), (swapped, false))
            u = rk_point(rk, 4N_EX, (; t = vec(T)))
            lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
            @test stan_vector(p, vec(T), "t") == vec(T)   # SBBRMI's point is vec(T)
            if finite
                @test lp ≈ raw_cell_oracle(T) rtol = 1e-10
                @test lp ≈ stan_lp(p, vec(T)) rtol = 1e-10
                @test all(isfinite, g)
            else
                @test raw_cell_oracle(T) == -Inf
                @test lp == -Inf
                @test stan_lp(p, vec(T)) == -Inf
            end
        end
    end
end

@testset "refusals name the explicit-threshold contract" begin
    bad = merge(DATA, (; y = vcat(0, DATA.y[2:end])))
    for build in (b -> SBBRMI(b; mod = @__MODULE__), RKBRMI)
        err = try build(model(MODEL_SCOPE, bad)); nothing catch e; sprint(showerror, e) end
        @test err !== nothing && occursin("category codes 1:(length(thresholds) + 1)", err)
    end
    no_thresholds = quote
        rows = groups(examiner => exids)
        @plate for e in eachindex(exids)
            y[rows[e]] ~ Ordinal(Cumulative(), ProbitLink(), eta_data[rows[e]])
        end
    end
    for build in (b -> SBBRMI(b; mod = @__MODULE__), RKBRMI)
        err = try build(model(no_thresholds)); nothing catch e; sprint(showerror, e) end
        @test err !== nothing && occursin("takes its thresholds explicitly", err)
    end
    effects = quote
        c[1:4] .~ Normal.(0.0, 1.0)
        b ~ Normal(0.0, 1.0)
        eta = b * x
        y ~ Ordinal(StoppingRatio(), ProbitLink(), eta, c; per_threshold = (x,))
    end
    err = try SBBRMI(model(effects); mod = @__MODULE__); nothing catch e; sprint(showerror, e) end
    @test err !== nothing && occursin("write threshold effects into the thresholds", err)
end

@testset "the reporter's model builds whole: strictness, discrimination, pooled gaps" begin
    p = stan_problem(model(FULL), "full")
    lps = [stan_lp(p, q) for q in points(length(p.names), 3; scale = 0.2)]
    @test all(isfinite, lps)
end
