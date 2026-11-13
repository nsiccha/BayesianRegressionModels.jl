# test/derived_term_parameters.jl — a term's own parameter set to an expression
# over model values: `sd(lp, hsgp(x)) = sqrt(log1pexp(s))` with
# `s ~ Normal(0, 1)`, and the same `=` on the `length_scale`, `ar`, `mixing`
# addresses of `hsgp`, `gp`, `dar`, `cdar`, `car` and (SBBRMI) `rw` terms (snag gp-raw-scale-hyp-2a375503: a
# Gaussian process whose amplitude and length scale are transforms of raw-scale
# normal parameters).
#
# Run: julia --project=test test/derived_term_parameters.jl [testset filter]
# Set BRM_DERIVED_TERM_RUNTIME=0 to skip the BridgeStan/RK density probes.
#
# Density checks:
# - an independent oracle: the HSGP log joint written out in Julia from
#   `StatisticalPreparation.hsgp_basis` and the squared-exponential spectral
#   weights, against SBBRMI, and RKBRMI against SBBRMI through the coordinate
#   transport (value and gradient);
# - the twin identity on each backend: the derived model is its twin that
#   samples the term parameter under an explicit prior, with that coordinate at
#   the derived value. With the Jacobian, the twin's log density minus the
#   derived model's is the prior's log density at the value plus the log
#   Jacobian of the parameter's transform there, at every point.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, LinearAlgebra
using Distributions: Exponential, Normal, LogNormal, Beta, logpdf
using LogExpFunctions: log1pexp, logistic
using BayesianRegressionModels.StatisticalPreparation: hsgp_basis

const DERIVED_RUNTIME = get(ENV, "BRM_DERIVED_TERM_RUNTIME", "1") != "0"
const DERIVED_CACHE = joinpath(tempdir(), "brm-derived-term-parameters")

Random.seed!(20261113)
const N = 30
const DF = (; x=round.(2 .* rand(N) .- 1; digits=3), u=round.(rand(N); digits=3),
            g=repeat(1:3, inner=10), t=collect(1.0:N), y=round.(randn(N); digits=3),
            step=repeat(1.0:10.0, 3), C=[1 0.3 0.1; 0.3 1 0.2; 0.1 0.2 1],
            from=[1, 2], to=[2, 3])

build(builder; data=DF) = SBBRMI(builder(data); mod=@__MODULE__, total_groups=())
stanc_ok(sb) = StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
function instantiate(sb)
    mkpath(DERIVED_CACHE)
    code = BRM.stan_code(sb)
    BRM.stan_instantiate(sb; path=joinpath(DERIVED_CACHE, string(hash(code)) * ".stan"))
end

# ---- the reporter's shape: raw-scale hyperparameters of an HSGP ------------

notebook = @brm begin
    mu ~ 0 + hsgp(x; k = 8, c = 1.5)
    sd(mu, hsgp(x)) = sqrt(log1pexp(sf))
    length_scale(mu, hsgp(x)) = 1 / log1pexp(l)
    sf ~ Normal(0, 1)
    l ~ Normal(0, 1)
    theta ~ Normal(0, 1)
    y ~ Normal(mu, exp(theta / 2))
end

# The log joint written out: basis and frequencies from the public helper,
# the squared-exponential spectral weights, `f = PHI * (w .* z)`.
function notebook_oracle(p)
    PHI, lambda = hsgp_basis(DF.x; k=8, c=1.5)
    amplitude, rho = sqrt(log1pexp(p.sf)), 1 / log1pexp(p.l)
    w = amplitude .* sqrt(rho * sqrt(2pi)) .* exp.(-rho^2 .* vec(lambda) ./ 4)
    f = Matrix(PHI) * (w .* p.z)
    logpdf(Normal(), p.sf) + logpdf(Normal(), p.l) + logpdf(Normal(), p.theta) +
        sum(logpdf.(Normal(), p.z)) + sum(logpdf.(Normal.(f, exp(p.theta / 2)), DF.y))
end

# ---- derived models and their sampled twins ---------------------------------
#
# Each case: the derived builder, its twin, and per extra twin coordinate
# (matched by a name pattern) the twin's prior, the derived value as a
# function of the shared point, and the parameter's transform.

hsgp_derived = @brm begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x; k = 5, c = 1.5)
    sd(mu, hsgp(x)) = exp(a)
    length_scale(mu, hsgp(x)) = 1 / log1pexp(b)
    y ~ Normal(mu, sigma)
end
hsgp_twin = @brm begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x; k = 5, c = 1.5)
    sd(mu, hsgp(x)) ~ Exponential(1.0)
    length_scale(mu, hsgp(x)) ~ LogNormal(0.0, 1.0)
    y ~ Normal(mu, sigma)
end
# One expression sets an anisotropic term's every length scale.
aniso_derived = @brm begin
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x, u; k = 3, c = 1.5, iso = false)
    length_scale(mu, hsgp(x, u)) = exp(b)
    y ~ Normal(mu, sigma)
end
aniso_twin = @brm begin
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x, u; k = 3, c = 1.5, iso = false)
    length_scale(mu, hsgp(x, u)) ~ LogNormal(0.0, 1.0)
    y ~ Normal(mu, sigma)
end
grouped_derived = @brm begin
    a ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x; k = 4, c = 1.5, by = g)
    sd(mu, hsgp(x)) = exp(a)
    y ~ Normal(mu, sigma)
end
grouped_twin = @brm begin
    a ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x; k = 4, c = 1.5, by = g)
    sd(mu, hsgp(x)) ~ Exponential(1.0)
    y ~ Normal(mu, sigma)
end
periodic_derived = @brm begin
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(t; k = 3, cov = :periodic, period = 12.0)
    length_scale(:, hsgp(t; cov = :periodic)) = exp(b)
    y ~ Normal(mu, sigma)
end
periodic_twin = @brm begin
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(t; k = 3, cov = :periodic, period = 12.0)
    length_scale(:, hsgp(t; cov = :periodic)) ~ LogNormal(0.0, 1.0)
    y ~ Normal(mu, sigma)
end
gp_derived = @brm begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + gp(x)
    sd(mu, gp(x)) = exp(a)
    length_scale(mu, gp(x)) = exp(b)
    y ~ Normal(mu, sigma)
end
gp_twin = @brm begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + gp(x)
    sd(mu, gp(x)) ~ Exponential(1.0)
    length_scale(mu, gp(x)) ~ LogNormal(0.0, 1.0)
    y ~ Normal(mu, sigma)
end
dar_derived = @brm begin
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + x + dar(t)
    ar(:, dar(t)) = logistic(b)
    y ~ Normal(mu, sigma)
end
dar_twin = @brm begin
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + x + dar(t)
    ar(:, dar(t)) ~ Normal(0.5, 0.2)
    y ~ Normal(mu, sigma)
end
cdar_derived = @brm begin
    a ~ Normal(-1.0, 0.5)
    sigma ~ Exponential(1.0)
    mu ~ 1 + cdar(step; by=g, cor=C)
    sd(:, cdar(step)) = exp(a)
    y ~ Normal(mu, sigma)
end
cdar_twin = @brm begin
    a ~ Normal(-1.0, 0.5)
    sigma ~ Exponential(1.0)
    mu ~ 1 + cdar(step; by=g, cor=C)
    sd(:, cdar(step)) ~ Exponential(1.0)
    y ~ Normal(mu, sigma)
end

car_derived = @brm begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + car(g; edges=(from, to), type=:bym2)
    sd(:, car(g)) = exp(a)
    mixing(:, car(g)) = logistic(b)
    y ~ Normal(mu, sigma)
end
car_twin = @brm begin
    a ~ Normal(0.0, 1.0)
    b ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu ~ 1 + car(g; edges=(from, to), type=:bym2)
    sd(:, car(g)) ~ Exponential(1.0)
    mixing(:, car(g)) ~ Beta(2.0, 2.0)
    y ~ Normal(mu, sigma)
end
# RKBRMI has no `rw` lowering (todo `1hq7nm0`): SBBRMI only.
rw_derived = @brm begin
    a ~ Normal(-1.0, 0.5)
    sigma ~ Exponential(1.0)
    mu ~ 1 + x + rw(t)
    sd(:, rw(t)) = exp(a)
    y ~ Normal(mu, sigma)
end
rw_twin = @brm begin
    a ~ Normal(-1.0, 0.5)
    sigma ~ Exponential(1.0)
    mu ~ 1 + x + rw(t)
    sd(:, rw(t)) ~ Exponential(1.0)
    y ~ Normal(mu, sigma)
end

positive(law, value) = logpdf(law, value) + log(value)          # exp transform
unit(law, value) = logpdf(law, value) + log(value) + log1p(-value) # logistic transform

const DERIVED_CASES = [
    ("hsgp", hsgp_derived, hsgp_twin, [
        ("rho", p -> positive(LogNormal(0.0, 1.0), 1 / log1pexp(p["b"])), p -> log(1 / log1pexp(p["b"]))),
        ("sigma", p -> positive(Exponential(1.0), exp(p["a"])), p -> p["a"])]),
    ("aniso", aniso_derived, aniso_twin, [
        ("rho", p -> positive(LogNormal(0.0, 1.0), exp(p["b"])), p -> p["b"])]),
    ("grouped", grouped_derived, grouped_twin, [
        ("sigma", p -> positive(Exponential(1.0), exp(p["a"])), p -> p["a"])]),
    ("periodic", periodic_derived, periodic_twin, [
        ("rho", p -> positive(LogNormal(0.0, 1.0), exp(p["b"])), p -> p["b"])]),
    ("gp", gp_derived, gp_twin, [
        ("rho", p -> positive(LogNormal(0.0, 1.0), exp(p["b"])), p -> p["b"]),
        ("sigma", p -> positive(Exponential(1.0), exp(p["a"])), p -> p["a"])]),
    # The twin's persistence prior is a Normal restricted to [0, 1], whose
    # normalizing mass is constant; the identity holds up to that constant.
    ("dar", dar_derived, dar_twin, [
        ("beta", p -> unit(Normal(0.5, 0.2), logistic(p["b"])), p -> p["b"])]),
    ("cdar", cdar_derived, cdar_twin, [
        ("sigma", p -> positive(Exponential(1.0), exp(p["a"])), p -> p["a"])]),
    ("car", car_derived, car_twin, [
        ("sigma", p -> positive(Exponential(1.0), exp(p["a"])), p -> p["a"]),
        ("rho", p -> unit(Beta(2.0, 2.0), logistic(p["b"])), p -> p["b"])]),
    ("rw", rw_derived, rw_twin, [
        ("sigma", p -> positive(Exponential(1.0), exp(p["a"])), p -> p["a"])]),
]
const OFFSET_RAW = Dict("cdar" => Dict("a" => Normal(-1.0, 0.5)),
                        "rw" => Dict("a" => Normal(-1.0, 0.5)))

# The raw parameters the derived values read. Nothing in a twin reads them, so
# a twin leaves them out and the identity subtracts their priors.
const RAW_PRIORS = Dict("a" => Normal(0.0, 1.0), "b" => Normal(0.0, 1.0))

# The twin identity at three points: `twin(v) - derived(u)` equals the summed
# role terms less the raw parameters' priors, up to one constant across points
# (`constant = true`) or exactly.
function check_twin(logdensity, names, twin_logdensity, twin_names, roles, raw, rng; constant)
    names, twin_names = string.(names), string.(twin_names)
    extra = setdiff(twin_names, names)
    own = setdiff(names, twin_names)
    @test Set(own) == Set(keys(raw))
    matched = [only(r for r in roles if occursin(first(r), n)) for n in extra]
    @test length(extra) >= length(roles)
    residuals = map(1:3) do _
        u = 0.5 .* randn(rng, length(names))
        point = Dict(zip(names, u))
        v = zeros(length(twin_names))
        for (j, n) in enumerate(names)
            n in own || (v[findfirst(==(n), twin_names)] = u[j])
        end
        expected = -sum(logpdf(raw[n], point[n]) for n in own)
        for (n, (_, term, coordinate)) in zip(extra, matched)
            v[findfirst(==(n), twin_names)] = coordinate(point)
            expected += term(point)
        end
        twin_logdensity(v) - logdensity(u) - expected
    end
    @test maximum(residuals) - minimum(residuals) <= 1e-9
    constant || @test first(residuals) ≈ 0.0 atol=1e-8
end

@stestset "a value over model values is accepted where a known value is" begin
    for (name, derived, _, _) in DERIVED_CASES
        sb = build(derived)
        @test stanc_ok(sb)
    end
    @test stanc_ok(build(notebook))
    # The emitted Stan computes the parameter where the twin samples it.
    code = BRM.stan_code(build(notebook))
    @test occursin("hsgp_x_sigma = sqrt(log1p_exp(sf))", code)
    @test occursin("hsgp_x_rho_iso = (1 / log1p_exp(l))", code)
    @test !occursin("hsgp_x_sigma;", code) && !occursin("hsgp_x_rho_iso;", code)
end

@stestset "refusals name the statement" begin
    # A distribution is a prior, not a value.
    dist = @brm begin
        mu ~ 1 + hsgp(x; k = 3)
        sd(mu, hsgp(x)) = Normal(0.0, 1.0)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "A distribution is not a value" build(dist)
    # A value and a hyper-predictor of the same parameter contradict each other.
    hyper = @brm begin
        a ~ Normal(0.0, 1.0)
        mu ~ 1 + hsgp(x; k = 3, by = g)
        log(length_scale(mu, hsgp(x))) ~ 1 + (1 | g)
        length_scale(mu, hsgp(x)) = exp(a)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "keep one of the two" build(hyper)
    @test_throws "keep one of the two" RKBRMI(hyper(DF))
    # One expression per term: per-axis expressions are not supported.
    per_axis = @brm begin
        a ~ Normal(0.0, 1.0)
        mu ~ 1 + hsgp(x, u; k = 3, iso = false)
        length_scale(mu, hsgp(x, u)) = [exp(a), exp(2a)]
        y ~ Normal(mu, 1.0)
    end
    @test_throws "per-axis expressions" build(per_axis)
    # The validity floor names a prior's support, not a value.
    floor = @brm begin
        a ~ Normal(0.0, 1.0)
        mu ~ 1 + hsgp(x; k = 3)
        length_scale(mu, hsgp(x)) = length_scale_floor + exp(a)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "length_scale_floor" build(floor)
    # A smoothing scale has no derived lowering yet.
    smooth = @brm begin
        a ~ Normal(0.0, 1.0)
        mu ~ 1 + s(x)
        sd(:, s(x)) = exp(a)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "does not lower for this term yet" build(smooth)
end

@stestset "readers: a derived role has no coordinate" begin
    d = brm_descriptor(build(hsgp_derived); name=:derived)
    @test_throws "computed from model values" brm_term_coordinates(d, :mu, String[];
        term=:hsgp_x, parameter=:sd)
    @test_throws "computed from model values" brm_term_coordinates(d, :mu, String[];
        term=:hsgp_x, parameter=:length_scale)
    @test_throws "parameters it reads" brm_start(RKBRMI(hsgp_derived(DF)), (;);
        term_inits=[:(sd(mu, hsgp(x))) => 1.0])
    # A frozen replay keeps the expression: the program is the same.
    sb = build(hsgp_derived)
    replay = reprocess(sb, merge(DF, (; x=DF.x .* 0.9)))
    @test BRM.stan_code(replay) == BRM.stan_code(sb)
end

DERIVED_RUNTIME && @stestset "SBBRMI: the oracle of the notebook GP" begin
    model = instantiate(build(notebook)).model
    names = BridgeStan.param_names(model)
    rng = MersenneTwister(7)
    for _ in 1:3
        q = 0.7 .* randn(rng, BridgeStan.param_unc_num(model))
        values = Dict(zip(names, BridgeStan.param_constrain(model, q)))
        z = [values["hsgp_x_beta_raw.$j"] for j in 1:8]
        p = (; sf=values["sf"], l=values["l"], theta=values["theta"], z)
        @test BridgeStan.log_density(model, q; propto=false, jacobian=false) ≈
            notebook_oracle(p) atol=1e-10 rtol=1e-12
    end
    # The HSGP draw readers read a sampled or known scale; a derived one is refused.
    d = brm_descriptor(build(notebook); name=:notebook)
    draws = permutedims(BridgeStan.param_constrain(model, zeros(BridgeStan.param_unc_num(model))))
    @test_throws "computed from model values" hsgp_coordinate_draws(d, draws, names;
        predictor=:mu, term=:hsgp_x)
end

DERIVED_RUNTIME && @stestset "RKBRMI equals SBBRMI on the HSGP shapes" begin
    rng = MersenneTwister(11)
    for builder in (notebook, hsgp_derived, aniso_derived, grouped_derived, periodic_derived)
        brmi = builder(DF)
        rk, problem = consumer_problem(brmi)
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        stan = instantiate(sb)
        stan_names = BridgeStan.param_unc_names(stan.model)
        transport = brm_coordinate_transport(rk, sb, stan_names)
        @test length(transport) == length(coordinate_names(rk.model.layout)) == length(stan_names)
        for _ in 1:2
            u = 0.6 .* randn(rng, length(transport))
            stan_u = brm_rk_point_to_stan(transport, u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_value, stan_gradient = BridgeStan.log_density_gradient!(stan.model, stan_u,
                zeros(length(stan_u)); propto=false, jacobian=true)
            @test value ≈ stan_value + transport.logdensity_offset atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        end
    end
end

DERIVED_RUNTIME && @stestset "SBBRMI: derived equals the twin at the derived value" begin
    rng = MersenneTwister(3)
    for (name, derived, twin, roles) in DERIVED_CASES
        model = instantiate(build(derived)).model
        twin_model = instantiate(build(twin)).model
        check_twin(u -> BridgeStan.log_density(model, u; propto=false, jacobian=true),
            BridgeStan.param_unc_names(model),
            v -> BridgeStan.log_density(twin_model, v; propto=false, jacobian=true),
            BridgeStan.param_unc_names(twin_model), roles,
            filter(p -> first(p) in BridgeStan.param_unc_names(model),
                get(OFFSET_RAW, name, RAW_PRIORS)), rng; constant=name == "dar")
    end
end

DERIVED_RUNTIME && @stestset "RKBRMI: derived equals the twin at the derived value" begin
    rng = MersenneTwister(5)
    for (name, derived, twin, roles) in DERIVED_CASES
        name == "rw" && continue
        rk, problem = consumer_problem(derived(DF))
        twin_rk, twin_problem = consumer_problem(twin(DF))
        check_twin(u -> LogDensityProblems.logdensity(problem, u),
            coordinate_names(rk.model.layout),
            v -> LogDensityProblems.logdensity(twin_problem, v),
            coordinate_names(twin_rk.model.layout), roles,
            filter(p -> first(p) in string.(coordinate_names(rk.model.layout)),
                get(OFFSET_RAW, name, RAW_PRIORS)), rng;
            constant=name == "dar")
    end
end
