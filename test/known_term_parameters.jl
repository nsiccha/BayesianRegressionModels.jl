# test/known_term_parameters.jl — a term's own parameter held at a known value:
# `length_scale(<lp|:>, hsgp(x)) = value`, `sd(<lp|:>, hsgp(x)) = value`, and
# the same `=` on the `gp`, `dar` and `car` term-parameter addresses (snag
# hsgp-length-scal-200dd332; the `=` spelling is user decision 1kicac2's).
#
# Run: julia --project=test test/known_term_parameters.jl
# Set BRM_KNOWN_TERM_RUNTIME=0 to skip the BridgeStan/RK density probes.
#
# The density checks compare each known-value model against the SAME model
# with the known parameter sampled under an explicit prior, evaluated at the
# known value: the two log densities (no Jacobian) must differ by exactly that
# prior's density of the value, at every point.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, LinearAlgebra
using Distributions: Exponential, Normal, LogNormal, Beta, Dirac, logpdf, truncated

const KNOWN_RUNTIME = get(ENV, "BRM_KNOWN_TERM_RUNTIME", "1") != "0"
const KNOWN_CACHE = joinpath(tempdir(), "brm-known-term-parameters")

Random.seed!(20261027)
const N = 30
const DF = (; x=round.(2 .* rand(N) .- 1; digits=3), u=round.(rand(N); digits=3),
            g=repeat(1:3, inner=10), t=collect(1.0:N), y=round.(randn(N); digits=3),
            step=repeat(1.0:10.0, 3), ell=0.7, C=[1 0.3 0.1; 0.3 1 0.2; 0.1 0.2 1],
            from=[1, 2], to=[2, 3])

build(builder; data=DF, kwargs...) =
    SBBRMI(builder(data); mod=@__MODULE__, total_groups=(), kwargs...)
stanc_ok(sb) = StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok

function instantiate(sb)
    mkpath(KNOWN_CACHE)
    code = BRM.stan_code(sb)
    BRM.stan_instantiate(sb; path=joinpath(KNOWN_CACHE, string(hash(code)) * ".stan"))
end

# The log density (propto=false, jacobian=false) of a compiled model at a
# constrained point given by name.
function stan_lp(model, values)
    theta = [values[n] for n in BridgeStan.param_names(model)]
    BridgeStan.log_density(model, BridgeStan.param_unconstrain(model, theta);
        propto=false, jacobian=false)
end

# Random values for every parameter `model` declares, positive where a name
# says it is a scale (a persistence or share below 1).
function random_point(rng, model)
    Dict(n => (occursin("sigma", n) || occursin("rho", n) || occursin("sd_pen", n) ?
                   0.3 + rand(rng) :
               occursin("beta", n) && !occursin("raw", n) && !occursin("pop", n) ?
                   0.2 + 0.6rand(rng) : randn(rng))
         for n in BridgeStan.param_names(model))
end

# ---- the models -------------------------------------------------------------

# The reporter's shape (ODLPlaus): a 2-D isotropic field read through an
# assignment, its length scale held known.
field_known = @brm begin
    m ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    field ~ 0 + hsgp(x, u; k = 4, c = 1.3)
    sd(field, hsgp(x, u)) ~ Exponential(2.0)
    length_scale(field, hsgp(x, u)) = 0.7
    mu = m .+ field .* t
    y ~ Normal(mu, sigma)
end
field_sampled = @brm begin
    m ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    field ~ 0 + hsgp(x, u; k = 4, c = 1.3)
    sd(field, hsgp(x, u)) ~ Exponential(2.0)
    length_scale(field, hsgp(x, u)) ~ LogNormal(0.0, 1.0)
    mu = m .+ field .* t
    y ~ Normal(mu, sigma)
end

# One axis, the length scale from a data field and the amplitude known.
both_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x; k = 5, c = 1.5)
    length_scale(:, hsgp(x)) = ell
    sd(:, hsgp(x)) = 0.4
    y ~ Normal(mu, sigma)
end
both_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x; k = 5, c = 1.5)
    length_scale(:, hsgp(x)) ~ LogNormal(0.0, 1.0)
    sd(:, hsgp(x)) ~ Exponential(1.0)
    y ~ Normal(mu, sigma)
end

# Anisotropic: one value per axis.
aniso_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x, u; k = 3, c = 1.5, iso = false)
    length_scale(mu, hsgp(x, u)) = [0.6, 0.9]
    y ~ Normal(mu, sigma)
end
aniso_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x, u; k = 3, c = 1.5, iso = false)
    length_scale(mu, hsgp(x, u)) ~ LogNormal(0.0, 1.0)
    y ~ Normal(mu, sigma)
end

# A grouped field shares the known length scale across its levels.
grouped_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x; k = 4, c = 1.5, by = g)
    length_scale(mu, hsgp(x)) = 0.8
    y ~ Normal(mu, sigma)
end
grouped_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(x; k = 4, c = 1.5, by = g)
    length_scale(mu, hsgp(x)) ~ LogNormal(0.0, 1.0)
    y ~ Normal(mu, sigma)
end

periodic_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(t; k = 3, cov = :periodic, period = 12.0)
    length_scale(:, hsgp(t; cov = :periodic)) = 1.2
    y ~ Normal(mu, sigma)
end
periodic_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + hsgp(t; k = 3, cov = :periodic, period = 12.0)
    length_scale(:, hsgp(t; cov = :periodic)) ~ LogNormal(0.0, 1.0)
    y ~ Normal(mu, sigma)
end

gp_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + gp(x)
    length_scale(:, gp(x)) = 0.5
    y ~ Normal(mu, sigma)
end
gp_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + gp(x)
    length_scale(:, gp(x)) ~ LogNormal(0.0, 1.0)
    y ~ Normal(mu, sigma)
end

dar_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + x + dar(t)
    ar(:, dar(t)) = 0.6
    y ~ Normal(mu, sigma)
end
dar_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + x + dar(t)
    ar(:, dar(t)) ~ Beta(2.0, 2.0)
    y ~ Normal(mu, sigma)
end

# A smooth's smoothing SD, a random walk's innovation scale, a damped walk's
# persistence and a BYM2 field's spatial share.
smooth_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + s(x)
    sd(:, s(x)) = 0.3
    y ~ Normal(mu, sigma)
end
smooth_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + s(x)
    sd(:, s(x)) ~ Exponential(1.0)
    y ~ Normal(mu, sigma)
end
rw_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + x + rw(t)
    sd(:, rw(t)) = 0.3
    y ~ Normal(mu, sigma)
end
rw_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + x + rw(t)
    sd(:, rw(t)) ~ Exponential(1.0)
    y ~ Normal(mu, sigma)
end
cdar_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + cdar(step; by=g, cor=C)
    ar(:, cdar(step)) = 0.5
    y ~ Normal(mu, sigma)
end
cdar_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + cdar(step; by=g, cor=C)
    ar(:, cdar(step)) ~ Beta(2.0, 2.0)
    y ~ Normal(mu, sigma)
end
car_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + car(g; edges=(from, to), type=:bym2)
    mixing(:, car(g)) = 0.4
    y ~ Normal(mu, sigma)
end
car_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + car(g; edges=(from, to), type=:bym2)
    mixing(:, car(g)) ~ Beta(2.0, 2.0)
    y ~ Normal(mu, sigma)
end

# One known length scale and one known amplitude per model, so the twins'
# difference is exactly one prior density each.
const DENSITY_CASES = (
    ("field", field_known, field_sampled,
        [(LogNormal(0.0, 1.0), 0.7)]),
    ("both", both_known, both_sampled,
        [(LogNormal(0.0, 1.0), 0.7), (Exponential(1.0), 0.4)]),
    ("aniso", aniso_known, aniso_sampled,
        [(LogNormal(0.0, 1.0), 0.6), (LogNormal(0.0, 1.0), 0.9)]),
    ("grouped", grouped_known, grouped_sampled, [(LogNormal(0.0, 1.0), 0.8)]),
    ("periodic", periodic_known, periodic_sampled, [(LogNormal(0.0, 1.0), 1.2)]),
    ("gp", gp_known, gp_sampled, [(LogNormal(0.0, 1.0), 0.5)]),
    ("dar", dar_known, dar_sampled, [(Beta(2.0, 2.0), 0.6)]),
    ("smooth", smooth_known, smooth_sampled, [(Exponential(1.0), 0.3)]),
    ("rw", rw_known, rw_sampled, [(Exponential(1.0), 0.3)]),
    ("cdar", cdar_known, cdar_sampled, [(Beta(2.0, 2.0), 0.5)]),
    ("car", car_known, car_sampled, [(Beta(2.0, 2.0), 0.4)]),
)

# A tensor smooth with one known smoothing SD of three: the other two stay
# sampled (`sd_pen_free`) with their priors.
t2_known = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + t2(x, u)
    sd(:, t2(x, u), rr) = 0.3
    sd(:, t2(x, u), nr) ~ Exponential(2.0)
    y ~ Normal(mu, sigma)
end
t2_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + t2(x, u)
    sd(:, t2(x, u), rr) ~ Exponential(1.0)
    sd(:, t2(x, u), nr) ~ Exponential(2.0)
    y ~ Normal(mu, sigma)
end

@stestset "parse and refusals" begin
    @test field_known(DF) isa BRMI
    # A known value takes the address of its `~` twin: both is a duplicate.
    @test_throws "duplicate" eval(:(@brm begin
        mu ~ 1 + hsgp(x)
        length_scale(:, hsgp(x)) = 0.5
        length_scale(:, hsgp(x)) ~ LogNormal(0.0, 1.0)
        y ~ Normal(mu, 1.0)
    end))
    # refused: a measurement-error latent is what the term infers; known, the
    # term is the plain covariate (stated in the parse error).
    @test_throws "nothing to infer" eval(:(@brm begin
        mu ~ 1 + me(x, 0.1)
        latent(:, me(x, 0.1)) = x
        y ~ Normal(mu, 1.0)
    end))
    # refused: known monotonic shares are a fixed scoring of the levels, which a
    # data covariate already writes (stated in the parse error).
    @test_throws "fixed scoring" eval(:(@brm begin
        mu ~ 1 + mo(g)
        simplex(:, mo(g)) = [0.5, 0.5]
        y ~ Normal(mu, 1.0)
    end))
    # refused: not a length scale (a length scale is finite and positive).
    @test_throws "strictly positive" build(@brm begin
        mu ~ 1 + hsgp(x; k = 3)
        length_scale(:, hsgp(x)) = -1.0
        y ~ Normal(mu, 1.0)
    end)
    # refused: an isotropic term has one length scale.
    @test_throws "holds one number" build(@brm begin
        mu ~ 1 + hsgp(x, u; k = 3)
        length_scale(:, hsgp(x, u)) = [1.0, 2.0]
        y ~ Normal(mu, 1.0)
    end)
    # refused: an anisotropic term's values are one per axis.
    @test_throws "one per axis" build(@brm begin
        mu ~ 1 + hsgp(x, u; k = 3, iso = false)
        length_scale(:, hsgp(x, u)) = [1.0, 2.0, 3.0]
        y ~ Normal(mu, 1.0)
    end)
    # refused: a persistence coefficient lies in [0, 1].
    @test_throws "[0, 1]" build(@brm begin
        mu ~ 1 + x + dar(t)
        ar(:, dar(t)) = 1.5
        y ~ Normal(mu, 1.0)
    end)
    # refused: a sampled quantity is not a known value.
    @test_throws "needs a known value" build(@brm begin
        tau ~ Exponential(1.0)
        mu ~ 1 + hsgp(x; k = 3)
        length_scale(:, hsgp(x)) = tau
        y ~ Normal(mu, 1.0)
    end)
    # refused: a known length scale and a hyper-predictor of the same length
    # scale contradict each other.
    hyper = @brm begin
        mu ~ 1 + hsgp(x; k = 3, by = g)
        log(length_scale(mu, hsgp(x))) ~ 1 + (1 | g)
        length_scale(mu, hsgp(x)) = 0.5
        y ~ Normal(mu, 1.0)
    end
    @test_throws "keep one of the two" build(hyper)
    @test_throws "keep one of the two" RKBRMI(hyper(DF))
end

@stestset "SBBRMI: the known value is data, not a parameter" begin
    for (name, known, sampled, _) in DENSITY_CASES
        sb, twin = build(known), build(sampled)
        @test stanc_ok(sb)
        known_names = Set(o.name for o in brm_descriptor(sb; name=:known).outputs
                          if o.kind === :parameter)
        sampled_names = Set(o.name for o in brm_descriptor(twin; name=:sampled).outputs
                            if o.kind === :parameter)
        # Exactly the known parameters disappear; nothing else changes.
        @test issubset(known_names, sampled_names)
        @test length(sampled_names) - length(known_names) ==
            (name == "both" ? 2 : 1)
    end
    # The HSGP floor does not bind a known value (the declaration wins,
    # decision 0v1op0o), and the floor accessor still reads it.
    d = brm_descriptor(build(field_known); name=:field)
    @test brm_length_scale_floor(d, :field; term=:hsgp_x_u) > 0
end

@stestset "descriptions: a known value is a point mass" begin
    desc = brm_description(brm_descriptor(build(field_known); name=:field))
    @test desc.complete
    known = [p for p in desc.priors if p.source.kind === :known]
    @test length(known) == 1
    @test only(known).id == (:parameter, :hsgp_x_u, :rho_iso)
    @test occursin("Dirac", brm_description_markdown(desc))
    aniso = brm_description(brm_descriptor(build(aniso_known); name=:aniso))
    @test only(p for p in aniso.priors if p.source.kind === :known).id ==
        (:parameter, :hsgp_x_u, :rho)
    t2 = brm_description(brm_descriptor(build(t2_known); name=:t2))
    @test only(p for p in t2.priors if p.source.kind === :known).id[end] === :known
end

@stestset "descriptors: a known role has no coordinate" begin
    d = brm_descriptor(build(field_known); name=:field)
    @test_throws "known value 0.7" brm_term_coordinates(d, :field, String[];
        term=:hsgp_x_u, parameter=:length_scale)
    d2 = brm_descriptor(build(both_known); name=:both)
    @test_throws "known value 0.4" brm_term_coordinates(d2, :mu, String[];
        term=:hsgp_x, parameter=:sd)
end

KNOWN_RUNTIME && @stestset "BridgeStan: known equals sampled at the known value" begin
    rng = MersenneTwister(3)
    for (name, known, sampled, laws) in DENSITY_CASES
        known_model = instantiate(build(known)).model
        twin = instantiate(build(sampled)).model
        extra = setdiff(BridgeStan.param_names(twin), BridgeStan.param_names(known_model))
        @test length(extra) == length(laws)
        expected = sum(logpdf(law, value) for (law, value) in laws)
        values = Dict(n => v for (n, (_, v)) in zip(sort(extra), laws))
        differences = map(1:3) do _
            base = random_point(rng, known_model)
            stan_lp(twin, merge(base, values)) - stan_lp(known_model, base)
        end
        @test maximum(differences) - minimum(differences) <= 1e-9
        @test first(differences) ≈ expected atol=1e-8
    end
end

# Shapes `brm_coordinate_transport` pairs (ungrouped HSGP terms): the RK and
# Stan lowerings of the known-value model are one law.
KNOWN_RUNTIME && @stestset "BridgeStan: one known smoothing SD beside sampled ones" begin
    rng = MersenneTwister(9)
    known_model = instantiate(build(t2_known)).model
    twin = instantiate(build(t2_sampled)).model
    free = [n for n in BridgeStan.param_names(known_model) if occursin("sd_pen_free", n)]
    @test length(free) == 2
    differences = map(1:3) do _
        base = random_point(rng, known_model)
        stem = replace(first(free), r"sd_pen_free\.1$" => "")
        merged = filter(p -> !occursin("sd_pen_free", first(p)), base)
        merged[stem * "sd_pen.1"] = 0.3
        merged[stem * "sd_pen.2"] = base[stem * "sd_pen_free.1"]
        merged[stem * "sd_pen.3"] = base[stem * "sd_pen_free.2"]
        stan_lp(twin, merged) - stan_lp(known_model, base)
    end
    @test maximum(differences) - minimum(differences) <= 1e-9
    @test first(differences) ≈ logpdf(Exponential(1.0), 0.3) atol=1e-8
end

# Adaptive centering reads a known length scale or amplitude as a constant:
# it claims no coordinate, and the spectral log-scale at any point equals the
# sampled twin's at the known value.
KNOWN_RUNTIME && @stestset "adaptive centering: a known value is a constant" begin
    sb, twin_sb = build(both_known), build(both_sampled)
    names = BridgeStan.param_unc_names(instantiate(sb).model)
    twin_names = BridgeStan.param_unc_names(instantiate(twin_sb).model)
    block = only(BRM._adaptive_hsgp_centering_blocks(sb, names))
    twin_block = only(BRM._adaptive_hsgp_centering_blocks(twin_sb, twin_names))
    @test block.length_scales == [0] && block.sd == 0
    rng = MersenneTwister(13)
    x = randn(rng, length(names))
    v = zeros(length(twin_names))
    for (j, n) in enumerate(names)
        v[findfirst(==(n), twin_names)] = x[j]
    end
    v[only(twin_block.length_scales)] = log(0.7)
    v[twin_block.sd] = log(0.4)
    for basis in eachindex(block.effects)
        @test BRM._adaptive_hsgp_log_scale(x, block, basis) ≈
            BRM._adaptive_hsgp_log_scale(v, twin_block, basis) atol=1e-12
    end
end

# The HSGP draw diagnostics read a known length scale in every draw.
KNOWN_RUNTIME && @stestset "hsgp_coordinate_draws: a known value in every draw" begin
    sb = build(both_known)
    problem = instantiate(sb)
    d = brm_descriptor(sb; name=:both)
    q = 0.1randn(MersenneTwister(17), 4, LogDensityProblems.dimension(problem))
    names = BridgeStan.param_names(problem.model)
    draws = permutedims(reduce(hcat, [BridgeStan.param_constrain(problem.model, collect(row))
                                      for row in eachrow(q)]))
    out = hsgp_coordinate_draws(d, draws, names; predictor=:mu, term=:hsgp_x)
    @test out.length_scales == fill(0.7, 4, 1)
    @test out.marginal_sd == fill(0.4, 4)
end

# A frozen replay keeps the known value as a fitted constant, on new rows and
# a new axis; the boundary check reads it in every draw.
@stestset "reprocess and hsgp_boundary_check keep the known value" begin
    sb = build(both_known)
    shifted = merge(DF, (; x=DF.x .* 0.9, ell=5.0))
    replay = reprocess(sb, shifted)
    data = BRM.stan_data(replay)
    known_keys = [k for k in keys(data) if occursin("_known_", string(k))]
    @test sort(known_keys) == [:hsgp_x_known_length_scale, :hsgp_x_known_sigma]
    @test data[:hsgp_x_known_length_scale] == 0.7
    @test data[:hsgp_x_known_sigma] == 0.4
    @test BRM.stan_code(replay) == BRM.stan_code(sb)
    d = brm_descriptor(sb; name=:both)
    check = hsgp_boundary_check(d, zeros(3, 0), String[]; predictor=:mu, term=:hsgp_x)
    @test check.mean_rho ≈ [0.7]
    @test check.floor_binding ≈ [0.7 < check.floor ? 1.0 : 0.0]
end

KNOWN_RUNTIME && @stestset "RKBRMI: the same law as SBBRMI" begin
    for (name, known, _, _) in DENSITY_CASES
        name in ("field", "both", "aniso", "grouped", "periodic", "cdar") || continue
        brmi = known(DF)
        rk, problem = consumer_problem(brmi)
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        stan = instantiate(sb)
        stan_names = BridgeStan.param_unc_names(stan.model)
        transport = brm_coordinate_transport(rk, sb, stan_names)
        @test length(transport) == length(coordinate_names(rk.model.layout)) == length(stan_names)
        for u in (zeros(length(transport)),
                  collect(range(-0.43, 0.51; length=length(transport))))
            stan_u = brm_rk_point_to_stan(transport, u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_value, stan_gradient = BridgeStan.log_density_gradient!(stan.model, stan_u,
                zeros(length(stan_u)); propto=false, jacobian=true)
            @test value ≈ stan_value + transport.logdensity_offset atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        end
    end
end

dar_rk_sampled = @brm begin
    sigma ~ Exponential(1.0)
    mu ~ 1 + x + dar(t)
    ar(:, dar(t)) ~ Normal(0.5, 0.2)
    y ~ Normal(mu, sigma)
end

# Every shape on RK alone: the known model is its sampled twin with the known
# parameter's coordinate held fixed, so their log densities differ by one
# constant at every point of the shared coordinates (the prior density and
# Jacobian at the value). A positive scale is `exp` of its coordinate and a
# persistence coefficient `logistic` of it, which fixes that constant.
KNOWN_RUNTIME && @stestset "RKBRMI: known equals sampled at the known value" begin
    rng = MersenneTwister(5)
    for (name, known, sampled, laws) in DENSITY_CASES
        # Not on RKBRMI before any known value is read: smoothing-scale priors
        # (`_rk_gate_spline_term_priors!`) and `rw` terms as a whole
        # (todo `1hq7nm0`).
        name in ("smooth", "rw") && continue
        twin_builder = name == "dar" ? dar_rk_sampled : sampled
        known_rk, problem = consumer_problem(known(DF))
        twin_rk, twin = consumer_problem(twin_builder(DF))
        names = coordinate_names(known_rk.model.layout)
        twin_names = coordinate_names(twin_rk.model.layout)
        extra = setdiff(twin_names, names)
        @test issubset(names, twin_names)
        @test length(extra) == length(laws)
        unit = name in ("dar", "car", "cdar")
        fixed = unit ? [log(v / (1 - v)) for (_, v) in laws] :
            [log(value) for (_, value) in laws]
        expected = name == "dar" ?
            logpdf(truncated(Normal(0.5, 0.2), 0.0, 1.0), 0.6) + log(0.6) + log(0.4) :
            unit ? sum(logpdf(law, v) + log(v) + log(1 - v) for (law, v) in laws) :
            sum(logpdf(law, value) + log(value) for (law, value) in laws)
        differences = map(1:3) do _
            u = randn(rng, length(names))
            v = zeros(length(twin_names))
            for (j, n) in enumerate(names)
                v[findfirst(==(n), twin_names)] = u[j]
            end
            for (n, value) in zip(sort(extra; by=string), fixed)
                v[findfirst(==(n), twin_names)] = value
            end
            LogDensityProblems.logdensity(twin, v) - LogDensityProblems.logdensity(problem, u)
        end
        @test maximum(differences) - minimum(differences) <= 1e-9
        @test first(differences) ≈ expected atol=1e-8
    end
end
