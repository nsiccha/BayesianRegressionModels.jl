# test/runtime_numeric_settings.jl — sizes and numeric settings are inputs of
# the emitted programs, not numbers they spell (user principle
# `ODLPlaus:precompile/decisions/2026-11-06T01-09-46-212-0ikeidi`; snag
# `runtime-numeric-f1388e0e`).
#
# `gmrf(...; neighbors=m)`, `gmrf(...; multiresolution=m)` and
# `hsgp(...; k, c)` take a literal or a scalar (per-axis) data field holding
# it. Oracles:
#   * one declaration, many values: models that differ only in such a value
#     emit the identical Stan program (SBBRMI) and the identical RK program
#     (RKBRMI), literal or data-field spelled;
#   * a data-field spelling is the literal twin: the same Stan program and Stan
#     data, and the same RK density at a point;
#   * a frozen replay keeps the fitted value and refuses a design holding
#     another one; a fresh replay takes the new value.
#
# The prior path is checked the same way (todo `0njkjzr`): a data scalar a prior
# reads, and a block's LKJ `eta`, are inputs. Besides the emitted source, the RK
# lowering's program and prepared query type stay one across values, and one
# retained build evaluates values A, B, A against an analytic oracle, so a stale
# bound number would show. A literal is the positive control: it is part of the
# author's source, so it does change the program and the query type.
#
# Run: julia --startup-file=no --project=test test/runtime_numeric_settings.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, LinearAlgebra

module RuntimeSettingsModels
using BayesianRegressionModels, Distributions
const FIELD_PRIORS = quote
    sd(field, gmrf((east, north), time)) ~ Exponential(0.7)
    length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(3.0), 0.4)
    length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(1.7), 0.4)
    y ~ Normal(field, 0.2)
end
footprint(m) = brm(:(begin
    field ~ 0 + gmrf((east, north), time; neighbors=$m)
    $(FIELD_PRIORS.args...)
end); mod=@__MODULE__)
multiresolution(m) = brm(:(begin
    field ~ 0 + gmrf((east, north), time; multiresolution=$m)
    $(FIELD_PRIORS.args...)
end); mod=@__MODULE__)
hsgp_model(k, c) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 0 + hsgp(t; k=$k, c=$c)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
tensor_model(k) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 0 + hsgp(t, s; k=$k, iso=false)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
lkj_model(K, eta) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(:, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky($K, $eta)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
domain_model(k, domain) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 0 + hsgp(t; k=$k, domain=$domain)
    length_scale(:, hsgp(t)) ~ truncated(LogNormal(0, 1); lower=length_scale_floor, upper=24.0)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
isolator() = brm(:(begin
    z[1:n] .~ LogNormal.(mu, sd)
    y ~ Normal(z[idx], 0.2)
end); mod=@__MODULE__)
isolator_literal(mu, sd) = brm(:(begin
    z[1:n] .~ LogNormal.($mu, $sd)
    y ~ Normal(z[idx], 0.2)
end); mod=@__MODULE__)
# A prior's scalar argument on each route that plans one.
scale_prior() = brm(:(begin
    sigma ~ Exponential(s)
    m ~ 1 + x
    y ~ Normal(m, sigma)
end); mod=@__MODULE__)
scale_prior_literal(s) = brm(:(begin
    sigma ~ Exponential($s)
    m ~ 1 + x
    y ~ Normal(m, sigma)
end); mod=@__MODULE__)
value_route_prior() = brm(:(begin
    sigma ~ Exponential(s)
    a ~ Normal(0, 1)
    c = a + 1
    y ~ Normal(c, sigma)
end); mod=@__MODULE__)
derived_prior() = brm(:(begin
    sigma ~ LogNormal(log(s), 0.3)
    m ~ 1 + x
    y ~ Normal(m, sigma)
end); mod=@__MODULE__)
derived_prior_literal(s) = brm(:(begin
    sigma ~ LogNormal(log($s), 0.3)
    m ~ 1 + x
    y ~ Normal(m, sigma)
end); mod=@__MODULE__)
assigned_prior() = brm(:(begin
    rate = 2 * s
    sigma ~ Exponential(rate)
    m ~ 1 + x
    y ~ Normal(m, sigma)
end); mod=@__MODULE__)
population_prior() = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    effect(mu, x) ~ Normal(loc, scale)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
population_prior_literal(loc, scale) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    effect(mu, x) ~ Normal($loc, $scale)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
term_priors() = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 0 + hsgp(x; k=5)
    sd(:, hsgp(x)) ~ Exponential(rate)
    length_scale(:, hsgp(x)) ~ LogNormal(lm, ls)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
term_priors_literal(rate, lm, ls) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 0 + hsgp(x; k=5)
    sd(:, hsgp(x)) ~ Exponential($rate)
    length_scale(:, hsgp(x)) ~ LogNormal($lm, $ls)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
block_sd_prior() = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(:, p) ~ Exponential(rate)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
stratified_lkj(K, eta) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | gr(subject, by=stratum))
    sd(:, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky($K, $eta)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
truncated_prior(lo, hi) = brm(:(begin
    sigma ~ truncated(Normal(0.2, 1.1); lower=$lo, upper=$hi)
    m ~ 1 + x
    y ~ Normal(m, sigma)
end); mod=@__MODULE__)
restricted_prior(lo) = brm(:(begin
    sigma ~ restricted(Normal(0.2, 1.1), $lo, Inf)
    m ~ 1 + x
    y ~ Normal(m, sigma)
end); mod=@__MODULE__)
uniform_prior(lo, hi) = brm(:(begin
    sigma ~ Uniform($lo, $hi)
    m ~ 1 + x
    y ~ Normal(m, sigma)
end); mod=@__MODULE__)
mvnormal_scale(s) = brm(:(begin
    w ~ MvNormal(zeros(3), $s)
    y ~ Normal(sum(w), 0.5)
end); mod=@__MODULE__)
dirichlet_prior(a) = brm(:(begin
    p ~ Dirichlet(3, $a)
    y ~ Categorical(p)
end); mod=@__MODULE__)
factor_effects() = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 1 + f
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
known_cor(C) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(:, p) ~ Exponential(1)
    cor(:, p) = $C
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
end
const RSM = RuntimeSettingsModels

function field_data(; seed=21)
    rng = MersenneTwister(seed)
    npositions, times = 7, [0.0, 1.0, 2.0, 80.0]
    x, y = 10 .* rand(rng, npositions), 10 .* rand(rng, npositions)
    east, north = repeat(x; outer=length(times)), repeat(y; outer=length(times))
    time = repeat(times; inner=npositions)
    (; east, north, time, y=0.3 .* randn(rng, length(time)))
end
const LDATA = (; subject=[1, 1, 2, 2, 3, 3], x=[-1.0, 0.5, 0.2, 1.1, -0.4, 0.9],
    y=[0.3, 0.1, -0.2, 0.6, 0.0, 0.4])
const HDATA = let rng = MersenneTwister(3), t = collect(range(0.0, 5.0; length=24))
    (; t, s=cos.(t), y=sin.(t) .+ 0.1 .* randn(rng, length(t)))
end

stan_program(brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod=RSM).model)
stan_inputs(brmi) = BRM.stan_data(SBBRMI(brmi; mod=RSM))
rk_program(brmi) = BRM._rk_program_statements(BRM._rk_emit_ast(RKBRMI(brmi).plan))
rk_point(rk) = collect(range(-0.6, 0.8; length=rk.model.layout.total))
rk_density(rk, u) = LogDensityProblems.logdensity(
    rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)

# The models of `values`, built from `build(value)` over `data(value)`, emit
# one Stan program and one RK program.
function check_one_program(label, build, values; data=_ -> HDATA)
    @testset "$label" begin
        models = [build(v)(data(v)) for v in values]
        stan = map(stan_program, models)
        @test all(==(first(stan)), stan)
        rk = map(rk_program, models)
        @test all(==(first(rk)), rk)
    end
end

@stestset "one program for every count and setting" begin
    D = field_data()
    check_one_program("gmrf neighbors literal", RSM.footprint, (3, 5); data=_ -> D)
    check_one_program("gmrf neighbors field", _ -> RSM.footprint(:nb), (3, 5);
        data=m -> merge(D, (; nb=m)))
    check_one_program("gmrf multiresolution field", _ -> RSM.multiresolution(:m), (2, 4);
        data=m -> merge(D, (; m)))
    check_one_program("hsgp k literal", k -> RSM.hsgp_model(k, 1.5), (6, 9))
    check_one_program("hsgp k and c fields", _ -> RSM.hsgp_model(:kk, :cc),
        ((6, 1.5), (9, 2.0)); data=v -> merge(HDATA, (; kk=v[1], cc=v[2])))
    check_one_program("hsgp per-axis k field", _ -> RSM.tensor_model(:kk),
        ([3, 4], [5, 2]); data=k -> merge(HDATA, (; kk=k)))
    check_one_program("hsgp fixed domain", d -> RSM.domain_model(7, d),
        ((-1.0, 6.0), (-2.0, 8.0)))
end

@stestset "a data-field setting is its literal twin" begin
    D = field_data()
    for (field, literal) in (
            (RSM.footprint(:nb)(merge(D, (; nb=3))), RSM.footprint(3)(D)),
            (RSM.multiresolution(:m)(merge(D, (; m=2))), RSM.multiresolution(2)(D)),
            (RSM.hsgp_model(:kk, :cc)(merge(HDATA, (; kk=7, cc=1.8))),
                RSM.hsgp_model(7, 1.8)(HDATA)),
            (RSM.tensor_model(:kk)(merge(HDATA, (; kk=[3, 4]))),
                RSM.tensor_model((3, 4))(HDATA)),
            # A random-effect block's correlation prior reads K and eta from the
            # data (ODLPlaus co-report on this snag).
            (RSM.lkj_model(:K, :eta)(merge(LDATA, (; K=2, eta=1.7))),
                RSM.lkj_model(2, 1.7)(LDATA)))
        @test stan_program(field) == stan_program(literal)
        stan_field, stan_literal = stan_inputs(field), stan_inputs(literal)
        for key in keys(stan_literal)
            @test isequal(stan_field[key], stan_literal[key])
        end
        rk_field, rk_literal = RKBRMI(field), RKBRMI(literal)
        @test coordinate_names(rk_field.model.layout) == coordinate_names(rk_literal.model.layout)
        u = rk_point(rk_literal)
        @test rk_density(rk_field, u) == rk_density(rk_literal, u)
    end
end

@stestset "different values are different laws" begin
    # Positive control for the identity tests: one program, but the bound
    # setting changes the basis size and the density.
    small, large = RKBRMI(RSM.hsgp_model(6, 1.5)(HDATA)), RKBRMI(RSM.hsgp_model(9, 1.5)(HDATA))
    @test small.model.layout.total + 3 == large.model.layout.total
    wide = RKBRMI(RSM.hsgp_model(6, 2.5)(HDATA))
    u = rk_point(small)
    @test rk_density(small, u) != rk_density(wide, u)
end

@stestset "a data-field LKJ dimension is checked against the block" begin
    @test_throws "does not match the addressed random-effect block width 2" RKBRMI(
        RSM.lkj_model(:K, :eta)(merge(LDATA, (; K=3, eta=1.7))))
    @test_throws "dimension K must be an integer" RKBRMI(
        RSM.lkj_model(:K, :eta)(merge(LDATA, (; K=2.5, eta=1.7))))
end

@stestset "a frozen replay keeps the fitted setting" begin
    D = field_data()
    build = RSM.hsgp_model(:kk, :cc)
    train = merge(HDATA, (; kk=6, cc=1.5))
    rk = RKBRMI(build(train))
    @test_throws "frozen replay keeps the fitted hsgp" reprocess(rk, merge(train, (; kk=9)))
    @test_throws "frozen replay keeps the fitted hsgp" reprocess(rk, merge(train, (; cc=2.0)))
    # SBBRMI's replay keeps the fitted basis in either mode and refuses too; a
    # new build from that data compiles the same Stan program.
    sb = SBBRMI(build(train); mod=RSM)
    @test_throws "replay keeps the fitted hsgp" reprocess(sb, merge(train, (; kk=9)))
    @test_throws "replay keeps the fitted hsgp" reprocess(sb, merge(train, (; kk=9));
        freeze_constants=false)
    @test stan_program(build(merge(train, (; kk=9)))) == stan_program(build(train))
    @test reprocess(sb, train) isa SBBRMI
    same = reprocess(rk, train)
    @test same.model.layout.total == rk.model.layout.total
    fresh = reprocess(rk, merge(train, (; kk=9)); freeze_constants=false)
    @test fresh.model.layout.total == rk.model.layout.total + 3
    field = RSM.footprint(:nb)
    fit = RKBRMI(field(merge(D, (; nb=3))))
    @test_throws "frozen replay keeps the fitted gmrf" reprocess(fit, merge(D, (; nb=5)))
    sb_field = SBBRMI(field(merge(D, (; nb=3))); mod=RSM)
    @test_throws "replay keeps the fitted gmrf" reprocess(sb_field, merge(D, (; nb=5)))
    # A fresh SBBRMI replay builds the support from the new data's count.
    fresh_sb = reprocess(sb_field, merge(D, (; nb=5)); freeze_constants=false)
    rebuilt = BRM.stan_data(SBBRMI(field(merge(D, (; nb=5))); mod=RSM))
    replayed = BRM.stan_data(fresh_sb)
    support = [k for k in keys(rebuilt) if startswith(string(k), "gmrf_")]
    @test !isempty(support)
    @test all(k -> isequal(replayed[k], rebuilt[k]), support)
    @test !all(k -> isequal(BRM.stan_data(sb_field)[k], rebuilt[k]), support)
end

# ---- prior-read scalars and a block's LKJ eta (todo `0njkjzr`) ---------------

const ISOLATOR = (; y=[0.3, 0.1, 0.2, 0.6], idx=[1, 2, 2, 3], n=3)
const VALUE_A, VALUE_B = (; mu=0.1, sd=0.5), (; mu=0.7, sd=1.3)
const STRATA = merge(LDATA, (; stratum=[1, 1, 1, 1, 2, 2]))

emitted_program(rk) = BRM._rk_program_statements(BRM._rk_emit_ast(rk.plan))
lowered_program(rk) = string(rk.model.program)
sampler_query(rk) = ReactiveKernelsPPL.prepare_query(rk.model, rk_bound_plan(rk), :sampler)
reverse_problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))

# The models `build(d)` for `d` in `data` emit one source, lower to one program
# and prepare one query type, read their numbers from bound inputs, and differ in
# their density (unless `differ=false`: the values are labels of one law).
function check_one_lowering(label, build, data, inputs; differ=true)
    @testset "$label" begin
        models = [RKBRMI(build(d)) for d in data]
        @test allequal(emitted_program(rk) for rk in models)
        @test allequal(lowered_program(rk) for rk in models)
        @test allequal(typeof(sampler_query(rk)) for rk in models)
        for name in inputs
            @test all(haskey(rk_bound_data(rk), name) for rk in models)
        end
        u = rk_point(first(models))
        @test differ != allequal(sampler_query(rk)(u) for rk in models)
    end
end

@stestset "a prior's data scalars are inputs: one source, program and query type" begin
    check_one_lowering("declared vector, the isolator", RSM.isolator(),
        [merge(ISOLATOR, VALUE_A), merge(ISOLATOR, VALUE_B)], (:mu, :sd))
    check_one_lowering("scalar parameter", RSM.scale_prior(),
        [merge(LDATA, (; s)) for s in (1.7, 2.9)], (:s,))
    check_one_lowering("scalar parameter on the value route", RSM.value_route_prior(),
        [(; y=LDATA.y, s) for s in (1.7, 2.9)], (:s,))
    check_one_lowering("an expression over one", RSM.derived_prior(),
        [merge(LDATA, (; s)) for s in (1.7, 2.9)], (:s,))
    check_one_lowering("through an assignment", RSM.assigned_prior(),
        [merge(LDATA, (; s)) for s in (1.7, 2.9)], (:s,))
    check_one_lowering("population effect", RSM.population_prior(),
        [merge(LDATA, v) for v in ((; loc=0.2, scale=1.1), (; loc=0.5, scale=1.9))], (:loc, :scale))
    check_one_lowering("term hyperparameters", RSM.term_priors(),
        [merge(LDATA, v) for v in ((; rate=1.7, lm=0.1, ls=0.5), (; rate=2.9, lm=0.2, ls=0.7))],
        (:rate, :lm, :ls))
    check_one_lowering("random-effect scale", RSM.block_sd_prior(),
        [merge(LDATA, (; rate)) for rate in (1.7, 2.9)], (:rate,))
end

@stestset "a literal changes the program and the query type: the probes can fail" begin
    # Positive control for the query-type and lowered-program identity above.
    a = RKBRMI(RSM.isolator_literal(0.1, 0.5)(ISOLATOR))
    b = RKBRMI(RSM.isolator_literal(0.7, 1.3)(ISOLATOR))
    @test emitted_program(a) != emitted_program(b)
    @test lowered_program(a) != lowered_program(b)
    @test typeof(sampler_query(a)) != typeof(sampler_query(b))
end

@stestset "a data scalar prior is its literal twin" begin
    for (field, literal) in (
            (RSM.isolator()(merge(ISOLATOR, VALUE_B)), RSM.isolator_literal(0.7, 1.3)(ISOLATOR)),
            (RSM.scale_prior()(merge(LDATA, (; s=1.7))), RSM.scale_prior_literal(1.7)(LDATA)),
            (RSM.derived_prior()(merge(LDATA, (; s=1.7))), RSM.derived_prior_literal(1.7)(LDATA)),
            (RSM.population_prior()(merge(LDATA, (; loc=0.2, scale=1.1))),
                RSM.population_prior_literal(0.2, 1.1)(LDATA)),
            (RSM.term_priors()(merge(LDATA, (; rate=1.7, lm=0.1, ls=0.5))),
                RSM.term_priors_literal(1.7, 0.1, 0.5)(LDATA)))
        rk_field, rk_literal = RKBRMI(field), RKBRMI(literal)
        @test coordinate_names(rk_field.model.layout) == coordinate_names(rk_literal.model.layout)
        u = rk_point(rk_literal)
        @test rk_density(rk_field, u) ≈ rk_density(rk_literal, u) atol=1e-12 rtol=1e-12
        check_rk_source_roundtrip(rk_field)
    end
end

# The isolator's density on the unconstrained scale. `z = exp(u)`, so the prior
# and its log Jacobian together are `Normal(mu, sd)` of `u`.
function isolator_oracle(data, u)
    z = exp.(u)
    prior = sum(logpdf.(Normal(data.mu, data.sd), u))
    prior + sum(logpdf.(Normal.(z[data.idx], 0.2), data.y))
end
function isolator_gradient(data, u)
    z = exp.(u)
    gradient = @. -(u - data.mu) / data.sd^2
    for (y, j) in zip(data.y, data.idx)
        gradient[j] += (y - z[j]) / 0.2^2 * z[j]
    end
    gradient
end

@stestset "one retained build reads the bound prior values: A, B, A" begin
    build = RSM.isolator()
    A, B = merge(ISOLATOR, VALUE_A), merge(ISOLATOR, VALUE_B)
    rk = RKBRMI(build(A))
    u = rk_point(rk)
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    replays = [reprocess(rk, d; freeze_constants=false) for d in (A, B, A)]
    @test allequal(lowered_program(replay) for replay in replays)
    @test all(replay -> typeof(replay.model) == typeof(rk.model), replays)
    # The fit's build evaluates each binding (`rkppl-use` §14, "Rebinding").
    plans = rk_bound_plan.(replays)
    @test allequal(typeof.(plans))
    @test allequal(typeof.(ReactiveKernelsPPL.prepare_query.(Ref(rk.model), plans, :sampler)))
    queries = [ReactiveKernelsPPL.prepare_sampler(rk.model, plan, u; backend) for plan in plans]
    # The same values prepare the same gradient query. Other values do not yet:
    # RK's `prepare_sampler` splices a bound scalar into its generated body
    # (ReactiveKernels snag `prepare-sampler-e69f1b38`).
    @test typeof(queries[1]) == typeof(queries[3])
    @test_broken typeof(queries[1]) == typeof(queries[2])
    for (data, query) in zip((A, B, A), queries)
        gradient = similar(u)
        value, _ = ReactiveKernelsPPL.sampler_value_and_gradient!(query, gradient, u)
        @test value ≈ isolator_oracle(data, u) atol=1e-11 rtol=1e-11
        @test gradient ≈ isolator_gradient(data, u) atol=1e-10 rtol=1e-10
    end
    # A and B are different laws, so the return to A cannot pass on stale values.
    @test !(isolator_oracle(A, u) ≈ isolator_oracle(B, u))
    @test !(isolator_gradient(A, u) ≈ isolator_gradient(B, u))
end

@stestset "a frozen replay keeps a prior's values, a fresh one takes new ones" begin
    build = RSM.isolator()
    A, B = merge(ISOLATOR, VALUE_A), merge(ISOLATOR, VALUE_B)
    rk = RKBRMI(build(A))
    u = rk_point(rk)
    @test_throws "prior input: mu = 0.7" reprocess(rk, B)
    @test_throws "prior input: sd = 1.3" reprocess(rk, merge(A, (; sd=1.3)))
    # Omitted or repeated values are the fit's.
    omitted = (; y=ISOLATOR.y, idx=ISOLATOR.idx, n=ISOLATOR.n)
    @test rk_density(reprocess(rk, omitted), u) == rk_density(rk, u)
    @test rk_density(reprocess(rk, A), u) == rk_density(rk, u)
    @test rk_density(reprocess(rk, B; freeze_constants=false), u) ≈
        isolator_oracle(B, u) atol=1e-11 rtol=1e-11
    # A data-free replay carries them in its state, and refuses others alike.
    state = brm_fitted_state(rk)
    @test rk_density(RKBRMI(build(omitted), state), u) == rk_density(rk, u)
    @test_throws "prior input: mu = 0.7" RKBRMI(build(B), state)
    # A design binder reuses them too.
    binder = rk_design_binder(rk)
    @test isequal(binder(omitted), rk_bound_data(reprocess(rk, omitted)))
    @test_throws "changes the fitted RK program" binder(B)
end

lkj_data(eta) = merge(LDATA, (; K=2, eta))

@stestset "one program for every LKJ eta, on both backends" begin
    field = RSM.lkj_model(:K, :eta)
    check_one_program("field", _ -> field, (1.7, 2.9); data=lkj_data)
    check_one_program("literal", eta -> RSM.lkj_model(2, eta), (1.7, 2.9); data=_ -> LDATA)
    check_one_lowering("field lowering", field, [lkj_data(1.7), lkj_data(2.9)], (:eta,))
    check_one_lowering("literal lowering", eta -> RSM.lkj_model(2, eta)(LDATA), [1.7, 2.9], ())
    @testset "stratified block" begin
        check_one_lowering("field", RSM.stratified_lkj(:K, :eta),
            [merge(STRATA, (; K=2, eta)) for eta in (1.7, 2.9)], (:eta,))
    end
    # Stan reads it as data: the value reaches the data block, not the program.
    field_sb = stan_inputs(field(lkj_data(1.7))), stan_inputs(field(lkj_data(2.9)))
    keys_eta = [k for k in keys(first(field_sb)) if endswith(string(k), "_lkj_eta")]
    @test length(keys_eta) == 1
    @test first(field_sb)[only(keys_eta)] == 1.7
    @test last(field_sb)[only(keys_eta)] == 2.9
    @test !occursin("1.7", stan_program(field(lkj_data(1.7))))
end

# A two-margin block's factor is `L[2, 1] = tanh(u)`, so two concentrations differ
# by the LKJ density of that one coordinate at each.
function lkj_coordinate_density(u, eta)
    r = tanh(u)
    logpdf(LKJCholesky(2, eta), Cholesky(LowerTriangular([1.0 0.0; r sqrt(1 - r^2)])))
end

@stestset "a block's eta reaches the density" begin
    field = RSM.lkj_model(:K, :eta)
    a, b = RKBRMI(field(lkj_data(1.7))), RKBRMI(field(lkj_data(2.9)))
    names = coordinate_names(a.model.layout)
    index = only(findall(name -> occursin(".L", string(name)), names))
    u = rk_point(a)
    @test rk_density(a, u) - rk_density(b, u) ≈
        lkj_coordinate_density(u[index], 1.7) - lkj_coordinate_density(u[index], 2.9) atol=1e-11
    @test rk_density(a, u) == rk_density(RKBRMI(RSM.lkj_model(2, 1.7)(LDATA)), u)
    # A frozen replay refuses another eta and a fresh one takes it; the literal's
    # generated input is the fit's on every replay.
    @test_throws "prior input: eta = 2.9" reprocess(a, lkj_data(2.9))
    @test rk_density(reprocess(a, lkj_data(2.9); freeze_constants=false), u) == rk_density(b, u)
    literal = RKBRMI(RSM.lkj_model(2, 1.7)(LDATA))
    @test rk_density(reprocess(literal, LDATA), u) == rk_density(a, u)
    binder = rk_design_binder(literal)
    design = (; subject=LDATA.subject, x=LDATA.x, y=LDATA.y)
    @test isequal(binder(design), rk_bound_data(reprocess(literal, design)))
    @test !occursin("replayed in full", sprint(show, binder))
    # A Stan replay keeps the fit's value in its data, as it keeps its program.
    sb = SBBRMI(field(lkj_data(1.7)); mod=RSM)
    replayed = BRM.stan_data(reprocess(sb, lkj_data(1.7)))
    key = only(k for k in keys(replayed) if endswith(string(k), "_lkj_eta"))
    @test replayed[key] == 1.7
end

@stestset "a checkpoint identity sees a prior's numbers" begin
    # The program no longer spells them, so what identifies a fit states them:
    # a resumed SVI run, a carried SMC posterior and a filter compare this.
    build = RSM.isolator()
    a, b = RKBRMI(build(merge(ISOLATOR, VALUE_A))), RKBRMI(build(merge(ISOLATOR, VALUE_B)))
    @test emitted_program(a) == emitted_program(b)
    @test BRM._rk_identity_statements(a) != BRM._rk_identity_statements(b)
    @test "prior input: mu = 0.1" in BRM._rk_identity_statements(a)
    @test BRM._brm_svi_source_id(a) != BRM._brm_svi_source_id(b)
    same = RKBRMI(build(merge(ISOLATOR, VALUE_A)))
    @test BRM._brm_svi_source_id(a) == BRM._brm_svi_source_id(same)
    # A number only the observation reads is the data's: it identifies no prior.
    observed = brm(:(begin
        z[1:n] .~ LogNormal.(0.1, 0.5)
        y ~ Normal(z[idx], noise)
    end); mod=RSM)
    x1, x2 = RKBRMI(observed(merge(ISOLATOR, (; noise=0.2)))), RKBRMI(observed(merge(ISOLATOR, (; noise=0.4))))
    @test BRM._rk_identity_statements(x1) == BRM._rk_identity_statements(x2)
end

@stestset "support bounds are spelled by name" begin
    # The emitted program serves every value. RKPPL's lowering still reads a bound
    # into the constrain transform (ReactiveKernels snag `rkppl-sized-vect-69604d32`).
    for (label, field, literal, values) in (
            ("truncated", RSM.truncated_prior(:lo, :hi), v -> RSM.truncated_prior(v.lo, v.hi),
                ((; lo=0.1, hi=9.0), (; lo=0.05, hi=7.0))),
            ("restricted", RSM.restricted_prior(:lo), v -> RSM.restricted_prior(v.lo),
                ((; lo=0.1,), (; lo=0.2,))),
            ("Uniform", RSM.uniform_prior(:lo, :hi), v -> RSM.uniform_prior(v.lo, v.hi),
                ((; lo=0.1, hi=9.0), (; lo=0.2, hi=7.0))))
        @testset "$label" begin
            models = [RKBRMI(field(merge(LDATA, v))) for v in values]
            @test allequal(emitted_program(rk) for rk in models)
            @test_broken allequal(lowered_program(rk) for rk in models)
            for (v, rk) in zip(values, models)
                twin = RKBRMI(literal(v)(LDATA))
                u = rk_point(twin)
                @test rk_density(rk, u) ≈ rk_density(twin, u) atol=1e-12 rtol=1e-12
                @test all(haskey(rk_bound_data(rk), name) for name in keys(v))
            end
            u = rk_point(first(models))
            @test rk_density(first(models), u) != rk_density(last(models), u)
            first_value = first(values)
            changed = map(x -> x + 0.01, first_value)
            @test_throws "prior input:" reprocess(first(models), merge(LDATA, first_value, changed))
        end
    end
    # The plan still reads the numbers for its support checks.
    @test_throws "lower < upper" RKBRMI(RSM.truncated_prior(:lo, :hi)(merge(LDATA, (; lo=2.0, hi=1.0))))
    @test_throws "`Uniform` bounds must be finite" RKBRMI(
        RSM.uniform_prior(:lo, :hi)(merge(LDATA, (; lo=2.0, hi=1.0))))
end

@stestset "a law's array built from scalars is an input" begin
    check_one_lowering("MvNormal scale", RSM.mvnormal_scale(:s),
        [(; y=[0.3, 0.1], s) for s in (1.7, 2.9)], (:s,))
    check_one_lowering("Dirichlet concentration", RSM.dirichlet_prior(:a),
        [(; y=[1, 2, 3, 1], a) for a in (1.7, 2.9)], (:a,))
    for (field, literal) in (
            (RSM.mvnormal_scale(:s)((; y=[0.3, 0.1], s=1.7)), RSM.mvnormal_scale(1.7)((; y=[0.3, 0.1]))),
            (RSM.dirichlet_prior(:a)((; y=[1, 2, 3, 1], a=1.7)), RSM.dirichlet_prior(1.7)((; y=[1, 2, 3, 1]))))
        rk_field, rk_literal = RKBRMI(field), RKBRMI(literal)
        u = rk_point(rk_literal)
        @test rk_density(rk_field, u) ≈ rk_density(rk_literal, u) atol=1e-12 rtol=1e-12
    end
end

@stestset "a factor's levels and count are inputs" begin
    base = (; y=LDATA.y)
    labels(a, b, c) = (; f=[a, a, b, b, c, c])
    build = RSM.factor_effects()
    # The same count under other labels: one emitted and one lowered program.
    check_one_lowering("other labels", build,
        [merge(base, labels("a", "b", "c")), merge(base, labels("a", "b", "z"))], ();
        differ=false)
    models = [RKBRMI(build(merge(base, d))) for d in (labels("a", "b", "c"), (; f=["a", "a", "b", "b", "c", "d"]))]
    @test allequal(emitted_program(rk) for rk in models)
    @test models[1].model.layout.total + 1 == models[2].model.layout.total
    # Positive control and the size RK owns: the layout spells the extent
    # (ReactiveKernels snag `rkppl-sized-vect-69604d32`).
    @test_broken allequal(lowered_program(rk) for rk in models)
    # A frozen replay keeps the fitted levels and refuses an unseen one.
    fit = first(models)
    @test rk_density(reprocess(fit, merge(base, labels("a", "b", "c"))), rk_point(fit)) ==
        rk_density(fit, rk_point(fit))
    @test_throws "is not a training level" reprocess(fit, merge(base, labels("a", "b", "unseen")))
    binder = rk_design_binder(fit)
    design = merge(base, labels("a", "b", "c"))
    @test isequal(binder(design), rk_bound_data(reprocess(fit, design)))
    @test !occursin("replayed in full", sprint(show, binder))
end

@stestset "a known correlation's factor is an input" begin
    C(c) = [1.0 c; c 1.0]
    check_one_lowering("data field", RSM.known_cor(:C),
        [merge(LDATA, (; C=C(c))) for c in (0.3, 0.5)], ())
    check_one_lowering("literal", c -> RSM.known_cor(:([1 $c; $c 1]))(LDATA), [0.3, 0.5], ())
    field, literal = RKBRMI(RSM.known_cor(:C)(merge(LDATA, (; C=C(0.3))))),
        RKBRMI(RSM.known_cor(:([1 0.3; 0.3 1]))(LDATA))
    u = rk_point(literal)
    @test rk_density(field, u) ≈ rk_density(literal, u) atol=1e-12 rtol=1e-12
    # Another matrix on a frozen replay is another fit.
    @test_throws "changes the fitted RK program" reprocess(field, merge(LDATA, (; C=C(0.5))))
    @test reprocess(field, merge(LDATA, (; C=C(0.3)))) isa RKBRMI
    state = brm_fitted_state(field)
    @test_throws "changes the fitted RK program" RKBRMI(RSM.known_cor(:C)(merge(LDATA, (; C=C(0.5)))), state)
    binder = rk_design_binder(literal)
    @test isequal(binder(LDATA), rk_bound_data(reprocess(literal, LDATA)))
    @test !occursin("replayed in full", sprint(show, binder))
end
