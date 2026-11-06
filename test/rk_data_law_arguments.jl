# test/rk_data_law_arguments.jl — a response law argument read from data on
# RKBRMI: a scalar data field, a `constant(...)` value, a per-row column or a
# plain alias as a Normal / Student-t / LogNormal scale, a censored response's
# scale, the scale of a model-scope `gmrf` field's readout, and the further
# arguments of the other response families.
#
# Such an argument is a runtime input of the RK program (user principle
# `0ikeidi`: numeric values are never baked into model source). Two data sets
# that differ only in the value therefore emit the same program, and the
# density follows the bound value. The formula route refused every one of
# these ("scale cannot be a data column"); the value route reads them by name
# (snag use-a-named-scal-99c4babc).
#
# Checks: an independent likelihood oracle, the compiled Stan program of the
# same BRMI through the coordinate transport (value and native Reverse
# gradient), complete printed-source replay, program identity across values,
# and frozen replay (`reprocess`) of a scalar field and of a constant.
#
# Run: julia --startup-file=no --project=test test/rk_data_law_arguments.jl [testset substring ...]

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

function law_data(; seed=5)
    rng = Xoshiro(seed)
    n = 8
    x = collect(range(-1.0, 1.0; length=n))
    lo, hi = fill(-0.5, n), fill(1.2, n)
    # Within the censoring bounds, with rows at each bound (censored rows).
    y = clamp.(0.4 .+ 0.7 .* x .+ 0.5 .* randn(rng, n), lo, hi)
    y[1], y[end] = lo[1], hi[end]
    pos = exp.(0.2 .* x .+ 0.3 .* randn(rng, n))
    sv = 0.3 .+ rand(rng, n)
    nuv = 3.0 .+ 4 .* rand(rng, n)
    (; x, y, pos, sv, nuv, s=0.6, half=0.3, alpha=3.0, lo, hi)
end

const SCALAR = @brm begin
    mu ~ 1 + x
    y ~ Normal(mu, s)
end
const CONSTANT = @brm begin
    sd_obs = constant(2 * half)
    mu ~ 1 + x
    y ~ Normal(mu, sd_obs)
end
const ALIAS = @brm begin
    sd_obs = s
    mu ~ 1 + x
    y ~ Normal(mu, sd_obs)
end
const PER_ROW = @brm begin
    mu ~ 1 + x
    y ~ Normal(mu, sv)
end
const CENSORED = @brm begin
    mu ~ 1 + x
    y ~ censored(Normal(mu, s); lower=lo, upper=hi)
end
const STUDENT = @brm begin
    mu ~ 1 + x
    y ~ LocationScale(mu, s, TDist(4.0))
end
const LOGNORMAL = @brm begin
    mu ~ 1 + x
    pos ~ LogNormal(mu, s)
end
const STUDENT_ROW_NU = @brm begin
    mu ~ 1 + x
    y ~ LocationScale(mu, s, TDist(nuv))
end
# A data shape inside Gamma's mean-shape spelling `Gamma(alpha, mu / alpha)`.
const GAMMA_SHAPE = @brm begin
    log(mu) ~ 1 + x
    pos ~ Gamma(alpha, mu / alpha)
end

# The linear predictor `beta_pop[1] + beta_pop[2] * x`, whatever its link names
# the one coefficient block (`pop_mu`, `pop_log_mu`, ...).
function law_location(backend, data, u)
    physical = constrain(backend.model.layout, u)
    intercept, slope = only(physical).beta_pop
    intercept .+ slope .* data.x
end

function law_oracle(label, backend, data, u)
    mu = law_location(backend, data, u)
    label in ("scalar", "alias") && return sum(logpdf.(Normal.(mu, data.s), data.y))
    label == "constant" && return sum(logpdf.(Normal.(mu, 2 * data.half), data.y))
    label == "per-row" && return sum(logpdf.(Normal.(mu, data.sv), data.y))
    label == "student" && return sum(logpdf.(LocationScale.(mu, data.s, TDist(4.0)), data.y))
    label == "lognormal" && return sum(logpdf.(LogNormal.(mu, data.s), data.pos))
    label == "student-row-nu" &&
        return sum(logpdf.(LocationScale.(mu, data.s, TDist.(data.nuv)), data.y))
    label == "gamma-shape" &&
        return sum(logpdf.(Gamma.(data.alpha, exp.(mu) ./ data.alpha), data.pos))
    label == "censored" || error("no oracle for $label")
    sum(map(eachindex(data.y)) do j
        d = Normal(mu[j], data.s)
        data.y[j] <= data.lo[j] ? logcdf(d, data.lo[j]) :
            data.y[j] >= data.hi[j] ? logccdf(d, data.hi[j]) : logpdf(d, data.y[j])
    end)
end

const CASES = (
    (label="scalar", builder=SCALAR, field=:s),
    (label="constant", builder=CONSTANT, field=:half),
    (label="alias", builder=ALIAS, field=:s),
    (label="per-row", builder=PER_ROW, field=:sv),
    (label="censored", builder=CENSORED, field=:s),
    (label="student", builder=STUDENT, field=:s),
    (label="lognormal", builder=LOGNORMAL, field=:s),
    (label="student-row-nu", builder=STUDENT_ROW_NU, field=:nuv),
    (label="gamma-shape", builder=GAMMA_SHAPE, field=:alpha),
)

const LAW_FOLDER = joinpath(tempdir(), "brm-rk-data-law-arguments")
function law_compiled(brmi, label)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    mkpath(LAW_FOLDER)
    code = BRM.stan_code(sb)
    stan = BRM.stan_instantiate(sb; path=joinpath(LAW_FOLDER,
        label * "_" * string(hash(code); base=16) * ".stan"))
    (; sb, stan, names=BridgeStan.param_unc_names(stan.model))
end

law_probes(n) = (zeros(n), collect(range(-0.3, 0.4; length=n)), fill(0.2, n))
law_likelihood(backend) = prepare_query(backend.model,
    Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)._rk_translated_plan(backend.plan),
    :likelihood)

# The reporter's shape: a model-scope `gmrf` field read by a Normal response
# whose scale is a named scalar input (and the same input as a constant).
function law_gmrf_data(; seed=11)
    rng = MersenneTwister(seed)
    npositions, times = 6, [0.0, 1.0, 2.0]
    ex, nx = 10 .* rand(rng, npositions), 10 .* rand(rng, npositions)
    east = repeat(ex; outer=length(times))
    north = repeat(nx; outer=length(times))
    time = repeat(times; inner=npositions)
    (; east, north, time, y=0.3 .* randn(rng, length(time)), observation_sd=0.6)
end
const GMRF_FIELD = quote
    field ~ 0 + gmrf((east, north), time; neighbors=3)
    sd(field, gmrf((east, north), time)) ~ Exponential(0.7)
    length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(3.0), 0.4)
    length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(1.7), 0.4)
end
const GMRF_SCALAR = brm(Expr(:block, GMRF_FIELD.args...,
    :(y ~ Normal(field, observation_sd))); mod=@__MODULE__)
const GMRF_CONSTANT = brm(Expr(:block, GMRF_FIELD.args...,
    :(field_sd_obs = constant(observation_sd)), :(y ~ Normal(field, field_sd_obs))); mod=@__MODULE__)

# The further arguments of the other response families, against an independent
# oracle of the linear predictor and the bound value.
law_inverse_logit(eta) = inv(1 + exp(-eta))
law_family_data() = let n = 6
    (; x=collect(range(-1.0, 1.0; length=n)), y=[-0.4, 0.1, 0.3, 0.2, 0.7, 1.1],
        pos=[0.7, 0.9, 1.3, 1.1, 2.2, 1.8], frac=[0.2, 0.35, 0.5, 0.45, 0.7, 0.6],
        k=[1, 0, 2, 3, 2, 5], c=[0, 2, 0, 1, 3, 0], ang=[-0.3, 0.1, 0.4, 0.2, 0.8, 1.0],
        trials=fill(8, n), zv=fill(0.3, n), pv=fill(0.4, n), s=0.6, nu=1.5, phi=2.5,
        shape=1.4, kappa=6.0, zi=0.3)
end
const FAMILIES = (
    (label="negative binomial", field=:phi, other=4.0,
        builder=(@brm begin
            log(mu) ~ 1 + x
            k ~ NegativeBinomial2(mu, phi)
        end),
        oracle=(eta, v, d) -> sum(logpdf.(NegativeBinomial.(v, v ./ (v .+ exp.(eta))), d.k))),
    (label="weibull", field=:shape, other=2.2,
        builder=(@brm begin
            log(mu) ~ 1 + x
            pos ~ Weibull(shape, mu)
        end),
        oracle=(eta, v, d) -> sum(logpdf.(Weibull.(v, exp.(eta)), d.pos))),
    (label="beta mean-precision", field=:kappa, other=11.0,
        builder=(@brm begin
            logit(mu) ~ 1 + x
            frac ~ Beta(mu * kappa, (1 - mu) * kappa)
        end),
        oracle=(eta, v, d) -> let p = law_inverse_logit.(eta)
            sum(logpdf.(Beta.(p .* v, (1 .- p) .* v), d.frac))
        end),
    (label="von mises", field=:kappa, other=11.0,
        builder=(@brm begin
            mu ~ 1 + x
            ang ~ VonMises(mu, kappa)
        end),
        oracle=(eta, v, d) -> sum(logpdf.(VonMises.(eta, v), d.ang))),
    (label="inverse gaussian", field=:shape, other=2.2,
        builder=(@brm begin
            log(mu) ~ 1 + x
            pos ~ InverseGaussian(mu, shape)
        end),
        oracle=(eta, v, d) -> sum(logpdf.(InverseGaussian.(exp.(eta), v), d.pos))),
    (label="beta binomial precision", field=:phi, other=4.0,
        builder=(@brm begin
            logit(mu) ~ 1 + x
            k ~ BetaBinomial2(trials, mu, phi)
        end),
        oracle=(eta, v, d) -> let p = law_inverse_logit.(eta)
            sum(logpdf.(BetaBinomial.(d.trials, p .* v, (1 .- p) .* v), d.k))
        end),
    (label="hurdle per-row", field=:zv, other=fill(0.5, 6),
        builder=(@brm begin
            log(lambda) ~ 1 + x
            c ~ HurdlePoisson(lambda, zv)
        end),
        oracle=(eta, v, d) -> sum(map(d.c, exp.(eta), v) do c, lambda, p
            c == 0 ? log(p) : log1p(-p) + logpdf(Poisson(lambda), c) - log1p(-exp(-lambda))
        end)),
    (label="zero inflated scalar", field=:zi, other=0.5,
        builder=(@brm begin
            log(lambda) ~ 1 + x
            c ~ ZeroInflatedPoisson(lambda, zi)
        end),
        oracle=(eta, v, d) -> sum(map(d.c, exp.(eta)) do c, lambda
            c == 0 ? log(v + (1 - v) * exp(-lambda)) : log1p(-v) + logpdf(Poisson(lambda), c)
        end)),
    (label="negative binomial success probability", field=:pv, other=fill(0.6, 6),
        builder=(@brm begin
            log(r) ~ 1 + x
            c ~ NegativeBinomial(r, pv)
        end),
        oracle=(eta, v, d) -> sum(logpdf.(NegativeBinomial.(exp.(eta), v), d.c))),
    # A deterministic wrapper of data is data: `exp(nu)` is computed at run time.
    (label="wrapped student-t", field=:nu, other=2.0,
        builder=(@brm begin
            mu ~ 1 + x
            y ~ LocationScale(mu, s, TDist(exp(nu)))
        end),
        oracle=(eta, v, d) -> sum(logpdf.(LocationScale.(eta, d.s, TDist(exp(v))), d.y))),
)

@testset "RKBRMI data law arguments" begin

@stestset "a data law argument binds at run time and agrees with SBBRMI" begin
    data = law_data()
    for case in CASES
        @testset "$(case.label)" begin
            original = deepcopy(data)
            brmi = case.builder(data)
            backend = check_rk_source_roundtrip(RKBRMI(brmi);
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            problem = rk_logdensity_problem(backend;
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            likelihood = law_likelihood(backend)
            c = law_compiled(brmi, replace(case.label, "-" => "_"))
            t = brm_coordinate_transport(backend, c.sb, String.(c.names))
            @test LogDensityProblems.dimension(problem) == 2
            for u in law_probes(LogDensityProblems.dimension(problem))
                before = copy(u)
                @test Base.invokelatest(likelihood, u) ≈
                    law_oracle(case.label, backend, data, u) atol=1e-10 rtol=1e-11
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                stan_value, stan_gradient = BridgeStan.log_density_gradient(c.stan.model,
                    brm_rk_to_stan(t, u); propto=false, jacobian=true)
                @test value ≈ stan_value atol=1e-10 rtol=1e-11
                @test gradient ≈ brm_stan_to_rk(t, stan_gradient) atol=1e-9 rtol=1e-10
                @test u == before
            end
            @test data == original
        end
    end
end

# The value is an input: another value emits the same program, binds the new
# value, and keeps the coordinates.
@stestset "another value emits the same program" begin
    data = law_data()
    for case in CASES
        @testset "$(case.label)" begin
            value = getproperty(data, case.field)
            changed = merge(data, NamedTuple{(case.field,)}((value isa Real ?
                1.7 * value : reverse(value),)))
            a, b = RKBRMI(case.builder(data)), RKBRMI(case.builder(changed))
            ea, eb = BRM._rk_emit_ast(a.plan), BRM._rk_emit_ast(b.plan)
            @test ea.main == eb.main
            @test ea.defs == eb.defs
            @test string(a.model.program) == string(b.model.program)
            @test coordinate_names(a.model.layout) == coordinate_names(b.model.layout)
            # No emitted statement spells the value.
            value isa Real && @test !occursin(string(value), sprint(Base.show_unquoted, ea.main))
            lb = law_likelihood(b)
            for u in law_probes(2)
                @test Base.invokelatest(lb, u) ≈
                    law_oracle(case.label, b, changed, u) atol=1e-10 rtol=1e-11
                @test !isapprox(Base.invokelatest(lb, u),
                    Base.invokelatest(law_likelihood(a), u); rtol=1e-6)
            end
        end
    end
end

# A frozen replay keeps a constant at its fitted value; a scalar data field
# read only by an observation is a design input: omitted, it keeps the fitted
# value, supplied, the design's (brm-use `reactivekernels.md`, frozen replay).
# An invalid value is refused when the backend binds it, naming the input.
@stestset "binding refuses an invalid data scale" begin
    data = law_data()
    for s in (0.0, -1.0, NaN, Inf)
        @test_throws "per-observation scale s must be finite positive numerics" RKBRMI(
            SCALAR(merge(data, (; s))))
    end
    @test_throws "per-observation scale sv must be finite positive numerics" RKBRMI(
        PER_ROW(merge(data, (; sv=[-1.0; data.sv[2:end]]))))
    family = law_family_data()
    builder(label) = only(case.builder for case in FAMILIES if case.label == label)
    @test_throws "per-observation p_zero zv must be finite numerics in [0, 1]" RKBRMI(
        builder("hurdle per-row")(merge(family, (; zv=[1.5; family.zv[2:end]]))))
    @test_throws "zi data zi must lie in [0, 1]" RKBRMI(
        builder("zero inflated scalar")(merge(family, (; zi=2.0))))
    @test_throws "per-observation NB1 p pv must be finite numerics in [0, 1]" RKBRMI(
        builder("negative binomial success probability")(
            merge(family, (; pv=[-0.2; family.pv[2:end]]))))
end

# The cutpoints and category predictors of a leveled response are not scalar law
# arguments: a data column there stays refused, with the formula route's message.
@stestset "a leveled response keeps its data refusals" begin
    data = (; x=collect(range(-1.0, 1.0; length=6)), x2=collect(range(0.0, 1.0; length=6)),
        level=[1, 1, 2, 2, 3, 3], outcome=[1, 2, 3, 4, 1, 2])
    @test_throws "needs one argument" BRM._brm_rk_plan((@brm begin
        eta ~ 0 + x
        level ~ OrderedLogistic(eta, x2)
    end)(data))
    @test_throws "not a declared linear predictor" BRM._brm_rk_plan((@brm begin
        eta1 ~ 1 + x
        eta2 ~ 1 + x
        outcome ~ CategoricalLogit(eta1, eta2, x2)
    end)(data))
end

@stestset "frozen replay of a scalar field and of a constant" begin
    data = law_data()
    u = collect(range(-0.3, 0.4; length=2))
    rows = (; x=data.x[1:3], y=data.y[1:3])
    fit = RKBRMI(SCALAR(data))
    omitted = reprocess(fit, rows)
    @test Base.invokelatest(law_likelihood(omitted), u) ≈
        law_oracle("scalar", omitted, merge(rows, (; s=data.s)), u) atol=1e-10 rtol=1e-11
    supplied = reprocess(fit, merge(rows, (; s=1.1)))
    @test Base.invokelatest(law_likelihood(supplied), u) ≈
        law_oracle("scalar", supplied, merge(rows, (; s=1.1)), u) atol=1e-10 rtol=1e-11
    constant_fit = RKBRMI(CONSTANT(data))
    frozen = reprocess(constant_fit, merge(rows, (; half=0.9)))
    @test Base.invokelatest(law_likelihood(frozen), u) ≈
        law_oracle("constant", frozen, merge(rows, (; half=data.half)), u) atol=1e-10 rtol=1e-11
end

@stestset "every response family reads a data argument at run time" begin
    data = law_family_data()
    for case in FAMILIES
        @testset "$(case.label)" begin
            value = getproperty(data, case.field)
            changed = merge(data, NamedTuple{(case.field,)}((case.other,)))
            a, b = RKBRMI(case.builder(data)), RKBRMI(case.builder(changed))
            @test a.plan isa BRM._RKValuePlan
            @test BRM._rk_emit_ast(a.plan).main == BRM._rk_emit_ast(b.plan).main
            la, lb = law_likelihood(a), law_likelihood(b)
            problem = rk_logdensity_problem(a; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            for u in law_probes(2)
                eta = law_location(a, data, u)
                @test Base.invokelatest(la, u) ≈ case.oracle(eta, value, data) atol=1e-10 rtol=1e-11
                @test Base.invokelatest(lb, u) ≈
                    case.oracle(eta, case.other, changed) atol=1e-10 rtol=1e-11
                _, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                h = 1e-6
                differences = map(eachindex(u)) do i
                    e = [j == i ? h : 0.0 for j in eachindex(u)]
                    (LogDensityProblems.logdensity(problem, u .+ e) -
                     LogDensityProblems.logdensity(problem, u .- e)) / 2h
                end
                @test gradient ≈ differences atol=1e-5 rtol=1e-5
            end
        end
    end
    # Binomial trials keep their formula spelling, already bound at run time.
    trials = @brm begin
        logit(p) ~ 1 + x
        k ~ Binomial(trials, p)
    end
    a = RKBRMI(trials(data))
    b = RKBRMI(trials(merge(data, (; trials=fill(9, length(data.k))))))
    @test !(a.plan isa BRM._RKValuePlan)
    @test BRM._rk_emit_ast(a.plan).main == BRM._rk_emit_ast(b.plan).main
    @test !isapprox(Base.invokelatest(law_likelihood(a), zeros(2)),
        Base.invokelatest(law_likelihood(b), zeros(2)); rtol=1e-6)
end

@stestset "a gmrf readout with a named scalar scale" begin
    data = law_gmrf_data()
    for (label, builder) in (("scalar", GMRF_SCALAR), ("constant", GMRF_CONSTANT))
        @testset "$label" begin
            brmi = builder(data)
            backend = RKBRMI(brmi)
            problem = rk_logdensity_problem(backend;
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            c = law_compiled(brmi, "gmrf_" * label)
            t = brm_coordinate_transport(backend, c.sb, String.(c.names))
            rng = Xoshiro(3)
            for _ in 1:3
                u = 0.3 .* randn(rng, LogDensityProblems.dimension(problem))
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                stan_value, stan_gradient = BridgeStan.log_density_gradient(c.stan.model,
                    brm_rk_to_stan(t, u); propto=false, jacobian=true)
                @test value ≈ stan_value atol=1e-8 rtol=1e-10
                @test gradient ≈ brm_stan_to_rk(t, stan_gradient) atol=1e-7 rtol=1e-8
            end
            other = RKBRMI(builder(merge(data, (; observation_sd=1.3))))
            @test BRM._rk_emit_ast(other.plan).main == BRM._rk_emit_ast(backend.plan).main
        end
    end
end

end
