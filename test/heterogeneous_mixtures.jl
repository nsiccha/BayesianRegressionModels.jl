# Native composition of different full-real scalar laws. The oracle composes
# Distributions.jl densities independently of either backend's emitter.
# julia --project=test test/heterogeneous_mixtures.jl [testset substring ...]
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
include(joinpath(@__DIR__, "mixture_stan_values.jl"))

function heterogeneous_builder(kind)
    if kind === :cauchy
        @brm begin
            mu ~ Normal(0, 1)
            sigma ~ Exponential(1)
            w ~ Beta(2, 3)
            y ~ MixtureModel([Normal(mu, sigma), Cauchy(mu, 1.7)], [1 - w, w])
        end
    else
        @brm begin
            mu ~ Normal(0, 1)
            sigma ~ Exponential(1)
            w ~ Beta(2, 3)
            # TDist has one Julia argument and three Stan arguments; the
            # mixture must retain both components' own translated signatures.
            y ~ MixtureModel([Normal(mu, sigma), TDist(4.5)], [1 - w, w])
        end
    end
end

function heterogeneous_terms(kind, y, mu, sigma, w)
    second = kind === :cauchy ? Cauchy(mu, 1.7) : TDist(4.5)
    [log1p(-w) + logpdf(Normal(mu, sigma), y), log(w) + logpdf(second, y)]
end

function heterogeneous_oracle(kind, data, names, u)
    mu = u[coordinate(names, :mu)]
    log_sigma = u[coordinate(names, :sigma)]
    logit_w = u[coordinate(names, :w)]
    sigma, w = exp(log_sigma), LogExpFunctions.logistic(logit_w)
    prior = logpdf(Normal(), mu) + logpdf(Exponential(), sigma) + log_sigma +
        logpdf(Beta(2, 3), w) + log(w) + log1p(-w)
    prior + sum(y -> LogExpFunctions.logsumexp(
        heterogeneous_terms(kind, y, mu, sigma, w)), data.y; init=0.0)
end

function heterogeneous_pointwise(kind, data, names, u)
    mu, sigma = u[coordinate(names, :mu)], exp(u[coordinate(names, :sigma)])
    w = LogExpFunctions.logistic(u[coordinate(names, :w)])
    [LogExpFunctions.logsumexp(heterogeneous_terms(kind, y, mu, sigma, w))
        for y in data.y]
end

function heterogeneous_check_stan(problem, u, oracle)
    gradient = similar(u)
    value, _ = BridgeStan.log_density_gradient!(problem.model, u, gradient;
        propto=false, jacobian=true)
    @test value ≈ oracle(u) atol=2e-11 rtol=2e-11
    h = 1e-5
    fd = map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += h; minus[j] -= h
        (oracle(plus) - oracle(minus)) / (2h)
    end
    @test gradient ≈ fd atol=2e-8 rtol=2e-8
end

@stestset "Stan heterogeneous scalar mixtures" begin
    data = (; y=[-8.0, -1.1, 0.2, 1.3, 11.0])
    for kind in (:cauchy, :student)
        brmi = heterogeneous_builder(kind)(data)
        descriptor = brm_descriptor(SBBRMI(brmi; mod=@__MODULE__); highlights=())
        output = brm_output(descriptor, :y; role=:membership)
        @test output.dims == (length(data.y), 2)
        problem = mixture_problem(descriptor)
        names = Symbol.(BridgeStan.param_unc_names(problem.model))
        for u in ([0.0, 0.0, 0.0], [0.3, -0.4, 0.7])
            heterogeneous_check_stan(problem, u,
                v -> heterogeneous_oracle(kind, data, names, v))
            values = mixture_values(problem, u)
            pointwise = [values["y_likelihood.$i"] for i in eachindex(data.y)]
            @test pointwise ≈ heterogeneous_pointwise(kind, data, names, u) atol=2e-11
            mu, sigma, w = values["mu"], values["sigma"], values["w"]
            for i in eachindex(data.y)
                terms = heterogeneous_terms(kind, data.y[i], mu, sigma, w)
                probabilities = exp.(terms .- LogExpFunctions.logsumexp(terms))
                @test [values["y_membership.$i.$k"] for k in 1:2] ≈ probabilities
                @test isfinite(values["y_gen.$i"])
            end
        end
    end
end

include(joinpath(@__DIR__, "heterogeneous_mixture_laws.jl"))
using .SyntheticSkewComponent: centered_skew
using .SyntheticExcessComponent: logscale_excess
using .SyntheticBoundedComponent: lower_shift

function custom_mixture_builder(row_alpha)
    if row_alpha
        @brm begin
            log_scale ~ Normal(0, 1)
            alpha ~ Normal(0, 1)
            shift ~ Normal(0, 1)
            log_rate ~ Normal(0, 1)
            w ~ Beta(2, 3)
            shape = alpha + x
            y ~ MixtureModel([centered_skew(log_scale, shape),
                logscale_excess(shift, log_scale, log_rate)], [1 - w, w])
        end
    else
        @brm begin
            log_scale ~ Normal(0, 1)
            alpha ~ Normal(0, 1)
            shift ~ Normal(0, 1)
            log_rate ~ Normal(0, 1)
            w ~ Beta(2, 3)
            y ~ MixtureModel([centered_skew(log_scale, alpha),
                logscale_excess(shift, log_scale, log_rate)], [1 - w, w])
        end
    end
end

# Independent normalized formulas. The test points keep lambda*sigma moderate,
# including overflowing sigma at log_scale=800 with log_rate=-800.
function custom_mixture_terms(y, log_scale, alpha, shift, log_rate, w)
    delta = alpha / hypot(1.0, alpha)
    center = sqrt(2 / pi) * delta
    sd = sqrt(1 - center^2)
    z = y * exp(-log_scale) * sd + center
    skew = log(2) + logpdf(Normal(), z) + logcdf(Normal(), alpha * z) +
        log(sd) - log_scale
    u = y * exp(-log_scale) - shift
    q = exp(log_scale + log_rate)
    excess = log_rate + q^2 / 2 - q * u + logcdf(Normal(), u - q)
    [log1p(-w) + skew, log(w) + excess]
end

function custom_mixture_oracle(data, names, u, row_alpha)
    p(name) = u[coordinate(names, name)]
    w = LogExpFunctions.logistic(p(:w))
    prior = sum(name -> logpdf(Normal(), p(name)),
        (:log_scale, :alpha, :shift, :log_rate)) +
        logpdf(Beta(2, 3), w) + log(w) + log1p(-w)
    prior + sum(eachindex(data.y); init=0.0) do i
        alpha = p(:alpha) + (row_alpha ? data.x[i] : 0.0)
        LogExpFunctions.logsumexp(custom_mixture_terms(data.y[i], p(:log_scale),
            alpha, p(:shift), p(:log_rate), w))
    end
end

@stestset "Stan heterogeneous custom components" begin
    # Exercise public BRM admission and ordinary lowering. The component
    # tokens stay in their own modules and have different argument signatures.
    data = (; y=[-1.1, 0.1, 1.3], x=[-0.3, 0.2, 0.7])
    for row_alpha in (false, true)
        brmi = custom_mixture_builder(row_alpha)(data)
        descriptor = brm_descriptor(SBBRMI(brmi; mod=@__MODULE__); highlights=())
        @test brm_output(descriptor, :y; role=:membership).dims == (length(data.y), 2)
        problem = mixture_problem(descriptor)
        names = Symbol.(BridgeStan.param_unc_names(problem.model))
        for u in (zeros(length(names)), collect(range(-0.2, 0.3; length=length(names))))
            heterogeneous_check_stan(problem, u,
                v -> custom_mixture_oracle(data, names, v, row_alpha))
            values = mixture_values(problem, u)
            w = values["w"]
            for i in eachindex(data.y)
                alpha = values["alpha"] + (row_alpha ? data.x[i] : 0.0)
                terms = custom_mixture_terms(data.y[i], values["log_scale"],
                    alpha, values["shift"], values["log_rate"], w)
                expected = LogExpFunctions.logsumexp(terms)
                @test values["y_likelihood.$i"] ≈ expected
                @test [values["y_membership.$i.$k"] for k in 1:2] ≈ exp.(terms .- expected)
                @test isfinite(values["y_gen.$i"])
            end
        end
        extreme = zeros(length(names))
        extreme[coordinate(names, :log_scale)] = 800.0
        extreme[coordinate(names, :log_rate)] = -800.0
        gradient = similar(extreme)
        value, _ = BridgeStan.log_density_gradient!(problem.model, extreme, gradient;
            propto=false, jacobian=true)
        @test isfinite(value)
        @test all(isfinite, gradient)
        @test value ≈ custom_mixture_oracle(data, names, extreme, row_alpha) rtol=2e-12
    end
end

@stestset "custom full-real components retain stable log scales and graph source" begin
    data = (; y=[-1.1, 0.1, 1.3], x=[-0.3, 0.2, 0.7])
    @testset "row_alpha=$row_alpha" for row_alpha in (false, true)
        brmi = custom_mixture_builder(row_alpha)(data)
        backend = check_rk_source_roundtrip(RKBRMI(brmi);
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        problem = rk_logdensity_problem(backend;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        names = coordinate_names(backend.model.layout)
        translated = Base.get_extension(BRM,
            :BayesianRegressionModelsReactiveKernelsExt)._rk_translated_plan(backend.plan)
        pointwise = prepare_query(backend.model, translated, :pointwise)
        stan = consumer_stan(brmi, "heterogeneous-custom-$row_alpha")
        mapping = [name => string(name) for name in names]
        for u in (zeros(length(names)), collect(range(-0.2, 0.3; length=length(names))))
            check_consumer_point(problem, u,
                v -> custom_mixture_oracle(data, names, v, row_alpha))
            check_consumer_stan(problem, stan, mapping, backend, u)
            p(name) = u[coordinate(names, name)]
            w = LogExpFunctions.logistic(p(:w))
            expected = map(eachindex(data.y)) do i
                shape = p(:alpha) + (row_alpha ? data.x[i] : 0.0)
                LogExpFunctions.logsumexp(custom_mixture_terms(data.y[i],
                    p(:log_scale), shape, p(:shift), p(:log_rate), w))
            end
            @test Base.invokelatest(pointwise, u).y ≈ expected atol=2e-11
        end
        extreme = zeros(length(names))
        extreme[coordinate(names, :log_scale)] = 800.0
        extreme[coordinate(names, :log_rate)] = -800.0
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, extreme)
        @test isfinite(value)
        @test all(isfinite, gradient)
        @test value ≈ custom_mixture_oracle(data, names, extreme, row_alpha) rtol=2e-12
        check_consumer_stan(problem, stan, mapping, backend, extreme)

        # Both shared and row-varying shape arguments keep one model when the
        # observation plate changes size. Empty domains have a separate
        # acceptance block so a failure there cannot hide the other cases.
        @testset "rebind n=$(length(rows))" for rows in
                ([0.6], [-2.0, 0.1, 1.0, 3.0, 5.0, 8.0])
            new_data = (; y=rows, x=[0.1 * i - 0.3 for i in eachindex(rows)])
            rebound = reprocess(backend, new_data)
            @test rebound.model === backend.model
            rebound_problem = rk_logdensity_problem(rebound;
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            for u in (zeros(length(names)), fill(0.2, length(names)))
                check_consumer_point(rebound_problem, u,
                    v -> custom_mixture_oracle(new_data, names, v, row_alpha))
            end
        end
    end
end

@stestset "RK heterogeneous scalar mixtures and complete source replay" begin
    data = (; y=[-8.0, -1.1, 0.2, 1.3, 11.0])
    @testset "$kind" for kind in (:cauchy, :student)
        brmi = heterogeneous_builder(kind)(data)
        backend = check_rk_source_roundtrip(RKBRMI(brmi);
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        problem = rk_logdensity_problem(backend;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        @test backend.plan isa BRM._RKValuePlan
        names = coordinate_names(backend.model.layout)
        stan = consumer_stan(brmi, "heterogeneous-$kind")
        translated = Base.get_extension(BRM,
            :BayesianRegressionModelsReactiveKernelsExt)._rk_translated_plan(backend.plan)
        pointwise = prepare_query(backend.model, translated, :pointwise)
        for u in ([0.0, 0.0, 0.0], [0.3, -0.4, 0.7])
            check_consumer_point(problem, u,
                v -> heterogeneous_oracle(kind, data, names, v))
            check_consumer_stan(problem, stan,
                [:mu => "mu", :sigma => "sigma", :w => "w"], backend, u)
            @test Base.invokelatest(pointwise, u).y ≈
                heterogeneous_pointwise(kind, data, names, u) atol=2e-11
        end
        # Rebinding keeps one executable graph and changes only observation
        # data. Empty likelihood plates are checked independently below.
        @testset "rebind n=$(length(rows))" for rows in
                ([0.6], [-2.0, 0.1, 1.0, 3.0, 5.0, 8.0])
            new_data = (; y=rows)
            replay = reprocess(backend, new_data)
            @test replay.model === backend.model
            replay_problem = rk_logdensity_problem(replay;
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            check_consumer_point(replay_problem, [0.2, -0.1, 0.3],
                v -> heterogeneous_oracle(kind, new_data, names, v))
        end
    end
end

module MixtureSchedulingProbe
const shape_calls = Ref(0)
const scale_calls = Ref(0)
measured_hypot(x, y) = (shape_calls[] += 1; hypot(x, y))
measured_inverse_scale(x) = (scale_calls[] += 1; exp(-x))
reset!() = (shape_calls[] = scale_calls[] = 0)

# Instrument the caller's source before normal lowering; never patch a
# translated graph. The uninstrumented graph's Reverse is tested above.
function measured_source(ex)
    ex isa Expr || return ex
    ex == :(hypot(1.0, alpha)) &&
        return :(brm_test_measured_hypot(1.0, alpha))
    ex == :(exp(-log_scale)) &&
        return :(brm_test_measured_inverse_scale(log_scale))
    Expr(ex.head, map(measured_source, ex.args)...)
end

function shape_loop_depths!(depths, ex, depth=0)
    ex isa Expr || return depths
    if ex.head === :call && occursin("brm_test_measured_hypot", string(ex.args[1]))
        push!(depths, depth)
    end
    next = depth + (ex.head in (:for, :while))
    foreach(arg -> shape_loop_depths!(depths, arg, next), ex.args)
    depths
end
end

@stestset "native mixture scheduling follows component dependencies" begin
    probe = MixtureSchedulingProbe
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    @testset "row_alpha=$row_alpha n=$n" for row_alpha in (false, true), n in (0, 1, 6, 32)
        data = (; y=[0.1i - 0.4 for i in 1:n], x=[0.2i - 0.3 for i in 1:n])
        brmi = custom_mixture_builder(row_alpha)(data)
        plan = BRM._brm_rk_plan(brmi)
        emitted = BRM._rk_emit_ast(plan)
        # Count the centered-skew component; the excess component is unchanged.
        definitions = [occursin("log_normalizer", string(d)) ?
            probe.measured_source(d) : d for d in emitted.defs]
        bindings = [emitted.bindings;
            :brm_test_measured_hypot => probe.measured_hypot;
            :brm_test_measured_inverse_scale => probe.measured_inverse_scale]
        instrumented = BRM._RKEmittedProgram(definitions, emitted.main, bindings)
        translated = ext._rk_translate_from_emitted(plan, instrumented)
        model = Base.invokelatest(build_kernel, translated)
        probe.reset!()
        query = prepare_query(model, translated, :likelihood)
        @test probe.shape_calls[] == probe.scale_calls[] == 0
        names = coordinate_names(model.layout)
        for shift in (0.0, 0.2)
            u = fill(shift, length(names))
            probe.reset!()
            value = Base.invokelatest(query, u)
            @test probe.shape_calls[] == (row_alpha ? n : min(n, 1))
            @test probe.scale_calls[] == min(n, 1)
            w = LogExpFunctions.logistic(shift)
            expected = sum((LogExpFunctions.logsumexp(custom_mixture_terms(
                data.y[i], shift, shift + (row_alpha ? data.x[i] : 0.0),
                shift, shift, w)) for i in eachindex(data.y)); init=0.0)
            @test value ≈ expected atol=2e-11
        end
        # A rejected probability vector must not execute either component.
        if n > 0
            invalid = zeros(length(names))
            invalid[coordinate(names, :w)] = NaN
            probe.reset!()
            @test Base.invokelatest(query, invalid) == -Inf
            @test probe.shape_calls[] == probe.scale_calls[] == 0
        end
        if n == 6
            generated = code_expr(query)
            depths = probe.shape_loop_depths!(Int[], generated)
            @test !isempty(depths)
            @test all(d -> row_alpha ? d > 0 : d == 0, depths)
            @test !occursin("_rng", string(generated))
        end
    end
end

@stestset "RK heterogeneous empty data rebinding" begin
    @testset "$kind" for kind in (:cauchy, :student)
        backend = RKBRMI(heterogeneous_builder(kind)((; y=[-0.2, 0.7])))
        names = coordinate_names(backend.model.layout)
        data = (; y=Float64[])
        rebound = reprocess(backend, data)
        @test rebound.model === backend.model
        problem = rk_logdensity_problem(rebound;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        check_consumer_point(problem, fill(0.2, length(names)),
            u -> heterogeneous_oracle(kind, data, names, u))
    end
    @testset "row_alpha=$row_alpha" for row_alpha in (false, true)
        backend = RKBRMI(custom_mixture_builder(row_alpha)(
            (; y=[-0.2, 0.7], x=[0.1, 0.3])))
        names = coordinate_names(backend.model.layout)
        data = (; y=Float64[], x=Float64[])
        rebound = reprocess(backend, data)
        @test rebound.model === backend.model
        problem = rk_logdensity_problem(rebound;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        check_consumer_point(problem, fill(0.2, length(names)),
            u -> custom_mixture_oracle(data, names, u, row_alpha))
    end
end

@stestset "heterogeneous mixtures reject incompatible supports" begin
    for builder in (
        @brm(begin y ~ MixtureModel([Normal(0, 1), Exponential(1)], [0.5, 0.5]) end),
        @brm(begin y ~ MixtureModel([Normal(0, 1), Uniform(-1, 1)], [0.5, 0.5]) end),
        @brm(begin y ~ MixtureModel([Poisson(1), Binomial(3, 0.5)], [0.5, 0.5]) end),
        @brm(begin
            a ~ Normal(0, 1)
            location = exp(a)
            y ~ MixtureModel([Normal(location, 1), Exponential(1)], [0.5, 0.5])
        end),
        @brm(begin
            lo ~ Normal(0, 1)
            y ~ MixtureModel([Normal(0, 1), lower_shift(lo)], [0.5, 0.5])
        end),
        @brm(begin
            lo ~ Normal(0, 1)
            y ~ MixtureModel([lower_shift(lo), lower_shift(0)], [0.5, 0.5])
        end))
        brmi = builder((; y=[0.0, 1.0]))
        @test_throws "full-real" SBBRMI(brmi)
        @test_throws "full-real" BRM._brm_rk_plan(brmi)
    end
end
