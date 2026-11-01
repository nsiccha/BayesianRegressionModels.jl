# Stable elementwise norms through the public formula surface and both backends.
# Run: julia --project=test test/hypot.jl [testset substring ...]
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module HypotFormulaSecurity
    include(joinpath(@__DIR__, "..", "web-macro", "src", "security.jl"))
end

@stestset "hypot formula spelling" begin
    @test isnothing(HypotFormulaSecurity.walk(:(scale = hypot(sigma, extra))))
    dotted = @brm begin
        sigma ~ Exponential(1)
        scale = hypot.(sigma, extra)
        y ~ Normal(0, scale)
    end
    @test_throws "A formula call already applies per row" dotted((; y=[0.1], extra=[0.2]))
end

@stestset "hypot assignment scalar and vector inputs" begin
    for (label, gain, extra) in (("scalar-vector", 1.0, [0.1, 0.2]),
            ("vector-scalar", [0.7, 1.3], 0.2),
            ("vector-vector", [0.7, 1.3], [0.1, 0.2]),
            ("scalar-scalar", 1.0, 0.2))
        @testset "$label" begin
            data = (; y=[0.2, 0.3], gain, extra)
            before = deepcopy(data)
            builder = @brm begin
                log_sigma ~ Normal(0, 1)
                sigma = exp(log_sigma) * gain
                scale = hypot(sigma, extra)
                y ~ Normal(0, scale)
            end
            model = builder(data)
            backend, problem = consumer_problem(model)
            stan = consumer_stan(model, "hypot-$label")
            oracle(u) = logpdf(Normal(), u[1]) +
                sum(logpdf.(Normal.(0, hypot.(exp(u[1]) .* gain, extra)), data.y))
            for log_sigma in (-2.0, 0.0, 1.2, 360.0, 430.0)
                u = [log_sigma]
                value, gradient = check_consumer_point(problem, u, oracle)
                @test isfinite(value)
                @test all(isfinite, gradient)
                check_consumer_stan(problem, stan, [:log_sigma => "log_sigma"], backend, u)
            end
            # Same declaration on a changed row count; no captured training extent.
            replay_data = (; y=[0.1, -0.2, 0.3],
                gain=gain isa AbstractVector ? [1.0, 1.2, 0.9] : 1.2,
                extra=extra isa AbstractVector ? [0.2, 0.3, 0.4] : 0.3)
            replay = reprocess(backend, replay_data)
            replay_problem = rk_logdensity_problem(replay; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            check_consumer_point(replay_problem, [0.2], u -> logpdf(Normal(), u[1]) +
                sum(logpdf.(Normal.(0, hypot.(replay_data.gain .* exp(u[1]),
                    replay_data.extra)), replay_data.y)))
            @test isequal(data, before)
        end
    end
end

@stestset "hypot square-root control and extreme magnitude" begin
    data = (; y=[0.2, 0.3], extra=[0.1, 0.2])
    control = @brm data begin
        log_sigma ~ Normal(0, 1)
        sigma = exp(log_sigma)
        scale = sqrt(sigma^2 + extra^2)
        y ~ Normal(0, scale)
    end
    stable = @brm data begin
        log_sigma ~ Normal(0, 1)
        sigma = exp(log_sigma)
        scale = hypot(sigma, extra)
        y ~ Normal(0, scale)
    end
    _, control_problem = consumer_problem(control)
    _, stable_problem = consumer_problem(stable)
    @test LogDensityProblems.logdensity(control_problem, [0.0]) ≈ -2.844472332584493
    @test LogDensityProblems.logdensity(stable_problem, [0.0]) ≈
        LogDensityProblems.logdensity(control_problem, [0.0])
    @test !isfinite(LogDensityProblems.logdensity(control_problem, [360.0]))
    @test isfinite(LogDensityProblems.logdensity(stable_problem, [360.0]))
    inline = @brm data begin
        log_sigma ~ Normal(0, 1)
        y ~ Normal(0, hypot(exp(log_sigma), extra))
    end
    _, inline_problem = consumer_problem(inline)
    for u in ([0.0], [360.0])
        @test LogDensityProblems.logdensity(inline_problem, u) ≈
            LogDensityProblems.logdensity(stable_problem, u)
    end
    # A fitted data-only assignment follows the same elementwise contract.
    data_only = @brm (; x=[1e200, 1e-200], z=[2e200, 2e-200], y=[0.0, 0.0]) begin
        scale = hypot(x, z)
        y ~ Normal(0, scale)
    end
    _, data_problem = consumer_problem(data_only)
    @test isfinite(LogDensityProblems.logdensity(data_problem, Float64[]))
    data_stan = consumer_stan(data_only, "hypot-data-extremes")
    @test LogDensityProblems.logdensity(data_problem, Float64[]) ≈
        BridgeStan.log_density(data_stan.model, Float64[]; propto=false, jacobian=true)
end

@stestset "hypot predictor terms and model offsets" begin
    data = (; x=[-0.4, 0.2, 0.7], z=[0.2, -0.3, 0.5], y=[0.1, -0.2, 0.3])
    for (label, builder) in (("data-term", @brm(begin
                mu ~ 1 + hypot(x, z)
                y ~ Normal(mu, 1)
            end)), ("offset", @brm(begin
                a ~ Normal(0, 1)
                mu ~ 1 + offset(hypot(a, x))
                y ~ Normal(mu, 1)
            end)))
        @testset "$label" begin
            model = builder(data)
            backend, problem = consumer_problem(model)
            stan = consumer_stan(model, "hypot-$label")
            mapping = brm_coordinate_transport(backend, SBBRMI(model),
                String.(BridgeStan.param_unc_names(stan.model)))
            for u in (zeros(LogDensityProblems.dimension(problem)),
                    fill(0.2, LogDensityProblems.dimension(problem)))
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                @test isfinite(value)
                @test all(isfinite, gradient)
                stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model,
                    brm_rk_point_to_stan(mapping, u); propto=false, jacobian=true)
                @test value ≈ brm_stan_logdensity_to_rk(mapping, u, stan_value)
                @test gradient ≈ brm_stan_gradient_to_rk(mapping, u, stan_gradient)
            end
        end
    end
end
