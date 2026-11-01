# A non-concave conditional density has two genuine local maxima. Returning
# to a population point after a distant proposal must not select another one.
# RUN: julia --project=test test/nlme_laplace_history.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems

@testset "Laplace target is independent of proposal history" begin
    rk = RKBRMI(@brm (; g=["one"], y=[4.0]) begin
        mu ~ 1 + (1 | g)
        effect(mu, Intercept) ~ Normal(0, 10)
        sd(mu, g) ~ LogNormal(0, 1)
        @plate for i in eachindex(mu)
            squared = mu[i]^2
            y[i] ~ Normal(squared, 0.5)
        end
    end)
    for options in ((; structured=false), (; integrate=:g))
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, options...)
        @test length(p.population) == 2
        mean_index = findfirst(==(:mu_Intercept), p.names)
        point = zeros(2); point[mean_index] = 0.5
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, point)
        gradient = copy(gradient)
        modes = copy(p.u)
        @test isfinite(value) && all(isfinite, gradient)

        # Independent scalar Newton oracle for the positive conditional mode:
        # f(z, φ) = log N(z;0,1) + log N(4;(φ+z)^2,0.5).
        # In m = φ+z, its stationary equation is φ + 31m - 8m^3 = 0.
        m = 2.0
        for _ in 1:10
            m -= (0.5 + 31m - 8m^3) / (31 - 24m^2)
        end
        z = m - 0.5
        curvature = 24m^2 - 31
        expected = logpdf(Normal(), z) + logpdf(Normal(m^2, 0.5), 4.0) +
            logpdf(Normal(0, 10), 0.5) + logpdf(LogNormal(), 1.0) +
            log(2π / curvature) / 2
        @test value ≈ expected atol=1e-10

        for distant_mean in (-40.0, 40.0)
            workspace = deepcopy(p)
            proposal = zeros(2); proposal[mean_index] = distant_mean
            @test isfinite(LogDensityProblems.logdensity(workspace, proposal))
            returned, returned_gradient = LogDensityProblems.logdensity_and_gradient(workspace, point)
            @test returned == value
            @test returned_gradient == gradient
            @test workspace.u == modes
            @test LogDensityProblems.logdensity(workspace, point) == value
        end

        # The selected branch's derivative includes the moving curvature.
        h = 1e-5
        for j in eachindex(point)
            step = zeros(2); step[j] = h
            fd = (LogDensityProblems.logdensity(p, point + step) -
                  LogDensityProblems.logdensity(p, point - step)) / (2h)
            @test gradient[j] ≈ fd atol=1e-7 rtol=1e-7
        end
    end
end
