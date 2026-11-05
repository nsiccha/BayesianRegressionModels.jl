#!/usr/bin/env julia

# Run with --project=web-macro/app to exercise the actual app dependency set.
# No server, backend preparation, or fit is needed.
using BRMMacroWeb, BayesianRegressionModels, HTMXObjects, Test

@testset "macro app loads and renders reusable model views" begin
    builder = @brm begin
        sigma ~ Exponential(1)
        mu = x
        y ~ Normal(mu, sigma)
    end
    model = builder((; x=[-1., 0., 1.], y=[0.1, 0.2, 0.3]))
    html = repr("text/html", BRMMacroWeb.brmi_card(model))
    @test occursin("Roles and ranks", html)
    @test occursin("Parameter", html)
    @test occursin("Data · response", html)
    @test occursin("Scalar", html)
    response = HTMXObjects.dispatch(:GET, "/")
    @test response.status == 200
    @test occursin("BRM", String(response.body))


end
println("MODEL_VIEW_APP_ACCEPTANCE_COMPLETE")
