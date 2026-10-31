# Strict unit partitions: structural ownership, including dependencies hidden
# at the attribution point, and units with an empty observation group.
# Run: julia --project=test test/nlme_partition.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_panel_fixture.jl"))
using Random

@stestset "a dormant cross-unit branch is refused before returning a model" begin
    data = (; subject=["a", "b", "c"], other=[2, 3, 1], y=[0.1, 0.2, 0.3])
    rk = RKBRMI(@brm data begin
        mu ~ 1 + (1 | subject)
        all_mu = mu
        @plate for i in eachindex(y)
            y[i] ~ Normal(ifelse(all_mu[other[i]] > 10, all_mu[other[i]], mu[i]), 1)
        end
    end)
    @test Base.invokelatest(likelihood_query(rk), zeros(rk.model.layout.total)) ≈
        sum(logpdf.(Normal(), data.y))
    # The branch is dormant at the unit point and the ordinary attribution
    # perturbations. Its structural footprint still includes the other unit.
    error = try
        brm_nlme_model(rk; ad_backend=NLME_AD)
        nothing
    catch e
        e
    end
    @test error isa BRMNLMEViewError
    @test occursin("another subject", sprint(showerror, error))
    @test occursin("b_subject", sprint(showerror, error))
end

@stestset "a unit can have no cells of one observation group" begin
    data = panel_data()
    x1 = copy(data.x1)
    x1[data.region .== 1] .= missing
    rk = RKBRMI(panel(merge(data, (; x1))); completions=:noncentered)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:region,
        nested_groups=PANEL_NESTED)
    observed = only(filter(c -> c.names == (:x1_obs,), m.cell_query))
    @test isempty(observed.positions[1])
    rng = Xoshiro(42)
    θ = 0.1 .* randn(rng, length(m.theta))
    σ = 0.1 .* randn(rng, length(m.sigma))
    H = [0.2 .* randn(rng, length(b)) for b in m.view.subject_coordinates]
    expected, gradients = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    for i in eachindex(H)
        value, gradient = brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, H[i])
        @test value ≈ expected[i] atol=1e-9 rtol=1e-10
        @test gradient ≈ gradients[i] atol=1e-8 rtol=1e-9
        @test brm_nlme_loglikelihood(m, i, θ, σ, H[i]) ≈ expected[i] atol=1e-9 rtol=1e-10
    end
end
