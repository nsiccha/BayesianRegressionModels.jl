# Run in the pinned NLMEEstimation environment described in test/README.md.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_connected_fixture.jl"))
using NLMEEstimation, LinearAlgebra, Random
const NE = NLMEEstimation
const NEEXT = Base.get_extension(BRM, :BayesianRegressionModelsNLMEEstimationExt)

@stestset "connected estimates preserve scientific labels and repeated covariance families" begin
    data = connected_data(5, [1, 2, 4, 5, 1], [2, 3, 5, 5, 1])
    rk = RKBRMI(connected_model(data))
    m = brm_nlme_model(rk; ad_backend=NLME_AD, unit_partition=:connected)
    start = brm_nlme_start(m)
    @test NE.nsubjects(m) == 2
    @test m.view.subject_families == [[1, 1, 1], [1, 1]]
    @test length(start.omega_matrix) == 1
    @test start.subjects == [("subject 1", "subject 2", "subject 3"), ("subject 4", "subject 5")]
    @test start.subject_levels == data.subject
    @test NE.check_protocol(m, start.theta, start.sigma, start.omega)
    @test NE.mu_referencing(m) === nothing

    point = m.unit .+ 0.15 .* randn(Xoshiro(92), length(m.unit))
    from_point = brm_nlme_start(m, point)
    @test from_point.max_error < 1e-10
    @test NEEXT._start_eta(m, from_point, [2]) == from_point.eta[[2]]
    original = BRM._brm_nlme_original_effects(m.view, [1, 2], from_point.eta)
    shuffled = [5, 2, 4, 1, 3]
    named = merge(from_point, (; subjects=data.subject[shuffled], eta=original.values[shuffled]))
    @test NEEXT._start_eta(m, named, nothing) == from_point.eta
    @test_throws r"missing subject" NEEXT._start_eta(m,
        merge(named, (; subjects=named.subjects[2:end], eta=named.eta[2:end])), nothing)

    fit = brm_nlme_fit(m, NE.empirical_bayes; start)
    A = zeros(length(data.y), length(data.subject))
    for row in eachindex(data.y)
        A[row, data.left[row]] += 1
        A[row, data.right[row]] += data.weight[row]
    end
    variance = only(only(start.omega_matrix))
    residual = data.y - (1 .+ data.weight) .* only(start.theta)
    oracle = (A' * A / 0.5^2 + I / variance) \ (A' * residual / 0.5^2)
    @test fit.estimates.eta.subjects == data.subject
    @test only.(fit.estimates.eta.values) ≈ oracle atol=1e-6
    @test all(names -> all(name -> name isa String, names), fit.estimates.eta.names)
    subset = brm_nlme_fit(m, NE.empirical_bayes; start, subjects=[2])
    @test subset.estimates.eta.subjects == data.subject[4:5]
    @test only.(subset.estimates.eta.values) ≈ oracle[4:5] atol=1e-6
    @test_throws ArgumentError brm_nlme_fit(m, NE.focei; start)

    # The public convenience route forwards the explicit preparation policy.
    short = brm_nlme_fit(rk, NE.saem; unit_partition=:connected,
        rng=Xoshiro(93), options=NE.SAEMOptions(iterations=3, burnin=1), ebes=false)
    @test short.result.iterations == 3
    @test short.model.view.subject_families == m.view.subject_families
    @test all(isfinite, short.result.theta)
end
