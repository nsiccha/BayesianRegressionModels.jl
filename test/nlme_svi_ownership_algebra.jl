using Test, BayesianRegressionModels
const _brm_svi_coordinate_owners = BayesianRegressionModels._brm_svi_coordinate_owners

@testset "structural SVI coordinate ownership" begin
    # 1 is a common scale; 2/3 are ordinary children; 4/5 are subject draws.
    footprint = [[1, 2, 4], [1, 3, 5]]
    subject = [[4], [5]]
    owned = [[j] for j in 1:5]
    support = [[1], [1, 2], [1, 3], [4], [5]]
    owners(; kwargs...) = _brm_svi_coordinate_owners(footprint, subject,
        owned, support, trues(5); kwargs...)
    @test owners() == [0, 1, 2, 1, 2]
    @test owners(global_coordinates=[2]) == [0, 0, 2, 1, 2]

    # Structural sharing, even with a numerically zero coefficient, stays
    # global. Atomic ownership closes the complete coupled block.
    @test _brm_svi_coordinate_owners([[2, 4], [2, 3, 5]], subject,
        owned, support, trues(5)) == [0, 0, 2, 1, 2]
    @test _brm_svi_coordinate_owners(footprint, subject,
        vcat(owned, [[2, 3]]), vcat(support, [[1, 2, 3]]), trues(5)) ==
        [0, 0, 0, 1, 2]

    # A parent can be local without an observation reading it directly;
    # ownership propagates transitively through authored prior reads.
    @test _brm_svi_coordinate_owners([[3], [5]], [[4], [5]],
        [[j] for j in 1:6], [[1], [1, 2], [2, 3], [4], [5], [6]],
        trues(6)) == [1, 1, 1, 1, 2, 0]
    # An unused child stays global and makes its otherwise-local parent
    # global, so no unit silently loses that child's prior density.
    @test _brm_svi_coordinate_owners([[2], [4]], [[3], [4]],
        [[j] for j in 1:4], [[1, 2], [2], [3], [4]], trues(4)) == [0, 0, 1, 2]

    # A derived factor with no owned coordinates still cannot cross units.
    @test _brm_svi_coordinate_owners(footprint, subject,
        vcat(owned, [Int[]]), vcat(support, [[2, 3]]), trues(5)) ==
        [0, 0, 0, 1, 2]
    @test _brm_svi_coordinate_owners([[1, 2], [1, 3]], [Int[], Int[]],
        [[1], [2], [3]], [[1], [1, 2], [1, 3]], BitVector([false, true, true])) ==
        [0, 1, 2]
end

@testset "empty Gaussian factors after promotion" begin
    brm = BayesianRegressionModels
    t = (; globals=Int[], units=[Int[]])
    q = brm._brm_svi_params(t, Float64[], MeanFieldGlobal();
        global_scale=1.0, unit_scale=1.0)
    @test brm._brm_svi_entropy(q) == 0.0
    prior = brm._BRMSVIStreamingPrior(brm._brm_svi_global_factor(q),
        brm._brm_svi_global_geometry(q), (; q.M, q.Λ, q.U), trues(1), nothing)
    gg, gz = Float64[], Float64[]
    @test brm._brm_svi_unit_prior!(gg, gz, prior, t, 1, Float64[], Float64[]) == 0.0
    @test brm._brm_svi_unit_step!(q, 1, gz, Float64[], Float64[], 0.1) === q
end
