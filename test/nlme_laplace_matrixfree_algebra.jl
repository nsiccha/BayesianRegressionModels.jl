using Test, BayesianRegressionModels, LinearAlgebra, SparseArrays, Random, Statistics
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

# A minimal algebra fixture, independent of model lowering and Enzyme. The
# 2x2 noncommuting example distinguishes the derivative of the fixed trace
# functional from the tempting but incorrect r' H^-1 H' r estimator.
function mf_factor(H, R; components=[collect(axes(H, 1))], large=trues(length(components)),
        options=MatrixFreeLaplace(; probes=size(R, 2)), wcols=Vector{Int}[], wλ=Float64[],
        references=nothing)
    component = zeros(Int, size(H, 1))
    for (c, I) in enumerate(components)
        component[I] .= c
    end
    p = (; pattern=sparse(H), components, component, wcols, wλ,
        algebra=(; options, large, references, probes=[R[I, :] for I in components], diagnostics=Dict{Symbol,Any}()))
    BRM._brm_operator_factor(p, sparse(H))
end
function mf_sensitivity(F)
    logdet(F)
    S = copy(F.problem.pattern)
    nonzeros(S) .= F.sensitivity
    Matrix(S)
end

@stestset "fixed preconditioner transforms the same determinant derivative and covariance" begin
    H(t) = [2.0 t; t 5.0]
    reference = BRM._brm_fixed_reference(only(mf_factor(H(0.3),ones(2,1)).parts).H,32)
    saved = deepcopy(reference)
    f(t) = logdet(mf_factor(H(t),ones(2,1);references=[reference]))
    F = mf_factor(H(0.5),ones(2,1);references=[reference])
    S = mf_sensitivity(F)
    @test 2S[1,2] ≈ (f(0.5+1e-5)-f(0.5-1e-5))/2e-5 atol=2e-7
    F = mf_factor(H(0.5),[1.0 1;1 -1];references=[reference])
    @test logdet(F) ≈ logdet(H(0.5)) atol=2e-7
    @test mf_sensitivity(F) ≈ inv(H(0.5)) atol=2e-7
    @test F \ [1.0,2.0] ≈ H(0.5) \ [1.0,2.0] atol=1e-10
    @test F \ zeros(2) == zeros(2)
    @test only(BRM._brm_operator_inverse_blocks(F,[[1,2]])) ≈ inv(H(0.5)) atol=1e-10
    # Dense independent oracle for C^-T A^-1/2 ξ, including the permutation.
    Cinv = Diagonal(reference.inverse_scale) * inv(Matrix(reference.lower)) *
        Matrix{Float64}(I,2,2)[reference.permutation,:]
    E = eigen(Symmetric(Cinv * H(0.5) * Cinv'))
    expected = Cinv' * (E.vectors * ((E.vectors' * randn(Xoshiro(41),2)) ./ sqrt.(E.values)))
    @test BRM._brm_operator_draw(F,Xoshiro(41)) ≈ expected atol=1e-9
    @test reference.lower == saved.lower && reference.permutation == saved.permutation
    @test_throws BRM._BRMOperatorSolveFailure logdet(mf_factor(H(0.5),ones(2,1);
        references=[reference],options=MatrixFreeLaplace(;probes=1,solve_maxiter=1)))
end
@stestset "matrix-free determinant differentiates its fixed probes" begin
    H(t) = [2.0 t; t 5.0]
    # Keep structural off-diagonal zeros in the sensitivity request.
    R = ones(2, 1)
    f(t) = logdet(mf_factor(sparse([1,2,1,2], [1,1,2,2],
        [2.0,t,t,5.0]), R))
    F = mf_factor(sparse([1,2,1,2], [1,1,2,2], [2.0,0.0,0.0,5.0]), R)
    S = mf_sensitivity(F)
    expected = 2log(2.5) / 3
    @test 2S[1,2] ≈ expected atol=2e-7
    @test (f(1e-5)-f(-1e-5))/2e-5 ≈ 2S[1,2] atol=2e-7
    @test abs(expected - 0.7) > 0.08
    @test logdet(F) == logdet(F)

    R = [1.0 1; 1 -1]
    for t in (0.0, 0.3, 1.2)
        F = mf_factor(sparse([1,2,1,2], [1,1,2,2], [2.0,t,t,5.0]), R)
        @test logdet(F) ≈ logdet(H(t)) atol=2e-7
        @test mf_sensitivity(F) ≈ inv(H(t)) atol=2e-7
        @test F \ [1.0, 2.0] ≈ H(t) \ [1.0, 2.0] atol=1e-10
    end
    F = mf_factor(Diagonal([0.01, 0.1, 10.0, 100.0]), ones(4, 1))
    @test logdet(F) ≈ 0 atol=2e-7
    @test diag(mf_sensitivity(F)) ≈ [100.0,10.0,0.1,0.01] rtol=2e-7
    mixed = mf_factor(Diagonal([2.0, 3.0, 4.0]), ones(3, 1);
        components=[[1], [2,3]], large=[false, true])
    @test logdet(mixed) ≈ log(24) atol=2e-7
    @test mixed.parts[1].factor !== nothing
    @test mixed.parts[2].factor === nothing
    @test_throws BRM._BRMOperatorSolveFailure logdet(mf_factor([1.0 2; 2 1], [1.0; -1.0;;]))
    @test_throws BRM._BRMOperatorSolveFailure logdet(mf_factor([2.0 0.3; 0.3 5.0], ones(2,1);
        options=MatrixFreeLaplace(; probes=1, solve_maxiter=1)))

    # ICAR soft constraints remain inside the operator even without sum cells.
    Hsoft = Matrix(Diagonal([2.0,3.0,4.0])) .+ 0.5
    F = mf_factor(Diagonal([2.0,3.0,4.0]), [1.0 1 1 1; 1 -1 1 -1; 1 1 -1 -1];
        wcols=[[1,2,3]], wλ=[0.5])
    @test F \ ones(3) ≈ Hsoft \ ones(3) atol=1e-10
    @test logdet(F) ≈ logdet(Hsoft) atol=2e-7
    E = eigen(Symmetric(Hsoft))
    expected_draw = E.vectors * ((E.vectors' * randn(Xoshiro(41),3)) ./ sqrt.(E.values))
    @test BRM._brm_operator_draw(F,Xoshiro(41)) ≈ expected_draw atol=1e-9
    @test only(BRM._brm_operator_inverse_blocks(F,[[1,3]])) ≈ inv(Hsoft)[[1,3],[1,3]] atol=1e-10
end
