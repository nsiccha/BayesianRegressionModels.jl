using Test, BayesianRegressionModels, LinearAlgebra, SparseArrays, Random
const BRM = BayesianRegressionModels

# One field's prepared state and precision at a point, without model lowering:
# `reads` maps (time, channel) to a list of (three vertices, three weights, rows).
function reference_case(mesh, times, ell_s, ell_t, L, reads; mode=:auto)
    nv, nc, nt = length(mesh.mass), size(L, 1), length(times)
    N = nv * nc * nt
    parts = BRM._brm_gmrf_spatial_parts(mesh)
    ns = length(nonzeros(parts.pattern))
    field = BRM._BRMSPLGMRFField(1, collect(1:N), (nv, nc, nt), parts, mesh, collect(Float64, times),
        nc > 1, ell_s, ell_t, L, ones(Int, N), Int[], zeros(ns), zeros(ns), zeros(nt),
        zeros(max(nt - 1, 0)), zeros(nt), zeros(max(nt - 1, 0)), Matrix(L),
        inv(Symmetric(L * L')), (ell_s, ell_t))
    BRM._brm_gmrf_spatial_values!(field.qs, field.dqs, parts, ell_s)
    BRM._brm_gmrf_temporal_values!(field.qt, field.qo, field.dqt, field.dqo, field.times, ell_t)
    gmrf = BRM._BRMSPLGMRF([field], (), nothing)
    state(i, c, t) = i + nv * ((c - 1) + nc * (t - 1))
    D = spzeros(N, N)
    for ((t, c), entries) in reads, (vertices, weights, rows) in entries
        I_ = [state(v, c, t) for v in vertices]
        D[I_, I_] += rows * weights * weights'
    end
    pairs = BRM._brm_gmrf_pattern_pairs(gmrf)
    full = sparse(vcat(first.(pairs), last.(pairs), 1:N), vcat(last.(pairs), first.(pairs), 1:N),
        ones(2length(pairs) + N), N, N) + abs.(D)
    h = SparseMatrixCSC(N, N, full.colptr, full.rowval, zeros(nnz(full)))
    BRM._brm_gmrf_pattern_positions!(gmrf, h)
    BRM._brm_spl_gmrf_precision!(nonzeros(h), (; gmrf))
    prior = copy(h)
    for j in 1:N, k in nzrange(D, j)
        h[rowvals(D)[k], j] += nonzeros(D)[k]
    end
    H = BRM._BRMConditionalOperator(h, spzeros(0, N), spzeros(0, 0), Vector(diag(h)), 0.0, Float64[], Float64[])
    reference = BRM._brm_gmrf_time_reference(field, H, h, collect(1:N); mode)
    (; field, h, prior, D, H, reference, N, A=BRM._BRMPreconditionedOperator(H, reference))
end

function reference_factors(A, N)
    basis = Matrix{Float64}(I, N, N)
    out = zeros(N)
    physical = hcat([copy(BRM._brm_to_physical!(out, A, x)) for x in eachcol(basis)]...)
    whitened = hcat([copy(BRM._brm_from_physical!(out, A, x)) for x in eachcol(basis)]...)
    physical, whitened
end

function exact_probe_factor(case; solve_maxiter=512)
    N = case.N
    options = MatrixFreeLaplace(; probes=N, solve_maxiter)
    p = (; pattern=case.h, components=[collect(1:N)], component=ones(Int, N), wcols=Vector{Int}[],
        wλ=Float64[], algebra=(; options, large=trues(1), references=[case.reference],
            probes=[sqrt(N) .* Matrix{Float64}(I, N, N)], diagnostics=Dict{Symbol,Any}()))
    BRM._brm_operator_factor(p, case.h)
end

const MESH = BRM._brm_gmrf_mesh([0.0 0; 1 0; 1 1; 0 1; 0.5 0.5], [1 2 5; 2 3 5; 3 4 5; 4 1 5])

@testset "The gmrf time reference is one congruence, with the determinant it reports" begin
    for mode in (:decoupled, :sequential, :auto), channels in (1, 3), times in ([0.0], [0.0, 0.3, 1.7, 2.0])
        L = Matrix(cholesky(Symmetric([0.4^abs(i - j) for i in 1:channels, j in 1:channels])).L)
        reads = Dict((length(times), channels) => [((1, 2, 5), [0.2, 0.3, 0.5], 4.0)],
            (1, 1) => [((2, 3, 5), [0.7, 0.1, 0.2], 9.0), ((3, 4, 5), [0.1, 0.1, 0.8], 2.0)])
        case = reference_case(MESH, times, 0.7, 1.3, L, reads; mode)
        saved = deepcopy(case.reference.lower)
        physical, whitened = reference_factors(case.A, case.N)
        @test whitened' ≈ physical atol=1e-10
        M = inv(Symmetric(physical * physical'))
        @test case.reference.logdet ≈ logdet(Symmetric(M)) atol=1e-8
        rng = Xoshiro(4112026)
        dense = Matrix(case.h)
        for _ in 1:3
            x, y = randn(rng, case.N), randn(rng, case.N)
            @test case.A(x) ≈ physical' * dense * physical * x atol=1e-8 rtol=1e-9
            @test dot(x, case.A(y)) ≈ dot(case.A(x), y) atol=1e-8
            solved = BRM._brm_operator_solve(case.A, x, MatrixFreeLaplace())
            @test solved ≈ dense \ x atol=1e-8 rtol=1e-8
        end
        # The Lanczos determinant with a complete orthogonal probe set is exact in
        # value and in its derivative, on the reference's coordinates.
        F = exact_probe_factor(case)
        @test logdet(F) ≈ logdet(Symmetric(dense)) atol=1e-6
        S = copy(case.h); nonzeros(S) .= F.sensitivity
        inverse = inv(Symmetric(dense))
        @test maximum(abs, (Matrix(S) .- inverse)[Matrix(case.h) .!= 0]) < 1e-7
        @test case.reference.lower == saved
    end
end

@testset "The prior alone is the reference exactly, and data keep the spectrum above one" begin
    times = [0.0, 0.3, 1.7, 2.0]
    L = Matrix(cholesky(Symmetric([1.0 0.6; 0.6 1.0])).L)
    for mode in (:decoupled, :sequential)
        prior = reference_case(MESH, times, 0.7, 1.3, L, Dict(); mode)
        physical, _ = reference_factors(prior.A, prior.N)
        @test physical' * Matrix(prior.h) * physical ≈ I atol=1e-9
        @test BRM._brm_lanczos(prior.A, ones(prior.N), MatrixFreeLaplace()).k == 1
        @test prior.reference.logdet ≈ logdet(Symmetric(Matrix(prior.h))) atol=1e-8
    end

    reads = Dict((t, 1) => [((1, 2, 5), [0.2, 0.3, 0.5], 50.0), ((3, 4, 5), [0.5, 0.25, 0.25], 3.0)] for t in 1:4)
    case = reference_case(MESH, times, 0.7, 1.3, ones(1, 1), reads)
    physical, _ = reference_factors(case.A, case.N)
    @test minimum(eigvals(Symmetric(physical' * Matrix(case.h) * physical))) > 1 - 1e-9
    # Blocks between neighbouring times are H's own.
    M = inv(Symmetric(physical * physical'))
    nv = length(MESH.mass)
    for t in 2:length(times)
        a, b = (t - 2) * nv .+ (1:nv), (t - 1) * nv .+ (1:nv)
        @test M[a, b] ≈ Matrix(case.h)[a, b] atol=1e-8
    end
end

@testset "The channels' combination follows where they are read" begin
    times = [0.0, 0.3, 1.7, 2.0]
    L = Matrix(cholesky(Symmetric([0.8^abs(i - j) for i in 1:3, j in 1:3])).L)
    site = ((1, 2, 5), [0.2, 0.3, 0.5], 400.0)
    same = Dict((t, c) => [site] for t in 1:4, c in 1:3)
    case = reference_case(MESH, times, 0.7, 1.3, L, same)
    @test case.reference.sequential === nothing
    # Decoupling is exact across channels here: only the times remain.
    @test BRM._brm_lanczos(case.A, ones(case.N), MatrixFreeLaplace()).k < 40
    # Uncorrelated channels need no choice, and are independent chains.
    independent = reference_case(MESH, times, 0.7, 1.3, Matrix(1.0I, 3, 3), same)
    @test independent.reference.sequential === nothing
    @test independent.reference.basis == I
end

@testset "A reference survives curvature that is not positive semidefinite" begin
    times = [0.0, 1.0]
    # Negative curvature large enough that a spatial factor needs its shift.
    reads = Dict((1, 1) => [((1, 2, 5), [0.5, 0.25, 0.25], -5000.0)])
    case = reference_case(MESH, times, 0.7, 1.3, ones(1, 1), reads)
    @test case.reference.diagonal_shift > 0
    physical, _ = reference_factors(case.A, case.N)
    @test case.reference.logdet ≈ -logdet(Symmetric(physical * physical')) atol=1e-7
end

@testset "A prepared reference is shared, read-only, by copies" begin
    case = reference_case(MESH, [0.0, 1.0], 0.7, 1.3, ones(1, 1),
        Dict((1, 1) => [((1, 2, 5), [0.2, 0.3, 0.5], 4.0)]))
    @test deepcopy(case.reference) === case.reference
end
