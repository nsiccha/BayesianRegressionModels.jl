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

# A problem's view for the determinant, with a chosen probe matrix. `gmrf` turns the
# control variate on for a decoupled reference.
function probe_factor(case, probes; control_variate=false, h=case.h, reference=case.reference)
    N = case.N
    options = MatrixFreeLaplace(; probes=size(probes, 2), control_variate)
    gmrf = control_variate ? BRM._BRMSPLGMRF([case.field], (), nothing) : nothing
    p = (; pattern=case.h, components=[collect(1:N)], component=ones(Int, N), wcols=Vector{Int}[],
        wλ=Float64[], gmrf, algebra=(; options, large=trues(1), references=[reference],
            probes=[probes], diagnostics=Dict{Symbol,Any}()))
    BRM._brm_operator_factor(p, h)
end
exact_probe_factor(case; kwargs...) = probe_factor(case, sqrt(case.N) .* Matrix{Float64}(I, case.N, case.N); kwargs...)

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

@testset "The rebuilt reference's derivative cancels its own estimate" begin
    # With a complete orthogonal probe set V(θ) is log det H exactly for any M, so
    # the exact derivative of log det M and the estimate of it, computed from
    # different formulas, must cancel entry by entry.
    times = [0.0, 0.3, 1.7, 2.0]
    for channels in (1, 3)
        L = Matrix(cholesky(Symmetric([0.4^abs(i - j) for i in 1:channels, j in 1:channels])).L)
        reads = Dict((t, c) => [((1, 2, 5), [0.2, 0.3, 0.5], 40.0), ((2, 3, 5), [0.7, 0.1, 0.2], 3.0)]
            for t in eachindex(times), c in 1:channels)
        case = reference_case(MESH, times, 0.7, 1.3, L, reads; mode=:decoupled)
        F = exact_probe_factor(case; control_variate=true)
        @test logdet(F) ≈ logdet(Symmetric(Matrix(case.h))) atol=1e-6
        tracked = only(F.tracked)
        @test tracked isa BRM._BRMGMRFTracked
        # The rebuilt reference at the reference's own point is that reference.
        @test tracked.logdet ≈ case.reference.logdet atol=1e-8
        @test maximum(abs, tracked.block) < 1e-6
        @test abs(tracked.dspatial) < 1e-5
        @test abs(tracked.dtemporal) < 1e-5
        @test maximum(abs, tracked.dscale) < 1e-5
        x = randn(Xoshiro(5), case.N)
        physical, _ = reference_factors(case.A, case.N)
        @test tracked(x) ≈ inv(Symmetric(physical * physical')) * x atol=1e-7 rtol=1e-8
    end
end

@testset "Few probes: the control variate's derivative is that of its value" begin
    # V's arguments move one at a time, with the reference and probes fixed; each
    # exact-plus-estimated derivative must match a central difference of V.
    times = [0.0, 0.3, 1.7, 2.0]
    reads = Dict((t, 1) => [((1, 2, 5), [0.2, 0.3, 0.5], 40.0), ((2, 3, 5), [0.7, 0.1, 0.2], 3.0)]
        for t in eachindex(times))
    ell_s, ell_t = 0.7, 1.3
    fixed = reference_case(MESH, times, ell_s, ell_t, ones(1, 1), reads; mode=:decoupled)
    probes = 2.0 .* rand(Xoshiro(11), Bool, fixed.N, 3) .- 1
    value(s, t) = begin
        moved = reference_case(MESH, times, s, t, ones(1, 1), reads; mode=:decoupled)
        logdet(probe_factor(moved, probes; control_variate=true, reference=fixed.reference))
    end
    moved = reference_case(MESH, times, ell_s, ell_t, ones(1, 1), reads; mode=:decoupled)
    F = probe_factor(moved, probes; control_variate=true, reference=fixed.reference)
    logdet(F)
    tracked = only(F.tracked)
    # S_H at fixed data and M's own arguments: the prior moves in both.
    function entry_list(field)
        out = Tuple{Int,Int,Int,Int}[]
        BRM._brm_gmrf_foreach_entry(field) do k, i, j, ks, ta, tb, ca, cb
            push!(out, (k, ks, ta, tb))
        end
        out
    end
    field = moved.field
    share = BRM._brm_gmrf_tracked_data(tracked)
    function prior_only(k, ks, ta, tb)
        ta == tb ? share[ks, 1, ta] : 0.0
    end
    sensitivity_of(entries) = sum((F.sensitivity[field.nz[k]] - prior_only(k, ks, ta, tb)) * entries(ks, ta, tb)
        for (k, ks, ta, tb) in entry_list(field))
    spatial = sensitivity_of((ks, ta, tb) -> BRM._brm_gmrf_temporal_entry(field, ta, tb) * field.dqs[ks]) + tracked.dspatial
    temporal = sensitivity_of((ks, ta, tb) -> BRM._brm_gmrf_dtemporal_entry(field, ta, tb) * field.qs[ks]) + tracked.dtemporal
    h = 1e-5
    @test spatial ≈ (value(ell_s + h, ell_t) - value(ell_s - h, ell_t)) / 2h atol=2e-4 rtol=2e-4
    @test temporal ≈ (value(ell_s, ell_t + h) - value(ell_s, ell_t - h)) / 2h atol=2e-4 rtol=2e-4
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

# Two fields on one mesh, read by the same rows: h = [Q₁+D D; D Q₂+D].
function star_case(mesh, times, lengths1, lengths2, reads)
    a = reference_case(mesh, times, lengths1..., ones(1, 1), reads; mode=:decoupled)
    b = reference_case(mesh, times, lengths2..., ones(1, 1), reads; mode=:decoupled)
    n = a.N
    h = [a.prior+a.D a.D; a.D b.prior+b.D]
    fields = [a.field, b.field]
    fields[2].z = collect((n + 1):2n)
    H = BRM._BRMConditionalOperator(h, spzeros(0, 2n), spzeros(0, 0), Vector(diag(h)), 0.0,
        Float64[], Float64[])
    gmrf = BRM._BRMSPLGMRF(fields, (), nothing)
    BRM._brm_gmrf_pattern_positions!(gmrf, h)
    (; fields, h, H, N=2n, reference=BRM._brm_gmrf_star_reference(fields, H, h, collect(1:2n)))
end

@testset "A star over fields that shared rows couple is one congruence" begin
    times = [0.0, 0.3, 1.7, 2.0]
    reads = Dict((t, 1) => [((1, 2, 5), [0.2, 0.3, 0.5], 40.0), ((2, 3, 5), [0.7, 0.1, 0.2], 3.0)]
        for t in eachindex(times))
    case = star_case(MESH, times, (0.9, 2.0), (0.3, 0.5), reads)
    R = case.reference
    @test R isa BRM._BRMGMRFStarReference
    @test deepcopy(R) === R
    A = BRM._BRMPreconditionedOperator(case.H, R)
    physical, whitened = reference_factors(A, case.N)
    @test whitened' ≈ physical atol=1e-10
    M = inv(Symmetric(physical * physical'))
    @test R.logdet ≈ logdet(Symmetric(M)) atol=1e-7
    dense = Matrix(case.h)
    # Its blocks between the hub and the other field are H's own.
    hub = R.members[R.hub].state
    other = setdiff(1:case.N, hub)
    @test M[other, hub] ≈ dense[other, hub] atol=1e-7
    x = randn(Xoshiro(3), case.N)
    @test A(x) ≈ physical' * dense * physical * x atol=1e-7 rtol=1e-8
    @test BRM._brm_operator_solve(A, x, MatrixFreeLaplace()) ≈ dense \ x atol=1e-7 rtol=1e-7
    # A complete probe set recovers the determinant through the star.
    options = MatrixFreeLaplace(; probes=case.N, control_variate=false)
    p = (; pattern=case.h, components=[collect(1:case.N)], component=ones(Int, case.N),
        wcols=Vector{Int}[], wλ=Float64[], gmrf=nothing, algebra=(; options, large=trues(1),
            references=[R], probes=[sqrt(case.N) .* Matrix{Float64}(I, case.N, case.N)],
            diagnostics=Dict{Symbol,Any}()))
    @test logdet(BRM._brm_operator_factor(p, case.h)) ≈ logdet(Symmetric(dense)) atol=1e-6
end
