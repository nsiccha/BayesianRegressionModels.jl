# Consumer-side reconstruction through RKPPL's public sum-factor boundary.
# Dense references are confined to the eight-row fixture below.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, Random
using SparseArrays
using DifferentiationInterface: SecondOrder

function spl_sum_fixture(n; rows=[collect(1:n)])
    data = (; unit=["u$i" for i in 1:n], rows,
        y=[1.2length(r) + 0.1 for r in rows])
    RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        eta ~ 1 + (1 | unit)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(sum(exp.(eta[rows[k]])), sigma)
        end
    end)
end

@stestset "augmented curvature keeps singular and indefinite outer blocks" begin
    B = sparse([1.0 0.2 0.0; 0.0 0.7 1.0])
    for d in ([2.0, 3.0, 4.0], [-0.1, 3.0, 4.0]),
            w in ([1.0 0.2; 0.2 1.0], [1.0 0.0; 0.0 0.0], [1.0 0.0; 0.0 -0.2])
        D = spdiagm(0 => d); W = sparse(w)
        H = Matrix(D + B' * W * B)
        @test isposdef(H)
        K = BRM._brm_spl_augmented_matrix(D, B, W)
        f = BRM._brm_spl_augmented_factor(K, 3)
        @test f.positive_definite
        @test f.logdet ≈ logdet(Symmetric(H)) atol=1e-12
        b = [0.3, -0.2, 0.7]
        @test BRM._brm_spl_augmented_solve(f, b) ≈ H \ b atol=1e-12
        S = BRM._brm_spl_selected(f)
        exact = inv(Matrix(K))
        i, j, _ = findnz(K)
        @test all(isapprox(S[a, b], exact[a, b]; atol=1e-12) for (a, b) in zip(i, j))
    end
    # Zero outer curvature still keeps B's support, needed by the derivative
    # of the log determinant as that curvature moves through zero.
    D = spdiagm(0 => [2.0, 3.0, 4.0]); W = spzeros(2, 2)
    K = BRM._brm_spl_augmented_matrix(D, B, W)
    f = BRM._brm_spl_augmented_factor(K, 3)
    Z = BRM._brm_spl_selected(f)
    for (j, i, _) in zip(findnz(B)...)
        @test (Z[i, 3 + 2 + j] - Z[i, 3 + j]) / sqrt(2) ≈ (D \ B')[i, j]
    end
    # Both a zero local diagonal and cancelled auxiliary diagonals remain
    # explicit requests; a pivoting backend must see this symbolic pattern.
    K = BRM._brm_spl_augmented_matrix(sparse([0.0;;]), sparse([1.0;;]), sparse([2.0;;]))
    @test nnz(K) == 9
    @test K[1,1] == K[3,3] == 0
end

@stestset "augmented covariance draws have a bounded square-root error" begin
    rng = Xoshiro(85)
    for n in (3, 32, 128), condition in (1.0, 1e2, 1e4)
        Q = Matrix(qr(randn(rng, n, n)).Q)
        values = exp.(range(0, log(condition); length=n))
        S = Symmetric(Q * Diagonal(values) * Q')
        b = randn(rng, n)
        out = BRM._brm_spl_covariance_root(x -> S * x, b)
        expected = Q * (sqrt.(values) .* (Q' * b))
        @test norm(out.draw - expected) <= 1.01out.error_bound + 1e-11norm(expected)
        @test norm(out.draw - expected) <= 1e-9norm(expected)
    end
    @test iszero(BRM._brm_spl_covariance_root(identity, zeros(4)).draw)
    @test_throws BRM.BRMNLMEViewError BRM._brm_spl_covariance_root(
        x -> [x[1], 100x[2]], ones(2); maxdim=1)
    for n in (128, 512, 4096)
        d = [i == 1 ? -0.1 : 1 + 2rand(rng) for i in 1:n]
        B = sparse(fill(1, n), 1:n, [i == 1 ? 1.0 : 0.01 for i in 1:n], 1, n)
        D = spdiagm(0 => d)
        K = BRM._brm_spl_augmented_matrix(D, B, sparse([1.0;;]))
        factor = BRM._brm_spl_augmented_factor(K, n)
        @test factor.positive_definite
        b = randn(rng, n)
        out = BRM._brm_spl_covariance_root(x -> BRM._brm_spl_augmented_solve(factor, x), b)
        @test out.iterations < 25
        @test out.error_bound <= 1e-10norm(out.draw)
        @test nnz(factor.F.L) + nnz(factor.F.U) < 7n
        n <= 512 || continue
        expected = sqrt(Symmetric(inv(Matrix(D + B' * B)))) * b
        @test norm(out.draw - expected) <= 1e-9norm(expected)
    end
end

@stestset "sum-cell block steps retain only incident auxiliary rows" begin
    B = sparse([0.0 1.0 0 0 0 0; 0 0.4 0 1.0 0 0;
                0 0 0 0 1.0 0; 0 0 0 0 0 1.0])
    W = sparse([2.0 0.2 0.3 0; 0.2 3.0 0 0.1;
                0.3 0 2.0 0.2; 0 0.1 0.2 2.0])
    members = [4,2]; g = [0.4,-0.2]
    for d in (1.0,-0.1)
        D = spdiagm(0 => fill(d,6))
        h = BRM._BRMSumPrecision(D,B,W,Any[])
        H = Matrix(D + B' * W * B)[members,members]
        @test BRM._brm_spl_block_direction(nothing,h,members,g) ≈ H \ g
    end
    # Damping and an empty incidence set still solve the same local system.
    h = BRM._BRMSumPrecision(-sparse(I,6,6),spzeros(4,6),spzeros(4,4),Any[])
    @test BRM._brm_spl_block_direction(nothing,h,members,g) ≈ g ./ 9
    allocations = map((128,4096)) do n
        h = BRM._BRMSumPrecision(spdiagm(0=>ones(n)),
            sparse(1:n,1:n,ones(n),n,n),spdiagm(0=>fill(0.5,n)),Any[])
        direction() = BRM._brm_spl_block_direction(nothing,h,members,g)
        @test direction() ≈ g ./ 1.5
        minimum(@allocated(direction()) for _ in 1:3)
    end
    @test allocations[2] < 2allocations[1]
end

@stestset "independent augmented workspaces permit concurrent chains" begin
    tasks = map(1:4) do chain
        Threads.@spawn begin
            rng = Xoshiro(100 + chain)
            previous = nothing
            checks = Bool[]
            for iteration in 1:12
                D = spdiagm(0 => 1 .+ rand(rng, 20))
                B = sparse(0.1randn(rng, 3, 20))
                W = spdiagm(0 => [-0.1, 0.0, 0.5 + iteration / 20])
                H = Matrix(D + B' * W * B)
                K = BRM._brm_spl_augmented_matrix(D, B, W)
                f = BRM._brm_spl_augmented_factor(K, 20; previous)
                b = randn(rng, 20)
                selected = BRM._brm_spl_selected(f)
                exact = inv(Matrix(K))
                ii, jj, _ = findnz(K)
                push!(checks, f.positive_definite,
                    isapprox(f.logdet, logdet(Symmetric(H)); atol=1e-10),
                    isapprox(f \ b, H \ b; atol=1e-10),
                    all(isapprox(selected[i,j], exact[i,j]; atol=1e-10) for (i,j) in zip(ii,jj)))
                previous = f
            end
            checks
        end
    end
    for task in tasks, result in fetch(task)
        @test result
    end
end

@stestset "sum factors preserve ragged, repeated and empty supports" begin
    rows = [[1, 2, 2], [3, 4], Int[], [2, 5, 6]]
    rk = spl_sum_fixture(8; rows)
    n = length(BRM._rk_layout_coordinate_names(rk))
    u = 0.1randn(Xoshiro(57), n)
    prepared = BRM._brm_spl_prepare_factors(rk, :y, u, length(rows), 1:n, Int[])
    @test length(prepared.q.terms) == 1
    @test [length(only(c.terms)) for c in prepared.cells] == length.(rows)
    expected = BRM._rk_pointwise_values(BRM._rk_pointwise_query(rk), u).y
    for c in prepared.cells
        values = BRM._brm_spl_factor_values(prepared, u, c)
        @test values.value ≈ sum(expected[c.i])
        @test length(c.direct.sel) == 1
        @test all(length(t.sel) <= 3 for terms in c.terms for t in terms)
    end
    @test prepared.cells[1].terms[1][2].sel == prepared.cells[1].terms[1][3].sel
    @test isempty(only(prepared.cells[3].terms))

    # Full-model AD is an independent reference, avoiding the separate
    # ordinary whole-cell shared-predictor query route.
    ops = BRM._brm_spl_factor_operators(prepared, u, NLME_AD)
    plan = rk_bound_plan(rk)
    dirs = ntuple(k -> Float64.(collect(1:n) .== k), n)
    forward = AutoEnzyme(; mode=Enzyme.Forward)
    so = SecondOrder(NLME_AD, forward)
    h = prepare_query_ad(prepare_ad_hvp, rk.model, plan, :likelihood, so, dirs, u)
    direction = randn(Xoshiro(58), n)
    third = prepare_query_ad(prepare_ad_third, rk.model, plan, :likelihood,
        SecondOrder(so, forward), (direction,), u)
    gref, href = ad_gradient_and_hvp(h, dirs, u)
    tref = only(ad_third(third, (direction,), u))
    gradient = zeros(n); hessian = zeros(n, n); contraction = zeros(n)
    for c in ops.cells
        sums = BRM._brm_spl_factor_values(ops, u, c).sums
        gf, hf = BRM._rk_spl_factor_block(c.outer, u, c.direct.sel, c.i; sums)
        r = length(sums)
        J = zeros(length(gf), n)
        for (k, i) in enumerate(c.direct.sel)
            J[r + k, i] = 1.0
        end
        terms = Any[]
        for reduction in c.terms, t in reduction
            gt, ht = BRM._rk_spl_factor_block(t.ad, u, t.sel, c.i; term=t.t)
            J[t.j, t.sel] .+= gt
            push!(terms, (; t, ht))
            hessian[t.sel, t.sel] .+= gf[t.j] .* ht
        end
        gradient .+= J' * gf
        hessian .+= J' * hf * J
        d = J * direction
        _, _, tf = BRM._rk_spl_factor_third(c.outer, d, u, c.direct.sel, c.i; sums)
        contraction .+= J' * tf
        for (; t, ht) in terms
            v = direction[t.sel]
            _, _, tt = BRM._rk_spl_factor_third(t.ad, v, u, t.sel, c.i; term=t.t)
            contraction[t.sel] .+= 2(hf * d)[t.j] .* (ht * v) + gf[t.j] .* tt
            contraction .+= dot(v, ht * v) .* (J' * hf[:, t.j])
        end
    end
    @test gradient ≈ gref rtol=1e-10 atol=1e-10
    @test hessian ≈ hcat(href...) rtol=1e-10 atol=1e-10
    @test contraction ≈ tref rtol=1e-9 atol=1e-9
    @test ops.cells[1].terms[1][1].ad === ops.cells[4].terms[1][1].ad
    # Exercise the production contraction assembly with a non-diagonal
    # covariance, including all Jacobian-derivative and mixed terms.
    Q = randn(Xoshiro(61), n, n)
    S = Q * Q' + I
    L = cholesky(Symmetric(S)).L
    expected_tau = sum(ad_third(third, (Vector(L[:, k]),), u)[1] for k in 1:n)
    tau = zeros(n); hv = zeros(n)
    for c in ops.cells
        state = BRM._brm_spl_factor_blocks(ops, u, c)
        blocks = [[S[t.lz, t.lz] for t in terms] for terms in c.terms]
        SB = S[c.lz, c.lz] * state.B'
        BRM._brm_spl_factor_contraction!(tau, Float64[], ops, u, c, state,
            blocks, SB, state.B * SB)
        BRM._brm_spl_factor_hvp!(hv, Float64[], c, state, direction)
    end
    @test tau ≈ expected_tau rtol=1e-9 atol=1e-9
    @test hv ≈ hessian * direction rtol=1e-10 atol=1e-10

    assembled = BRM._brm_spl_factor_curvature(ops, u, n, 0)
    @test assembled.value ≈ sum(sum, expected)
    @test assembled.gradient ≈ gref rtol=1e-10 atol=1e-10
    @test Matrix(assembled.D + assembled.B' * assembled.W * assembled.B) ≈ -hessian
    @test assembled.rowptr == collect(1:2:9)

    # Each chain owns an independent preparation/selection map, while terms
    # of one chain still share the same compact operator by selection width.
    replica = deepcopy(ops)
    @test replica.cells[1].terms[1][1].ad !== ops.cells[1].terms[1][1].ad
    @test replica.cells[1].terms[1][1].ad === replica.cells[4].terms[1][1].ad
    u2 = u .+ 0.05
    moved = BRM._brm_spl_factor_curvature(replica, u2, n, 0)
    @test moved.gradient ≈ first(ad_gradient_and_hvp(h, dirs, u2)) rtol=1e-10 atol=1e-10
    original = BRM._brm_spl_factor_curvature(ops, u, n, 0)
    @test original.gradient ≈ assembled.gradient rtol=1e-10 atol=1e-10

    # Only the tiny reference is dense. Every production projection is read
    # from the augmented selected inverse, including zero Jacobian entries.
    precision = assembled.D + assembled.B' * assembled.W * assembled.B
    shift = 1 + opnorm(Matrix(precision))
    D = copy(assembled.D)
    for j in 1:n
        D[j,j] += shift
    end
    K = BRM._brm_spl_augmented_matrix(D, assembled.B, assembled.W)
    factor = BRM._brm_spl_augmented_factor(K, n)
    @test factor.positive_definite
    selected = BRM._brm_spl_selected(factor)
    covariance = inv(Symmetric(Matrix(precision + shift*I)))
    for (k,c) in enumerate(ops.cells)
        range = assembled.rowptr[k]:(assembled.rowptr[k+1]-1)
        projected = BRM._brm_spl_factor_covariances(c, range, selected, n, size(assembled.B,1))
        target = covariance[c.lz,:] * assembled.B[range,:]'
        @test projected.SB ≈ target
        @test projected.BSB ≈ assembled.B[range,:] * covariance * assembled.B[range,:]'
        for (reduction, blocks) in zip(c.terms, projected.blocks), (t, block) in zip(reduction, blocks)
            @test block ≈ covariance[t.lz,t.lz]
        end
    end

    # Mixed population/effect coordinates must use the global effect map,
    # not packed indices or one cell's local order, in both the sparse
    # precision and the third-derivative contraction.
    integrated = collect(2:2:n)
    population = setdiff(1:n, integrated)
    split = BRM._brm_spl_prepare_factors(rk, :y, u, length(rows), integrated, population)
    split = BRM._brm_spl_factor_operators(split, u, NLME_AD)
    a = BRM._brm_spl_factor_curvature(split, u, length(integrated), length(population))
    @test a.gradient ≈ gref[integrated]
    @test a.population ≈ gref[population]
    @test Matrix(a.D + a.B' * a.W * a.B) ≈ -hessian[integrated, integrated]
    sz = S[integrated, integrated]
    lz = cholesky(Symmetric(sz)).L
    directions = ntuple(length(integrated)) do k
        v = zeros(n); v[integrated] .= lz[:, k]; v
    end
    expected_split_tau = sum(ad_third(third, (d,), u)[1] for d in directions)
    tz = zeros(length(integrated)); tp = zeros(length(population))
    hz = zero(tz); hp = zero(tp)
    for (c, state) in zip(split.cells, a.states)
        blocks = [[sz[t.lz, t.lz] for t in reduction] for reduction in c.terms]
        SB = sz[c.lz, c.lz] * state.B'
        BRM._brm_spl_factor_contraction!(tz, tp, split, u, c, state,
            blocks, SB, state.B * SB)
        BRM._brm_spl_factor_hvp!(hz, hp, c, state, direction[integrated])
    end
    @test tz ≈ expected_split_tau[integrated] rtol=1e-9 atol=1e-9
    @test tp ≈ expected_split_tau[population] rtol=1e-9 atol=1e-9
    @test hz ≈ hessian[integrated, integrated] * direction[integrated]
    @test hp ≈ hessian[population, integrated] * direction[integrated]
end

@stestset "sum-factor selections stay compact as the total grows" begin
    widths = map((32, 4096)) do n
        rk = spl_sum_fixture(n)
        packed = length(BRM._rk_layout_coordinate_names(rk))
        u = zeros(packed)
        f = BRM._brm_spl_prepare_factors(rk, :y, u, 1, 1:packed, Int[])
        c = only(f.cells)
        @test length(only(c.terms)) == n
        @test BRM._brm_spl_factor_values(f, u, c).sums == [Float64(n)]
        (length(c.direct.sel), maximum(length(t.sel) for t in only(c.terms)))
    end
    @test widths[1] == widths[2] == (1, 3)
end

@stestset "assembled sum curvature retains sparse storage" begin
    for n in (32, 4096)
        rk = spl_sum_fixture(n)
        packed = length(BRM._rk_layout_coordinate_names(rk))
        u = zeros(packed)
        f = BRM._brm_spl_prepare_factors(rk, :y, u, 1, 1:packed, Int[])
        f = BRM._brm_spl_factor_operators(f, u, NLME_AD)
        a = BRM._brm_spl_factor_curvature(f, u, packed, 0)
        @test size(a.B) == (2, packed)
        @test nnz(a.B) <= 2packed
        @test nnz(a.D) <= 9packed
        @test nnz(a.W) == 4
        @test length(only(a.states).terms[1]) == n
        @test all(size(t.h,1) <= 3 for t in only(a.states).terms[1])
    end
end
