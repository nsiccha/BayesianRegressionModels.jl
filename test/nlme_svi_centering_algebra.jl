# Learned SVI centering (src/nlme_svi_centering.jl): the cell map u → y, its
# inverse and log-Jacobian, and the reverse sweep's gradients in u, the controls,
# the natural factors, the vector laws' means and scales and the location
# globals, against central finite differences of the composed target
# F(g, y(u)) + log|∂y/∂u| on a smooth test F. No model and no RK: this pins the
# algebra only.
#
# RUN: julia --project=test test/nlme_svi_centering_algebra.jl
using Test, LinearAlgebra, Random, BayesianRegressionModels
const BRM = BayesianRegressionModels
using ReparametrizableDistributions: PartiallyCentered, reparam

# A smooth non-Gaussian test target over the unit block y and the globals g.
F(g, y) = -sum(abs2, y .- 0.3) / 2 + sum(sin, y) + 0.2 * sum(y)^2 * g[1] - sum(abs2, g) / 4
F_y(g, y) = -(y .- 0.3) .+ cos.(y) .+ 0.4 * g[1] * sum(y)
F_g(g, y) = (v = -g ./ 2; v[1] += 0.2 * sum(y)^2; v)

# One unit with three instances: a correlated K = 3 block (two margins located
# on globals, one also through a fixed packed value), a scalar block, and two
# authored vector elements (a per-element mean, a shared scale); position 3 is
# no cell.
function fixture()
    L3 = cholesky(Symmetric([1.0 0.4 -0.2; 0.4 1.0 0.3; -0.2 0.3 1.0])).L
    frame = (; A=[Diagonal([0.7, 1.9, 0.4]) * Matrix(L3), fill(2.3, 1, 1)],
        values=([0.4, -1.1, 0.25], 0.6))
    inst = [BRM._BRMSVICenteringBlock(1, [2, 4, 5], 1:3,
                [[(1, 1.0)], [(2, 0.5), (-1, 2.0)], Tuple{Int,Float64}[]]),
            BRM._BRMSVICenteringBlock(2, [1], 4:4, [[(2, -1.5)]]),
            BRM._BRMSVICenteringVector(1, 2, [6, 7], [2, 3], 5:6)]
    (; frame, inst, base=[0.8], d=7, ncells=6)
end

forward(f, g, u, c, frame) = begin
    y = similar(u)
    lj, μ, ℓ = BRM._brm_svi_unit_centering!(y, u, f.inst, f.ncells, c, frame, g, f.base)
    y, μ, ℓ, lj
end
target(f, g, u, c, frame) = ((y, _, _, lj) = forward(f, g, u, c, frame); F(g, y) + lj)

function fd(h, x; ε=1e-6)
    map(eachindex(x)) do j
        xp, xm = copy(x), copy(x)
        xp[j] += ε; xm[j] -= ε
        (h(xp) - h(xm)) / (2ε)
    end
end

@testset "learned centering: map, inverse and log-Jacobian" begin
    rng = Xoshiro(7)
    f = fixture()
    A, (mean, scale) = f.frame.A, f.frame.values
    for c in (zeros(6), ones(6), rand(rng, 6))
        g, u = randn(rng, 2), randn(rng, f.d)
        y, μ, ℓ, lj = forward(f, g, u, c, f.frame)
        @test y[3] == u[3]
        # c = 0 keeps a draw's packed coordinate and standardizes a vector element.
        if all(iszero, c)
            @test y[1:5] == u[1:5]
            @test y[6:7] ≈ mean[2:3] .+ scale .* u[6:7] rtol = 1e-12
        end
        # c = 1 makes u the physical value: b = μ0 + A z, or the element itself.
        if all(isone, c)
            b = A[1] * y[[2, 4, 5]] .+ [g[1], 0.5g[2] + 2.0 * f.base[1], 0.0]
            @test u[[2, 4, 5]] ≈ b rtol = 1e-12
            @test u[1] ≈ A[2][1, 1] * y[1] - 1.5g[2] rtol = 1e-12
            @test y[6:7] ≈ u[6:7] rtol = 1e-12
        end
        # Each cell's location and log scale are its conditional ones.
        @test ℓ ≈ log.([A[1][1, 1], A[1][2, 2], A[1][3, 3], A[2][1, 1], scale, scale])
        @test μ[3] ≈ A[1][3, 1] * y[2] + A[1][3, 2] * y[4]
        @test μ[5:6] == mean[2:3]
        back, μi, ℓi = copy(y), zeros(6), zeros(6)
        ljback = sum(inst -> BRM._brm_svi_centering_inverse!(back, μi, ℓi, y, inst, c,
            f.frame, g, f.base), f.inst)
        @test back ≈ u rtol = 1e-12
        @test μi ≈ μ rtol = 1e-12
        @test ℓi == ℓ
        # The log-Jacobian is that of the triangular map.
        J = reduce(hcat, fd(x -> forward(f, g, x, c, f.frame)[1], u))'
        @test lj ≈ logabsdet(J)[1] atol = 1e-7
        @test lj ≈ -sum(c[1:4] .* ℓ[1:4]) + sum((1 .- c[5:6]) .* ℓ[5:6]) rtol = 1e-12
        @test ljback ≈ -lj rtol = 1e-12 atol = 1e-14
    end
end

@testset "learned centering: reverse sweep against finite differences" begin
    rng = Xoshiro(11)
    f = fixture()
    for trial in 1:6
        g, u, c = randn(rng, 2), randn(rng, f.d), rand(rng, 6)
        trial == 1 && fill!(c, 0.0)
        trial == 2 && fill!(c, 1.0)
        y, μ, ℓ, _ = forward(f, g, u, c, f.frame)
        gy, gg, gc = F_y(g, y), F_g(g, y), zeros(6)
        adjoint = BRM._BRMSVICenteringAdjoint(f.frame)
        BRM._brm_svi_unit_centering_backward!(gy, gc, adjoint, y, μ, ℓ, f.inst, c, f.frame)
        for (k, v) in adjoint.globals
            gg[k] += v
        end
        @test gy ≈ fd(x -> target(f, g, x, c, f.frame), u) rtol = 1e-6 atol = 1e-8
        @test gc ≈ fd(x -> target(f, g, u, x, f.frame), c) rtol = 1e-6 atol = 1e-8
        @test gg ≈ fd(x -> target(f, x, u, c, f.frame), g) rtol = 1e-6 atol = 1e-8
        for b in 1:2, j in eachindex(f.frame.A[b])
            r, s = Tuple(CartesianIndices(f.frame.A[b])[j])
            r >= s || continue
            h = function (x)
                A = deepcopy(f.frame.A)
                A[b][r, s] = x
                target(f, g, u, c, (; A, f.frame.values))
            end
            ε = 1e-6
            @test adjoint.A[b][r, s] ≈ (h(f.frame.A[b][r, s] + ε) - h(f.frame.A[b][r, s] - ε)) /
                (2ε) rtol = 1e-6 atol = 1e-8
        end
        # The vector ports' adjoints, summed by (port, element).
        port = zeros(4)   # mean[1:3], scale
        for (p, j, v) in adjoint.ports
            port[p == 1 ? j : 4] += v
        end
        mean0, scale0 = f.frame.values
        @test port[1:3] ≈ fd(x -> target(f, g, u, c, (; f.frame.A, values=(x, scale0))), mean0) rtol = 1e-6 atol = 1e-8
        @test port[4] ≈ only(fd(x -> target(f, g, u, c, (; f.frame.A, values=(mean0, x[1]))),
            [scale0])) rtol = 1e-6 atol = 1e-8
    end
end

@testset "learned centering: the shared primitive's scalar map" begin
    # The map is ReparametrizableDistributions' `reparam`, not a BRM copy.
    for (u, loc, ℓ, c, c0) in ((0.3, -1.2, 0.7, 0.25, 0.0), (-2.0, 0.4, -1.1, 0.9, 1.0))
        lj, y = reparam(PartiallyCentered(c0), PartiallyCentered(c), u, loc, ℓ)
        @test (lj, y) == BRM._brm_svi_cell(c0, c, u, loc, ℓ)
        @test y ≈ c0 * loc + (u - c * loc) * exp((c0 - c) * ℓ)
        @test lj ≈ (c0 - c) * ℓ
    end
    @test_throws ArgumentError LearnedCentering(; start=0.0)
    @test_throws ArgumentError LearnedCentering(; start=1.0)
    @test LearnedCentering().start == 0.5
    @test BRM._brm_svi_control(BRM._brm_svi_control_logit(0.3)) ≈ 0.3
end

@testset "learned centering: affine factor transport and conditional checkpoint" begin
    rng, f = Xoshiro(32), fixture()
    g, c = randn(rng, 2), rand(rng, f.ncells)
    maps = [BRM._brm_svi_centering_affine(inst, c, f.frame, g, f.base) for inst in f.inst]
    C, b = Matrix{Float64}(I, f.d, f.d), zeros(f.d)
    for map in maps
        C[map.positions, map.positions] .= map.C
        b[map.positions] .= map.b
    end
    μ = randn(rng, f.d)
    Z = randn(rng, f.d, f.d)
    Σ = Z * Z' + I
    Λ = inv(Σ)
    q = (; M=[copy(μ)], Λ=[copy(Λ)], U=[Matrix(cholesky(Symmetric(Λ)).L)])
    BRM._brm_svi_unit_affine!(q, 1, maps)
    @test forward(f, g, only(q.M), c, f.frame)[1] ≈ μ
    @test only(q.M) ≈ C * μ + b
    @test inv(only(q.Λ)) ≈ C * Σ * C'
    @test only(q.U) * only(q.U)' ≈ only(q.Λ)
    saved = (; semantic=collect(Any, 1:f.d), M=copy(only(q.M)), Λ=copy(only(q.Λ)),
        centering=zeros(f.d), centering_maps=maps)
    before = deepcopy(saved)
    packed = BRM._brm_svi_source_unit("test", saved, "slice 1")
    @test packed.M ≈ μ
    @test packed.Λ ≈ Λ
    @test saved == before
    @test length(packed.natural) == 2
    old = Base.structdiff(saved, NamedTuple{(:centering_maps,)})
    @test_throws "no saved affine maps" BRM._brm_svi_source_unit("test", old, "slice 1")

    # Structured factors preserve transformed rank covariance and residual
    # marginal variances. Correlated residual noise is a stated projection.
    ρ, B = randn(rng, f.d), randn(rng, f.d, 2)
    factor = BRM._BRMSVILowRankUnit(copy(ρ), copy(B))
    structured = (; M=[copy(μ)], Λ=[BRM._BRMSVIUnitPrecision(factor)],
        U=[BRM._BRMSVIUnitRoot(factor)])
    BRM._brm_svi_unit_affine!(structured, 1, maps)
    @test only(structured.M) ≈ C * μ + b
    @test factor.Β ≈ C * B
    exact = C * (Diagonal(exp.(2 .* ρ)) + B * B') * C'
    projected = Diagonal(exp.(2 .* factor.ρ)) + factor.Β * factor.Β'
    @test diag(projected) ≈ diag(exact)
    @test !isapprox(projected, exact)
end

@testset "learned centering: omission and legacy policies" begin
    @test BRM._brm_svi_resolve_centering(nothing, missing) isa LearnedCentering
    @test BRM._brm_svi_resolve_centering(nothing, nothing) === nothing
    for keyword in (missing, nothing)
        @test BRM._brm_svi_resolve_centering((; layout=(;)), keyword) === nothing
        inherited = BRM._brm_svi_resolve_centering(
            (; layout=(; centering=(; start=0.2, step=0.03))), keyword)
        @test inherited.start == 0.2
        @test inherited.step == 0.03
    end
end

@testset "Gaussian population width: weak, strong and unbalanced groups" begin
    # Independent exact reference after integrating unit deviations of SD 1:
    # Var(a | y) = (prior_sd^-2 + sum(1 / (1 + sigma^2 / n_i)))^-1.
    # Here the optimal VIP controls are n_i / (sigma^2 + n_i), which remove
    # every global-unit precision cross term. Evaluate the production map,
    # rather than substituting that cancellation into the target.
    for (counts, sigma) in ((fill(1, 32), 5.0), (fill(50, 32), 1.0),
            (repeat([1, 50], 16), 1.0))
        prior_sd = 5.0
        controls = counts ./ (sigma^2 .+ counts)
        frame = (; A=[ones(1, 1)], values=())
        inst = BRM._BRMSVICenteringBlock(1, [1], 1:1, [[(1, 1.0)]])
        function logtarget(a, cs)
            value = -a^2 / (2prior_sd^2)
            for (n, c) in zip(counts, cs)
                y = zeros(1)
                BRM._brm_svi_unit_centering!(y, zeros(1), [inst], 1, [c],
                    frame, [a], Float64[])
                value -= (only(y)^2 + n * (a + only(y))^2 / sigma^2) / 2
            end
            value
        end
        sd(cs) = inv(sqrt(-(logtarget(1.0, cs) + logtarget(-1.0, cs) -
            2logtarget(0.0, cs))))
        exact = inv(sqrt(inv(prior_sd^2) + sum(inv(1 + sigma^2 / n) for n in counts)))
        @test sd(controls) ≈ exact rtol=1e-12
        @test sd(zeros(length(counts))) < exact
        if all(==(50), counts)
            @test sd(zeros(length(counts))) / exact < 0.15
        end
    end
end
