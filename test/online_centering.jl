# julia --project=test test/online_centering.jl
include("online_centering_fixture.jl")
using .OnlineCenteringFixture: prepare, wrap, scalar_model, correlated_model
using BayesianRegressionModels, WarmupHMC, LinearAlgebra, Test
import LogDensityProblems as LDP

# Conditional on log SD, (mu, z...) has a zero-mean Gaussian posterior.
# Opposite points on its Cholesky axes integrate every conditional second
# moment exactly. Gauss-Hermite integrates over the Normal(0, 0.4) log SD.
function scalar_points(precision)
    n = length(precision) + 1
    order = 24
    rule = eigen(SymTridiagonal(zeros(order), sqrt.(1:order-1)))
    points = NamedTuple[]
    for (ell, mass) in zip(0.4 .* rule.values, rule.vectors[1, :].^2)
        scale = exp(ell)
        X = hcat(ones(n-1), scale * I(n-1))
        F = cholesky(Symmetric(I + X' * Diagonal(precision) * X))
        weight = mass / (2n * prod(diag(F.U)))
        for axis in eachcol(F.U \ Matrix{Float64}(I, n, n)), sign in (-1, 1)
            v = sign * sqrt(n) * axis
            mu, z = first(v), v[2:end]
            residual = precision .* (mu .+ scale .* z)
            h = -z - scale .* residual
            push!(points, (; ell, mu, z, h, weight, g_mu=-mu-sum(residual),
                g_ell=-ell/0.4^2-scale*dot(z, residual)))
        end
    end
    points
end

function scalar_scores(fit, swapped)
    cells = fit.schema.cells
    block = only(fit.schema.blocks)
    levels = getproperty.(cells, :level)
    @test levels == block.levels
    @test block.target == 0
    @test all(c -> c.group == :group && c.family == :ranef, cells)
    @test Set(levels) == Set(["weak", "strong", "prior", "unused"])
    effect = getproperty.(cells, :effect)
    mu = only(unique(c.location.coefficient for c in cells))
    sd = only(only(block.log_scales).linear).coordinate
    n = LDP.dimension(fit.target)
    @test sort([mu, sd, effect...]) == 1:n
    weak_sd, strong_sd = swapped ? (0.2, 5.0) : (5.0, 0.2)
    by_level = Dict("weak"=>weak_sd^-2, "strong"=>strong_sd^-2, "prior"=>0.0, "unused"=>0.0)
    points = scalar_points([by_level[l] for l in levels])
    weights = getproperty.(points, :weight)
    seeds = Dict("weak"=>0.23, "strong"=>0.74, "prior"=>0.41, "unused"=>0.62)
    for controls in (zeros(length(cells)), [seeds[l] for l in levels], ones(length(cells)))
        wrapped = wrap(fit, controls)
        reported = adaptive_centering_controls(wrapped, fit.model)
        @test getproperty.(reported, :c) == controls
        @test getproperty.(reported, :index) == effect
        Q = zeros(n, length(points))
        G = similar(Q)
        for (i, p) in enumerate(points)
            q = zeros(n)
            q[mu], q[sd] = p.mu, p.ell
            q[effect] = controls .* p.mu + exp.(controls .* p.ell) .* p.z
            Q[:, i] = q
            _, grad = LDP.logdensity_and_gradient(wrapped, q)
            G[:, i] = grad
            # Include the source Jacobian and the dependence on the sampled mu and SD.
            g_effect = p.h ./ exp.(controls .* p.ell)
            @test grad[effect] ≈ g_effect atol=1e-8
            @test grad[mu] ≈ p.g_mu - dot(controls, g_effect) atol=1e-8
            @test grad[sd] ≈ p.g_ell - dot(controls .* p.z, p.h) - sum(controls) atol=1e-8
        end
        rows = WarmupHMC.candidate_scoring_losses(wrapped, Q, G; weights)
        @test length(rows) == 11 * length(cells)
        for c in cells
            curve = filter(r -> r.pair_number == c.cell, rows)
            @test sort(getproperty.(curve, :candidate)) ≈ 0:0.1:1
            for r in curve
                t = r.candidate
                x = [t*p.mu + exp(t*p.ell)*p.z[c.cell] for p in points]
                g = [exp(-t*p.ell)*p.h[c.cell] for p in points]
                # Both means vanish by symmetry.
                loss = dot(weights, x .* g) / sqrt(dot(weights, x.^2) * dot(weights, g.^2))
                @test r.index == c.effect
                @test r.loss ≈ loss atol=1e-10
            end
            winner = argmin(r -> r.loss, curve)
            @test winner.candidate ≈ (by_level[c.level] > 1 ? 0.9 : 0.0)
        end
    end
    LDP.logdensity_and_gradient(fit.target, zeros(n))
end

function correlated_transport(fit)
    blocks, cells = fit.schema
    pair = only(b for b in blocks if b.id == :pair)
    pc = filter(c -> c.block == pair.block, cells)
    @test getproperty.(pc, :term) == [1, 2]
    @test pc[2].conditioning == [pc[1].cell]
    @test length(blocks) == 2
    # Give each block its own seed; then move the two correlated margins
    # separately, keeping the other block's source fixed.
    old = [c.block == pair.block ? 0.2 : 0.6 for c in cells]
    new = copy(old)
    idx = getproperty.(pc, :cell)
    new[idx] = [0.75, 0.3]
    from, to = wrap(fit, old), wrap(fit, new)
    effect = getproperty.(pc, :effect)
    model_point = zeros(LDP.dimension(fit.target))
    beta = [0.7, -0.4]
    z = [1.2, -0.6]
    model_point[[c.location.coefficient for c in pc]] = beta
    model_point[[only(s.linear).coordinate for s in pair.log_scales]] = [0.27, -0.31]
    model_point[only(pair.correlation.free)] = 0.8
    model_point[effect] = z
    C = [exp(0.27) 0.0; exp(-0.31)*tanh(0.8) exp(-0.31)/cosh(0.8)]
    s = diag(C)
    A(c) = Diagonal(s.^c) + Diagonal(c) * tril(C, -1)
    oldmap, newmap = WarmupHMC.reparametrizer.((from, to))
    _, q = WarmupHMC.InverseFunctions.inverse(oldmap)(model_point)
    lj_old, target = oldmap(q)
    new_inverse = WarmupHMC.InverseFunctions.inverse(newmap)
    lj_inv, u = new_inverse(target)
    @test target ≈ model_point atol=1e-12
    @test last(newmap(u)) ≈ model_point atol=1e-12
    @test q[effect] ≈ old[idx] .* beta + A(old[idx])*z
    @test u[effect] ≈ new[idx] .* beta + A(new[idx])*z
    other = setdiff(eachindex(q), effect)
    @test u[other] ≈ q[other] atol=1e-12
    delta = dot(new[idx] - old[idx], log.(s))
    @test lj_old + lj_inv ≈ delta
    old_lp, old_g = LDP.logdensity_and_gradient(from, q)
    new_lp, new_g = LDP.logdensity_and_gradient(to, u)
    @test old_lp - new_lp ≈ delta
    # Differentiate the entire old-to-new transform, including its change
    # of volume with the sampled scales and correlation.
    function transport(x)
        lj1, y = oldmap(x)
        lj2, u_new = new_inverse(y)
        [u_new; lj1 + lj2]
    end
    step = 1e-5
    derivatives = map(eachindex(q)) do j
        dx = zeros(length(q))
        dx[j] = step
        (transport(q + dx) - transport(q - dx)) / (2step)
    end
    J = reduce(hcat, derivatives)
    @test first(logabsdet(J[1:end-1, :])) ≈ delta atol=1e-8
    @test J' * [new_g; 1.0] ≈ old_g atol=1e-8
    _, target_g = LDP.logdensity_and_gradient(fit.target, target)
    h = A(old[idx])' * old_g[effect]
    @test h ≈ target_g[effect]
    @test h ≉ old_g[effect]
    @test A(new[idx])' \ h ≈ new_g[effect]
    plan = WarmupHMC.candidate_scoring_plan(from)
    frame = plan.prepare(oldmap, q, old_g)
    proxy = h ./ s.^new[idx]
    for (k, c) in enumerate(pc)
        raw, value = oldmap.pairs[c.cell]
        t = WarmupHMC.PartiallyCentered(new[c.cell])
        jac, pos, score = plan.score(frame, c.cell, raw, value, t)
        @test jac ≈ (t.c - old[c.cell])*log(s[k])
        @test pos ≈ u[c.effect]
        @test score ≈ proxy[k]
    end
    @test abs(proxy[1] - new_g[effect[1]]) > 0.1
    @test proxy[2] ≈ new_g[effect[2]]
end

@testset "ordinary Gaussian centering observations" begin
    for backend in (RKBRMI, SBBRMI)
        @testset "$backend" begin
            original = prepare(backend, scalar_model())
            @info "Fitted scalar coordinates" backend levels=[c.level => c.effect for c in original.schema.cells]
            swapped = prepare(backend, scalar_model(; swapped=true))
            lp, g = scalar_scores(original, false)
            lp_swap, g_swap = scalar_scores(swapped, true)
            @test lp ≈ lp_swap
            @test g ≈ g_swap atol=1e-12
            correlated_transport(prepare(backend, correlated_model()))
        end
    end
end
