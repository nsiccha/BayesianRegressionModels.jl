# Per-step cost of each field family's draw, read, pull-back and update at field
# sizes up to the requested scale, on a synthetic grid mesh. Nothing here is a fit.
#
# A batch reads `reads` (vertex triple, time) rows with synthetic stations. Prior
# terms are the families': whitened ones have KL(q(v) ‖ N(0, I)) in closed form;
# `diagonal` evaluates E log p(z | θ) through BRM's separable operator, which is
# what a hand-written mesh rule would cost, not the RK graph's measured 10-14 s.
#
#     julia --startup-file=no research/svi_field_units/family_cost.jl
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
include("family_oracle.jl")
BLAS.set_num_threads(1)

timed(f, n=5) = begin
    f()
    best = Inf; bytes = 0
    for _ in 1:n
        s = @timed f()
        best = min(best, s.time); bytes = s.bytes
    end
    (; seconds=best, bytes)
end

function ou!(state, a, v)
    N, T = size(state)
    @views state[:, 1] .= state[:, 1]
    for t in 2:T
        @views state[:, t] .= a[t] .* state[:, t-1] .+ sqrt(v[t]) .* state[:, t]
    end
    state
end
function ou_adjoint!(g, a, v)
    N, T = size(g)
    for t in T:-1:2
        @views g[:, t-1] .+= a[t] .* g[:, t]
        @views g[:, t] .*= sqrt(v[t])
    end
    g
end

# The read of a batch and its adjoint at full typed speed: one pass per read vertex
# over the retained modes, on an e_t already through the OU recursion.
function spectral_read!(out, Ut, D, state, verts, weights, rtime)
    k = size(Ut, 1)
    @inbounds for r in eachindex(out)
        t, value = rtime[r], 0.0
        for c in 1:3
            i, w = verts[r, c], weights[r, c]
            @simd for j in 1:k
                value += w * Ut[j, i] * D[j] * state[j, t]
            end
        end
        out[r] = value
    end
    out
end
function spectral_adjoint!(g, Ut, D, out, verts, weights, rtime)
    k = size(Ut, 1)
    fill!(g, 0.0)
    @inbounds for r in eachindex(out)
        t, value = rtime[r], out[r]
        for c in 1:3
            i, w = verts[r, c], weights[r, c]
            @simd for j in 1:k
                g[j, t] += w * value * Ut[j, i] * D[j]
            end
        end
    end
    g
end
# The prior tail at a batch's reads: a whole-field prior draw through sparse solves
# per time, less its K low modes' part, read at the batch's vertices.
function tail_draw!(out, rng, white, work, coefficients, Ut, D, C12, B, a, vv, verts, weights, rtime)
    N, T = size(white)
    randn!(rng, white)
    for t in 1:T
        @views work[:, t] .= sqrt(2π) .* (B \ (C12 .* white[:, t]))
    end
    # Low-mode coefficients of the same noise: u_kᵀ C^½ w, scaled like the modes.
    mul!(coefficients, Ut, C12 .* white)
    coefficients .*= D
    ou!(work, a, vv)
    ou!(coefficients, a, vv)
    k = size(Ut, 1)
    @inbounds for r in eachindex(out)
        t, value = rtime[r], 0.0
        for c in 1:3
            i, w = verts[r, c], weights[r, c]
            low = 0.0
            @simd for j in 1:k
                low += Ut[j, i] * coefficients[j, t]
            end
            value += w * (work[i, t] - low)
        end
        out[r] = value
    end
    out
end
function adam_like!(x, m, v)
    @. m = 0.9 * m + 0.1 * x
    @. v = 0.999 * v + 0.001 * x^2
    @. x += 0.01 * m / (sqrt(v) + 1e-8)
    x
end
function spectral_step!(rng, state, g, out, m, v, Ut, D, a, vv, verts, weights, rtime, C)
    for _ in 1:C
        randn!(rng, state)
        ou!(state, a, vv)
        spectral_read!(out, Ut, D, state, verts, weights, rtime)
        spectral_adjoint!(g, Ut, D, out, verts, weights, rtime)
        ou_adjoint!(g, a, vv)
        adam_like!(g, m, v)
    end
    nothing
end

function bench(N1, T, C; K=N1^2, reads=2432, spectral=true)
    mesh = grid_mesh(N1)
    N = length(mesh.mass)
    times = collect(0.0:T-1)
    ℓs, ℓt = 0.3, 20.0
    (; a, v) = temporal_parts(times, ℓt)
    rng = Xoshiro(1)
    n = N * T
    verts = rand(rng, 1:N, reads, 3)
    weights = rand(rng, reads, 3); weights ./= sum(weights; dims=2)
    rtime = rand(rng, 1:T, reads)
    results = Dict{Symbol,Any}(:states => n * C, :vertices => N, :times => T, :channels => C, :reads => reads)
    Adam(x) = (x .+= 0.01 .* x ./ (sqrt.(x .^ 2) .+ 1e-8))

    # A one-channel pass; channels repeat it with their small triangular products.
    if spectral
        setup = @elapsed modes = spatial_modes(mesh)
        results[:spectral_setup_seconds] = setup
        k = min(K, N)
        D = sqrt(2π) ./ (1 / ℓs .+ ℓs .* modes.μ[1:k] ./ 2)
        U = Matrix(modes.U[:, 1:k]')
        state = randn(rng, k, T)
        out = zeros(reads)
        g, mm, vv2 = zeros(k, T), zeros(k, T), zeros(k, T)
        cost = timed() do
            spectral_step!(rng, state, g, out, mm, vv2, U, D, a, v, verts, weights, rtime, C)
        end
        results[:space_spectral] = (; cost..., retained_state_bytes=C * k * T * 8 * 6 + sizeof(U))
        if k < N
            B0 = cholesky(Symmetric(spatial_b(mesh, ℓs)))
            white, workspace, coefficients = zeros(N, T), zeros(N, T), zeros(k, T)
            C12 = sqrt.(mesh.mass)
            tail = timed() do
                for _ in 1:C
                    tail_draw!(out, rng, white, workspace, coefficients, U, D, C12, B0, a, v, verts, weights, rtime)
                end
            end
            results[:prior_tail] = (; tail..., retained_state_bytes=0)
        end
    end

    # Whitened: the whole field per draw, one pair of sparse solves per time.
    B = cholesky(Symmetric(spatial_b(mesh, ℓs)))
    C12 = sqrt.(mesh.mass)
    state = randn(rng, N, T)
    work = zeros(N, T)
    g = zeros(N, T)
    out = zeros(reads)
    cost = timed() do
        for _ in 1:C
            randn!(rng, state)
            for t in 1:T
                @views work[:, t] .= sqrt(2π) .* (B \ (C12 .* state[:, t]))
            end
            ou!(work, a, v)
            for r in 1:reads
                out[r] = sum(weights[r, c] * work[verts[r, c], rtime[r]] for c in 1:3)
            end
            fill!(g, 0.0)
            for r in 1:reads, c in 1:3
                g[verts[r, c], rtime[r]] += weights[r, c] * out[r]
            end
            ou_adjoint!(g, a, v)
            for t in 1:T
                @views g[:, t] .= sqrt(2π) .* C12 .* (B \ g[:, t])
            end
            Adam(g)
        end
    end
    results[:whitened] = (; cost..., retained_state_bytes=C * N * T * 8 * 6)

    # Diagonal in field coordinates: reads are local, the prior gradient is not.
    state3 = randn(rng, N, 1, T)
    outp = similar(state3)
    L = ones(1, 1)
    workspace = GM._brm_gmrf_precision_workspace(state3)
    cost = timed() do
        for _ in 1:C
            randn!(rng, state3)
            GM._brm_gmrf_precision_mul!(outp, state3, mesh, times, ℓs, ℓt, L, workspace)
            Adam(outp)
        end
    end
    results[:diagonal_whole_prior] = (; cost..., retained_state_bytes=C * N * T * 8 * 6)
    results
end

for (N1, T, C, K, spectral) in ((52, 40, 1, 2704, true), (52, 122, 1, 2704, true), (52, 122, 9, 2704, true),
        (52, 122, 9, 256, true), (107, 122, 1, 256, true))
    r = bench(N1, T, C; K, spectral)
    @printf("states=%-9d vertices=%-6d times=%-4d channels=%-2d reads=%d modes=%d\n", r[:states], r[:vertices], r[:times], r[:channels], r[:reads], K)
    haskey(r, :spectral_setup_seconds) && @printf("   space_spectral setup (dense eigendecomposition): %.1f s\n", r[:spectral_setup_seconds])
    for key in (:space_spectral, :prior_tail, :whitened, :diagonal_whole_prior)
        haskey(r, key) || continue
        x = r[key]
        @printf("   %-22s %9.4f s per step  %9.1f MB allocated  %9.1f MB state\n", key, x.seconds, x.bytes / 1e6, x.retained_state_bytes / 1e6)
    end
    flush(stdout)
end
