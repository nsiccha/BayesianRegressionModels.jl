# What a variational family over a gmrf field means at its optimum, exactly.
#
# Toy only: a synthetic mesh field read by synthetic stations, y = α_s + σ A z + ε,
# z ~ N(0, Q_t(ℓ_t) ⊗ Q_s(ℓ_s))⁻¹, using BRM's own precision (`src/gmrf_precision.jl`).
# The target is Gaussian given θ = (σ, ℓ_s, ℓ_t, τ), so the exact posterior and
# marginal likelihood are available, and each family's optimum is available in
# closed form: every factorized Gaussian family has the exact posterior mean, and
# its factor precision is the corresponding block of the target's precision. There
# is no optimizer and no Monte Carlo noise in this file.
#
#     julia --startup-file=no research/svi_field_units/family_oracle.jl out.json [regime] [nseeds] [family,family,...]
#
# Families (every one keeps one independent factor per station effect α_s):
#   exact      the conjugate posterior
#   diagonal   z_j independent
#   tile_k     z independent between space-time tiles, dense within (k vertices a
#              side × 4 times)
#   whitened   z = T v with T Tᵀ = Q⁻¹, v_j independent. T = T_t ⊗ T_s, with
#              T_s = √(2π) B⁻¹ C^½ and T_t the causal OU recursion.
#   space_whitened  z = (I ⊗ T_s) e, e_j independent: whitened in space only.
#   spectral   z = T v with T = (Φ ω^-½) ⊗ (U D): the prior's own eigenbasis, v_j
#              independent. U solves G u = μ C u and does not depend on ℓ_s;
#              D = √(2π)/(1/ℓ_s + ℓ_s μ/2). Φ, ω are the eigenpairs of Q_t(ℓ_t).
#   space_spectral  the same in space, with the causal OU root in time.
#   space_hier the eigenbasis in space with a hierarchical (Lévy) root in time: the two
#              ends, then midpoints conditional on their interval's ends. OU is Markov, so
#              this is an exact root for every ℓ_t on any lattice, and the prior cancels.
#              Time costs O(T) with no dense matrix.
#   hier_K     space_hier on the lowest K spatial modes, every other mode at its prior.
#   time_dct   like spectral, but time uses a fixed DCT-II basis Ψ scaled per ℓ_t so each
#              coordinate's prior variance is one: T_t = Ψ S(ℓ_t). The prior no longer
#              cancels exactly; its term stays closed form. It needs no eigenvector
#              derivative and costs T log T, on any lattice.
#   truncated_K  space_spectral with variational parameters on the lowest K spatial
#              modes only (K × times of them). Every other mode stays exactly at its
#              prior N(0, 1), so the state is K × times, whatever the vertex count, and
#              U is n × K. Here the mean is not exact: it is the optimum in that subspace.
#   whitened_rK  whitened, with q(v)'s precision diagonal plus rank K, from the top K
#              eigenpairs of the data's precision in v and the matching diagonal. It is
#              one member of that family, so its gap is an upper bound on the family's.
using LinearAlgebra, SparseArrays, Random, Statistics, Printf

module GM
using LinearAlgebra, SparseArrays
const SRC = joinpath(@__DIR__, "..", "..", "src")
include(joinpath(SRC, "gmrf_precision.jl"))
include(joinpath(SRC, "gmrf_interpolation.jl"))
end

function grid_mesh(v)
    xy = zeros(v^2, 2)
    id(i, j) = i + v * (j - 1)
    for j in 1:v, i in 1:v
        xy[id(i, j), 1], xy[id(i, j), 2] = (i - 1) / (v - 1), (j - 1) / (v - 1)
    end
    cells = Matrix{Int}(undef, 2(v - 1)^2, 3)
    c = 0
    for j in 1:v-1, i in 1:v-1
        cells[c+=1, :] = [id(i, j), id(i + 1, j), id(i + 1, j + 1)]
        cells[c+=1, :] = [id(i, j), id(i + 1, j + 1), id(i, j + 1)]
    end
    GM._brm_gmrf_mesh(xy, cells)
end

spatial_b(mesh, ℓ) = sparse(Diagonal(mesh.mass)) ./ ℓ .+ (ℓ / 2) .* mesh.stiffness
function spatial_q(mesh, ℓ)
    B = spatial_b(mesh, ℓ)
    sparse(Symmetric(B * Diagonal(1 ./ mesh.mass) * B ./ 2π))
end
function temporal_parts(times, ℓ)
    n = length(times)
    a, v = zeros(n), ones(n)
    for t in 2:n
        gap = times[t] - times[t-1]
        a[t], v[t] = exp(-gap / ℓ), -expm1(-2gap / ℓ)
    end
    (; a, v)
end
function temporal_q(times, ℓ)
    (; a, v) = temporal_parts(times, ℓ)
    n = length(times)
    Q = spzeros(n, n)
    for t in 1:n
        Q[t, t] += 1 / v[t]
        if t > 1
            Q[t-1, t-1] += a[t]^2 / v[t]
            Q[t, t-1] -= a[t] / v[t]
            Q[t-1, t] -= a[t] / v[t]
        end
    end
    Q
end
function temporal_root(times, ℓ)
    (; a, v) = temporal_parts(times, ℓ)
    n = length(times)
    T = zeros(n, n)
    for t in 1:n
        T[t, t] = sqrt(v[t])
        for s in t-1:-1:1
            T[t, s] = a[t] * T[t-1, s]
        end
    end
    T
end
spatial_root(mesh, ℓ) = sqrt(2π) .* (Matrix(spatial_b(mesh, ℓ)) \ Matrix(Diagonal(sqrt.(mesh.mass))))
# G u = μ C u: with C diagonal the eigenvectors of C^-½ G C^-½ give u, free of ℓ.
function spatial_modes(mesh)
    h = 1 ./ sqrt.(mesh.mass)
    E = eigen(Symmetric(Matrix(mesh.stiffness) .* h .* h'))
    (; U=E.vectors .* h, μ=max.(E.values, 0.0))
end
spatial_spectral_root(modes, ℓ) = modes.U .* (sqrt(2π) ./ (1 / ℓ .+ ℓ .* modes.μ ./ 2))'
function temporal_spectral_root(times, ℓ)
    E = eigen(Symmetric(Matrix(temporal_q(times, ℓ))))
    E.vectors .* (1 ./ sqrt.(E.values))'
end

# Reads: each station sits at one position and reads a random subset of times.
function toy_reads(mesh, times, stations, rows, rng)
    N, nt = length(mesh.mass), length(times)
    east, north = rand(rng, stations) .* 0.9 .+ 0.05, rand(rng, stations) .* 0.9 .+ 0.05
    (; indices, weights) = GM._brm_gmrf_interpolation(mesh, east, north)
    I, J, V, S = Int[], Int[], Float64[], Int[]
    r = 0
    for s in 1:stations, t in sort!(randperm(rng, nt)[1:min(rows, nt)])
        r += 1
        push!(S, s)
        for k in 1:3
            push!(I, r); push!(J, indices[s, k] + N * (t - 1)); push!(V, weights[s, k])
        end
    end
    A = sparse(I, J, V, r, N * nt)
    Z = sparse(1:r, S, ones(r), r, stations)
    (; A, Z)
end

function target(mesh, times, reads, θ)
    (; σ, ℓs, ℓt, τ, σα) = θ
    Qs, Qt = spatial_q(mesh, ℓs), temporal_q(times, ℓt)
    Q = kron(Qt, Qs)
    (; A, Z) = reads
    nz, S = size(A, 2), size(Z, 2)
    λ = 1 / τ^2
    Λzz = Q + σ^2 * λ * (A' * A)
    Λza = σ * λ * (A' * Z)
    Λaa = sparse(Diagonal(fill(1 / σα^2, S))) + λ * (Z' * Z)
    Λ = [Λzz Λza; Λza' Λaa]
    F = cholesky(Symmetric(Λ))
    logQs = 2logdet(cholesky(Symmetric(Matrix(spatial_b(mesh, ℓs))))) -
        sum(log, mesh.mass) - length(mesh.mass) * log(2π)
    logQt = -sum(log, temporal_parts(times, ℓt).v)
    logQ = length(times) * logQs + length(mesh.mass) * logQt
    (; Q, Qs, Qt, Λ, Λzz, Λza, Λaa, F, logdetΛ=logdet(F), logQ, logQs, logQt, nz, S, λ, σ, θ)
end

function log_marginal(tg, reads, y)
    (; A, Z) = reads
    b = vcat(tg.σ .* (A' * y), Z' * y) .* tg.λ
    n = length(y)
    -n * log(2π / tg.λ) / 2 - tg.λ * dot(y, y) / 2 + dot(b, tg.F \ b) / 2 -
        tg.logdetΛ / 2 + tg.logQ / 2 - tg.S * log(tg.θ.σα^2) / 2
end

# Tiles: k vertices a side and `span` times. `v` is the mesh's vertices per side.
function tiles(v, nt, k, span)
    N = v^2
    groups = Dict{NTuple{3,Int},Vector{Int}}()
    for t in 1:nt, j in 1:v, i in 1:v
        push!(get!(groups, ((i - 1) ÷ k, (j - 1) ÷ k, (t - 1) ÷ span), Int[]), i + v * (j - 1) + N * (t - 1))
    end
    collect(values(groups))
end

const _LOWRANK = (; whitened_r8=8, whitened_r32=32)
const _TRUNCATED = (; truncated_16=16, truncated_36=36, truncated_64=64, hier_16=16, hier_36=36, hier_64=64)
_truncated_root(name) = startswith(String(name), "hier") ? :space_hier : :space_spectral
const _DCT_TRUNCATED = (; dct_16=16, dct_36=36)

# The hierarchical root of an OU lattice: T[t, j] is coordinate j's effect on time t.
function temporal_hier_root(times, ℓ)
    n = length(times)
    T = zeros(n, n)
    T[1, 1] = 1.0
    n == 1 && return T
    ρ = exp(-(times[n] - times[1]) / ℓ)
    T[n, :] .= ρ .* T[1, :]
    T[n, 2] = sqrt(-expm1(-2(times[n] - times[1]) / ℓ))
    next = Ref(3)
    function fill!(l, r)
        r - l <= 1 && return
        m = (l + r) ÷ 2
        a, b = exp(-(times[m] - times[l]) / ℓ), exp(-(times[r] - times[m]) / ℓ)
        d = 1 - a^2 * b^2
        T[m, :] .= (a * (1 - b^2) / d) .* T[l, :] .+ (b * (1 - a^2) / d) .* T[r, :]
        T[m, next[]] = sqrt((1 - a^2) * (1 - b^2) / d)
        next[] += 1
        fill!(l, m)
        fill!(m, r)
    end
    fill!(1, n)
    T
end

dct_basis(n) = [(k == 1 ? sqrt(1 / n) : sqrt(2 / n)) * cos(π * (t - 0.5) * (k - 1) / n) for t in 1:n, k in 1:n]
function time_dct_root(times, ℓ)
    Ψ = dct_basis(length(times))
    Q = Matrix(temporal_q(times, ℓ))
    Ψ ./ sqrt.(vec(sum(Ψ .* (Q * Ψ); dims=1)))'
end
# The DCT family's own log|T|²: the root is no longer an exact root of Q⁻¹.
time_dct_logdet2(times, ℓ, ctx_logQs, N) = begin
    R = time_dct_root(times, ℓ)
    N * 2logabsdet(R)[1] - length(times) * ctx_logQs
end

# Retained spatial modes get the DCT family; every other mode stays exactly at its own
# prior time series, independent across modes because the prior is Q_t ⊗ Q_s.
function _dct_truncated(tg, reads, ctx, K)
    θ, times, N = tg.θ, ctx.times, ctx.N
    nt = length(times)
    Φ = spatial_spectral_root(ctx.modes, θ.ℓs)
    retained = kron(time_dct_root(times, θ.ℓt), Φ[:, 1:K])
    AR = Matrix(reads.A * retained) .* (tg.σ * sqrt(tg.λ))
    tail = kron(temporal_root(times, θ.ℓt), Φ[:, K+1:N])
    AT = Matrix(reads.A * tail) .* (tg.σ * sqrt(tg.λ))
    (; retained, tail, AR, AT, K)
end

# The truncated family's optimum: P_jj on retained modes, 1 on the tail, and the
# subspace mean over retained modes and station effects. Returns its KL pieces.
function _truncated(tg, reads, ctx, K, root=:space_spectral)
    T = ctx.roots[root](tg.θ)
    N, nt = ctx.N, length(ctx.times)
    keep = [k + N * (t - 1) for t in 1:nt for k in 1:K]
    AT = Matrix(reads.A * T) .* (tg.σ * sqrt(tg.λ))
    Pdiag = 1 .+ vec(sum(abs2, AT; dims=1))
    tail = setdiff(1:size(T, 2), keep)
    (; T, AT, keep, tail, Pdiag)
end
function _truncated_mean(tg, reads, y, f)
    Z = Matrix(reads.Z)
    sqrtλ = sqrt(tg.λ)
    AS = f.AT[:, f.keep]
    P = [Matrix{Float64}(I, length(f.keep), length(f.keep)) + AS' * AS  sqrtλ .* (AS' * Z);
         sqrtλ .* (Z' * AS)  Matrix(tg.Λaa)]
    b = vcat(sqrtλ .* (AS' * y), tg.λ .* (Z' * y))
    m = cholesky(Symmetric(P)) \ b
    (; m, b)
end

# In v the target's precision is P = I + H, H = σ²λ (A T)ᵀ(A T), rank ≤ reads. Take
# P_q = D + U Uᵀ with U the top r eigenpairs of H and D matching P's diagonal, so
# Woodbury gives log|P_q|, tr(P P_q⁻¹) and the marginal variance.
function _whitened_lowrank(tg, reads, ctx, r)
    AT = Matrix(reads.A * ctx.T(tg.θ)) .* (tg.σ * sqrt(tg.λ))
    F = svd(AT)
    k = min(r, length(F.S))
    U = F.V[:, 1:k] .* F.S[1:k]'
    D = 1 .+ vec(sum(abs2, AT; dims=1)) .- vec(sum(abs2, U; dims=2))
    Di = 1 ./ D
    W = cholesky(Symmetric(Matrix{Float64}(I, k, k) + U' * (U .* Di)))
    logdetq = sum(log, D) + logdet(W)
    apply(X) = (X .* Di) .- (U .* Di) * (W \ (U' * (X .* Di)))
    # tr(P P_q⁻¹) with P = I + ATᵀAT.
    V = U .* Di
    tr_inv = sum(Di) - sum(V .* (W \ V')')
    Y = apply(AT')
    trace = tr_inv + sum(AT' .* Y)
    (; H=AT, U, D, W, Di, logdetq, trace, apply)
end

# KL(q* ‖ posterior) = ½(Σ_b log|Λ_bb| − log|Λ|), in the family's own coordinates.
function family_gap(name, tg, reads, ctx)
    logΛaa = sum(log, diag(tg.Λaa))
    if name === :diagonal
        (sum(log, diag(tg.Λzz)) + logΛaa - tg.logdetΛ) / 2
    elseif name in (:whitened, :spectral, :space_spectral, :space_hier)
        AT = reads.A * ctx.roots[name](tg.θ)
        P = 1 .+ tg.σ^2 * tg.λ .* vec(sum(abs2, AT; dims=1))
        (sum(log, P) + logΛaa - tg.logdetΛ + tg.logQ) / 2
    elseif name === :time_dct
        T = ctx.roots[:time_dct](tg.θ)
        AT = reads.A * T
        P = 1 .+ tg.σ^2 * tg.λ .* vec(sum(abs2, AT; dims=1))
        (sum(log, P) + logΛaa - tg.logdetΛ - time_dct_logdet2(ctx.times, tg.θ.ℓt, tg.logQs, ctx.N)) / 2
    elseif name in keys(_DCT_TRUNCATED)
        # KL(q ‖ posterior) in the coordinates (retained v, tail w): the prior is N(0, R) on
        # the retained block and N(0, I) on the tail, and the tail's q is its prior.
        f = _dct_truncated(tg, reads, ctx, _DCT_TRUNCATED[name])
        Z = Matrix(reads.Z)
        sqrtλ = sqrt(tg.λ)
        n = size(f.retained, 2)
        Qmodes = Matrix(Symmetric(f.retained' * (tg.Q * f.retained)))
        Pdiag = diag(Qmodes) .+ vec(sum(abs2, f.AR; dims=1))
        H = Qmodes + f.AR' * f.AR
        P = [H sqrtλ .* (f.AR' * Z); sqrtλ .* (Z' * f.AR) Matrix(tg.Λaa)]
        b = vcat(sqrtλ .* (f.AR' * ctx.y), tg.λ .* (Z' * ctx.y))
        m = cholesky(Symmetric(P)) \ b
        bfull = vcat(tg.σ .* (reads.A' * ctx.y), reads.Z' * ctx.y) .* tg.λ
        quadratic = dot(tg.F \ bfull, bfull) - dot(m, b)
        taildata = sum(abs2, f.AT)
        # log|T|²: the retained modes' DCT root, the tail's exact root, and the spatial part.
        K = ctx.K(name)
        logT2 = K * 2logabsdet(time_dct_root(ctx.times, tg.θ.ℓt))[1] -
            (ctx.N - K) * tg.logQt - length(ctx.times) * tg.logQs
        (taildata - tg.logdetΛ - logT2 + sum(log, Pdiag) + logΛaa + quadratic) / 2
    elseif name in keys(_TRUNCATED)
        f = _truncated(tg, reads, ctx, _TRUNCATED[name], _truncated_root(name))
        fit = _truncated_mean(tg, reads, ctx.y, f)
        bfull = vcat(tg.σ .* (reads.A' * ctx.y), reads.Z' * ctx.y) .* tg.λ
        quadratic = dot(tg.F \ bfull, bfull) - dot(fit.m, fit.b)
        (sum(f.Pdiag[f.tail] .- 1) - tg.logdetΛ + tg.logQ + sum(log, f.Pdiag[f.keep]) + logΛaa + quadratic) / 2
    elseif name in keys(_LOWRANK)
        factor = _whitened_lowrank(tg, reads, ctx, _LOWRANK[name])
        # KL(N(μ, P_q⁻¹) ‖ N(μ, P⁻¹)) = ½(tr(P P_q⁻¹) − n + log|P_q| − log|P|), α-blocks exact.
        (factor.trace - size(factor.H, 2) + factor.logdetq + logΛaa - tg.logdetΛ + tg.logQ) / 2
    elseif name === :space_whitened
        AT = reads.A * ctx.Ts(tg.θ)
        P = diag(kron(tg.Qt, sparse(I, size(tg.Qs)...))) .+ tg.σ^2 * tg.λ .* vec(sum(abs2, AT; dims=1))
        (sum(log, P) + logΛaa - tg.logdetΛ + length(ctx.times) * tg.logQs) / 2
    else
        total = logΛaa
        for b in ctx.tiles[name]
            total += logdet(cholesky(Symmetric(Matrix(tg.Λzz[b, b]))))
        end
        (total - tg.logdetΛ) / 2
    end
end

# Posterior or family marginal SDs on the field and on the latent reads A z.
function family_sds(name, tg, reads, ctx, exact)
    A = reads.A
    if name === :exact
        Σ = exact.Σ
        return (; z=sqrt.(diag(Σ)), read=sqrt.(max.(0, diag(A * Σ * A'))))
    elseif name === :diagonal
        s2 = 1 ./ diag(tg.Λzz)
        return (; z=sqrt.(s2), read=sqrt.(vec(sum(abs2, A; dims=2)' ) .* 0 .+ [dot(A[r, :] .^ 2, s2) for r in axes(A, 1)]))
    elseif name in keys(_DCT_TRUNCATED)
        f = _dct_truncated(tg, reads, ctx, _DCT_TRUNCATED[name])
        Qmodes = f.retained' * (tg.Q * f.retained)
        Pdiag = diag(Qmodes) .+ vec(sum(abs2, f.AR; dims=1))
        return (; z=sqrt.(vec(sum(abs2, f.retained ./ sqrt.(Pdiag)'; dims=2)) .+ vec(sum(abs2, f.tail; dims=2))),
            read=sqrt.(vec(sum(abs2, Matrix(A * f.retained) ./ sqrt.(Pdiag)'; dims=2)) .+
                vec(sum(abs2, Matrix(A * f.tail); dims=2))))
    elseif name in keys(_TRUNCATED)
        f = _truncated(tg, reads, ctx, _TRUNCATED[name], _truncated_root(name))
        scale = ones(size(f.T, 2))
        scale[f.keep] .= 1 ./ sqrt.(f.Pdiag[f.keep])
        return (; z=sqrt.(vec(sum(abs2, f.T .* scale'; dims=2))),
            read=sqrt.(vec(sum(abs2, Matrix(A * f.T) .* scale'; dims=2))))
    elseif name in keys(_LOWRANK)
        T = ctx.T(tg.θ)
        factor = _whitened_lowrank(tg, reads, ctx, _LOWRANK[name])
        Y, X = factor.apply(T'), factor.apply((reads.A * T)')
        return (; z=sqrt.(max.(0, vec(sum(T' .* Y; dims=1)))),
            read=sqrt.(max.(0, vec(sum((reads.A * T)' .* X; dims=1)))))
    elseif name in (:whitened, :spectral, :space_spectral, :space_hier, :space_whitened, :time_dct)
        T = name === :space_whitened ? ctx.Ts(tg.θ) : ctx.roots[name](tg.θ)
        AT = A * T
        base = name === :space_whitened ? diag(kron(tg.Qt, sparse(I, size(tg.Qs)...))) : ones(size(T, 2))
        P = base .+ tg.σ^2 * tg.λ .* vec(sum(abs2, AT; dims=1))
        return (; z=sqrt.(vec(sum(abs2, T ./ sqrt.(P)'; dims=2))),
            read=sqrt.(vec(sum(abs2, AT ./ sqrt.(P)'; dims=2))))
    else
        n = tg.nz
        rows, cols, vals = Int[], Int[], Float64[]
        for b in ctx.tiles[name]
            C = inv(Symmetric(Matrix(tg.Λzz[b, b])))
            for (x, j) in enumerate(b), (w, i) in enumerate(b)
                push!(rows, i); push!(cols, j); push!(vals, C[w, x])
            end
        end
        Σ = sparse(rows, cols, vals, n, n)
        return (; z=sqrt.(diag(Σ)), read=sqrt.(max.(0, diag(A * Σ * A'))))
    end
end

function nelder_mead(f, x0; step=0.3, iterations=220)
    n = length(x0)
    simplex = [copy(x0) for _ in 1:n+1]
    for i in 1:n
        simplex[i+1][i] += step
    end
    values = f.(simplex)
    for _ in 1:iterations
        order = sortperm(values)
        simplex, values = simplex[order], values[order]
        c = sum(simplex[1:n]) / n
        r = c + (c - simplex[end])
        fr = f(r)
        if fr < values[1]
            e = c + 2(c - simplex[end])
            fe = f(e)
            simplex[end], values[end] = fe < fr ? (e, fe) : (r, fr)
        elseif fr < values[n]
            simplex[end], values[end] = r, fr
        else
            k = c + 0.5(simplex[end] - c)
            fk = f(k)
            if fk < values[end]
                simplex[end], values[end] = k, fk
            else
                for i in 2:n+1
                    simplex[i] = simplex[1] + 0.5(simplex[i] - simplex[1])
                    values[i] = f(simplex[i])
                end
            end
        end
    end
    i = argmin(values)
    simplex[i], values[i]
end

quantile3(x) = (; lo=quantile(x, 0.05), median=median(x), hi=quantile(x, 0.95))

# Positive controls: this file's Q, log|Q| and root T against BRM's own prior.
function self_check(mesh, times, θ, T, rng)
    tg_Qs, tg_Qt = spatial_q(mesh, θ.ℓs), temporal_q(times, θ.ℓt)
    Q = kron(tg_Qt, tg_Qs)
    N, nt = length(mesh.mass), length(times)
    z = randn(rng, N * nt)
    state = reshape(z, N, 1, nt)
    L = ones(1, 1)
    out = similar(state)
    GM._brm_gmrf_precision_mul!(out, state, mesh, times, θ.ℓs, θ.ℓt, L)
    @assert norm(vec(out) - Q * z) <= 1e-9 * norm(Q * z) "precision disagrees with BRM"
    logQ = nt * (2logdet(cholesky(Symmetric(Matrix(spatial_b(mesh, θ.ℓs))))) -
        sum(log, mesh.mass) - N * log(2π)) - N * sum(log, temporal_parts(times, θ.ℓt).v)
    value = logQ / 2 - dot(z, Q * z) / 2 - N * nt * log(2π) / 2
    brm = GM._brm_gmrf_logpdf(state, mesh, times, θ.ℓs, θ.ℓt, L)
    @assert abs(value - brm) <= 1e-8 * abs(brm) "log density disagrees with BRM: $value vs $brm"
    R = T(θ)
    residual = norm(R' * (Q * R) - I) / sqrt(N * nt)
    @assert residual <= 1e-5 "T Tᵀ is not Q⁻¹: residual $residual"
    nothing
end

const ALL_FAMILIES = (:exact, :diagonal, :tile_2, :tile_4, :space_whitened, :whitened, :space_spectral, :spectral, :space_hier,
    :hier_16, :hier_36, :hier_64, :time_dct, :dct_16, :dct_36, :truncated_16, :truncated_36, :truncated_64, :whitened_r8, :whitened_r32)
function run_regime(label, truth; v=11, nt=16, stations=12, rows=8, seed=1, families=ALL_FAMILIES, irregular=false)
    mesh = grid_mesh(v)
    times = irregular ? cumsum(vcat(0.0, 0.25 .+ 2.5 .* rand(Xoshiro(99), nt - 1))) : collect(0.0:nt-1)
    rng = Xoshiro(seed)
    reads = toy_reads(mesh, times, stations, rows, rng)
    N = length(mesh.mass)
    nz = size(reads.A, 2)
    T(θ) = kron(temporal_root(times, θ.ℓt), spatial_root(mesh, θ.ℓs))
    Ts(θ) = kron(Matrix{Float64}(I, nt, nt), spatial_root(mesh, θ.ℓs))
    modes = spatial_modes(mesh)
    roots = Dict(:whitened => T,
        :spectral => (θ -> kron(temporal_spectral_root(times, θ.ℓt), spatial_spectral_root(modes, θ.ℓs))),
        :space_spectral => (θ -> kron(temporal_root(times, θ.ℓt), spatial_spectral_root(modes, θ.ℓs))),
        :space_hier => (θ -> kron(temporal_hier_root(times, θ.ℓt), spatial_spectral_root(modes, θ.ℓs))),
        :time_dct => (θ -> kron(time_dct_root(times, θ.ℓt), spatial_spectral_root(modes, θ.ℓs))))
    ctx = (; T, Ts, roots, times, N, modes, K=(name -> _DCT_TRUNCATED[name]), y=Float64[], tiles=Dict(:tile_2 => tiles(v, nt, 2, 4), :tile_4 => tiles(v, nt, 4, 4)))
    # Draw the truth from the model.
    Tz = T(truth)
    z = Tz * randn(rng, nz)
    α = truth.σα .* randn(rng, stations)
    y = truth.σ .* (reads.A * z) .+ reads.Z * α .+ truth.τ .* randn(rng, size(reads.A, 1))
    ctx = merge(ctx, (; y))
    tg = target(mesh, times, reads, truth)
    exact = (; Σ=Matrix(inv(cholesky(Symmetric(Matrix(tg.Λ)))))[1:nz, 1:nz])
    b = vcat(tg.σ .* (reads.A' * y), reads.Z' * y) .* tg.λ
    mean_z = (tg.F \ b)[1:nz]
    is_read = vec(sum(abs, reads.A; dims=1)) .> 0
    names = families
    for (name, root) in roots
        name === :time_dct || self_check(mesh, times, truth, root, rng)
    end
    fixed = map(names) do name
        sds = family_sds(name, tg, reads, ctx, exact)
        ref = family_sds(:exact, tg, reads, ctx, exact)
        ratio, read_ratio = sds.z ./ ref.z, sds.read ./ ref.read
        family_mean = if name in keys(_DCT_TRUNCATED)
            f = _dct_truncated(tg, reads, ctx, _DCT_TRUNCATED[name])
            Z, sqrtλ = Matrix(reads.Z), sqrt(tg.λ)
            H = f.retained' * (tg.Q * f.retained) + f.AR' * f.AR
            P = [H sqrtλ .* (f.AR' * Z); sqrtλ .* (Z' * f.AR) Matrix(tg.Λaa)]
            b = vcat(sqrtλ .* (f.AR' * y), tg.λ .* (Z' * y))
            f.retained * (cholesky(Symmetric(P)) \ b)[1:size(f.retained, 2)]
        elseif name in keys(_TRUNCATED)
            f = _truncated(tg, reads, ctx, _TRUNCATED[name], _truncated_root(name))
            f.T[:, f.keep] * _truncated_mean(tg, reads, y, f).m[1:length(f.keep)]
        else
            mean_z
        end
        mean_error = median(abs.(family_mean .- mean_z) ./ ref.z)
        cover = mean(abs.(z .- family_mean) .<= 1.645 .* sds.z)
        (; family=name, mean_error, kl=name === :exact ? 0.0 : family_gap(name, tg, reads, ctx),
           sd_ratio_read_states=quantile3(ratio[is_read]),
           sd_ratio_unread_states=quantile3(ratio[.!is_read]),
           sd_ratio_latent_reads=quantile3(read_ratio),
           coverage90=cover)
    end
    # Closed-form gap against the dense Gaussian KL of the diagonal family.
    let Λd = Matrix(tg.Λ)
        qprec = Diagonal(diag(Λd))
        # tr(Λ Σ_q) = n for the optimal block factor, leaving the log determinants.
        forward = (logdet(qprec) - logdet(cholesky(Symmetric(Λd)))) / 2
        @assert abs(forward - family_gap(:diagonal, tg, reads, ctx)) <= 1e-6 * max(1, abs(forward)) "gap formula"
    end
    # Hyperparameter optimum of each family's ELBO (flat on log θ; σα known).
    pack(θ) = log.([θ.σ, θ.ℓs, θ.ℓt, θ.τ])
    unpack(x) = (; σ=exp(x[1]), ℓs=exp(x[2]), ℓt=exp(x[3]), τ=exp(x[4]), σα=truth.σα)
    box(x) = any(abs.(x .- pack(truth)) .> log(30))
    fits = map(names) do name
        objective(x) = begin
            box(x) && return 1e12
            θ = unpack(x)
            tgx = target(mesh, times, reads, θ)
            -(log_marginal(tgx, reads, y) - (name === :exact ? 0.0 : family_gap(name, tgx, reads, ctx)))
        end
        x, value = nelder_mead(objective, pack(truth))
        θ = unpack(x)
        (; family=name, σ=θ.σ, ℓs=θ.ℓs, ℓt=θ.ℓt, τ=θ.τ, elbo=-value)
    end
    (; label, seed, v, nt, stations, rows, nz, nread=count(is_read),
       truth=(; truth.σ, truth.ℓs, truth.ℓt, truth.τ, truth.σα), fixed, fits)
end

function json(io, x)
    if x isa NamedTuple
        print(io, "{")
        for (i, k) in enumerate(keys(x))
            i > 1 && print(io, ",")
            print(io, '"', k, "\":")
            json(io, x[k])
        end
        print(io, "}")
    elseif x isa Union{Tuple,AbstractVector}
        print(io, "[")
        for (i, e) in enumerate(x)
            i > 1 && print(io, ",")
            json(io, e)
        end
        print(io, "]")
    elseif x isa Symbol || x isa AbstractString
        print(io, '"', x, '"')
    else
        print(io, x)
    end
end

function main(out, which, nseeds, families=ALL_FAMILIES)
    regimes = [
        ("slow, moderately informative", (; σ=1.0, ℓs=0.5, ℓt=10.0, τ=0.2, σα=0.5)),
        ("slow, weakly informative", (; σ=1.0, ℓs=0.5, ℓt=10.0, τ=1.0, σα=0.5)),
        ("shorter, moderately informative", (; σ=1.0, ℓs=0.2, ℓt=3.0, τ=0.2, σα=0.5)),
        ("slow, irregular times", (; σ=1.0, ℓs=0.5, ℓt=10.0, τ=0.2, σα=0.5)),
    ]
    results = []
    for (label, truth) in regimes[which], seed in 1:nseeds
        t = @elapsed r = run_regime(label, truth; seed, families, irregular=occursin("irregular", label))
        push!(results, r)
        println("== ", label, " seed=", seed, " nz=", r.nz, " read states=", r.nread, " (", round(t; digits=1), " s)")
        @printf("  %-15s %9s %8s %24s %24s %24s %7s\n", "family", "KL", "mean err", "sd ratio read states", "sd ratio unread states", "sd ratio latent reads", "cover")
        for f in r.fixed
            @printf("  %-15s %9.1f %8.3f   %5.2f [%5.2f,%5.2f]    %5.2f [%5.2f,%5.2f]    %5.2f [%5.2f,%5.2f]  %6.3f\n", f.family, f.kl, f.mean_error,
                f.sd_ratio_read_states.median, f.sd_ratio_read_states.lo, f.sd_ratio_read_states.hi,
                f.sd_ratio_unread_states.median, f.sd_ratio_unread_states.lo, f.sd_ratio_unread_states.hi,
                f.sd_ratio_latent_reads.median, f.sd_ratio_latent_reads.lo, f.sd_ratio_latent_reads.hi, f.coverage90)
        end
        @printf("  %-15s %8s %8s %8s %8s   (truth σ=%.2f ℓs=%.2f ℓt=%.2f τ=%.2f)\n", "θ̂ family", "σ", "ℓs", "ℓt", "τ", truth.σ, truth.ℓs, truth.ℓt, truth.τ)
        for f in r.fits
            @printf("  %-15s %8.3f %8.3f %8.3f %8.3f\n", f.family, f.σ, f.ℓs, f.ℓt, f.τ)
        end
        flush(stdout)
    end
    open(io -> json(io, results), out, "w")
    println("wrote ", out)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(isempty(ARGS) ? joinpath(tempdir(), "family_oracle.json") : ARGS[1],
    length(ARGS) >= 2 ? (parse(Int, ARGS[2]):parse(Int, ARGS[2])) : (1:3),
    length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3,
    length(ARGS) >= 4 ? Tuple(Symbol.(split(ARGS[4], ","))) : ALL_FAMILIES)
