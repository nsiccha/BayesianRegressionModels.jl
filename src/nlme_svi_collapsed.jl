# ── Sparse globals with their readers integrated out (`sparse_globals = :collapsed`) ──
#
# A sparse global (src/nlme_svi_sites.jl: a window's boundary state, a read-pair
# `gmrf` field state, a separator) steps by default on one diagonal site per
# reader, measured with the reader's own coordinates held at its factor. Where the
# data pin a sparse global only together with its readers' coordinates (a field
# state plus each window's level, a boundary plus the window's path), that is
# block coordinate ascent: units against sparse globals, and each sparse global
# against the others, and it crawls. On a space-time toy of 2,048 window units with
# 2,192 sparse globals (snag brm-svi-fit-gmrf-ab199f57; exact Gaussian at fixed
# hyperparameters) an exact emulation of today's sites closed 1.1% of the gap to
# the exact means per pass over the data, and per-coordinate sites with the units
# integrated out 3.1%. Joint sites with the units integrated out and a joint
# update of the sparse globals were exact once every unit had been visited. A
# diagonal factor there also put the field states' sds at 0.20-0.35 of their exact
# marginal sds, so the sparse globals take a joint factor too.
#
# Under `:collapsed` each reader i keeps one Gaussian site over all the sparse
# globals it reads, g_i (k_i of them), with its own coordinates z_i integrated out:
#
#     site_i(g_i) ∝ exp(ν_iᵀ g_i - g_iᵀ Λ_i g_i / 2),
#     Λ_i = H_gg - H_gz H_zz⁻¹ H_zg,      (the Schur complement of the unit's
#                                          curvature over [z_i; g_i])
#     gradient at the visit's point = G_g - H_gz H_zz⁻¹ G_z  (the profile gradient),
#
# for H the negative Hessian of the unit's term and G its gradient there. H_zz is
# the curvature the orthogonal design already estimates (`S`, whitened: H_zz =
# U S Uᵀ for the factor's root U); k_i antithetic probes along the unit's sparse
# globals, at the unit's mean, give H_gg and H_zg (2k_i more unit evaluations). A
# unit with no whole curvature (a structured unit, or the single pair) keeps a
# joint site conditional on its factor: Λ_i = H_gg, gradient G_g. On a term
# quadratic in (z_i, g_i) the site is exact, whatever the point. A site replaces the
# previous one by the unit's own step β (whole under the orthogonal design's
# default), as incremental EM stores and replaces each unit's statistics (Neal and
# Hinton 1998) and conjugate-computation VI takes its site steps (Khan and Lin
# 2017); like `src/nlme_svi_sites.jl` it damps the gradient at the current mean, not
# ν, so a site does not depend on where the coordinates' origin is.
#
# The sparse globals then take a joint Gaussian factor q(g_s) = N(m, P⁻¹) with the
# sparse precision P = H_prior + Σ_i Λ_i (Tan and Nott 2018, "Gaussian variational
# approximation with sparse precision matrices"; on a Gaussian model it is INLA's
# Gaussian of a latent field given its hyperparameters, Rue, Martino and Chopin
# 2009). H_prior is the global term's curvature over the sparse globals, at the
# dense globals' means, assembled from one Hessian-vector product per color of a
# star coloring of its structural pattern: the cliques of the model's density
# factors, a read-pair `gmrf` field's joint law replaced by its conditioning sets
# (`neighbors=`), so a field's block stays as sparse as its Vecchia precision. Each
# product is a central difference of the global term's gradient, exact on a
# Gaussian prior; it is refreshed once per stationarity window, as the dense
# globals move. Each iteration, after the batch's sites, the means take one Newton
# step on the global term plus the summed sites, P Δ = r, by a sparse Cholesky
# factorization of P, whose factor also draws g_s, gives the factor's entropy and,
# by selected inversion, the marginal sds that `q.ρ` reports. A Gaussian prior with
# exact sites gives the exact means and covariance in one step. The sparse globals'
# μ, ρ and rank rows are not stepped on any gradient.
#
# A reader stays coupled to its sparse globals (`LinearCoupling`), with its slope
# set from the same curvature, -H_zz⁻¹ H_zg per unit of the global's deviation, on
# every visit from the first: its conditional mean then follows the sparse globals'
# draw. That is what `_brm_svi_uncouple_sparse!` gives up under `:sites`. The slope
# is computed, not learned, so it does not wait for exploration's end as the
# learned columns do: held at zero, a unit's mean is re-solved at each drawn value
# of its sparse globals and jitters with their draws (on the test toy, unit means
# 0.14 exact sds off after 20 passes while the sparse means were exact).
#
# Learned centering holds its frame of cell locations and scales fixed in these
# probes, so `:collapsed` refuses a model whose centred cells read a sparse global
# (a stratum's sd few units read), naming it; `centering = nothing` or
# `sparse_globals = :sites` fits such a model. A joint law over many sparse globals
# other than a read-pair field (a `car` field, a `multiresolution=` basis) enters
# H_prior as one dense block.

const _BRM_SVI_COLLAPSED_PROBE = 1e-2     # finite-difference step, in current global sds
const _BRM_SVI_COLLAPSED_HALVINGS = 20    # step halvings before a non-finite step is skipped

# The current sparse factor of P, kept for the next iteration's draws. Workspace: a
# copy or a serialized run starts empty and factors P again from its stored values.
mutable struct _BRMSVIFactorCache
    factor::Any
    _BRMSVIFactorCache() = new(nothing)
end
Base.deepcopy_internal(::_BRMSVIFactorCache, ::IdDict) = _BRMSVIFactorCache()
Serialization.serialize(s::Serialization.AbstractSerializer, ::_BRMSVIFactorCache) =
    Serialization.serialize_type(s, _BRMSVIFactorCache)
Serialization.deserialize(::Serialization.AbstractSerializer, ::Type{_BRMSVIFactorCache}) =
    _BRMSVIFactorCache()

mutable struct _BRMSVICollapsedSites
    base::_BRMSVISites                     # which globals are sparse; each unit's entries
    slots::Vector{Vector{Int}}             # unit i's sparse slots, in entry order
    precision::Vector{Matrix{Float64}}     # Λ_i
    pull::Vector{Vector{Float64}}          # ν_i
    seen::BitVector
    sum::SparseMatrixCSC{Float64,Int}      # Σ_i Λ_i on P's pattern
    places::Vector{Matrix{Int}}            # unit i's entries' positions in `sum.nzval`
    pull_sum::Vector{Float64}              # Σ_i ν_i
    prior::Vector{Float64}                 # H_prior on P's pattern
    directions::Vector{Vector{Int}}        # the prior pattern's colors, as slots
    sources::Vector{NTuple{3,Int}}         # per entry of P: (product row, color, scale slot)
    refresh::Int                           # solves between H_prior assemblies
    prior_solve::Int                       # the solve that last assembled H_prior (-1: none)
    logdet::Float64                        # log det P after the last solve
    solves::Int
    prior_assemblies::Int
    fallbacks::Int                         # P not positive definite: its sites' part used
    skipped::Int
    global_term_calls::Int
    cache::_BRMSVIFactorCache
end

_brm_svi_sparse_mode(c) = get(c, :sparse_globals, :sites)

# The structural cliques of the global term over the sparse slots: each density
# factor's support, a read-pair field's joint law replaced by its conditionals.
_brm_svi_collapsed_cliques(t, base) = hasproperty(t, :m) ?
    _brm_svi_collapsed_cliques(t, base, t.m) : Vector{Int}[]
function _brm_svi_collapsed_cliques(t, base, m)
    slot = Dict{Int,Int}(t.globals[p] => s for (s, p) in enumerate(base.globals))
    fields = _brm_svi_footprint_fields(m)
    factors = _brm_svi_strip_fields(_rk_svi_factors(m.backend, m.unit), fields)
    cliques = Vector{Int}[]
    keep!(coordinates) = (c = unique!([slot[j] for j in coordinates if haskey(slot, j)]);
        length(c) > 1 && push!(cliques, c))
    foreach(keep!, factors.support)
    for field in fields
        f, C = field.footprint, field.footprint.nchannels
        for s in eachindex(f.position)
            states = vcat(s, f.neighbor[f.neighbor_ptr[s]:(f.neighbor_ptr[s + 1] - 1)])
            keep!([field.positions[C * (Int(x) - 1) + c] for x in states for c in 1:C])
        end
    end
    cliques
end

# A model's read-pair `gmrf` fields and their packed coordinates, as
# `_brm_svi_field_inventory` reads them; a field whose coordinates do not line up
# stays one factor.
function _brm_svi_footprint_fields(m)
    names = m.view.coordinates
    fields = NamedTuple[]
    for record in _brm_nlme_records(m.backend.plan)
        record.kind === :gmrf_footprint || continue
        support = only(m.backend.plan.columns[record.fitted])
        base = Symbol(record.declaration, ".", record.locals.field)
        positions = findall(name -> _brm_nlme_owned(name, base), names)
        length(positions) == support.geometry.ncoordinates || continue
        push!(fields, (; record, footprint=support.geometry, positions))
    end
    Tuple(fields)
end

# The diagonal and every pair within each clique, as row and column indices.
function _brm_svi_clique_pairs(n, cliques)
    rows, columns = collect(1:n), collect(1:n)
    for c in cliques, a in c, b in c
        push!(rows, a); push!(columns, b)
    end
    rows, columns
end

# P's pattern (the prior's cliques, the readers' blocks and the diagonal), a star
# coloring of the prior's, and where each entry of P reads its prior value.
function _brm_svi_collapsed_sites(base::_BRMSVISites, t; refresh=100)
    n, N = length(base.globals), length(base.entries)
    slots = [[base.slot[_brm_svi_unit_touched(t, i)[e]] for e in base.entries[i]] for i in 1:N]
    rows₁, columns₁ = _brm_svi_clique_pairs(n, _brm_svi_collapsed_cliques(t, base))
    prior = sparse(rows₁, columns₁, trues(length(rows₁)), n, n, |)
    rows₂, columns₂ = _brm_svi_clique_pairs(n, slots)
    S = sparse(vcat(rows₁, rows₂), vcat(columns₁, columns₂),
        zeros(length(rows₁) + length(rows₂)), n, n)
    result = SparseMatrixColorings.coloring(prior,
        SparseMatrixColorings.ColoringProblem(; structure=:symmetric, partition=:column),
        SparseMatrixColorings.GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst();
            decompression=:direct))
    colors = n == 0 ? Int[] : SparseMatrixColorings.column_colors(result)
    ncolors = maximum(colors; init=0)
    # How many neighbours (the entry itself included) of each slot carry each color:
    # an entry is read where its column's color is unique.
    multiplicity = [Dict{Int,Int}() for _ in 1:n]
    for a in 1:n, k in nzrange(prior, a)
        color = colors[rowvals(prior)[k]]
        multiplicity[a][color] = get(multiplicity[a], color, 0) + 1
    end
    sources = NTuple{3,Int}[]
    for b in 1:n, k in nzrange(S, b)
        a = rowvals(S)[k]
        if !prior[a, b]
            push!(sources, (0, 0, 0))
        elseif multiplicity[a][colors[b]] == 1
            push!(sources, (a, colors[b], b))
        elseif multiplicity[b][colors[a]] == 1
            push!(sources, (b, colors[a], a))
        else
            error("brm_svi_fit: internal: SparseMatrixColorings returned a coloring " *
                "that is not a star coloring; entry ($a, $b) has no direct source")
        end
    end
    rows = rowvals(S)
    place(a, b) = (r = nzrange(S, b); r[searchsortedfirst(view(rows, r), a)])
    places = [[place(a, b) for a in s, b in s] for s in slots]
    _BRMSVICollapsedSites(base, slots, [zeros(length(s), length(s)) for s in slots],
        [zeros(length(s)) for s in slots], falses(N), S, places, zeros(n),
        zeros(SparseArrays.nnz(S)), [findall(==(c), colors) for c in 1:ncolors], sources,
        max(Int(refresh), 1), -1, 0.0, 0, 0, 0, 0, 0, _BRMSVIFactorCache())
end

_brm_svi_minibatch_global(sites::_BRMSVICollapsedSites, p) = sites.base.slot[p] == 0
_brm_svi_update_sites!(::_BRMSVICollapsedSites, t, q, g, ε, batch, terms, β, scale) = nothing
# The centring chain through a sparse global is zero: `_brm_svi_collapsed_check_centering`.
_brm_svi_centering_site_terms!(terms, ::_BRMSVICollapsedSites, plan, t, frame, centered,
    batch, weights, chain) = terms

# Learned centering's frame must not read a sparse global (see the header): one
# pullback of unit cotangents on every port, at the starting globals, finds any
# global the ports read.
_brm_svi_collapsed_check_centering(sites, t, q, ::Nothing) = nothing
function _brm_svi_collapsed_check_centering(sites, t, q, plan)
    plan.pullback === nothing && return nothing
    frame = _brm_svi_centering_frame(plan, t, q.μ)
    cot = Any[x isa Number ? 1.0 : ones(Float64, size(x)) for x in frame.values]
    reach = plan.pullback(frame.point, Tuple(cot))[t.globals]
    read = [p for p in sites.base.globals if reach[p] != 0]
    isempty(read) && return nothing
    names = _brm_svi_coordinate_names(t, read)
    throw(ArgumentError("brm_svi_fit: sparse_globals = :collapsed holds learned " *
        "centering's frame fixed in its curvature probes, and centred cells read " *
        "$(length(read)) sparse global(s) as a location or scale ($(join(first(names, 5), ", "))" *
        "$(length(names) > 5 ? ", …" : "")); pass centering = nothing, or " *
        "sparse_globals = :sites"))
end
function _brm_svi_coordinate_names(t, positions)
    names = hasproperty(t, :m) ? t.m.view.coordinates : nothing
    names === nothing && return string.(t.globals[positions])
    [string(names[t.globals[p]]) for p in positions]
end

# 2k evaluations per visit of a unit that reads k sparse globals.
_brm_svi_collapsed_unit_calls(sites, i) = 0
_brm_svi_collapsed_unit_calls(sites::_BRMSVICollapsedSites, i) = 2length(sites.slots[i])
_brm_svi_collapsed_calls(sites, t, batch) =
    sum(i -> _brm_svi_collapsed_unit_calls(sites, i), batch; init=0)

# The unit's term and its gradients at (g, z), as the iteration evaluates it.
function _brm_svi_collapsed_term!(gg, gz, t, q, i, g, z, plan, work, centering, frame)
    centering === nothing && return _brm_svi_unit_local!(gg, gz, t, i, g, z, plan, work)
    first(_brm_svi_centered_unit_term!(gg, gz, centering, frame, t, q, i, g, z, plan, work))
end

# Unit i's collapsed site at the iteration's draw `g`, its mean `zmean`, and the
# gradients `Gg` (at its sparse entries) and `Gz` the iteration read there; `S` is
# the orthogonal design's whitened curvature, or `nothing` for a unit without one.
# Returns `nothing` when a probe is not finite: the unit's site then keeps its last
# value, and the iteration's own terms report a non-finite term.
function _brm_svi_collapsed_probe(sites, t, q, i, g, zmean, plan, work, D, S, Gg, Gz;
        centering=nothing, frame=nothing)
    touched = _brm_svi_unit_touched(t, i)
    entries = sites.base.entries[i]
    k, d = length(entries), length(zmean)
    positions = [touched[e] for e in entries]
    Hgg, Hzg = zeros(k, k), zeros(d, k)
    gg₊, gg₋ = _brm_svi_global_buffers(work, D)
    gz₊, gz₋ = zeros(d), zeros(d)
    finite = true
    for (a, p) in enumerate(positions)
        h = _BRM_SVI_COLLAPSED_PROBE * exp(q.ρ[p])
        g₊, g₋ = copy(g), copy(g)
        g₊[p] += h; g₋[p] -= h
        fill!(gz₊, 0.0); fill!(gz₋, 0.0)
        v₊ = _brm_svi_collapsed_term!(gg₊, gz₊, t, q, i, g₊, zmean, plan, work, centering, frame)
        v₋ = _brm_svi_collapsed_term!(gg₋, gz₋, t, q, i, g₋, zmean, plan, work, centering, frame)
        for (b, e) in enumerate(entries)
            Hgg[b, a] = -(gg₊[touched[e]] - gg₋[touched[e]]) / 2h
        end
        Hzg[:, a] .= .-(gz₊ .- gz₋) ./ 2h
        for p′ in touched
            gg₊[p′] = gg₋[p′] = 0.0
        end
        finite &= isfinite(v₊) && isfinite(v₋)
    end
    finite && all(isfinite, Hgg) && all(isfinite, Hzg) || return nothing
    Hgg = (Hgg .+ Hgg') ./ 2
    if S === nothing || d == 0
        Λ, G, C = Hgg, Gg, nothing
    else
        # H_zz⁻¹ H_zg = U⁻ᵀ S₊⁺ U⁻¹ H_zg, on the positive part of the estimate.
        U = LowerTriangular(q.U[i])
        F = eigen(Symmetric((S .+ S') ./ 2))
        keep = F.values .> sqrt(eps()) * max(maximum(F.values; init=0.0), 0.0)
        V, w = F.vectors[:, keep], F.values[keep]
        solve(X) = U' \ (V * (Diagonal(1 ./ w) * (V' * (U \ X))))
        C = solve(Hzg)
        Λ = Hgg .- Hzg' * C
        G = Gg .- C' * Gz
    end
    all(isfinite, Λ) && all(isfinite, G) || return nothing
    Λ₊ = Matrix(_brm_svi_positive_part(Symmetric((Λ .+ Λ') ./ 2)))
    (; Λ=Λ₊, G, point=g[positions], C)
end

# Replaces each probed batch unit's site by step β (whole on a first visit), in
# batch order, damping the gradient at the current means as src/nlme_svi_sites.jl
# does.
function _brm_svi_collapsed_update!(sites::_BRMSVICollapsedSites, q, batch, probes, β)
    nz = nonzeros(sites.sum)
    for (k, i) in enumerate(batch)
        probe = probes[k]
        probe === nothing && continue
        slots = sites.slots[i]
        positions = sites.base.globals[slots]
        μ = q.μ[positions]
        Λ, ν = sites.precision[i], sites.pull[i]
        b = sites.seen[i] ? β : 1.0
        old = ν .- Λ * μ
        new = probe.G .+ probe.Λ * (probe.point .- μ)
        Λ′ = (1 - b) .* Λ .+ b .* probe.Λ
        ν′ = (1 - b) .* old .+ b .* new .+ Λ′ * μ
        places = sites.places[i]
        for c in axes(Λ′, 2), r in axes(Λ′, 1)
            nz[places[r, c]] += Λ′[r, c] - Λ[r, c]
        end
        sites.pull_sum[slots] .+= ν′ .- ν
        Λ .= Λ′
        ν .= ν′
        sites.seen[i] = true
    end
    sites
end

# The sparse globals' μ, ρ and rank rows are set by the factor, not stepped: their
# entries of the gradient of [μ; ρ; Β] are zeroed, so neither Adam nor the whitened
# step moves them.
function _brm_svi_add_site_gradient!(grad, sites::_BRMSVICollapsedSites, q)
    D, r = length(q.μ), size(q.Β, 2)
    for p in sites.base.globals, k in 0:(r + 1)
        grad[k * D + p] = 0.0
    end
    grad
end

# H_prior on P's pattern: for each color, the central difference of the global
# term's gradient at the sparse means ± h s on the color's slots (s the current
# sds), its product row read at each entry the color is unique for.
function _brm_svi_collapsed_prior!(sites::_BRMSVICollapsedSites, t, q)
    globals = sites.base.globals
    s = exp.(q.ρ[globals])
    h = _BRM_SVI_COLLAPSED_PROBE
    products = map(sites.directions) do slots
        x₊, x₋ = copy(q.μ), copy(q.μ)
        for a in slots
            x₊[globals[a]] += h * s[a]
            x₋[globals[a]] -= h * s[a]
        end
        _, g₊ = _brm_svi_global_term(t, x₊)
        _, g₋ = _brm_svi_global_term(t, x₋)
        sites.global_term_calls += 2
        .-(view(g₊, globals) .- view(g₋, globals)) ./ 2h
    end
    prior = similar(sites.prior)
    for (k, (row, color, scale)) in enumerate(sites.sources)
        prior[k] = row == 0 ? 0.0 : products[color][row] / s[scale]
    end
    all(isfinite, prior) || return false
    # The two triangles' reads of an entry agree on a quadratic; keep P symmetric.
    S = sites.sum
    rows = rowvals(S)
    for b in axes(S, 2), k in nzrange(S, b)
        a = rows[k]
        r = nzrange(S, a)
        mirror = r[searchsortedfirst(view(rows, r), b)]
        sites.prior[k] = (prior[k] + prior[mirror]) / 2
    end
    sites.prior_assemblies += 1
    true
end

# P on its pattern, and its Cholesky factor; where P is not positive definite (a
# global term that is not log-concave there, or a sparse global with neither a
# prior curvature nor a site yet), the factor of the sites' part plus the prior's
# diagonal clipped at zero and a jitter of 1e-8 of the mean diagonal.
function _brm_svi_collapsed_factor(sites::_BRMSVICollapsedSites)
    S = sites.sum
    P = SparseMatrixCSC(size(S)..., S.colptr, S.rowval, sites.prior .+ nonzeros(S))
    F = cholesky(Symmetric(P); check=false)
    issuccess(F) && return F, false
    d = [max(P[a, a] - S[a, a], 0.0) + S[a, a] for a in axes(P, 1)]
    jitter = 1e-8 * max(sum(d) / max(length(d), 1), eps())
    F = cholesky(Symmetric(S + spdiagm(0 => d .- diag(S) .+ jitter)); check=false)
    issuccess(F) ? (F, true) : (nothing, true)
end

# One Newton step of the sparse globals' means on the global term plus the summed
# sites, the dense globals at their means, by the factor of P; then the factor's
# entropy and marginal sds.
function _brm_svi_collapsed_solve!(sites::_BRMSVICollapsedSites, t, q)
    globals = sites.base.globals
    isempty(globals) && return q
    if sites.prior_solve < 0 || sites.solves - sites.prior_solve >= sites.refresh
        _brm_svi_collapsed_prior!(sites, t, q) && (sites.prior_solve = sites.solves)
    end
    sites.solves += 1
    sites.cache.factor = nothing
    μ = copy(q.μ)
    v0, g0 = _brm_svi_global_term(t, μ)
    sites.global_term_calls += 1
    F, fallback = _brm_svi_collapsed_factor(sites)
    fallback && (sites.fallbacks += 1)
    if !(isfinite(v0) && all(isfinite, view(g0, globals))) || F === nothing
        sites.skipped += 1
        return q
    end
    r = g0[globals] .+ sites.pull_sum .- sites.sum * μ[globals]
    Δ = F \ r
    applied = false
    for _ in 0:_BRM_SVI_COLLAPSED_HALVINGS
        all(isfinite, Δ) || break
        x = copy(μ)
        x[globals] .+= Δ
        v, gx = _brm_svi_global_term(t, x)
        sites.global_term_calls += 1
        if isfinite(v) && all(isfinite, gx)
            q.μ[globals] .= view(x, globals)
            applied = true
            break
        end
        Δ ./= 2
    end
    applied || (sites.skipped += 1)
    q.ρ[globals] .= log.(sqrt.(SelectedInversion.selinv_diag(F)))
    sites.logdet = logdet(F)
    sites.cache.factor = F
    q
end

# The draw of the sparse globals from their joint factor, in place in `g`, from the
# same standard normals `ε` the diagonal draw used; returns the standard normals the
# coupling reads, the sparse globals' deviations in units of their marginal sds.
# Before the first solve the diagonal draw stands.
_brm_svi_sparse_draw!(g, ε, sites, q) = ε
function _brm_svi_sparse_draw!(g, ε, sites::_BRMSVICollapsedSites, q)
    sites.solves == 0 && return ε
    F = sites.cache.factor
    if F === nothing
        F, _ = _brm_svi_collapsed_factor(sites)
        F === nothing && return ε
        sites.cache.factor = F
    end
    globals = sites.base.globals
    x = F.UP \ ε[globals]
    g[globals] .= view(q.μ, globals) .+ x
    noise = copy(ε)
    noise[globals] .= x ./ exp.(view(q.ρ, globals))
    noise
end

# What the joint factor changes in the global factor's entropy: -log det P / 2 in
# place of the sparse globals' Σ log s.
_brm_svi_sparse_entropy(sites, q) = 0.0
_brm_svi_sparse_entropy(sites::_BRMSVICollapsedSites, q) = sites.solves == 0 ? 0.0 :
    -sites.logdet / 2 - sum(i -> q.ρ[i], sites.base.globals; init=0.0)

# Under linear coupling, each probed batch unit's slope on its sparse globals is
# -H_zz⁻¹ H_zg per unit of the global's deviation (the coupling reads the sparse
# globals' deviations in their marginal sds, `_brm_svi_sparse_draw!`), set on every
# visit.
_brm_svi_collapsed_couple!(sites, coupling, q, t, batch, probes) = nothing
function _brm_svi_collapsed_couple!(sites::_BRMSVICollapsedSites, coupling, q, t, batch,
        probes)
    coupling === nothing && return nothing
    for (k, i) in enumerate(batch)
        probe = probes[k]
        (probe === nothing || probe.C === nothing) && continue
        V, rows, columns = coupling.V[i], coupling.rows[i], coupling.columns[i]
        isempty(V) && continue
        positions = sites.base.globals[sites.slots[i]]
        for (a, p) in enumerate(positions)
            col = findfirst(==(p), columns)
            col === nothing && continue
            V[:, col] .= .-view(probe.C, rows, a) .* exp(q.ρ[p])
        end
    end
    nothing
end

function _brm_svi_sites_diagnostics(sites::_BRMSVICollapsedSites, t)
    merge(_brm_svi_sites_diagnostics(sites.base, t), (; mode=:collapsed,
        units_seen=count(sites.seen), solves=sites.solves,
        prior_colors=length(sites.directions), prior_assemblies=sites.prior_assemblies,
        precision_nonzeros=SparseArrays.nnz(sites.sum), fallbacks=sites.fallbacks,
        skipped_steps=sites.skipped, global_term_calls=sites.global_term_calls))
end
