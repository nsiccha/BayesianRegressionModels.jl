# Sum cells share the ordinary sparse engine's mode search and row terms.
# Only curvature storage, factorization and the sum-factor chain rule differ.
function _brm_spl_precision(p::BRMSparseLaplaceProblem)
    D = copy(p.pattern)
    any(g -> g isa _BRMSumCellGroup, p.cells) || return D
    isdefined(SelectedInversion, :LUSelInvWorkspace) || _brm_nlme_error(
        "sum-cell Laplace needs SelectedInversion's pivoted KLU API; " *
        "pin fleet revision 8b1ee1077cea94c1aff746e182681ae39ce91526 or a later release carrying it")
    _BRMSumPrecision(D, spzeros(0, size(D, 1)), spzeros(0, 0), Any[])
end
_brm_spl_finite(h::SparseMatrixCSC) = all(isfinite, nonzeros(h))
_brm_spl_finite(h::_BRMSumPrecision) = all(_brm_spl_finite, (h.D, h.B, h.W))

# A finite density does not imply finite derivatives (for example an
# overflowing inner exponential whose outer exponential saturates at zero).
# Attribute invalid derivatives to the blocks they touch so backtracking a
# bad trial never shortens an unrelated block's step. Inspect sparse storage,
# including the augmented sum factors, without forming B' W B.
function _brm_spl_finite_blocks(p, g, h)
    valid = trues(length(p.sblocks))
    for i in eachindex(g)
        isfinite(g[i]) || (valid[p.sblock[i]] = false)
    end
    _brm_spl_finite_blocks!(valid, p, h)
    valid
end
function _brm_spl_finite_blocks!(valid, p, h::SparseMatrixCSC)
    for j in axes(h, 2), k in nzrange(h, j)
        if !isfinite(nonzeros(h)[k])
            valid[p.sblock[j]] = false
            valid[p.sblock[rowvals(h)[k]]] = false
        end
    end
    valid
end
function _brm_spl_finite_blocks!(valid, p, h::_BRMSumPrecision)
    _brm_spl_finite_blocks!(valid, p, h.D)
    for j in axes(h.B, 2), k in nzrange(h.B, j)
        isfinite(nonzeros(h.B)[k]) || (valid[p.sblock[j]] = false)
    end
    # Each outer curvature block belongs to one cell. Its nonfinite entries
    # invalidate every step block that supplies that cell's integrated inputs.
    offset = 0
    for piece in h.parts
        for (i, c) in enumerate(piece.group.cells)
            cols = (piece.offset + piece.part.rowptr[i]):(piece.offset + piece.part.rowptr[i + 1] - 1)
            if any(j -> any(k -> !isfinite(nonzeros(h.W)[k]), nzrange(h.W, j)), cols)
                for z in c.lz
                    valid[p.sblock[z]] = false
                end
            end
        end
        offset += size(piece.part.W, 1)
    end
    for (j, blocks) in enumerate(p.wsblocks)
        if !isfinite(h.W[offset + j, offset + j])
            valid[blocks] .= false
        end
    end
    valid
end

_brm_spl_copy_precision!(h::SparseMatrixCSC, trial::SparseMatrixCSC) =
    copyto!(nonzeros(h), nonzeros(trial))
function _brm_spl_copy_precision!(h::_BRMSumPrecision, trial::_BRMSumPrecision)
    _brm_spl_copy_precision!(h.D, trial.D)
    h.B = trial.B; h.W = trial.W; h.parts = trial.parts
    h
end

_brm_spl_precision_block(h::SparseMatrixCSC, I) = Matrix(Symmetric(_brm_spl_entries(h, I, I)))
function _brm_spl_precision_block(h::_BRMSumPrecision, I)
    rows = sort!(unique!([rowvals(h.B)[k] for j in I for k in nzrange(h.B, j)]))
    B = _brm_spl_entries(h.B, rows, I)
    W = _brm_spl_submatrix(h.W, rows, rows)
    Matrix(Symmetric(_brm_spl_entries(h.D, I, I) + B' * W * B))
end

# Cache baseline sums and incidence once. A block diagnostic changes only
# the summands reading that block, rather than re-evaluating a large cell
# once per level. The prior's edges and soft constraints use the same rule.
function _brm_plan_local_cache(p, state)
    cells = Dict(c.term => (group, c) for group in p.cells for c in group.cells)
    sums = Dict{Int,Any}()
    for group in p.cells
        group isa _BRMSumCellGroup || continue
        for c in group.cells
            entries = [(; t, value=Base.invokelatest(group.prepared.q.terms[t.j].value,
                state.u, c.i, t.t)) for reduction in c.terms for t in reduction]
            by_effect = Dict{Int,Vector{Int}}()
            for (k, entry) in enumerate(entries), i in entry.t.lz
                push!(get!(by_effect, i, Int[]), k)
            end
            values = _brm_spl_factor_values(group.prepared, state.u, c).sums
            sums[c.term] = (; entries, by_effect, values)
        end
    end
    n = length(p.integrated)
    identity = falses(n)
    edges = Tuple{Int,Int}[]
    edges_by_effect = [Int[] for _ in 1:n]
    for b in p.blocks
        if _brm_spl_identity(b)
            identity[b.z] .= true
        else
            for e in axes(b.edges, 2)
                i, j = b.z[b.edges[1, e], 1], b.z[b.edges[2, e], 1]
                push!(edges, (i,j))
                push!(edges_by_effect[i], length(edges))
                push!(edges_by_effect[j], length(edges))
            end
        end
    end
    soft_by_effect = [Int[] for _ in 1:n]
    for (c, cols) in enumerate(p.wcols), i in cols
        push!(soft_by_effect[i], c)
    end
    (; cells, sums, identity, edges, edges_by_effect, soft_by_effect,
        gmrf=_brm_spl_gmrf_sparse(p, n),
        soft_members=Set.(p.wcols), soft_values=[sum(state.z[cols]) for cols in p.wcols],
        mmax=maximum(t -> p.tptr[t + 1] - p.tptr[t], eachindex(p.family); init=0))
end
_brm_plan_cell_value(group, c, I, u, cache) = _rk_nlme_cell_value(group.query, u, c.i)
function _brm_plan_cell_value(group::_BRMSumCellGroup, c, I, u, cache)
    cached = cache[c.term]
    sums = copy(cached.values)
    for k in unique(k for i in I for k in get(cached.by_effect, i, Int[]))
        (; t, value) = cached.entries[k]
        sums[t.j] += Base.invokelatest(group.prepared.q.terms[t.j].value, u, c.i, t.t) - value
    end
    Base.invokelatest(group.prepared.q.outer.value, u, sums, c.i)
end

# `weight` scales the cells' likelihood (values, gradient, local curvature D
# and outer curvature W) as in `_brm_spl_assemble!`; the soft constraints
# below are prior and keep their weight.
function _brm_spl_assemble_sums!(fc, g, h, p, u; derivatives, third, fb, tv,
        correction=nothing, weight::Float64=1.0)
    rest = 0.0
    parts = Any[]; jacobians = SparseMatrixCSC{Float64,Int}[]
    outers = SparseMatrixCSC{Float64,Int}[]
    offset = 0
    for group in (weight == 0 ? empty(p.cells) : p.cells)
        group isa _BRMSumCellGroup || continue
        if derivatives
            part = _brm_spl_factor_curvature(group.prepared, u,
                length(p.integrated), length(p.population))
            g .+= weight .* part.gradient
            # Every local summand clique is already in the fixed D pattern.
            for j in axes(part.D, 2), k in nzrange(part.D, j)
                h.D[rowvals(part.D)[k], j] += weight * nonzeros(part.D)[k]
            end
            push!(parts, (; group, part, offset))
            push!(jacobians, part.B); push!(outers, weight == 1 ? part.W : weight * part.W)
            offset += size(part.B, 1)
        end
        for (k, c) in enumerate(group.cells)
            comp = p.term_component[c.term]
            comp == 0 && !third && continue
            value = derivatives ? part.states[k].value :
                _brm_spl_factor_values(group.prepared, u, c).value
            tv === nothing || (tv[c.term] = value)
            if comp == 0
                rest += value
            else
                _brm_spl_add_value!(fc, correction, comp, weight * value)
                if fb !== nothing
                    for a in p.tsbptr[c.term]:(p.tsbptr[c.term + 1] - 1)
                        fb[p.tsb[a]] += weight * value
                    end
                end
            end
        end
    end
    if derivatives && h isa _BRMSumPrecision
        # ICAR soft constraints are also low-rank terms. Including them here
        # handles a singular local D without a Woodbury inverse of D.
        if !isempty(p.wcols)
            bi = [k for (k, cols) in enumerate(p.wcols) for _ in cols]
            bj = [i for cols in p.wcols for i in cols]
            push!(jacobians, sparse(bi, bj, ones(length(bi)), length(p.wcols), length(g)))
            push!(outers, spdiagm(0 => p.wλ))
        end
        # Sparse vcat discards explicit zeros. Those entries request S*B'
        # for the third derivative even when the current Jacobian vanishes.
        bi = Int[]; bj = Int[]; bv = Float64[]; offset = 0
        for B in jacobians
            for j in axes(B, 2), k in nzrange(B, j)
                push!(bi, offset + rowvals(B)[k]); push!(bj, j)
                push!(bv, nonzeros(B)[k])
            end
            offset += size(B, 1)
        end
        h.B = sparse(bi, bj, bv, offset, length(g))
        h.W = blockdiag(outers...)
        h.parts = parts
    end
    rest
end

_brm_spl_woodbury(p::BRMSparseLaplaceProblem, F::_BRMAugmentedFactor) = nothing
function _brm_spl_factor!(p::BRMSparseLaplaceProblem, h::_BRMSumPrecision; shift=0.0)
    p.algebra === nothing || return _brm_operator_factor(p, h; shift)
    D = copy(h.D)
    nonzeros(D)[p.diagonal] .+= _brm_spl_effect_shift(p, shift)
    K = _brm_spl_augmented_matrix(D, h.B, h.W)
    F = _brm_spl_augmented_factor(K, length(p.integrated); previous=p.factor[])
    p.factor[] = F
    F
end

function _brm_spl_selected_precision(p::BRMSparseLaplaceProblem, F)
    nonzeros(selinv_extract(selinv(F; depermute=true).Z, p.pattern))
end
function _brm_spl_selected_precision(p::BRMSparseLaplaceProblem, F::_BRMAugmentedFactor)
    selected = _brm_spl_selected(F)
    [selected[rowvals(p.pattern)[k], j] for j in axes(p.pattern, 2)
        for k in nzrange(p.pattern, j)]
end

function _brm_spl_sum_gradient!(s, gradient, p, state, u, on)
    state.h isa _BRMSumPrecision || return
    selected = state.F isa _BRMOperatorFactor ? nothing : _brm_spl_selected(state.F)
    n = length(s); r = size(state.h.B, 1)
    scratch = zeros(n); tz = zeros(n); tp = zeros(length(gradient))
    for (; group, part, offset) in state.h.parts
        for (k, c) in enumerate(group.cells)
            on(c.term) || continue
            cs = part.states[k]
            _brm_spl_factor_pullback!(scratch, gradient, c, cs, cs.gf)
            rows = (offset + part.rowptr[k]):(offset + part.rowptr[k + 1] - 1)
            cov = state.F isa _BRMOperatorFactor ?
                _brm_operator_sum_covariances(p, state.F, c) :
                _brm_spl_factor_covariances(c, rows, selected, n, r)
            _brm_spl_factor_contraction!(tz, tp, group.prepared, u, c, cs,
                cov.blocks, cov.SB, cov.BSB)
        end
    end
    s .+= tz ./ 2
    gradient .+= tp ./ 2
end

function _brm_spl_sum_implicit!(gradient, p, state, v, on)
    state.h isa _BRMSumPrecision || return
    z = zeros(length(v))
    for (; group, part) in state.h.parts, (k, c) in enumerate(group.cells)
        on(c.term) || continue
        _brm_spl_factor_hvp!(z, gradient, c, part.states[k], v)
    end
end

# A single batched selected-inverse request for all effect blocks, including
# blocks absent from the likelihood pattern. No one-hot solve per level.
function _brm_spl_sum_inverse_blocks(F::_BRMAugmentedFactor, sets)
    ii = [i for set in sets for j in set for i in set]
    jj = [j for set in sets for j in set for i in set]
    request = sparse(ii, jj, ones(length(ii)), size(F.K)...)
    selected = selinv(F.F, request)
    [Matrix(Symmetric(_brm_spl_entries(selected, set, set))) for set in sets]
end

function _brm_spl_component_logdets(p::BRMSparseLaplaceProblem, F::_BRMAugmentedFactor)
    n = F.n; r = (size(F.K, 1) - n) ÷ 2
    labels = vcat(p.component, zeros(Int, 2r))
    for j in 1:n, k in nzrange(F.K, j)
        i = rowvals(F.K)[k]
        i > n && (labels[i] = p.component[j])
    end
    out = zeros(length(p.components))
    # KLU's scaled identity is L*U+F = Diagonal(1 ./ Rs)*K[p,q].
    # Absolute determinants ignore permutation parity. Pivots remain within
    # a connected component; BTF off-diagonal blocks do not affect det(U).
    d = diag(F.F.U)
    for (k, i) in enumerate(F.F.p)
        c = labels[i]
        c == 0 || (out[c] += log(abs(d[k])) + log(F.F.Rs[k]))
    end
    out
end

function _brm_spl_block_direction(p, h::SparseMatrixCSC, members, g)
    H = Matrix(h[members, members])
    for (w, cols) in enumerate(p.wcols)
        inside = findall(in(Set(cols)), members)
        isempty(inside) || (H[inside, inside] .+= p.wλ[w])
    end
    H = Symmetric(H)
    F = cholesky(H; check=false)
    μ = 1e-4
    while !issuccess(F) && μ < 1e9
        F = cholesky(H + μ * I; check=false)
        μ *= 10
    end
    issuccess(F) ? F \ g : nothing
end
# Component `members`' augmented block with D + μ I, restricted like
# `_brm_spl_block_direction` below: whether it is positive definite.
function _brm_spl_block_definite(p::BRMSparseLaplaceProblem, h::_BRMSumPrecision, members,
        μ)
    rows = sort!(unique!([rowvals(h.B)[k] for j in members for k in nzrange(h.B, j)]))
    D = _brm_spl_submatrix(h.D, members, members)
    B = _brm_spl_submatrix(h.B, rows, members)
    W = _brm_spl_submatrix(h.W, rows, rows)
    issuccess(_brm_spl_augmented_factor(_brm_spl_augmented_matrix(D + μ * I, B, W),
        length(members)))
end
function _brm_spl_block_direction(p, h::_BRMSumPrecision, members, g)
    # Auxiliary rows that do not read this block have zero Jacobian here.
    # Restrict before factoring, including when the original W couples rows.
    rows = sort!(unique!([rowvals(h.B)[k] for j in members for k in nzrange(h.B,j)]))
    D = _brm_spl_submatrix(h.D, members, members)
    B = _brm_spl_submatrix(h.B, rows, members)
    W = _brm_spl_submatrix(h.W, rows, rows)
    μ = 0.0
    for _ in 1:14
        K = _brm_spl_augmented_matrix(D + μ * I, B, W)
        F = _brm_spl_augmented_factor(K, length(members))
        issuccess(F) && return F \ g
        μ = μ == 0 ? 1e-4 : 10μ
    end
    nothing
end
