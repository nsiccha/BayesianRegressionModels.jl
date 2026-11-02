# Sum-factored likelihood cells. Each summand keeps its own compact packed
# selection; the outer density sees the sums and its direct coordinates.
# These preparation records are independent of the Newton/factor workspace.
function _rk_spl_factor_queries end
function _rk_spl_factor_rows end
function _rk_spl_factor_reads end
function _rk_spl_factor_ad end
function _rk_spl_factor_block end
function _rk_spl_factor_third end

struct _BRMSumCellGroup
    name::Symbol
    query::Any
    cells::Vector
    prepared::Any
end

# Keep the local sparse precision and low-rank outer blocks separate even
# during line searches and block Newton fallback. Multiplication and solves
# never expand a sum cell into a clique.
mutable struct _BRMSumPrecision
    D::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    W::SparseMatrixCSC{Float64,Int}
    parts::Vector{Any}
end
SparseArrays.nonzeros(h::_BRMSumPrecision) = nonzeros(h.D)
Base.:*(h::_BRMSumPrecision, x::AbstractVector) = h.D * x + h.B' * (h.W * (h.B * x))

mutable struct _BRMAugmentedFactor
    F::Any
    n::Int
    positive_definite::Bool
    logdet::Float64
    K::SparseMatrixCSC{Float64,Int}
    workspace::Any
    selected::SparseMatrixCSC{Float64,Int}
    selected_ready::Bool
end
LinearAlgebra.issuccess(f::_BRMAugmentedFactor) = f.positive_definite
LinearAlgebra.logdet(f::_BRMAugmentedFactor) = f.logdet
Base.:\(f::_BRMAugmentedFactor, b::AbstractVector) = _brm_spl_augmented_solve(f, b)

# MUMPS's sequential backend still requires MPI initialization. Keep every
# MUMPS operation under one lock, and destroy its temporary native workspace
# before returning, including on errors. A chain retains only its KLU factor
# and selected-inverse workspace; deepcopy resets that whole cache.
const _BRM_INERTIA_LOCK = ReentrantLock()
function _brm_spl_inertia(K)
    lock(_BRM_INERTIA_LOCK) do
        MUMPS.MPI.Initialized() || MUMPS.MPI.Init(; threadlevel=:serialized)
        m = MUMPS.Mumps{Float64}(MUMPS.mumps_symmetric,
            MUMPS.get_icntl(det=true, nthreads=1, tree_parallelism=false),
            copy(MUMPS.default_cntl64); backend=MUMPS.Sequential())
        try
            MUMPS.set_icntl!(m, 13, 1; displaylevel=0)
            # The rotation can leave roundoff at an exactly singular Schur
            # complement. Reject numerical null pivots at MUMPS's default
            # scale-aware tolerance, even if its remaining inertia is right.
            MUMPS.set_icntl!(m, 24, 1; displaylevel=0)
            MUMPS.set_icntl!(m, 31, 1; displaylevel=0)
            MUMPS.associate_matrix!(m, K)
            MUMPS.set_job!(m, 4)
            MUMPS.invoke_mumps!(m)
            (; negative=m.infog[28] == 0 ? Int(m.infog[12]) : -1,
                logdet=log(abs(m.rinfog[12])) + m.infog[34] * log(2.0))
        catch err
            # A singular trial curvature is handled by Newton damping. Other
            # solver failures must remain visible, rather than looking like
            # a statistical rejection.
            err isa MUMPS.MUMPSException && m.infog[1] == -10 || rethrow()
            (; negative=-1, logdet=NaN)
        finally
            Base.finalize(m)
        end
    end
end

function _brm_spl_factor_selection(reads, integrated, population)
    sel = sort!(unique!(collect(Int, reads)))
    z = [(get(integrated, i, 0), k) for (k, i) in enumerate(sel)
        if haskey(integrated, i)]
    sort!(z)
    ps = [(get(population, i, 0), k) for (k, i) in enumerate(sel)
        if haskey(population, i)]
    (; sel, lz=first.(z), lpos=last.(z), pidx=first.(ps), ppos=last.(ps))
end

# These position/population maps are read-only after preparation and commonly
# identical across rows and reductions. Intern them within this preparation,
# never globally; deepcopy still gives each chain its own arrays. Absolute
# packed/effect selections usually differ per row, so do not cache those.
_brm_spl_share_indices(maps, indices) = get!(maps, indices, indices)
function _brm_spl_factor_selection(reads, integrated, population, maps)
    s = _brm_spl_factor_selection(reads, integrated, population)
    merge(s, (; lpos=_brm_spl_share_indices(maps, s.lpos),
        pidx=_brm_spl_share_indices(maps, s.pidx),
        ppos=_brm_spl_share_indices(maps, s.ppos)))
end

# Each reduction can have a different generated callable type. Enter its
# current world once, then specialize this whole batch on both callables:
# dispatching through q.terms in the inner loop erases their types and also
# makes the selection/record construction allocate dynamically per summand.
function _brm_spl_prepare_reduction(value::V, footprint::F, q, j, u0, ncells,
        zof, pof, maps) where {V,F}
    sums = zeros(ncells)
    terms = map(1:ncells) do i
        map(eachindex(_rk_spl_factor_rows(q, j, i))) do t
            sums[i] += value(u0, i, t)
            reads = footprint(u0, i, t).unconstrained
            reads = reads === :whole ? eachindex(u0) : reads
            merge((; j, t), _brm_spl_factor_selection(reads, zof, pof, maps))
        end
    end
    (; sums, terms)
end

function _brm_spl_prepare_factor_cells(footprint::F, reductions, u0, ncells,
        zof, pof, maps) where {F}
    map(1:ncells) do i
        sums = [r.sums[i] for r in reductions]
        terms = [r.terms[i] for r in reductions]
        reads = footprint(u0, sums, i).unconstrained
        reads = reads === :whole ? eachindex(u0) : reads
        direct = _brm_spl_factor_selection(reads, zof, pof, maps)
        lz = sort!(unique!(vcat(direct.lz,
            [z for reduction in terms for term in reduction for z in term.lz])))
        localof = Dict(z => k for (k, z) in enumerate(lz))
        direct = merge(direct, (; loc=_brm_spl_share_indices(maps,
            [localof[z] for z in direct.lz])))
        terms = [map(t -> merge(t, (; loc=_brm_spl_share_indices(maps,
                [localof[z] for z in t.lz]))), reduction)
            for reduction in terms]
        (; i, direct, terms, lz)
    end
end

# Discover support structurally before preparing AD. Numeric zeros therefore
# cannot remove a coordinate, and empty sums need no exemplar term call.
function _brm_spl_prepare_factors(backend, name, u0, ncells, integrated, population;
        query=nothing)
    q = query === nothing ? _rk_spl_factor_queries(backend, name) : query
    isempty(q.terms) && return nothing
    zof = Dict(i => k for (k, i) in enumerate(integrated))
    pof = Dict(i => k for (k, i) in enumerate(population))
    maps = Dict{Vector{Int},Vector{Int}}()
    reductions = map(eachindex(q.terms)) do j
        term = q.terms[j]
        Base.invokelatest(_brm_spl_prepare_reduction, term.value, term.footprint,
            q, j, u0, ncells, zof, pof, maps)
    end
    cells = Base.invokelatest(_brm_spl_prepare_factor_cells, q.outer.footprint,
        reductions, u0, ncells, zof, pof, maps)
    (; q, cells)
end

# The Jacobian has one row per sum or direct outer input and one column per
# effect this cell reads. No cell-wide Hessian is formed: second derivatives
# remain the outer's small block and each summand's small block.
function _brm_spl_factor_blocks(prepared, u, c)
    values = _brm_spl_factor_values(prepared, u, c)
    gf, hf = _rk_spl_factor_block(c.outer, u, c.direct.sel, c.i; sums=values.sums, columns=c.active)
    ri = Int[]; ci = Int[]; bv = Float64[]
    r = length(values.sums)
    for (k, loc) in enumerate(c.direct.loc)
        push!(ri, r + k); push!(ci, loc); push!(bv, 1.0)
    end
    terms = map(c.terms) do reduction
        map(reduction) do t
            g, h = t.ad === nothing ? (Float64[], zeros(0, 0)) :
                _rk_spl_factor_block(t.ad, u, t.sel, c.i; term=t.t, columns=t.lpos)
            for (k, loc) in zip(t.lpos, t.loc)
                push!(ri, t.j); push!(ci, loc); push!(bv, g[k])
            end
            merge(t, (; g, h))
        end
    end
    B = sparse(ri, ci, bv, length(c.active), length(c.lz))
    # The third-order chain rule needs S*B' for every outer input at every
    # effect this cell reads, including pairs where the current Jacobian is
    # zero. Keep that rectangular request in the structural pattern.
    ri = [i for j in axes(B, 2) for i in axes(B, 1)]
    ci = [j for j in axes(B, 2) for i in axes(B, 1)]
    B = sparse(ri, ci, Float64[B[i,j] for (i,j) in zip(ri,ci)], size(B)...)
    (; values..., gf, hf, B, terms, active=c.active)
end

# Assemble likelihood precision as D + B' W B. Only each summand's small
# clique enters D; each outer block enters W once. Structural zero entries
# are retained for selected inverse requests as the curvature changes sign.
# This is shared by Newton assembly and the saved state for its derivatives.
function _brm_spl_factor_curvature(prepared, u, n, np)
    gradient = zeros(n); population = zeros(np)
    di = Int[]; dj = Int[]; dv = Float64[]
    bi = Int[]; bj = Int[]; bv = Float64[]
    wi = Int[]; wj = Int[]; wv = Float64[]
    rowptr = [1]
    states = map(prepared.cells) do c
        state = _brm_spl_factor_blocks(prepared, u, c)
        _brm_spl_factor_pullback!(gradient, population, c, state, state.gf)
        for reduction in state.terms, t in reduction
            for (b, j) in enumerate(t.lz), (a, i) in zip(t.lpos, t.lz)
                push!(di, i); push!(dj, j); push!(dv, -state.gf[t.j] * t.h[a,b])
            end
        end
        offset = last(rowptr) - 1
        for j in axes(state.B, 2), k in nzrange(state.B, j)
            push!(bi, offset + rowvals(state.B)[k])
            push!(bj, c.lz[j]); push!(bv, nonzeros(state.B)[k])
        end
        for j in axes(state.hf, 2), (i, a) in enumerate(state.active)
            push!(wi, offset + i); push!(wj, offset + j); push!(wv, -state.hf[a,j])
        end
        push!(rowptr, offset + length(state.active) + 1)
        state
    end
    r = last(rowptr) - 1
    (; value=sum(s -> s.value, states; init=0.0), gradient, population,
        D=sparse(di,dj,dv,n,n), B=sparse(bi,bj,bv,r,n), W=sparse(wi,wj,wv,r,r),
        rowptr, states)
end

function _brm_spl_factor_scatter!(z, population, selection, value)
    z[selection.lz] .+= value[selection.lpos]
    population[selection.pidx] .+= value[selection.ppos]
    nothing
end

# Pull a covector on (sums, direct coordinates) through their packed reads.
function _brm_spl_factor_pullback!(z, population, c, state, covector)
    r = length(state.sums)
    _brm_spl_factor_scatter!(z, population, c.direct, view(covector, (r + 1):length(covector)))
    for reduction in state.terms, t in reduction
        _brm_spl_factor_scatter!(z, population, t, covector[t.j] .* t.g)
    end
    nothing
end

# Contract D³ F(s(x), direct(x)) with the covariance on integrated effects.
# `blocks[j][t]` is S on the summand's tiny effect clique; SB is S B' at
# this cell's effects, and BSB is B S B'. The augmented inverse supplies all
# three without a dense inverse of the cell's entire support.
function _brm_spl_factor_contraction!(tz, tp, prepared, u, c, state, blocks, SB, BSB)
    eig = eigen(Symmetric(BSB))
    scale = max(1.0, maximum(abs, eig.values; init=0.0))
    minimum(eig.values; init=0.0) >= -1e-10scale ||
        error("sum-cell covariance projection is not positive semidefinite")
    outer_tau = zeros(length(state.gf))
    for k in eachindex(eig.values)
        eig.values[k] <= 0 && continue
        d = zeros(length(state.gf))
        d[state.active] .= sqrt(eig.values[k]) .* eig.vectors[:, k]
        _, _, tv = _rk_spl_factor_third(c.outer, d, u, c.direct.sel, c.i;
            sums=state.sums)
        outer_tau .+= tv
    end
    traces = zeros(length(state.active))
    for (j, reduction) in enumerate(state.terms), (k, t) in enumerate(reduction)
        isempty(t.lz) && continue
        S = blocks[j][k]
        traces[j] += sum(S .* t.h[t.lpos, :])
        # The two differentiated-Jacobian terms in the chain rule.
        cross = 2 .* (t.h * (SB[t.loc, :] * state.hf[state.active, j]))
        L = cholesky(Symmetric(S)).L
        for a in axes(L, 2)
            d = zeros(length(t.sel)); d[t.lpos] .= L[:, a]
            _, _, tv = _rk_spl_factor_third(t.ad, d, u, t.sel, c.i; term=t.t)
            cross .+= state.gf[j] .* tv
        end
        _brm_spl_factor_scatter!(tz, tp, t, cross)
    end
    _brm_spl_factor_pullback!(tz, tp, c, state, outer_tau .+ state.hf * traces)
    nothing
end

function _brm_spl_factor_hvp!(z, population, c, state, direction)
    _brm_spl_factor_pullback!(z, population, c, state,
        state.hf * (state.B * direction[c.lz]))
    for reduction in state.terms, t in reduction
        local_hv = state.gf[t.j] .* (t.h * direction[t.lz])
        _brm_spl_factor_scatter!(z, population, t, local_hv)
    end
    nothing
end

function _brm_spl_factor_values(prepared, u, cell)
    sums = [sum(t -> Base.invokelatest(prepared.q.terms[j].value, u, cell.i, t.t),
        terms; init=0.0) for (j, terms) in enumerate(cell.terms)]
    value = Base.invokelatest(prepared.q.outer.value, u, sums, cell.i)
    (; value, sums)
end

function _brm_spl_factor_operators(prepared, u0, ad_backend)
    cache = Dict{Tuple{Any,Int,Int},Any}()
    operator(part, selection, K, i, t=1) = get!(cache, (part, length(selection.sel), K)) do
        _rk_spl_factor_ad(prepared.q, part, u0, selection.sel, ad_backend;
            cell0=i, term0=t, K)
    end
    cells = map(prepared.cells) do c
        r = length(c.terms)
        active = vcat(collect(1:r), r .+ c.direct.lpos)
        outer = operator(:outer, c.direct, length(active), c.i)
        terms = map(c.terms) do reduction
            map(reduction) do t
                ad = isempty(t.sel) ? nothing : operator(t.j, t, length(t.lpos), c.i, t.t)
                merge(t, (; ad))
            end
        end
        merge(c, (; outer, terms, active))
    end
    merge(prepared, (; cells))
end

# H = D + B' W B without materializing the outer cliques. The two auxiliary
# blocks are a 45-degree rotation of [0 I; I W], whose inverse is [-W I; I 0].
# Its determinant has magnitude one and its inertia is (r, r), even when W
# is singular or indefinite. Thus K's Schur complement is H; H is positive
# definite exactly when K has n+r positive and r negative pivots.
#
# Keeping both copies of B in the input pattern also keeps the selected
# inverse entries needed for S B' and B S B', including at zero curvature.
function _brm_spl_augmented_matrix(D::SparseMatrixCSC, B::SparseMatrixCSC,
        W::SparseMatrixCSC)
    n = size(D, 1); r = size(B, 1)
    size(D) == (n, n) && size(B, 2) == n && size(W) == (r, r) ||
        throw(DimensionMismatch("incompatible local, sum-Jacobian and outer blocks"))
    # Construct from structural entries: sparse block addition can discard
    # cancelled zeros, although their inverse entries are still needed by
    # derivatives at the cancellation point.
    ii = collect(1:(n + 2r)); jj = copy(ii); vv = zeros(n + 2r)
    vv[(n + 1):(n + r)] .= 1.0
    vv[(n + r + 1):end] .= -1.0
    for j in axes(D, 2), k in nzrange(D, j)
        push!(ii, rowvals(D)[k]); push!(jj, j); push!(vv, nonzeros(D)[k])
    end
    for j in axes(B, 2), k in nzrange(B, j)
        i = rowvals(B)[k]; value = nonzeros(B)[k] / sqrt(2.0)
        for a in (n + i, n + r + i)
            push!(ii, a, j); push!(jj, j, a); push!(vv, value, value)
        end
    end
    for j in axes(W, 2), k in nzrange(W, j)
        i = rowvals(W)[k]; value = nonzeros(W)[k] / 2
        for (a,b,v) in ((n+i,n+j,value), (n+r+i,n+r+j,value),
                (n+i,n+r+j,-value), (n+r+i,n+j,-value))
            push!(ii,a); push!(jj,b); push!(vv,v)
        end
    end
    sparse(ii,jj,vv,n+2r,n+2r)
end

# Read the small covariance blocks needed by one cell's chain rule from
# selected entries of the rotated augmented inverse. The original effects
# are first, followed by the two copies of the outer-input rows.
function _brm_spl_factor_covariances(c, rows, selected, n, r)
    # CSC submatrix indexing allocates work arrays as long as the original
    # matrix. Read these small requested blocks directly, so doing this once
    # per cell never allocates a model-sized index map per cell.
    SB = Float64[(selected[i,n+r+j] - selected[i,n+j]) / sqrt(2.0)
        for i in c.lz, j in rows]
    BSB = Float64[(selected[n+i,n+j] - selected[n+i,n+r+j] -
        selected[n+r+i,n+j] + selected[n+r+i,n+r+j]) / 2 for i in rows, j in rows]
    blocks = [[Matrix(Symmetric(_brm_spl_entries(selected,t.lz,t.lz))) for t in reduction]
        for reduction in c.terms]
    (; blocks, SB, BSB=Matrix(Symmetric(BSB)))
end

_brm_spl_entries(A, rows, cols) = Float64[A[i,j] for i in rows, j in cols]

# Sparse restriction with workspace proportional to the requested rows and
# columns, retaining stored zeros. Julia's general CSC indexing uses a row
# map proportional to size(A,1), which is too costly inside per-level work.
function _brm_spl_submatrix(A::SparseMatrixCSC, rows, cols)
    rowof = Dict(i => k for (k,i) in enumerate(rows))
    ii = Int[]; jj = Int[]; vv = Float64[]
    for (j,col) in enumerate(cols), k in nzrange(A,col)
        i = get(rowof,rowvals(A)[k],0)
        i == 0 && continue
        push!(ii,i); push!(jj,j); push!(vv,nonzeros(A)[k])
    end
    sparse(ii,jj,vv,length(rows),length(cols))
end

function _brm_spl_augmented_factor(K, n; previous=nothing)
    inertia = _brm_spl_inertia(K)
    r = (size(K, 1) - n) ÷ 2
    positive_definite = inertia.negative == r && isfinite(inertia.logdet)
    same = previous !== nothing && previous.F !== nothing &&
        size(K) == size(previous.K) && K.colptr == previous.K.colptr &&
        rowvals(K) == rowvals(previous.K)
    # klu! reuses the old numeric pivots; a nonsingular updated matrix can
    # then hit a zero pivot. Re-pivot each new curvature with klu. The
    # selected-inverse workspace can still reuse an unchanged factor pattern.
    F = positive_definite ? KLU.klu(K) : nothing
    _BRMAugmentedFactor(F, n, positive_definite, inertia.logdet, K,
        same ? previous.workspace : nothing, same ? previous.selected : similar(K), false)
end

function _brm_spl_selected(f::_BRMAugmentedFactor)
    f.selected_ready && return f.selected
    f.workspace === nothing &&
        (f.workspace = SelectedInversion.LUSelInvWorkspace(f.F, f.K))
    SelectedInversion.selinv!(f.selected, f.workspace, f.F)
    f.selected_ready = true
    f.selected
end

function _brm_spl_augmented_solve(factor, b::AbstractVector)
    length(b) == factor.n || throw(DimensionMismatch("wrong integrated-vector length"))
    rhs = zeros(size(factor.F, 1)); rhs[1:factor.n] .= b
    (factor.F \ rhs)[1:factor.n]
end

# Apply S^(1/2) to a standard-normal vector through solves for S = H^(-1).
# This remains valid when the local part D is indefinite: only H must be
# positive definite. KrylovKit owns the reorthogonalized Lanczos recurrence.
# The basis is capped independently of model size, so memory stays O(n).
function _brm_spl_covariance_root(solve, b; rtol=1e-10, maxdim=256)
    β = norm(b)
    β == 0 && return (; draw=copy(b), iterations=0, error_bound=0.0)
    it = KrylovKit.LanczosIterator(solve, b, KrylovKit.ModifiedGramSchmidt2())
    fact = KrylovKit.initialize(it)
    bound = Inf
    limit = min(length(b), maxdim)
    for k in 1:limit
        T = KrylovKit.rayleighquotient(fact)
        E = eigen(T)
        minimum(E.values) > 0 || _brm_nlme_error(
            "brm_laplace_draws: the covariance projection is not positive definite")
        roots = sqrt.(E.values)
        weights = E.vectors * (roots .* E.vectors[1, :])
        draw = β .* (KrylovKit.basis(fact) * weights)
        # X = sqrt(S)V - V sqrt(T) solves sqrt(S)X + X sqrt(T) = r e_k'.
        # Sign alternation makes the Jacobi T a Stieltjes matrix, so the
        # scalar e_k' exp(-sqrt(T)t)e_1 has constant sign. Bound its integral
        # using ||exp(-sqrt(S)t)|| <= 1: ||X e_1|| <= ||r|| |T^-1/2[k,1]|.
        # No estimated smallest eigenvalue enters the stopping condition.
        corner = dot(E.vectors[end, :], E.vectors[1, :] ./ roots)
        bound = β * KrylovKit.normres(fact) * abs(corner)
        bound <= rtol * norm(draw) && return (; draw, iterations=k, error_bound=bound)
        k < limit && KrylovKit.expand!(it, fact)
    end
    _brm_nlme_error("brm_laplace_draws: the covariance square-root action did not " *
        "reach relative tolerance $rtol in $limit Lanczos steps (error bound $bound)")
end
