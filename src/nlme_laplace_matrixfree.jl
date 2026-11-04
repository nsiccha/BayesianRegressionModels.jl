# Bounded conditional algebra. Curvature is sparse local D plus B' W B;
# no large dense matrix or unbounded-fill sparse factor is formed.

"""
    MatrixFreeLaplace(; exact_limit=1024, probes=8, seed=20261102,
        solve_tolerance=1e-10, solve_maxiter=512,
        preconditioner_memory=32, preconditioner_ridge=0.01, control_variate=true)

Numerical policy for `brm_laplace_problem(...; solver=:auto)`. Components
above `exact_limit` use preconditioned CG and a fixed Rademacher trace
approximation of `log`. `solver=:exact` retains sparse factorization;
`solver=:matrixfree` forces the operator path. Seeds, probes and iteration caps
are fixed for a prepared target.

Every large component has a reference P = C*C', prepared once and fixed during
target evaluations, at zero latent coordinates and zero population coordinates
(a problem with a `gmrf` field: at its population prior medians). A component
that is exactly one `gmrf` field uses that field's time-block reference
(`src/gmrf_preconditioner.jl`), which sees the data's curvature. Any other
component uses a bounded-fill LDL with `preconditioner_memory` extra factor
entries per column and relative diagonal ridge `preconditioner_ridge`; reduction
cliques are not expanded to build it. The reference transforms the determinant and its derivative together; it does
not regularize the model's precision. Set `preconditioner_memory=nothing` to
use no reference: diagonal Newton solves and an untransformed determinant.

The determinant is the reference's exact log determinant plus the mean over the
probes r of `r' log(A) r`, `A = C^-1 H C^-T` (without a reference, `A = H`).
Each r takes one Lanczos recurrence, A V = V T + β v e', stopped when the solve
of `A x = r` it implies reaches `solve_tolerance`. On that space
`r' log(A) r = ‖r‖² e₁' log(T) e₁`, and the determinant's sensitivity to H is
`‖r‖² V G V'` in the same coordinates, G being the derivative of `log` at T in
the direction e₁e₁'. No quadrature enters. Probe error and solve error are
distinct; neither is an exact determinant certificate, and increasing dimension
does not increase the work caps or guarantee accuracy. Lanczos checks curvature
in its own directions; this is not a proof of global positive definiteness for
a nonconcave conditional density. A recurrence retains its basis while its
probe runs, so that probe's memory is the iteration count times the coordinates.

Fixed probes leave noise in the gradient that does not shrink relative to the
posterior as a field grows. With `control_variate=true` a field whose reference
decouples its channels (one channel included) also rebuilds that reference at
each point and estimates only the difference from it
(`src/gmrf_tracked_reference.jl`). This costs about twice as much per probe and
cuts the noise several times over. The value and its gradient stay one
fixed-probe function and its exact derivative.
"""
struct MatrixFreeLaplace
    exact_limit::Int
    probes::Int
    seed::UInt64
    solve_tolerance::Float64
    solve_maxiter::Int
    preconditioner_memory::Union{Nothing,Int}
    preconditioner_ridge::Float64
    control_variate::Bool
    function MatrixFreeLaplace(; exact_limit=1024, probes=8, seed=20261102,
            solve_tolerance=1e-10, solve_maxiter=512,
            preconditioner_memory=32, preconditioner_ridge=0.01, control_variate=true)
        exact_limit >= 0 && probes > 0 && solve_maxiter > 0 ||
            throw(ArgumentError("Laplace dimensions and iteration budgets must be positive"))
        0 < solve_tolerance < 1 && isfinite(solve_tolerance) ||
            throw(ArgumentError("invalid matrix-free Laplace numerical tolerances"))
        (preconditioner_memory === nothing || preconditioner_memory >= 0) &&
            isfinite(preconditioner_ridge) && preconditioner_ridge >= 0 ||
            throw(ArgumentError("invalid fixed conditional preconditioner policy"))
        new(exact_limit, probes, seed, solve_tolerance, solve_maxiter,
            preconditioner_memory, preconditioner_ridge, control_variate)
    end
end
export MatrixFreeLaplace

function _brm_spl_algebra!(p, solver, options::MatrixFreeLaplace)
    solver in (:auto, :exact, :matrixfree) || throw(ArgumentError(
        "brm_laplace_problem: solver must be :auto, :exact or :matrixfree"))
    large = [solver === :matrixfree || length(I) > options.exact_limit for I in p.components]
    if solver === :exact || !any(large)
        p.algebra = nothing
    else
        # Retaining the actual probes makes copies and repeated evaluations
        # independent of RNG history and of evaluations at other points.
        rng = Random.Xoshiro(options.seed)
        probes = [large[c] ? 2.0 .* rand(rng, Bool, length(I), options.probes) .- 1 :
            zeros(0, 0) for (c, I) in enumerate(p.components)]
        references = Any[nothing for _ in p.components]
        p.algebra = (; solver, options, large, probes, references, diagnostics=Dict{Symbol,Any}())
        if options.preconditioner_memory !== nothing
            # One deterministic reference, independent of the first evaluated
            # point, caller RNG, mode-search history and later evaluations.
            pop = _brm_spl_population(p, _brm_spl_reference_point(p))
            n = length(p.integrated)
            h = _brm_spl_precision(p)
            _brm_spl_assemble!(zeros(length(p.components)), zeros(n), h, p,
                pop.A, pop.C, zeros(n); derivatives=true, uA=pop.uA)
            F = _brm_operator_factor(p, h)
            for (c, part) in enumerate(F.parts)
                large[c] || continue
                references[c] = _brm_component_reference(p, c, part.H, h, F.localindex, options)
            end
        end
    end
    p.factor[] = nothing
    p
end

# Where a problem's reference is prepared. A field's curvature and prior move by
# orders of magnitude between its lengths' priors and the unit point of the zero
# coordinates, so a problem with a field takes its prior medians.
_brm_spl_reference_point(p) = p.gmrf === nothing ? zeros(length(p.population)) :
    _brm_laplace_prior_median(p)

# Backtrack all unresolved components together, with a fixed cap. A local
# Gauss-Seidel fallback would reassemble the whole field once per level.
# Like the exact step, require finite derivatives and return them at the
# accepted point for the next Newton iteration.
function _brm_operator_step!(p, pop, z, direction, g, h, fc, converged; weight::Float64=1.0,
        search=nothing)
    accepted = copy(converged)
    decrement = [dot(view(g, I), view(direction, I)) for I in p.components]
    steps = ones(length(fc))
    trial = copy(z); ft = similar(fc); gt = similar(g); ht = _brm_spl_precision(p)
    for _ in 1:60
        for i in eachindex(z)
            c = p.component[i]
            accepted[c] || (trial[i] = z[i] + steps[c] * direction[i])
        end
        _brm_spl_assemble!(ft, gt, ht, p, pop.A, pop.C, trial;
            derivatives=true, uA=pop.uA, weight)
        valid = _brm_spl_finite_blocks(p, gt, ht)
        for c in eachindex(fc)
            accepted[c] && continue
            if isfinite(ft[c]) && all(valid[p.sblock[i]] for i in p.components[c]) &&
                    (decrement[c] <= 1e-8 ||
                    ft[c] >= fc[c] + 1e-4 * steps[c] * decrement[c])
                accepted[c] = true
                z[p.components[c]] .= trial[p.components[c]]
            else
                search === nothing || steps[c] < 1 || (search.backtracks[c] += 1)
                steps[c] /= 2
            end
        end
        if all(accepted)
            copyto!(fc, ft); copyto!(g, gt); _brm_spl_copy_precision!(h, ht)
            return true
        end
    end
    false
end

struct _BRMConditionalOperator{P}
    D::P
    B::SparseMatrixCSC{Float64,Int}
    W::SparseMatrixCSC{Float64,Int}
    diagonal::Vector{Float64}
    shift::Float64
    scratch_b::Vector{Float64}
    scratch_w::Vector{Float64}
end
Base.size(H::_BRMConditionalOperator) = size(H.D)
Base.size(H::_BRMConditionalOperator, d::Integer) = size(H.D, d)
Base.eltype(::Type{<:_BRMConditionalOperator}) = Float64
Base.eltype(::_BRMConditionalOperator) = Float64
function LinearAlgebra.mul!(y, H::_BRMConditionalOperator, x)
    mul!(y, H.D, x)
    if !isempty(H.W)
        mul!(H.scratch_b, H.B, x)
        mul!(H.scratch_w, H.W, H.scratch_b)
        mul!(y, H.B', H.scratch_w, 1.0, 1.0)
    end
    H.shift == 0 || axpy!(H.shift, x, y)
    y
end
function (H::_BRMConditionalOperator)(x)
    mul!(similar(x), H, x)
end

struct _BRMOperatorSolveFailure <: Exception
    message::String
end
Base.showerror(io::IO, e::_BRMOperatorSolveFailure) = print(io, e.message)

function _brm_checked_cg(H, b, options; M=I, rtol=options.solve_tolerance / 10,
        diagnostics=nothing)
    iszero(norm(b)) && return zeros(length(b))
    x, info = Krylov.cg_lanczos(H, collect(b); M,
        check_curvature=true, rtol,
        atol=0.0, itmax=options.solve_maxiter)
    residual = norm(H(x) .- b) / norm(b)
    if diagnostics !== nothing
        old = get(diagnostics, :linear_solves, (; attempts=0, max_iterations=0,
            max_relative_residual=0.0))
        diagnostics[:linear_solves] = (; attempts=old.attempts + 1,
            max_iterations=max(old.max_iterations, info.niter),
            max_relative_residual=max(old.max_relative_residual, residual))
    end
    info.solved && info.niter <= options.solve_maxiter &&
        residual <= options.solve_tolerance ||
        throw(_BRMOperatorSolveFailure("conditional solve failed within " *
            "$(options.solve_maxiter) iterations (used $(info.niter), relative residual $residual; $(info.status))"))
    x
end
function _brm_operator_solve(H::_BRMConditionalOperator, b, options; diagnostics=nothing)
    d = H.diagonal .+ H.shift
    all(>(0), d) || throw(_BRMOperatorSolveFailure("nonpositive conditional curvature diagonal"))
    _brm_checked_cg(H, b, options; M=Diagonal(1 ./ d), diagnostics)
end

mutable struct _BRMOperatorFactor
    parts::Vector{Any}
    components::Vector{Vector{Int}}
    component::Vector{Int}
    localindex::Vector{Int}
    options::MatrixFreeLaplace
    logdets::Vector{Float64}
    sensitivity::Vector{Float64}
    sumcov::Dict{Int,Any}
    ready::Bool
    problem::Any
    curvature::Any
    # Each part's control variate (`_BRMGMRFTracked`), once its determinant is computed.
    tracked::Vector{Any}
end

# `shift`: one number, or one per component (each part takes its own).
function _brm_operator_factor(p, h; shift=0.0)
    D = h isa _BRMSumPrecision ? h.D : h
    if h isa _BRMSumPrecision
        B, W = h.B, h.W
    else
        bi = [k for (k, cols) in enumerate(p.wcols) for _ in cols]
        bj = [i for cols in p.wcols for i in cols]
        B = sparse(bi, bj, ones(length(bi)), length(p.wcols), size(D, 1))
        W = spdiagm(0 => p.wλ)
    end
    localindex = zeros(Int, size(D, 1))
    parts = Any[]
    for (c, I) in enumerate(p.components)
        localindex[I] .= eachindex(I)
        rows = sort!(unique!([rowvals(B)[k] for j in I for k in nzrange(B, j)]))
        Dc = _brm_spl_submatrix(D, I, I)
        Bc = _brm_spl_submatrix(B, rows, I)
        Wc = _brm_spl_submatrix(W, rows, rows)
        WB = Wc * Bc
        diagonal = collect(diag(Dc) .+ vec(sum(Bc .* WB; dims=1)))
        sc = shift isa AbstractVector ? Float64(shift[c]) : Float64(shift)
        H = _BRMConditionalOperator(Dc, Bc, Wc, diagonal, sc,
            zeros(length(rows)), zeros(length(rows)))
        factor = if p.algebra.large[c]
            nothing
        else
            # Only a bounded component is ever materialized densely.
            dense = Matrix(Dc) + Matrix(Bc' * WB)
            dense[diagind(dense)] .+= sc
            cholesky(Symmetric(dense); check=false)
        end
        references = get(p.algebra, :references, nothing)
        reference = references === nothing ? nothing : references[c]
        operator = reference === nothing ? H : _BRMPreconditionedOperator(H, reference)
        push!(parts, (; H, operator, factor, probes=p.algebra.probes[c]))
    end
    _BRMOperatorFactor(parts, p.components, p.component, localindex, p.algebra.options,
        zeros(length(parts)), zeros(SparseArrays.nnz(p.pattern)), Dict{Int,Any}(), false, p, h,
        Any[nothing for _ in parts])
end
LinearAlgebra.issuccess(F::_BRMOperatorFactor) = all(F.parts) do part
    part.factor === nothing ? all(>(0), part.H.diagonal .+ part.H.shift) : issuccess(part.factor)
end
# A component whose right-hand side is not finite gets a nonfinite solution,
# as a direct factor's solve gives, instead of a failed iterative solve: the
# marginal gradient then reports the term that made it nonfinite
# (`_brm_spl_nonfinite_gradient!`). Other components are unaffected.
function Base.:\(F::_BRMOperatorFactor, b::AbstractVector)
    x = zeros(length(b))
    for (part, I) in zip(F.parts, F.components)
        bI = b[I]
        x[I] .= !all(isfinite, bI) ? NaN :
            part.factor === nothing ? _brm_operator_solve(part.operator, bI, F.options;
                diagnostics=F.problem.algebra.diagnostics) :
            part.factor \ bI
    end
    x
end
_brm_spl_woodbury(p::BRMSparseLaplaceProblem, F::_BRMOperatorFactor) = nothing

# The operator algebra's Newton direction and each component's shift
# (`_brm_spl_shifted_direction`): every part tries 0 and then the shifts in
# `_BRM_SPL_SHIFTS` on its own, so one part's curvature never damps another's.
function _brm_operator_direction(p, h, g)
    nc = length(p.components)
    x = zeros(length(g)); shift = zeros(nc); done = falses(nc)
    for μ in (0.0, _BRM_SPL_SHIFTS...)
        F = _brm_operator_factor(p, h; shift=[done[c] ? shift[c] : μ for c in 1:nc])
        for (c, (part, I)) in enumerate(zip(F.parts, F.components))
            done[c] && continue
            definite = part.factor === nothing ?
                all(>(0), part.H.diagonal .+ part.H.shift) : issuccess(part.factor)
            definite || continue
            try
                x[I] .= part.factor === nothing ? _brm_operator_solve(part.operator, g[I],
                    F.options; diagnostics=p.algebra.diagnostics) : part.factor \ g[I]
                shift[c] = μ; done[c] = true
            catch err
                err isa _BRMOperatorSolveFailure || rethrow()
                p.algebra.diagnostics[:last_damped_solve_failure] = sprint(showerror, err)
            end
        end
        all(done) && return (x, shift)
    end
    nothing
end

# The eigen-decomposition of the derivative's small matrix drops directions
# holding less than this share of its trace.
const _BRM_LOGDET_RANK_TOLERANCE = 1e-12

# The Loewner matrix of `log` is positive semidefinite, so G below is too: the
# determinant's sensitivity stays a positive sum of rank-one terms.
_brm_log_divided_difference(a, b) = a == b ? 1 / a : log1p((a - b) / b) / (a - b)

# One probe's Lanczos recurrence on `operator`, stopped when the solve of
# `A x = r` it implies reaches `rtol`, at `maxiter` iterations, or at nonpositive
# curvature in its directions. That solve's residual is β_k |e_k' T_k⁻¹ e₁|, read
# from T_k = L D L' as T grows, whose pivots also check the curvature.
function _brm_lanczos_run(operator, r, rtol, maxiter)
    iterator = KrylovKit.LanczosIterator(operator, collect(r), KrylovKit.ModifiedGramSchmidt())
    state = KrylovKit.initialize(iterator; verbosity=0)
    pivot = first(state.αs)
    last_entry = 1.0
    while true
        k = length(state)
        pivot > 0 && isfinite(pivot) || return (; state, residual=Inf, curved=false)
        β = KrylovKit.normres(state)
        residual = β * abs(last_entry) / pivot
        (residual <= rtol || k >= maxiter) && return (; state, residual, curved=true)
        KrylovKit.expand!(iterator, state; verbosity=0)
        last_entry *= -β / pivot
        pivot = state.αs[k + 1] - β^2 / pivot
    end
end

function _brm_lanczos(operator, r, options)
    (; state, residual, curved) = _brm_lanczos_run(operator, r, options.solve_tolerance / 10,
        options.solve_maxiter)
    curved || throw(_BRMOperatorSolveFailure("conditional determinant recurrence found " *
        "nonpositive curvature at iteration $(length(state))"))
    residual <= options.solve_tolerance / 10 || throw(_BRMOperatorSolveFailure("conditional " *
        "determinant recurrence failed within $(options.solve_maxiter) iterations: residual $residual"))
    state
end

# `r' log(A) r` on r's Krylov space, the residual of the solve that recurrence
# implies, and the factorization G = P Γ P' (scaled by ‖r‖²) of the sensitivity
# ‖r‖² V G V', with `V G V'` the sensitivity of `e₁' log(T) e₁` to a change of A
# (Daleckii–Krein: G = Q [log[θᵢ,θⱼ] cᵢ cⱼ] Q', T = Q Θ Q', c = Q'e₁).
function _brm_lanczos_log(state, r)
    T = KrylovKit.rayleighquotient(state)
    k = length(state)
    E = eigen(Symmetric(Matrix(T)))
    θ, Q = E.values, E.vectors
    all(>(0), θ) || throw(_BRMOperatorSolveFailure(
        "conditional determinant recurrence found nonpositive curvature: smallest Ritz value $(first(θ))"))
    c = view(Q, 1, :)
    scale = dot(r, r)
    value = scale * sum(i -> c[i]^2 * log(θ[i]), 1:k)
    G = Q * [_brm_log_divided_difference(θ[i], θ[j]) * c[i] * c[j] for i in 1:k, j in 1:k] * Q'
    Γ = eigen(Symmetric(G))
    keep = _brm_lanczos_rank(Γ.values)
    start = zeros(k); start[1] = sqrt(scale)
    (; value, ritz=(first(θ), last(θ)), weights=scale .* Γ.values[keep],
        directions=Γ.vectors[:, keep], solve=T \ start)
end

# Which eigenvalues of G to keep: the largest, down to a tail of a negligible
# share of the trace. Rounding's tiny negative eigenvalues are never kept.
function _brm_lanczos_rank(values)
    total = sum(v -> max(v, 0.0), values)
    tail = 0.0
    first_kept = length(values)
    for k in eachindex(values)
        tail += max(values[k], 0.0)
        tail > _BRM_LOGDET_RANK_TOLERANCE * total && (first_kept = k; break)
    end
    first_kept:length(values)
end

_brm_lanczos_combine!(out, V, coefficients) = begin
    fill!(out, 0.0)
    for (a, coefficient) in enumerate(coefficients)
        axpy!(coefficient, V[a], out)
    end
    out
end

# This part's control variate, for a field or star whose references decouple their
# channels. A sum's reductions stay out of the rebuilt reference, whose derivative
# they would need.
function _brm_operator_tracked(F, component, operator)
    F.options.control_variate || return nothing
    operator isa _BRMPreconditionedOperator || return nothing
    F.curvature isa _BRMSumPrecision && return nothing
    iszero(operator.h.shift) || return nothing
    p = F.problem
    hasproperty(p, :gmrf) && p.gmrf !== nothing || return nothing
    if operator.reference isa _BRMGMRFTimeReference
        field = _brm_gmrf_component_field(p, component)
        return field === nothing ? nothing : _brm_gmrf_tracked(field, operator.reference, F.curvature)
    elseif operator.reference isa _BRMGMRFStarReference
        fields = _brm_gmrf_component_fields(p, component)
        return fields === nothing ? nothing : _brm_gmrf_tracked(fields, operator.reference, F.curvature)
    end
    nothing
end

# S[k] += weight x[row] x[col] over the component's stored entries, without a
# stored index per entry: those would outweigh the precision itself.
function _brm_operator_sensitivity!(S, pattern, localindex, I, x, weight)
    rows = rowvals(pattern)
    @inbounds for (position, j) in enumerate(I)
        scaled = weight * x[position]
        for k in nzrange(pattern, j)
            S[k] += scaled * x[localindex[rows[k]]]
        end
    end
    S
end

# S is d(logdet approximation)/dH, not an independent inverse-trace estimator.
# Stream its rank-one contributions; never retain n * probes vectors.
function _brm_operator_logdet!(F::_BRMOperatorFactor)
    F.ready && return F
    p = F.problem
    sums = [Any[] for _ in F.parts]
    if F.curvature isa _BRMSumPrecision
        for (; group, part) in F.curvature.parts, (k, c) in enumerate(group.cells)
            isempty(c.lz) && continue
            cs = part.states[k]
            cov = (; SB=zeros(length(c.lz), length(cs.active)),
                BSB=zeros(length(cs.active), length(cs.active)))
            F.sumcov[c.term] = cov
            push!(sums[p.term_component[c.term]], (; c, cs, cov,
                localrows=F.localindex[c.lz]))
        end
    end
    for (component, (part, I)) in enumerate(zip(F.parts, F.components))
        function accumulate(x, weight)
            _brm_operator_sensitivity!(F.sensitivity, p.pattern, F.localindex, I, x, weight)
            for (; cs, cov, localrows) in sums[component]
                localx = x[localrows]
                projected = cs.B * localx
                cov.SB .+= weight .* localx .* projected'
                cov.BSB .+= weight .* projected .* projected'
            end
        end
        if part.factor !== nothing
            F.logdets[component] = logdet(part.factor)
            L = inv(part.factor.U)
            for j in axes(L, 2)
                accumulate(view(L, :, j), 1.0)
            end
            continue
        end
        options = F.options
        estimates = zeros(options.probes)
        operator = part.operator
        transformed = operator isa _BRMPreconditionedOperator
        direction = zeros(length(I)); physical = similar(direction); solve = similar(direction)
        max_residual = 0.0
        max_iterations = 0
        max_rank = 0
        smallest, largest = Inf, -Inf
        tracked = _brm_operator_tracked(F, component, operator)
        tracked_operator = tracked === nothing ? nothing :
            _BRMPreconditionedOperator(tracked, operator.reference)
        tracked_iterations = 0
        for (q, r) in enumerate(eachcol(part.probes))
            state = _brm_lanczos(operator, r, options)
            V = KrylovKit.basis(state)
            log_r = _brm_lanczos_log(state, r)
            _brm_lanczos_combine!(solve, V, log_r.solve)
            residual = norm(operator(solve) .- r) / norm(r)
            residual <= options.solve_tolerance || throw(_BRMOperatorSolveFailure(
                "conditional determinant recurrence has solve residual $residual, " *
                "exceeding $(options.solve_tolerance)"))
            max_residual = max(max_residual, residual)
            max_iterations = max(max_iterations, length(state))
            max_rank = max(max_rank, length(log_r.weights))
            smallest = min(smallest, first(log_r.ritz)); largest = max(largest, last(log_r.ritz))
            estimates[q] = log_r.value
            for (k, weight) in enumerate(log_r.weights)
                _brm_lanczos_combine!(direction, V, view(log_r.directions, :, k))
                if transformed
                    _brm_to_physical!(physical, operator, direction)
                    accumulate(physical, weight / options.probes)
                else
                    accumulate(direction, weight / options.probes)
                end
            end
            tracked === nothing && continue
            state_m = _brm_lanczos(tracked_operator, r, options)
            V_m = KrylovKit.basis(state_m)
            log_m = _brm_lanczos_log(state_m, r)
            estimates[q] -= log_m.value
            tracked_iterations = max(tracked_iterations, length(state_m))
            for (k, weight) in enumerate(log_m.weights)
                _brm_lanczos_combine!(direction, V_m, view(log_m.directions, :, k))
                _brm_to_physical!(physical, tracked_operator, direction)
                _brm_gmrf_tracked_contract!(tracked, physical, -weight / options.probes)
            end
        end
        reference_logdet = transformed ? operator.reference.logdet : 0.0
        F.logdets[component] = (tracked === nothing ? reference_logdet : tracked.logdet) + mean(estimates)
        if tracked !== nothing
            _brm_gmrf_tracked_exact!(tracked)
            _brm_gmrf_tracked_sensitivity!(F.sensitivity, tracked, p, component)
            F.tracked[component] = tracked
        end
        p.algebra.diagnostics[Symbol("component_", component)] =
            (; logdet=F.logdets[component], probe_mcse=options.probes > 1 ?
                std(estimates) / sqrt(options.probes) : NaN,
                probes=options.probes, gradient_rank=max_rank,
                ritz=(smallest, largest), tracked_iterations,
                tracked_shift=tracked === nothing ? 0.0 : tracked.shift,
                max_solve_residual=max_residual, max_solve_iterations=max_iterations,
                curvature_check=:krylov_directions,
                reference_entries=transformed ? operator.reference.factor_entries : 0,
                reference_shift=transformed ? operator.reference.diagonal_shift : 0.0)
    end
    F.ready = true
    F
end
LinearAlgebra.logdet(F::_BRMOperatorFactor) = sum(_brm_operator_logdet!(F).logdets)
_brm_spl_component_logdets(p::BRMSparseLaplaceProblem, F::_BRMOperatorFactor) =
    _brm_operator_logdet!(F).logdets
_brm_spl_selected_precision(p::BRMSparseLaplaceProblem, F::_BRMOperatorFactor) =
    _brm_operator_logdet!(F).sensitivity

function _brm_operator_sum_covariances(p, F, c)
    cov = get(F.sumcov, c.term, (; SB=zeros(length(c.lz), length(c.active)),
        BSB=zeros(length(c.active), length(c.active))))
    S = F.sensitivity
    blocks = [[[S[_brm_spl_nzindex(p.pattern, i, j)] for i in t.lz, j in t.lz]
        for t in reduction] for reduction in c.terms]
    (; blocks, cov.SB, cov.BSB)
end

function _brm_operator_draw(F, rng)
    result = zeros(length(F.component))
    for (part, I) in zip(F.parts, F.components)
        r = randn(rng, length(I))
        if part.factor !== nothing
            result[I] .= part.factor.U \ r
        elseif part.operator isa _BRMPreconditionedOperator
            A = part.operator
            draw = _brm_spl_covariance_root(b -> _brm_checked_cg(A, b, F.options;
                diagnostics=F.problem.algebra.diagnostics), r).draw
            _brm_to_physical!(view(result, I), A, draw)
        else
            result[I] .= _brm_spl_covariance_root(
                b -> _brm_operator_solve(part.H, b, F.options;
                    diagnostics=F.problem.algebra.diagnostics), r).draw
        end
    end
    result
end

# Covariance diagnostics use actual inverse actions, not the derivative of
# the approximate determinant. Each request is bounded by a level's width.
function _brm_operator_inverse_blocks(F, sets)
    map(sets) do set
        c = F.component[first(set)]
        all(i -> F.component[i] == c, set) || error("covariance set crosses components")
        part = F.parts[c]; localrows = F.localindex[set]
        S = zeros(length(set), length(set))
        for (k, j) in enumerate(localrows)
            r = zeros(length(F.components[c])); r[j] = 1
            x = part.factor === nothing ? _brm_operator_solve(part.operator, r, F.options;
                diagnostics=F.problem.algebra.diagnostics) : part.factor \ r
            S[:, k] .= x[localrows]
        end
        Matrix(Symmetric(S))
    end
end

# A square root of a sensitivity block and each column's sign: a control variate's
# difference of estimates need not be positive semidefinite, as a sum of its own
# rank-one terms is.
_brm_spl_sensitivity_root(F, S) = (cholesky(Symmetric(S)).L, nothing)
function _brm_spl_sensitivity_root(F::_BRMOperatorFactor, S)
    all(isnothing, F.tracked) && return (_brm_spl_psd_root(S), nothing)
    E = eigen(Symmetric(S))
    (E.vectors * Diagonal(sqrt.(abs.(E.values))), sign.(E.values))
end

# Probe sensitivities can be rank deficient (e.g. H=I and fewer probes than
# a cell's width). No inverse of this square root belongs in the chain rule.
function _brm_spl_psd_root(S)
    E = eigen(Symmetric(S))
    scale = max(1.0, maximum(abs, E.values; init=0.0))
    minimum(E.values; init=0.0) >= -1e-10scale || error("indefinite Laplace sensitivity")
    E.vectors * Diagonal(sqrt.(max.(0.0, E.values)))
end
