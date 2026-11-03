# Bounded conditional algebra. Curvature is sparse local D plus B' W B;
# no large dense matrix or unbounded-fill sparse factor is formed.

"""
    MatrixFreeLaplace(; exact_limit=1024, probes=8, seed=20261102,
        solve_tolerance=1e-10, solve_maxiter=512,
        logdet_step=1.0, logdet_extent=32.0,
        preconditioner_memory=32, preconditioner_ridge=0.01)

Numerical policy for `brm_laplace_problem(...; solver=:auto)`. Components
above `exact_limit` use preconditioned CG and a fixed Rademacher
trace approximation to a rational approximation of `log`. `solver=:exact`
retains sparse factorization; `solver=:matrixfree` forces the operator path.
Seeds, probes, quadrature and iteration caps are fixed for a prepared target.

A bounded-fill LDL reference is prepared at zero population and latent
coordinates, with `preconditioner_memory` extra factor entries per column and
relative diagonal ridge `preconditioner_ridge`. It stays fixed during target
evaluations. The reference transforms the determinant and its derivative
together; it does not regularize the model's precision. Reduction cliques
are not expanded to build it. Set `preconditioner_memory=nothing` to use
diagonal Newton solves and an untransformed determinant instead.

The quadrature is the trapezoidal rule in log(t) for
`log(A) = integral_0^Inf ((1+t)^-1 I - (A+tI)^-1) dt`, where
`A = C^-1 H C^-T` and `C C'` is the fixed reference. Its exact log determinant
is added separately. Without a reference, `A = H`.
Its derivative uses the *same* shifted solves. Probe error, finite quadrature
error and solve error are distinct; none is an exact determinant certificate.
Increasing dimension does not increase the work caps or guarantee accuracy.
CG checks curvature in its search directions; this is not a proof of global
positive definiteness for a nonconcave conditional density.
"""
struct MatrixFreeLaplace
    exact_limit::Int
    probes::Int
    seed::UInt64
    solve_tolerance::Float64
    solve_maxiter::Int
    logdet_step::Float64
    logdet_extent::Float64
    preconditioner_memory::Union{Nothing,Int}
    preconditioner_ridge::Float64
    function MatrixFreeLaplace(; exact_limit=1024, probes=8, seed=20261102,
            solve_tolerance=1e-10, solve_maxiter=512,
            logdet_step=1.0, logdet_extent=32.0,
            preconditioner_memory=32, preconditioner_ridge=0.01)
        exact_limit >= 0 && probes > 0 && solve_maxiter > 0 ||
            throw(ArgumentError("Laplace dimensions and iteration budgets must be positive"))
        0 < solve_tolerance < 1 && isfinite(solve_tolerance) &&
            0 < logdet_step <= 2 && isfinite(logdet_extent) &&
            logdet_step <= logdet_extent <= 64 ||
            throw(ArgumentError("invalid matrix-free Laplace numerical tolerances"))
        (preconditioner_memory === nothing || preconditioner_memory >= 0) &&
            isfinite(preconditioner_ridge) && preconditioner_ridge >= 0 ||
            throw(ArgumentError("invalid fixed conditional preconditioner policy"))
        new(exact_limit, probes, seed, solve_tolerance, solve_maxiter,
            logdet_step, logdet_extent, preconditioner_memory, preconditioner_ridge)
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
            pop = _brm_spl_population(p, zeros(length(p.population)))
            n = length(p.integrated)
            h = _brm_spl_precision(p)
            _brm_spl_assemble!(zeros(length(p.components)), zeros(n), h, p,
                pop.A, pop.C, zeros(n); derivatives=true, uA=pop.uA)
            F = _brm_operator_factor(p, h)
            for (c, part) in enumerate(F.parts)
                large[c] || continue
                references[c] = _brm_fixed_reference(part.H, options.preconditioner_memory;
                    ridge=options.preconditioner_ridge)
            end
        end
    end
    p.factor[] = nothing
    p
end

# Backtrack all unresolved components together, with a fixed cap. A local
# Gauss-Seidel fallback would reassemble the whole field once per level.
# Like the exact step, require finite derivatives and return them at the
# accepted point for the next Newton iteration.
function _brm_operator_step!(p, pop, z, direction, g, h, fc, converged)
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
            derivatives=true, uA=pop.uA)
        valid = _brm_spl_finite_blocks(p, gt, ht)
        for c in eachindex(fc)
            accepted[c] && continue
            if isfinite(ft[c]) && all(valid[p.sblock[i]] for i in p.components[c]) &&
                    (decrement[c] <= 1e-8 ||
                    ft[c] >= fc[c] + 1e-4 * steps[c] * decrement[c])
                accepted[c] = true
                z[p.components[c]] .= trial[p.components[c]]
            else
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

struct _BRMConditionalOperator
    D::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    W::SparseMatrixCSC{Float64,Int}
    diagonal::Vector{Float64}
    shift::Float64
    scratch_b::Vector{Float64}
    scratch_w::Vector{Float64}
end
Base.size(H::_BRMConditionalOperator) = size(H.D)
Base.size(H::_BRMConditionalOperator, d::Integer) = size(H.D, d)
Base.eltype(::Type{_BRMConditionalOperator}) = Float64
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
end

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
        diagonal = diag(Dc) .+ vec(sum(Bc .* WB; dims=1))
        H = _BRMConditionalOperator(Dc, Bc, Wc, diagonal, shift,
            zeros(length(rows)), zeros(length(rows)))
        factor = if p.algebra.large[c]
            nothing
        else
            # Only a bounded component is ever materialized densely.
            dense = Matrix(Dc) + Matrix(Bc' * WB)
            dense[diagind(dense)] .+= shift
            cholesky(Symmetric(dense); check=false)
        end
        references = get(p.algebra, :references, nothing)
        reference = references === nothing ? nothing : references[c]
        operator = reference === nothing ? H : _BRMPreconditionedOperator(H, reference)
        push!(parts, (; H, operator, factor, probes=p.algebra.probes[c]))
    end
    _BRMOperatorFactor(parts, p.components, p.component, localindex, p.algebra.options,
        zeros(length(parts)), zeros(SparseArrays.nnz(p.pattern)), Dict{Int,Any}(), false, p, h)
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

function _brm_operator_direction(p, h, g)
    for shift in (0.0, (10.0^k for k in -4:7)...)
        F = _brm_operator_factor(p, h; shift)
        issuccess(F) || continue
        try
            return F \ g
        catch err
            err isa _BRMOperatorSolveFailure || rethrow()
            p.algebra.diagnostics[:last_damped_solve_failure] = sprint(showerror, err)
        end
    end
    nothing
end

# Factor parts have different concrete operator/workspace types. Keep the
# coordinate loop behind a function barrier so dynamic dispatch there does
# not allocate boxed scalars once per coordinate, probe and quadrature node.
function _brm_shifted_probe_statistics!(hx, operator, x, r, t)
    mul!(hx, operator, x)
    residual2 = 0.0; value = 0.0
    @inbounds for i in eachindex(x)
        residual2 += (hx[i] + t*x[i] - r[i])^2
        value += r[i] * (hx[i] - x[i])
    end
    sqrt(residual2) / norm(r), value
end

# S is d(logdet approximation)/dH, not an independent inverse-trace
# estimator. For each fixed probe r and quadrature node t,
# d[-t r'(H+tI)^-1 r] = t x' dH x, x=(H+tI)^-1 r.
# Stream these rank-one contributions; never retain n * probes * nodes.
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
        # Pattern positions and local indices are prepared once per component.
        positions = [k for j in I for k in nzrange(p.pattern, j)]
        rows = [F.localindex[rowvals(p.pattern)[k]] for k in positions]
        cols = [F.localindex[j] for j in I for _ in nzrange(p.pattern, j)]
        function accumulate(x, weight)
            @inbounds for a in eachindex(positions)
                F.sensitivity[positions[a]] += weight * x[rows[a]] * x[cols[a]]
            end
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
        K = ceil(Int, options.logdet_extent / options.logdet_step)
        estimates = zeros(options.probes)
        shifts = exp.(collect(-K:K) .* options.logdet_step)
        operator = part.operator
        transformed = operator isa _BRMPreconditionedOperator
        probe_rhs = zeros(length(I)); hx = similar(probe_rhs); physical = similar(probe_rhs)
        workspace = Krylov.CgLanczosShiftWorkspace(operator, probe_rhs, shifts)
        max_residual = 0.0
        max_iterations = 0
        for (q, r) in enumerate(eachcol(part.probes))
            # One common Krylov recurrence for every shift. This avoids one
            # full field traversal per shift per iteration. The congruence
            # reference is fixed; shifts are identities in these coordinates.
            probe_rhs .= r
            Krylov.cg_lanczos_shift!(workspace, operator, probe_rhs, shifts;
                check_curvature=true, atol=0.0, rtol=options.solve_tolerance / 10,
                itmax=options.solve_maxiter)
            xs = Krylov.solution(workspace)
            info = Krylov.statistics(workspace)
            info.solved && info.niter <= options.solve_maxiter ||
                throw(_BRMOperatorSolveFailure("conditional shifted solves failed " *
                    "within $(options.solve_maxiter) iterations: $(info.status)"))
            max_iterations = max(max_iterations, info.niter)
            for (j, k) in enumerate(-K:K)
                t = shifts[j]; x = xs[j]
                residual, value = _brm_shifted_probe_statistics!(hx, operator, x, r, t)
                max_residual = max(max_residual, residual)
                residual <= options.solve_tolerance || throw(_BRMOperatorSolveFailure(
                    "conditional shifted solve residual $residual exceeds $(options.solve_tolerance)"))
                weight = options.logdet_step * (abs(k) == K ? 0.5 : 1.0)
                # This equivalent form avoids cancellation of n-sized constants
                # at large shifts, and is zero for an identity precision.
                estimates[q] += weight * t / (1 + t) * value
                if transformed
                    _brm_to_physical!(physical, operator, x)
                    accumulate(physical, weight * t / options.probes)
                else
                    accumulate(x, weight * t / options.probes)
                end
            end
        end
        reference_logdet = transformed ? operator.reference.logdet : 0.0
        F.logdets[component] = reference_logdet + mean(estimates)
        p.algebra.diagnostics[Symbol("component_", component)] =
            (; logdet=F.logdets[component], probe_mcse=options.probes > 1 ?
                std(estimates) / sqrt(options.probes) : NaN,
                probes=options.probes, quadrature_nodes=2K+1,
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

# Probe sensitivities can be rank deficient (e.g. H=I and fewer probes than
# a cell's width). No inverse of this square root belongs in the chain rule.
function _brm_spl_psd_root(S)
    E = eigen(Symmetric(S))
    scale = max(1.0, maximum(abs, E.values; init=0.0))
    minimum(E.values; init=0.0) >= -1e-10scale || error("indefinite Laplace sensitivity")
    E.vectors * Diagonal(sqrt.(max.(0.0, E.values)))
end
