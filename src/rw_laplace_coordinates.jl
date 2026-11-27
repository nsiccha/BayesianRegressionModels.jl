"""
    brm_rw2_coordinates(sigma, phi)

Convert a sum-to-zero order-2 walk's standardized fitted field `phi = x / sigma`
(in sorted step order) to its current `:structured` sampling coordinates. The
first two are the standardized soft mean and the physical least-squares slope
per step. The remaining coordinates encode the standardized second differences
in a scale-dependent banded basis. Use the walk scale from the same draw.

If `D` takes second differences, let `A` and `C` be the lower Cholesky factors
of `D * D'` and `A' * A + sigma^2 * I`. For packed curvature `z = basis[3:end]`,
the standardized differences are `D * phi = A * (C' \\ z)`. This choice uses a
unit of reference curvature per fitted step in physical field units. The model
uses its stated likelihood and walk prior. The conversion takes linear time and
storage in the number of steps.
"""
function brm_rw2_coordinates(sigma::Real, phi::AbstractVector{<:Real})
    n = length(phi)
    n >= 3 || throw(ArgumentError("a second-order walk needs at least three fitted steps"))
    isfinite(sigma) && sigma > 0 || throw(ArgumentError("the walk scale must be positive and finite"))
    m = n - 2
    bands = brm_rw2_factors(sigma, m)
    # Undo r = A * (C' \ z) by forward substitution followed by C' * work.
    work = diff(diff(promote_type(Float64, eltype(phi), typeof(sigma)).(phi)))
    for i in 1:m
        if i > 1
            work[i] -= bands[i, 2] * work[i - 1]
        end
        if i > 2
            work[i] -= bands[i, 3] * work[i - 2]
        end
        work[i] /= bands[i, 1]
    end
    trend = (1:n) .- (n + 1) / 2
    basis = vcat(mean(phi) / 0.001, sigma * dot(trend, phi) / sum(abs2, trend), work)
    for i in 1:m
        z = bands[i, 4] * work[i]
        if i < m
            z += bands[i + 1, 5] * work[i + 1]
        end
        if i + 1 < m
            z += bands[i + 2, 6] * work[i + 2]
        end
        basis[i + 2] = z
    end
    basis
end

# Selected intrinsic RW2 and independent AR(1) walks integrate in standardized
# field values. Other population coordinates keep the public model's meaning.
_brm_laplace_field_plan(p::_RKHeldOutPlan, selected) =
    _RKHeldOutPlan(_brm_laplace_field_plan(p.parent, selected), p.held_out)
_brm_laplace_field_plan(p::_RKValuePlan, selected) = _RKValuePlan(
    _brm_laplace_field_plan(p.regression, selected), p.assignments, p.observations, p.columns, p.completions)
function _brm_laplace_field_plan(p::_RKStructuralPlan, selected)
    predictors = map(p.predictors) do predictor
        terms = map(predictor.terms) do t
            t.kind in (:rw, :cdar) && t.options.value in selected || return t
            _RKTermSpec(t.kind, t.columns, merge(t.options, (; field_coordinates=true)),
                t.addressee, t.label)
        end
        _RKPredictorSpec(predictor.name, predictor.link, terms, predictor.label, predictor.row_source)
    end
    _RKStructuralPlan(p.responses, predictors, p.population_priors, p.parameters,
        p.assignments, p.derived, p.columns, p.n_obs, p.ranef_buckets, p.vector_parameters,
        p.r2d2_priors, p.horseshoe_priors, p.inputs, p.prior_reads)
end

function _brm_laplace_field_backend(rk, declarations)
    walks = [r for r in _rk_coordinate_records(rk) if
        ((r.kind === :rw && r.sum_to_zero && r.order == 2) ||
         (r.kind === :cdar && r.cor_factor === nothing)) && r.declaration in declarations]
    isempty(walks) && return rk, walks
    plan = _brm_laplace_field_plan(rk.plan, Set(r.declaration for r in walks))
    _brm_laplace_working_backend(rk, plan), walks
end

_rw2_field_backend(rk, declarations) = _brm_laplace_field_backend(rk, declarations)

# Re-emit a private integration plan with the public model's fitted binding.
function _brm_laplace_working_backend(rk, plan)
    emitted = _rk_emit_ast(plan; documentation=brm_documentation(brm_source(rk)))
    frozen = rk.fitted === nothing ? nothing : rk.fitted.frozen
    program, bound, model = _rk_fitted_build(emitted, plan, frozen, nothing)
    working = RKBRMI(parent(rk), plan, model; program, bound, source=brm_source(rk), fitted=rk.fitted)
    _rk_layout_coordinate_names(working) == _rk_layout_coordinate_names(rk) ||
        error("Laplace integration changed the model's coordinate order")
    working
end

function _brm_laplace_draw_frame(working, walks, integrated)
    isempty(walks) && return nothing
    position = Dict(s => i for (i, s) in enumerate(_rk_layout_coordinate_names(working)))
    blocks = map(walks) do r
        n = length(r.steps)
        if r.kind === :rw
            ids = [position[Symbol(r.declaration, ".", r.locals.basis, ".", i)] for i in 1:n]
            (; kind=:rw, r.declaration, ids, shape=(1, n),
                port=Symbol(r.declaration, ".", r.locals.sigma))
        else
            ids = vec([position[Symbol(r.declaration, ".", r.locals.eta, ".", g, ".", t)]
                for g in eachindex(r.levels), t in 1:n])
            (; kind=:cdar, r.declaration, ids, shape=(length(r.levels), n),
                port=Symbol(r.declaration, ".", r.locals.rho))
        end
    end
    _BRMLaplaceFieldFrame(working, blocks, integrated)
end

_brm_laplace_public_point!(p, u) = u
function _brm_laplace_public_point!(p::BRMSparseLaplaceProblem, u)
    frame = p.walk_frame
    isnothing(frame) && return u
    values = _brm_laplace_field_values(frame, u)
    for b in frame.blocks
        if b.kind === :rw
            field = view(u, b.ids)
            field .= brm_rw2_coordinates(values[b.port], field)
        else
            rho = values[b.port]
            scale = sqrt(1 - rho^2)
            field = reshape(view(u, b.ids), b.shape)
            # Reverse order leaves the preceding state intact in this in-place
            # conversion back to the public standardized innovations.
            for t in size(field, 2):-1:2, g in axes(field, 1)
                field[g, t] = (field[g, t] - rho * field[g, t - 1]) / scale
            end
        end
    end
    u
end

_brm_laplace_releases_field(p, selected) = p.walk_frame !== nothing &&
    any(b -> b.declaration ∉ selected, p.walk_frame.blocks)
