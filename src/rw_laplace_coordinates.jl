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

# The sparse solver integrates the intrinsic RW2 in standardized field values.
# Only selected walks change frame. Other population coordinates keep the
# public model's meaning, even when several walks share a time column.
_rw2_field_plan(p::_RKHeldOutPlan, selected) =
    _RKHeldOutPlan(_rw2_field_plan(p.parent, selected), p.held_out)
_rw2_field_plan(p::_RKValuePlan, selected) = _RKValuePlan(
    _rw2_field_plan(p.regression, selected), p.assignments, p.observations, p.columns, p.completions)
function _rw2_field_plan(p::_RKStructuralPlan, selected)
    predictors = map(p.predictors) do predictor
        terms = map(predictor.terms) do t
            t.kind === :rw && t.options.value in selected || return t
            _RKTermSpec(t.kind, t.columns, merge(t.options, (; field_coordinates=true)),
                t.addressee, t.label)
        end
        _RKPredictorSpec(predictor.name, predictor.link, terms, predictor.label, predictor.row_source)
    end
    _RKStructuralPlan(p.responses, predictors, p.population_priors, p.parameters,
        p.assignments, p.derived, p.columns, p.n_obs, p.ranef_buckets, p.vector_parameters,
        p.r2d2_priors, p.horseshoe_priors, p.inputs, p.prior_reads)
end

function _rw2_field_backend(rk, declarations)
    walks = [r for r in _rk_coordinate_records(rk) if r.kind === :rw &&
        r.sum_to_zero && r.order == 2 && r.declaration in declarations]
    isempty(walks) && return rk, walks
    plan = _rw2_field_plan(rk.plan, Set(r.declaration for r in walks))
    emitted = _rk_emit_ast(plan; documentation=brm_documentation(brm_source(rk)))
    frozen = rk.fitted === nothing ? nothing : rk.fitted.frozen
    program, bound, model = _rk_fitted_build(emitted, plan, frozen, nothing)
    working = RKBRMI(parent(rk), plan, model; program, bound, source=brm_source(rk), fitted=rk.fitted)
    _rk_layout_coordinate_names(working) == _rk_layout_coordinate_names(rk) ||
        error("RW2 integration changed the model's coordinate order")
    working, walks
end

function _rw2_draw_frame(working, walks)
    isempty(walks) && return nothing
    position = Dict(s => i for (i, s) in enumerate(_rk_layout_coordinate_names(working)))
    blocks = map(walks) do r
        n = length(r.steps)
        ids = [position[Symbol(r.declaration, ".", r.locals.basis, ".", i)] for i in 1:n]
        (; r.declaration, ids, scale=Symbol(r.declaration, ".", r.locals.sigma))
    end
    # Read the model's scale port. It can be sampled, fixed or derived from
    # another parameter, and several walks can use the same observed grid.
    (; blocks, scales=_rk_value_reader(working, Tuple(b.scale for b in blocks)))
end

_brm_laplace_public_point!(p, u) = u
function _brm_laplace_public_point!(p::BRMSparseLaplaceProblem, u)
    frame = p.walk_frame
    isnothing(frame) && return u
    scales = frame.scales(u)
    for b in frame.blocks
        field = view(u, b.ids)
        field .= brm_rw2_coordinates(scales[b.scale], field)
    end
    u
end

_rw2_releases_walk(p, selected) = p.walk_frame !== nothing &&
    any(b -> b.declaration ∉ selected, p.walk_frame.blocks)
