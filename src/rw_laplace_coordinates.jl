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
        # Reading the physical path also covers fixed and model-dependent walk
        # scales. Repeated observations of a step count once in the projection.
        rows = Dict(v => i for (i, v) in enumerate(rk_bound_data(working)[r.time]))
        trend = collect(1:n) .- (n + 1) / 2
        (; r.declaration, ids, rows=[rows[t] for t in r.steps], slope=trend ./ sum(abs2, trend))
    end
    (; blocks, read=_rk_value_reader(working, Tuple(b.declaration for b in blocks)))
end

_brm_laplace_public_point!(p, u) = u
function _brm_laplace_public_point!(p::BRMSparseLaplaceProblem, u)
    frame = p.walk_frame
    isnothing(frame) && return u
    paths = frame.read(u)
    for b in frame.blocks
        field = view(u, b.ids)
        path = paths[b.declaration]
        level = sum(field) / (0.001 * length(field))
        slope = dot(b.slope, view(path, b.rows))
        for i in length(field):-1:3
            field[i] = field[i] - 2 * field[i-1] + field[i-2]
        end
        field[1], field[2] = level, slope
    end
    u
end

_rw2_releases_walk(p, selected) = p.walk_frame !== nothing &&
    any(b -> b.declaration ∉ selected, p.walk_frame.blocks)
