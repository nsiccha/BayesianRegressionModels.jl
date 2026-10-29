# `RKBRMI(brmi; car_units)`: every `car(...)` field sampled by its unit border
# separator (`_brm_car_separation`), so that each unit's interior areas are
# that unit's standard-normal draws and only the separator values and the
# units' interior sums are shared. The field, its density (up to a constant
# Jacobian) and its graph are those of the default lowering.
#
# The units are labels row-aligned with the model rows: the name of a column
# the model reads, or the labels themselves. An area's unit is the unit of its
# rows; an area with no rows, or with rows in several units, is a separator
# area.

# A fit's separations, which a frozen replay reuses with its fitted graphs, by
# each field's predictor and area column; `units` is the caller's `car_units`.
struct _RKFittedCarUnits
    units::Any
    separations::Dict{Tuple{Symbol,Symbol},Any}
end

_rk_plan_car_units(plan, ::Nothing) = plan
_rk_plan_car_units(plan, units) = _rk_separate_car_fields(plan, units)
_rk_separate_car_fields(plan::_RKHeldOutPlan, units) =
    _RKHeldOutPlan(_rk_separate_car_fields(plan.parent, units), plan.held_out)
function _rk_separate_car_fields(plan::_RKValuePlan, units)
    columns = Dict{Symbol,Any}(plan.columns)
    regression = _rk_separate_car_fields(plan.regression, units;
        labels_from=plan.columns, taken=_rk_plan_taken(plan), added=columns)
    _RKValuePlan(regression, plan.assignments, plan.observations, columns,
        plan.completions)
end
_rk_separate_car_fields(plan, _units) = error(
    "RK backend: `car_units` samples `car(...)` fields by their unit border " *
    "separator, but this model lowers to a $(nameof(typeof(plan))), which has no " *
    "`car(...)` term")
function _rk_separate_car_fields(plan::_RKStructuralPlan, units;
        labels_from=plan.columns, taken=_rk_plan_taken(plan), added=nothing)
    any(t -> t.kind === :car, (t for p in plan.predictors for t in p.terms)) || error(
        "RK backend: `car_units` samples `car(...)` fields by their unit border " *
        "separator, but this model has no `car(...)` term")
    columns = Dict{Symbol,AbstractVector}(plan.columns)
    predictors = map(plan.predictors) do predictor
        terms = map(predictor.terms) do term
            term.kind === :car || return term
            fit = _rk_car_unit_separation(units, predictor.name, term.options, labels_from)
            separated = _rk_car_separated_options(term.options, fit, columns, taken)
            _RKTermSpec(term.kind, term.columns, separated, term.addressee, term.label)
        end
        _RKPredictorSpec(predictor.name, predictor.link, terms, predictor.label,
            predictor.row_source)
    end
    added === nothing || merge!(added, columns)
    _RKStructuralPlan(plan.responses, predictors, plan.population_priors,
        plan.parameters, plan.assignments, plan.derived, columns, plan.n_obs,
        plan.ranef_buckets, plan.vector_parameters, plan.r2d2_priors,
        plan.horseshoe_priors)
end

function _rk_car_unit_labels(units::Symbol, columns, area::Symbol)
    haskey(columns, units) || error(
        "RK backend: `car_units = :$units` names no column the model reads; pass " *
        "the labels themselves (`car_units = data.$units`), row-aligned with the " *
        "rows of the `car($area)` areas")
    columns[units]
end
_rk_car_unit_labels(units::AbstractVector, _columns, _area::Symbol) = units
_rk_car_unit_labels(units, _columns, _area::Symbol) = throw(ArgumentError(
    "car_units must name a column (a Symbol) or be a vector of unit labels, got " *
    "$(typeof(units))"))

# A field's separation over the caller's unit labels: the units in their level
# order, each area's unit (`0`: none or several), and `_brm_car_separation`.
function _rk_car_unit_separation(units, predictor::Symbol, options, labels_from)
    area_labels = labels_from[options.area]
    labels = _rk_car_unit_labels(units, labels_from, options.source)
    length(labels) == length(area_labels) || error(
        "RK backend: `car_units` has $(length(labels)) labels, but `car($(options.source))` " *
        "has $(length(area_labels)) rows")
    levels = Any[_rk_grouping_levels(labels)...]
    unit_index = Dict(_brm_transport_level(l) => k for (k, l) in enumerate(levels))
    nodes = options.nodes
    node_unit = fill(-1, length(nodes))
    for (node, label) in zip(_brm_car_row_idx(nodes, area_labels), labels)
        u = unit_index[_brm_transport_level(label)]
        node_unit[node] = node_unit[node] == -1 || node_unit[node] == u ? u : 0
    end
    replace!(node_unit, -1 => 0)
    n_struct = options.n_struct
    struct_idx = [p <= n_struct ? p : 0 for p in labels_from[options.struct_pos]]
    unit = zeros(Int, n_struct)
    for (k, p) in enumerate(struct_idx)
        p > 0 && (unit[p] = node_unit[k])
    end
    graph = (; n_struct, node1=labels_from[options.node1], node2=labels_from[options.node2],
        comp_start=first.(options.components), comp_size=last.(options.components))
    (; units, levels, nodes, node_unit,
       separation=_brm_car_separation(graph, unit, length(levels)))
end

# A frozen replay keeps the fit's graph, and with it the fit's separation.
function _rk_car_unit_separation(fitted::_RKFittedCarUnits, predictor::Symbol, options,
        _labels_from)
    fit = get(fitted.separations, (predictor, options.source), nothing)
    fit === nothing && error("RK reprocess: internal: `car($(options.source))` of " *
        "`$predictor` has no fitted separation")
    isequal(fit.nodes, options.nodes) || error(
        "RK reprocess: `car($(options.source))` was refitted on the new data, so the " *
        "fit's unit border separation does not apply to it")
    fit
end

function _rk_car_separated_options(options, fit, columns, taken)
    sep, levels = fit.separation, fit.levels
    stem = string(options.value, "_sep")
    column(field, value) = begin
        name = _rk_mint_generated!(taken, columns, string(stem, "_", field))
        columns[name] = value
        name
    end
    blocks = NamedTuple[]
    first_draw = 1
    for (k, u) in enumerate(sep.units)
        isempty(u.areas) && continue
        draws = u.factor.ndraws
        # The unit's sparse interior map, read whole by its in-graph call.
        push!(blocks, (; unit=k, label=levels[k], areas=length(u.areas),
            ncols=length(u.cols), ndraws=draws,
            cols=isempty(u.cols) ? nothing : column(Symbol(:cols_, k), u.cols),
            factor=column(Symbol(:U_, k), [u.factor]),
            draw_range=first_draw:(first_draw + draws - 1)))
        first_draw += draws
    end
    x_phi = findall(>(0), sep.x_areas)
    separation = (; fit, levels, m=sep.m, ndraws=first_draw - 1,
        F=sep.m == 0 ? nothing : column(:F, [sep.F]),
        x_phi=isempty(x_phi) ? nothing : column(:x_phi, x_phi),
        # `phi` in graph order is `vcat(x[x_phi], phi_I_1, …)[restore]`
        restore=column(:restore, invperm(sep.order)), blocks=Tuple(blocks),
        x_areas=sep.x_areas, x_sums=sep.x_sums,
        # the unit label of each draw, and of each area's BYM2 iid part
        draw_units=Any[levels[b.unit] for b in blocks for _ in b.draw_range],
        node_units=Any[u == 0 ? nothing : levels[u] for u in fit.node_unit],
        shared=sep.shared)
    generated = Symbol[name for name in (separation.F, separation.x_phi,
        separation.restore) if name !== nothing]
    for b in blocks, name in (b.cols, b.factor)
        name === nothing || push!(generated, name)
    end
    merge(options, (; separation, inputs=(options.inputs..., generated...)))
end

# The separations a lowering used, which a frozen replay reuses; `nothing` for
# a lowering without `car_units`.
function _rk_fitted_car_units(plan)
    fits = Dict{Tuple{Symbol,Symbol},Any}()
    for predictor in _rk_replay_predictors(plan), term in predictor.terms
        term.kind === :car && haskey(term.options, :separation) || continue
        fits[(predictor.name, term.options.source)] = term.options.separation.fit
    end
    isempty(fits) ? nothing : _RKFittedCarUnits(first(values(fits)).units, fits)
end

# The `car_units` a rebuild on new data (`freeze_constants=false`) passes again:
# a column name; labels of the fit's rows do not label the new ones.
_rk_rebuilt_car_units(::Nothing) = nothing
function _rk_rebuilt_car_units(fitted::_RKFittedCarUnits)
    fitted.units isa Symbol && return fitted.units
    error("RK reprocess: this backend was lowered with `car_units` labels for the " *
        "fit's rows, which do not label the new data's rows. Rebuild it with " *
        "`RKBRMI(brmi; car_units = <labels of the new rows>)`, or lower the fit " *
        "with `car_units` naming a column.")
end
