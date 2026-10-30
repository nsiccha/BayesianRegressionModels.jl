# `RKBRMI(brmi; car_units)`: every `car(...)` field sampled by its unit border
# separator (`_brm_car_separation`), so that each unit's interior areas are
# read from that unit's standard-normal draws, `⌈log₂ n⌉` per area of a piece
# of `n`, and only the separator values and the units' interior means are shared. The field, its
# density (up to a constant) and its graph are those of the default lowering.
#
# The units are labels row-aligned with a field's rows: the name of a column
# the model reads, or the labels themselves, for every field; or one such entry
# per field, keyed by its area column (`car_units = (area = :units, …)`), for
# fields on different row sets; a field without an entry keeps the default
# lowering. An area's unit is the unit of its rows; an area with no rows, or
# with rows in several units, is a separator area.

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
    areas = unique!(Symbol[t.options.source for p in plan.predictors for t in p.terms
        if t.kind === :car])
    isempty(areas) && error(
        "RK backend: `car_units` samples `car(...)` fields by their unit border " *
        "separator, but this model has no `car(...)` term")
    _rk_check_car_unit_keys(units, areas)
    columns = Dict{Symbol,AbstractVector}(plan.columns)
    predictors = map(plan.predictors) do predictor
        terms = map(predictor.terms) do term
            term.kind === :car &&
                _rk_car_separates(units, predictor.name, term.options.source) || return term
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

# Which fields `car_units` separates, and the units of each: one entry for
# every field, or a field's own entry keyed by its area column.
_rk_car_separates(_units, _predictor::Symbol, _area::Symbol) = true
_rk_car_separates(units::NamedTuple, _predictor::Symbol, area::Symbol) = haskey(units, area)
_rk_car_field_units(units, area::Symbol) = units
_rk_car_field_units(units::NamedTuple, area::Symbol) = units[area]

_rk_check_car_unit_keys(_units, _areas) = nothing
function _rk_check_car_unit_keys(units::NamedTuple, areas)
    unknown = setdiff(keys(units), areas)
    isempty(unknown) || error(
        "RK backend: `car_units` names $(join(("`$k`" for k in unknown), ", ")), which " *
        "is no `car(...)` area column of this model; its `car` fields are over " *
        "$(join(("`$a`" for a in areas), ", "))")
    nothing
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
    "car_units must name a column (a Symbol), be a vector of unit labels, or give " *
    "each `car(...)` area column one of those (`car_units = (area = :units, …)`), " *
    "got $(typeof(units))"))

_rk_car_units_rows_error(_units, area::Symbol, nlabels, nrows) = error(
    "RK backend: `car_units` has $nlabels labels, but `car($area)` has $nrows rows. " *
    "Labels row-aligned with one row set cannot label a `car` field on another; " *
    "give each field labels of its own rows, keyed by its area column: " *
    "`car_units = ($area = <labels of its rows>, …)`")
_rk_car_units_rows_error(_units::NamedTuple, area::Symbol, nlabels, nrows) = error(
    "RK backend: `car_units.$area` has $nlabels labels, but `car($area)` has " *
    "$nrows rows; give each field labels row-aligned with its own rows")

# A field's separation over the caller's unit labels: the units in their level
# order, each area's unit (`0`: none or several), and `_brm_car_separation`.
function _rk_car_unit_separation(units, predictor::Symbol, options, labels_from)
    area_labels = labels_from[options.area]
    labels = _rk_car_unit_labels(_rk_car_field_units(units, options.source),
        labels_from, options.source)
    length(labels) == length(area_labels) || _rk_car_units_rows_error(units,
        options.source, length(labels), length(area_labels))
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

# A frozen replay keeps the fit's graph, and with it the fit's separation; a
# field the fit lowered without one keeps the default lowering.
_rk_car_separates(fitted::_RKFittedCarUnits, predictor::Symbol, area::Symbol) =
    haskey(fitted.separations, (predictor, area))
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
    optional(field, value) = isempty(value) ? nothing : column(field, value)
    blocks = NamedTuple[]
    first_draw = 1
    for (k, u) in enumerate(sep.units)
        isempty(u.areas) && continue
        push!(blocks, (; unit=k, label=levels[k], areas=length(u.areas), ndraws=u.ndraws,
            draw_range=first_draw:(first_draw + u.ndraws - 1)))
        first_draw += u.ndraws
    end
    shared = sep.m == 0 ? (; last=nothing, scale=nothing, weight=nothing,
            shared_first=nothing, shared_second=nothing) : begin
        data = _brm_car_shared_data(sep.blocks, sep.m)
        (; last=column(:last, data.last), scale=column(:scale, data.scale),
           weight=column(:weight, data.weight),
           shared_first=column(:shared_first, sep.shared_edges[1]),
           shared_second=column(:shared_second, sep.shared_edges[2]))
    end
    edge_first, edge_second, edge_units = sep.unit_edges
    nterms = length(edge_first) + length(sep.cancel)
    separation = (; fit, levels, m=sep.m, ndraws=sep.ndraws, shared...,
        msrc=column(:msrc, sep.msrc),
        reads=Tuple(column(Symbol(:read_, l), r) for (l, r) in enumerate(sep.reads)),
        left=sep.ndraws == 0 ? nothing : column(:left, sep.left),
        right=sep.ndraws == 0 ? nothing : column(:right, sep.right),
        edge_first=optional(:edge_first, edge_first),
        edge_second=optional(:edge_second, edge_second),
        cancel=optional(:cancel, sep.cancel),
        # the units' terms observe one placeholder each
        terms=nterms == 0 ? nothing : column(:terms, zeros(nterms)),
        blocks=Tuple(blocks), x_areas=sep.x_areas, x_means=sep.x_means,
        # the unit label of each draw, each unit term, and each area's BYM2 iid part
        draw_units=Any[levels[b.unit] for b in blocks for _ in b.draw_range],
        term_units=Any[levels[u] for u in vcat(edge_units, sep.cancel_units)],
        node_units=Any[u == 0 ? nothing : levels[u] for u in fit.node_unit],
        shared=sep.shared)
    generated = Symbol[name for name in (separation.last, separation.scale,
        separation.weight, separation.shared_first, separation.shared_second,
        separation.msrc, separation.reads..., separation.left, separation.right,
        separation.edge_first, separation.edge_second, separation.cancel,
        separation.terms) if name !== nothing]
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
# column names; labels of the fit's rows do not label the new ones.
_rk_rebuilt_car_units(::Nothing) = nothing
function _rk_rebuilt_car_units(fitted::_RKFittedCarUnits)
    units = fitted.units
    units isa Symbol && return units
    units isa NamedTuple && all(u -> u isa Symbol, values(units)) && return units
    error("RK reprocess: this backend was lowered with `car_units` labels for the " *
        "fit's rows, which do not label the new data's rows. Rebuild it with " *
        "`RKBRMI(brmi; car_units = <labels of the new rows>)`, or lower the fit " *
        "with `car_units` naming columns.")
end
