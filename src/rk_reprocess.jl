# src/rk_reprocess.jl — frozen replay of a fitted RKBRMI on new data.
#
# RK preprocessing is in-graph: the emitted program computes standardization
# anchors, basis fits and level positions from its bound inputs. A replay binds
# new data to the SAME program, with the fitted values frozen at the training
# data's (user principle 077lr1b: preprocessing in-graph, frozen by partial
# evaluation):
#
# - every in-graph anchor reads a fit-role input (`_rk_fit_twin!`): the same
#   array at fit time, the training column at replay, so RKPPL's bind-time
#   evaluation computes the training anchors while the per-row transform reads
#   the new rows;
# - planning the new data runs under `_BRMFrozenReplay`, so every Julia-side
#   fit (level sets, the planner's own value checks) and every
#   `name = constant(expr)` takes its training value.
#
# The replay's emitted program must equal the fit's, statement for statement:
# anything a fit site still derived from the new data would show up as a
# different literal, and is refused rather than evaluated. A numeric scalar a
# sampled parameter's law reads by name has no literal to differ, so its value
# must equal the fit's too (`_rk_check_replay_priors`).
#
# A design may omit responses (prediction). Each omitted response plans as a
# `_BRMAbsentResponse`: the fit's route, no values. It is never bound, and
# RKPPL binds its observation without a value (rkppl-use §14), so values that
# do not read it evaluate and the density is refused, naming it.

"""
    reprocess(rk::RKBRMI, new_data; freeze_constants=true, resample_groups=(),
              held_out=<the fit's hold-outs>) -> RKBRMI

Bind `new_data` to the fitted RK model `rk` with the TRAINING fit's
preprocessing, reusing its executable graph (`rk.model`; no rebuild) and its
translated program: a replay only binds its own data. The
returned backend evaluates the same draws on the new rows: densities through
[`rk_logdensity_problem`](@ref), authored values through
`ReactiveKernels.prepare(rk2.model.spec; have, want, bound=rk_bound_data(rk2))`,
and RKPPL's queries over a binding, posterior-predictive draws included,
through [`rk_bound_plan`](@ref).

- `freeze_constants=true` (default; prediction and replay): `center` /
  `zscale` / `standardize` anchors, completed-covariate anchors, factor,
  `mo` and group level sets, spline fits, HSGP intervals and length-scale
  floors, R2D2 variances and `name = constant(expr)` values are the training
  fit's. A one-row design is fine. An unseen factor, `mo` or group level is
  an error.
- `freeze_constants=false`: fresh-fit semantics, equivalent to rebuilding on
  `new_data` (`RKBRMI` of the rebound model); its coordinates need not match
  `rk`'s.
- `resample_groups=[:g]`: group `g` takes its levels from `new_data` (new
  subjects). Its effects are not the fit's, so supply them in `have`
  (`have=(:unconstrained, :b_p_g, ports...)`); everything else stays frozen.
- `held_out`: the response hold-outs of the replay, as in `RKBRMI`.

`new_data` supplies every data column the model reads. With
`freeze_constants=true` it may omit numeric scalar inputs (their fitted values
are kept) and responses: responses are bound without values,
so authored values (`want=(:mu,)`, `@plate` cell outputs) evaluate on the new
rows, while the density and other queries that read an omitted response are
refused, naming it. A model whose planning reads the response values (for
example an `mi(y)` response or a bounded likelihood) needs them.

A supplied numeric scalar a prior reads must equal the fitted value, or the
replay is refused: a design cannot change a fitted prior. That covers every
sampled parameter's law, a `@plate` cell latent's included, and a scalar it
reads through an assignment. A scalar only a `@plate` cell's values or
observations read (a dose, a known observation scale) is a design input, and
takes the design's value.
"""
function reprocess(rk::RKBRMI, new_data; freeze_constants::Bool=true,
        resample_groups=(), held_out=_rk_held_out_names(rk.plan))
    prefix = "RK reprocess"
    car_units = _rk_fitted_car_units(rk.plan)
    freeze_constants || return RKBRMI(_brm_rebind_brmi(parent(rk), new_data); held_out,
        completions=_rk_completion_mode(rk.plan),
        car_units=_rk_rebuilt_car_units(car_units))
    _rk_refuse_level_correlated_resample(parent(rk),
        _rk_resample_group_set(resample_groups; prefix); prefix)
    _rk_refuse_grouped_hsgp_resample(parent(rk),
        _rk_resample_group_set(resample_groups; prefix); prefix)
    _brm_refuse_joint_resample(parent(rk),
        _rk_resample_group_set(resample_groups; prefix); prefix)
    # A data-free backend (`RKBRMI(brmi, state)`) replays from its fitted state.
    rk.fitted === nothing || return _rk_fitted_replay(brm_source(rk), new_data,
        _rk_fitted_state(rk); resample_groups, held_out, previous=rk, prefix)
    groups = _rk_resample_group_set(resample_groups; prefix)
    training = _rk_frozen_training(parent(rk))
    unknown = setdiff(groups, keys(training))
    isempty(unknown) || error(
        "$prefix: `resample_groups` names $(sort!(collect(unknown))), which " *
        "is not a data column of the fitted model")
    absent = Dict{Symbol,Any}(name => _BRMAbsentResponse{eltype(training[name])}(name)
        for name in _rk_response_names(rk.plan)
        if haskey(training, name) && !_brm_df_has_column(new_data, name))
    merge!(absent, _rk_car_fitted_edges(parent(rk), new_data, training))
    merge!(absent, _rk_omitted_scalars(_rk_scalar_data(parent(rk)), new_data))
    rebound = _brm_rebind_brmi(parent(rk), _BRMReplayData(new_data, absent))
    _, fitted = _brm_resolve_constants(parent(rk); prefix)
    constants = Dict{Symbol,Any}(c.name => c.value for c in fitted)
    plan = _brm_with_frozen_replay(_BRMFrozenReplay(training, groups)) do
        _rk_plan_car_units(_rk_plan_completions(_brm_rk_plan(rebound; held_out,
            fitted_constants=(name, _source) -> get(constants, name, nothing)),
            _rk_completion_mode(rk.plan)), car_units)
    end
    retained = _rk_retained_program(rk)
    replayed = _rk_emit_ast(plan)
    _rk_check_replay_program(retained, replayed; prefix)
    _rk_check_replay_priors(replayed, plan, _rk_scalar_data(parent(rk)), new_data; prefix)
    _rk_check_replay_terms(plan; prefix)
    _rk_check_replay_groupings(plan, groups; prefix)
    # The fit's program, translated once, binds the replay's data too, unless
    # a callable differs under the same name.
    program = _rk_same_callables(retained, replayed) ? rk.program : _RKOnce()
    replay = RKBRMI(parent(rk), plan, rk.model; program, source=brm_source(rk))
    # Same coordinates, so the fit's draws mean the same thing on the replay.
    # Resampled groups size their effects by the new data's levels.
    if isempty(groups)
        fitted_names = _rk_layout_coordinate_names(rk)
        replay_names = _rk_bound_coordinate_names(replay)
        fitted_names == replay_names || _rk_refuse_replay_coordinates(plan,
            fitted_names, replay_names; prefix)
    end
    replay
end

# A replay whose coordinates differ from the fit's. A `@plate` cell latent has
# one draw per cell entry, so new cells change it; no grouping resample covers
# that, so the refusal names the latent.
function _rk_refuse_replay_coordinates(plan, fitted_names, replay_names; prefix)
    count_of(names, root) = count(name -> _rk_record_root(name) === root, names)
    latents = [record.name for record in _rk_coordinate_records(plan)
        if record.kind === :cell_latent && count_of(fitted_names, record.declaration) !=
            count_of(replay_names, record.declaration)]
    isempty(latents) || error(
        "$prefix: the new data's `@plate` cells give the cell latent(s) " *
        "$(join(("`$name`" for name in latents), ", ")) a different number of " *
        "draws than the fit's cells ($(length(fitted_names)) fitted coordinates, " *
        "$(length(replay_names)) on the replay). A cell latent's draws belong to " *
        "the fit's cells: replay onto those cells, or rebuild with " *
        "`freeze_constants=false` for new cells.")
    error("$prefix: the new data changes the fitted coordinates " *
        "($(length(fitted_names)) fitted, $(length(replay_names)) on the " *
        "replay), so the fit's draws do not apply to it. A grouping whose " *
        "levels the replay does not freeze must be named in " *
        "`resample_groups`.")
end

# The emitted program a backend's queries read, and the coordinate names its
# binding packs (ReactiveKernels package extension).
function _rk_retained_program end
function _rk_bound_coordinate_names end

_rk_same_callables(a::_RKEmittedProgram, b::_RKEmittedProgram) =
    length(a.bindings) == length(b.bindings) &&
    all(((x, y),) -> first(x) === first(y) && last(x) === last(y),
        zip(a.bindings, b.bindings))

_rk_replay_predictors(plan::_RKStructuralPlan) = plan.predictors
_rk_replay_predictors(plan::_RKValuePlan) = plan.regression.predictors
_rk_replay_predictors(plan::_RKHeldOutPlan) = _rk_replay_predictors(plan.parent)
_rk_replay_predictors(_plan) = ()

_rk_replay_responses(plan::_RKStructuralPlan) = plan.responses
_rk_replay_responses(plan::_RKHeldOutPlan) = _rk_replay_responses(plan.parent)
_rk_replay_responses(_plan) = ()

# Latent values that belong to the training rows themselves (an exact GP's
# values at its locations, an AR/DAR path over its time steps, measurement-
# error latents, structured group fields) have no meaning at other rows, and
# a replay with the same number of rows would silently reinterpret them.
const _RK_ROW_LATENT_TERMS = (gp="an exact `gp(...)`", ar="an `ar(...)`/`rw(...)` path",
    dar="a `dar(...)` path", cdar="a `cdar(...)` walk", me="a measurement-error `me(...)`",
    structured="a structured group field")
function _rk_check_replay_terms(plan; prefix)
    for predictor in _rk_replay_predictors(plan), term in predictor.terms
        haskey(_RK_ROW_LATENT_TERMS, term.kind) || continue
        error("$prefix: $(_RK_ROW_LATENT_TERMS[term.kind]) term of `$(predictor.name)` " *
            "holds latent values of the training rows, which do not carry to new " *
            "rows; rebuild with `freeze_constants=false`")
    end
end

_rk_replay_buckets(plan::_RKStructuralPlan) = plan.ranef_buckets
_rk_replay_buckets(plan::_RKValuePlan) = plan.regression.ranef_buckets
_rk_replay_buckets(plan::_RKHeldOutPlan) = _rk_replay_buckets(plan.parent)
_rk_replay_buckets(_plan) = ()

# Plain blocks read their levels from a fit-role copy of the grouping column;
# `mm(...)` and `gr(...; by=...)` blocks still read the bound columns' levels,
# so a replay of their groups is refused unless they are resampled.
# Finite level-correlation factors cover the fitted level set; new levels
# would need a larger one. Coordinate multigroup fields instead share fixed
# spatial weights and evaluate the field at the new levels' locations.
function _rk_refuse_level_correlated_resample(brmi, groups::Set{Symbol}; prefix)
    isempty(groups) && return nothing
    correlated = Symbol[_brm_grouping_column(d) for d in _brm_group_declarations(brmi)
                        if !isnothing(_brm_declaration_level_cor(d))]
    for predictor in linear_predictors(brmi)
        operation = linear_predictor_op(brmi, predictor.name)
        isnothing(operation) && continue
        for term in _brm_additive_terms(last(getargs(operation, 2)))
            term isa ExprColumn && getf(term) === hsgp || continue
            kw = getkwargs(term)
            haskey(kw, :cor) && !_brm_multigroup_coordinates_form(kw.cor) &&
                push!(correlated, name(kw.by))
        end
    end
    hit = sort!(collect(intersect(groups, correlated)))
    isempty(hit) || error(
        "$prefix: `resample_groups` names $(hit), whose levels are correlated " *
        "(`gr(...; cor=...)` / `hsgp(...; by=..., cor=...)`): the correlation " *
        "covers the fitted levels only, and new levels of a level-correlated " *
        "grouping are not implemented")
    nothing
end

# An independent grouped HSGP (`hsgp(...; by=g)`) draws its basis weights, and
# the level effects of a `log(sd(hsgp(...))) ~ 1 + (1 | g)` hyper-predictor,
# inside its own block, one row per level: unlike a `(… | g)` block's effects,
# a replay has no input through which a caller supplies new levels' values.
# A design that reuses the fitted labels keeps the fit's program and would read
# the fitted levels' weights as the new levels' (measured: equal values); any
# other design changes the program. Refuse both by name. SBBRMI draws new
# levels in generated quantities (`_sb_emit_group_blocks!`).
function _rk_refuse_grouped_hsgp_resample(brmi, groups::Set{Symbol}; prefix)
    isempty(groups) && return nothing
    grouped = Symbol[]
    for predictor in linear_predictors(brmi)
        operation = linear_predictor_op(brmi, predictor.name)
        isnothing(operation) && continue
        for term in _brm_additive_terms(last(getargs(operation, 2)))
            term isa ExprColumn && getf(term) === hsgp || continue
            kw = getkwargs(term)
            haskey(kw, :by) || continue
            haskey(kw, :cor) && _brm_multigroup_coordinates_form(kw.cor) && continue
            push!(grouped, name(kw.by))
        end
    end
    hit = sort!(collect(intersect(groups, grouped)))
    isempty(hit) || error(
        "$prefix: `resample_groups` names $(hit), the grouping of a grouped " *
        "`hsgp(...; by=...)`: its new levels' basis weights have no input on " *
        "RKBRMI (a design reusing the fitted labels would read the fitted " *
        "levels' weights). SBBRMI draws new levels of a grouped HSGP: " *
        "`reprocess(sb, rows; resample_groups=...)`")
    nothing
end

function _rk_check_replay_groupings(plan, groups::Set{Symbol}; prefix)
    for bucket in _rk_replay_buckets(plan)
        bucket.decomposition === nothing || error(
            "$prefix: a random-effect R2D2M2 budget (`sd(...) ~ r2d2(...)`) does " *
            "not freeze its column variances yet; rebuild with `freeze_constants=false`")
        grouping = bucket.grouping
        grouping.form === :plain && continue
        columns = Symbol[grouping.columns...]
        grouping.by === nothing || push!(columns, grouping.by)
        all(in(groups), columns) && continue
        error("$prefix: `$(grouping.form)(...)` random-effect blocks do not " *
            "freeze their group levels yet; name $(columns) in " *
            "`resample_groups` and supply their effects, or rebuild with " *
            "`freeze_constants=false`")
    end
end

# A replay value the backend could not compute from the replay rows. The
# replay data lacks it, and reading it raises the cause.
struct _BRMReplayFailure
    message::String
end

# Replay data with some values supplied by the backend: an omitted response
# standing in as a `_BRMAbsentResponse`, a resolved constant, or a data-only
# assignment evaluated on the replay rows (`_BRMReplayFailure` when it cannot be).
struct _BRMReplayData{D}
    data::D
    values::Dict{Symbol,Any}
end
function Base.hasproperty(d::_BRMReplayData, key::Symbol)
    values = getfield(d, :values)
    haskey(values, key) ? !(values[key] isa _BRMReplayFailure) :
        hasproperty(getfield(d, :data), key)
end
function Base.getproperty(d::_BRMReplayData, key::Symbol)
    values = getfield(d, :values)
    haskey(values, key) || return getproperty(getfield(d, :data), key)
    value = values[key]
    value isa _BRMReplayFailure && error(value.message)
    value
end

# Raise why the replay data lacks `key` when the backend could not compute it.
_brm_raise_replay_failure(_data, _key::Symbol) = nothing
function _brm_raise_replay_failure(d::_BRMReplayData, key::Symbol)
    value = get(getfield(d, :values), key, nothing)
    value isa _BRMReplayFailure && error(value.message)
    nothing
end

# A frozen replay keeps each `car(area; edges=..., nodes=...)` term's fitted
# graph and reads only `area`, so edge vectors, and a declared `nodes=` vector,
# the new data omits keep their fitted values. A fresh replay rebuilds the
# graph from the new data and needs them.
function _rk_car_fitted_edges(brmi::BRMI, new_data, training)
    edges = Dict{Symbol,Any}()
    for predictor in linear_predictors(brmi)
        operation = linear_predictor_op(brmi, predictor.name)
        isnothing(operation) && continue
        for term in _brm_additive_terms(last(getargs(operation, 2)))
            term isa ExprColumn && getf(term) === car || continue
            kw = getkwargs(term)
            for column in (haskey(kw, :nodes) ? (kw.edges..., kw.nodes) : kw.edges)
                key = name(column)
                _brm_df_has_column(new_data, key) || (edges[key] = training[key])
            end
        end
    end
    edges
end

# Every response the fit conditions on, held out or not.
_rk_response_names(plan) = _rk_observed_names(plan)
_rk_response_names(plan::_RKHeldOutPlan) = _rk_observed_names(plan.parent)

_rk_held_out_names(plan) = ()
_rk_held_out_names(plan::_RKHeldOutPlan) = Tuple(sort!(collect(plan.held_out)))

function _rk_resample_group_set(groups; prefix)
    groups === nothing && return Set{Symbol}()
    groups isa Symbol && return Set{Symbol}((groups,))
    names = collect(groups)
    all(g -> g isa Symbol, names) || error(
        "$prefix: `resample_groups` expects a Symbol or a collection of Symbols")
    Set{Symbol}(names)
end

# The training data a frozen fit reads, by column name: the materialized
# columns planning reads, and the raw value of any other data column.
function _rk_training_columns(brmi::BRMI)
    columns = Dict{Symbol,Any}()
    for op in _brm_operation_values(brmi)
        _rk_raw_data_columns!(columns, op)
    end
    merge!(columns, _brm_backend_context(brmi; retain_mm_sources=true).data)
end

# A fit's numeric scalar inputs. They carry no training rows and can supply
# omitted replay fields. A present field cannot silently change a prior: where
# the fitted program holds it as a literal, the program identity check sees the
# change; where the program reads it by name, `_rk_check_replay_priors` does.
_rk_scalar_data(brmi::BRMI) = Dict{Symbol,Any}(name => deepcopy(value)
    for (name, value) in _rk_training_columns(brmi) if value isa Real)
_rk_omitted_scalars(scalars, new_data) = Dict{Symbol,Any}(name => value
    for (name, value) in scalars if !_brm_df_has_column(new_data, name))

# The names the laws of a program's sampled declarations read, directly or
# through the assignments they read. An observation's law is no prior, and a
# callee's arguments count only where its own laws read them: a component's
# design matrix reads no prior, its location and scale do.
_rk_prior_reads(emitted::_RKEmittedProgram, observed) = _rk_law_inputs(emitted.main,
    Set{Symbol}(observed), Dict{Symbol,Any}(part.name => part
        for part in map(_rk_definition_parts, emitted.defs)), Dict{Symbol,Set{Symbol}}())

function _rk_definition_parts(definition)
    definition = _brm_undoc(definition)
    body = Meta.isexpr(definition, :macrocall) ?
        only(filter(arg -> !(arg isa LineNumberNode), definition.args[2:end])) : definition
    signature, statements = body.args
    parameter(arg) = Meta.isexpr(arg, :(::)) ? first(arg.args) : arg
    (; name=first(signature.args), parameters=map(parameter, signature.args[2:end]), statements)
end

function _rk_law_inputs(statements, observed, callees, inputs)
    reads = Set{Symbol}()
    assigned = Dict{Symbol,Vector{Any}}()
    _rk_scan_laws!(reads, assigned, statements, observed, callees, inputs)
    pending = collect(reads)
    while !isempty(pending)
        for value in get(assigned, pop!(pending), ()), name in _rk_value_names!(Symbol[], value)
            name in reads || (push!(reads, name); push!(pending, name))
        end
    end
    reads
end

_rk_scan_laws!(_reads, _assigned, _statement, _observed, _callees, _inputs) = nothing
function _rk_scan_laws!(reads, assigned, statement::Expr, observed, callees, inputs)
    scan(s) = _rk_scan_laws!(reads, assigned, s, observed, callees, inputs)
    if statement.head === :call && length(statement.args) == 3 &&
            first(statement.args) in (:~, :.~)
        _rk_emitted_observation_name(statement.args[2]) in observed ||
            _rk_law_reads!(reads, statement.args[3], callees, inputs)
    elseif statement.head === :(=)
        for name in _rk_source_lhs!(Set{Symbol}(), first(statement.args))
            push!(get!(assigned, name, Any[]), last(statement.args))
        end
    elseif statement.head === :for
        scan(last(statement.args))
    elseif statement.head in (:macrocall, :block)
        foreach(scan, statement.args)
    end
    nothing
end

function _rk_law_reads!(reads, law, callees, inputs)
    callee = Meta.isexpr(law, :call) ? get(callees, first(law.args), nothing) : nothing
    callee === nothing && return _rk_value_names!(reads, law)
    # Inside a definition, `~` on a parameter observes what the caller passes.
    used = get!(inputs, callee.name) do
        _rk_law_inputs(callee.statements, Set{Symbol}(callee.parameters), callees, inputs)
    end
    arguments = law.args[2:end]
    length(arguments) == length(callee.parameters) || return _rk_value_names!(reads, law)
    for (parameter, argument) in zip(callee.parameters, arguments)
        parameter in used && _rk_value_names!(reads, argument)
    end
    reads
end

# The names a value reads: not call heads, broadcast callees or keyword names.
_rk_value_names!(names, _value) = names
_rk_value_names!(names, value::Symbol) = push!(names, value)
function _rk_value_names!(names, value::Expr)
    skip = value.head in (:call, :kw, :macrocall) ||
        (value.head === :. && !(last(value.args) isa QuoteNode)) ? 1 : 0
    foreach(arg -> _rk_value_names!(names, arg), value.args[skip+1:end])
    names
end

# A scalar the program reads by name is the same program at any value, so the
# identity check cannot see a design change the law that reads it. Refuse it
# with that check's error. A scalar no law reads is a design input.
function _rk_check_replay_priors(emitted, plan, scalars, new_data; prefix)
    reads = _rk_prior_reads(emitted, _rk_response_names(plan))
    fitted, replay = String[], String[]
    for name in sort!(collect(intersect(reads, keys(scalars))))
        _brm_df_has_column(new_data, name) || continue
        value = _brm_df_column(new_data, name)
        isequal(value, scalars[name]) && continue
        shown = sprint(show, value; context=:limit => true)
        push!(fitted, "prior input: $name = $(scalars[name])")
        push!(replay, "prior input: $name = $shown")
    end
    _rk_check_replay_program(fitted, replay; prefix)
end

# What a frozen replay fits from (`_BRMFrozenReplay`): the fit's data columns
# and each data-only assignment's training value, which a fit
# (`standardize(tt)`) reads like a column (`_brm_resolve_data_assignments`).
function _rk_frozen_training(brmi::BRMI)
    training = _rk_training_columns(brmi)
    resolved, _ = _brm_resolve_constants(brmi; prefix="RK replay")
    _, assignments = _brm_resolve_data_assignments(resolved; prefix="RK replay")
    foreach(a -> training[a.name] = a.value, assignments)
    training
end

_rk_raw_data_columns!(columns, _x) = columns
function _rk_raw_data_columns!(columns, x::NamedColumn)
    backing = parent(x)
    backing isa DataColumn && (columns[name(x)] = parent(backing))
    _rk_raw_data_columns!(columns, backing)
end
_rk_raw_data_columns!(columns, x::ExprColumn) =
    (_brm_foreach_child(a -> _rk_raw_data_columns!(columns, a), x); columns)

function _rk_program_statements(emitted::_RKEmittedProgram)
    text(x) = sprint(Base.show_unquoted, Base.remove_linenums!(_brm_strip_docs(x)))
    vcat(["definition: " * text(d) for d in emitted.defs],
        ["statement: " * text(s) for s in emitted.main.args if !(s isa LineNumberNode)],
        ["binding: " * string(first(b)) for b in emitted.bindings])
end

_rk_check_replay_program(retained::_RKEmittedProgram, replayed::_RKEmittedProgram;
    prefix) = _rk_check_replay_program(_rk_program_statements(retained),
        _rk_program_statements(replayed); prefix)
function _rk_check_replay_program(fitted::Vector{String}, replay::Vector{String}; prefix)
    fitted == replay && return nothing
    k = something(findfirst(i -> i > length(replay) || fitted[i] != replay[i],
        eachindex(fitted)), length(fitted) + 1)
    error("$prefix: the new data changes the fitted RK program, so its " *
        "preprocessing cannot be frozen. First difference:\n" *
        "  fitted: $(get(fitted, k, "(end of program)"))\n" *
        "  replay: $(get(replay, k, "(end of program)"))\n" *
        "Check that the new data keeps the fitted levels and column types, or " *
        "rebuild with `freeze_constants=false`.")
end

"""
    rk_bound_data(rk::RKBRMI) -> NamedTuple

The bound inputs of `rk`'s executable graph for its own data: pass them as
`bound=` to `ReactiveKernels.prepare(rk.model.spec; have=(:unconstrained,
keys(bound)...), want, bound)`. For [`reprocess`](@ref)'d backends these are
the new data's inputs with the training preprocessing frozen. They are the
columns of [`rk_bound_plan`](@ref)`(rk)`. Implemented by the ReactiveKernels
package extension.
"""
function rk_bound_data end

# The identity of the density a backend binds: its emitted program and
# callables, its complete bound inputs (data and frozen preprocessing, not a
# truncated display or just their shapes) and its packed coordinate names,
# hashed under `tag`. SMC checkpoints identify their bindings with
# `tag="brm-smc-v1"`, SVI optimizer checkpoints with `tag="brm-svi-v1"`.
function _brm_rk_binding_identity(rk, tag)
    emitted = _rk_retained_program(rk)
    io = IOBuffer()
    serialize(io, (format=1, program=_rk_program_statements(emitted),
        callables=emitted.bindings, data=rk_bound_data(rk),
        coordinates=_rk_bound_coordinate_names(rk)))
    "$tag:" * bytes2hex(sha256(take!(io)))
end

"""
    rk_bound_plan(rk::RKBRMI) -> ReactiveKernelsPPL.StructuralPlan

The RKPPL binding of `rk`'s program to its own data: the bound plan every query
of `rk` evaluates on `rk.model`. For a [`reprocess`](@ref)'d backend it binds the
new data, with the training preprocessing frozen and the responses the data
omit left absent. `(rk.model, rk_bound_plan(rk))` is the `(built, plan)` pair
RKPPL's queries take (rkppl-use §14), so each of them applies to `rk` as
documented there. Posterior-predictive draws on a new design, for example:

```julia
replay = reprocess(rk, design; resample_groups=[:subject])
targets = predictive_targets(replay.model, rk_bound_plan(replay))
# WANT predictive_node(replay.model) in a cut over replay.model.spec, then
draws = predictive_draws(value, targets, rng)
```

Without `resample_groups`, `prepare_predictive(replay.model, rk_bound_plan(replay))`
draws the whole model at a draw `u` of the fit. Every query of `rk` reads this
one plan: read it, never mutate it. Implemented by the ReactiveKernels package
extension.
"""
function rk_bound_plan end
