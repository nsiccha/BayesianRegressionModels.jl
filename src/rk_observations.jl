# Keep the full statistical program and its coordinate declarations when a
# response is withheld. Selection changes ordinary emitted observation source;
# it never edits an already-lowered RKPPL plan or caller-owned data.
struct _RKHeldOutPlan{P}
    parent::P
    held_out::Set{Symbol}
end
_rk_extent_bindings(plan::_RKHeldOutPlan) = _rk_extent_bindings(plan.parent)
_rk_input_bindings(plan::_RKHeldOutPlan) = _rk_input_bindings(plan.parent)
_rk_generated_inputs(plan::_RKHeldOutPlan) = _rk_generated_inputs(plan.parent)

# An unconditioned BRM program forward-simulates on the Stan route. Its
# declarations belong to generated quantities, not to an HMC coordinate pack.
# Retain the authoritative generative snapshot while emitting only its empty
# fitted density. This does not implement native generated draws.
struct _RKUnconditionedPlan{G}
    generative::G
    columns::Dict{Symbol,Any}
end
_rk_plan_summary(plan::_RKUnconditionedPlan) =
    "unconditioned program (zero fitted coordinates; " *
    string(length(plan.generative.declarations)) * " retained generative declarations)"
_rk_observed_names(::_RKUnconditionedPlan) = ()
_rk_emit_ast(::_RKUnconditionedPlan; documentation=()) =
    _RKEmittedProgram(Expr[], Expr(:block), Pair{Symbol,Any}[])

# The module authoring the model's cells, where the Stan program reads the
# functions they call; no keyword without a cell.
Base.@nospecializeinfer function _rk_cell_module(@nospecialize(brmi::BRMI))
    operations = _rk_kernel_ops(brmi)
    isempty(operations) && return (;)
    (; mod=_brm_inline_scope(first(getargs(last(first(operations))))))
end

Base.@nospecializeinfer function _rk_has_bound_observation(@nospecialize(brmi::BRMI))
    # Bound formula observations, including kernel-cell inputs, use the
    # ordinary planner. Empty bound arrays still retain this fitted role.
    for node in _brm_operation_values(brmi)
        node isa NamedColumn || continue
        operation = parent(node)
        operation isa ExprColumn{typeof(~)} || continue
        lhs, rhs = getargs(operation, 2)
        _brm_observation_name(lhs) === nothing || return true
        if rhs isa ExprColumn && getf(rhs) isa BRMComponent
            nested = _brm_component_instance(getf(rhs), rhs)
            _rk_has_bound_observation(nested) && return true
        end
        rhs isa ExprColumn && getf(rhs) === kernel || continue
        args = getargs(rhs)
        isempty(args) && continue
        parts = _sb_kernel_lambda_parts(first(args))
        parts === nothing && continue
        params, body = parts
        for (param, column) in zip(params, args[2:end])
            _sb_cell_param_observed(body, param) || continue
            _brm_observation_name(column) === nothing || return true
        end
    end
    false
end
Base.@nospecializeinfer function _rk_unconditioned_plan(@nospecialize(brmi::BRMI))
    _rk_has_bound_observation(brmi) && return nothing
    # The emitted declaration inventory also finds synthetic observations
    # (for example a modeled baseline). Endpoint absence alone must never
    # discard their densities or their inference coordinates.
    generated = generative_plan(SBBRMI(brmi; _rk_cell_module(brmi)...))
    any(d -> d.role === :observation && d.data_source !== nothing,
        generated.declarations) && return nothing
    _RKUnconditionedPlan(generated, Dict{Symbol,Any}(generated.data))
end
function Base.getproperty(plan::_RKHeldOutPlan, field::Symbol)
    field in (:parent, :held_out) ? getfield(plan, field) :
        getproperty(getfield(plan, :parent), field)
end
_rk_plan_summary(plan::_RKHeldOutPlan) =
    _rk_plan_summary(plan.parent) * " (withheld: " *
    join(string.(sort!(collect(plan.held_out))), ", ") * ")"

# Compilation of preparation must not scale with the entire BRMI source type.
# This barrier applies only to planning; emitted numerical kernels retain their
# ordinary specialization and contain exactly the same source and data.
Base.@nospecializeinfer function _brm_rk_plan(@nospecialize(brmi::BRMI); held_out=(),
        fitted_constants=(_name, _source) -> nothing)
    brmi = _brm_resolve_prior_addresses(brmi)
    # A `constant(...)` becomes a named data constant (a number or an array),
    # which the graph binds: evaluated here, or the fitted value on a frozen
    # `reprocess` replay.
    brmi, constants = _brm_resolve_constants(brmi; fitted=fitted_constants,
        prefix="RK backend")
    # A data-only assignment is planned as a data column and computed in the
    # graph from its own inputs (user decision `1tb1jrg`).
    brmi, assignments = _brm_resolve_data_assignments(brmi; prefix="RK backend")
    selected = _rk_held_out_selection(brmi, held_out)
    known_scalars = Ref{Any}(nothing)
    original = _brm_rk_unselected_plan(brmi, known_scalars)
    _rk_attach_data_assignments!(original, assignments, constants)
    _rk_define_read_scalars!(original, known_scalars[])
    isempty(selected) ? original : _RKHeldOutPlan(original, selected)
end

# The completion parametrization a lowering asks for: `:centered` (the default)
# samples each missing value under its law; `:noncentered` samples a standard
# normal draw per missing value, so a completion is a unit draw for the NLME
# paths (`brm_nlme_view`); a collection of `mi` sources non-centers those
# completions only. Location-scale laws (`Normal`/`LogNormal(location, scale)`)
# directly have that form. Every other continuous univariate law keeps its
# support transform on the draw (the coordinates are the centred ones) and its
# free entries observe the exact density remainder (`_rk_mi_free_part`). The
# grouped entries of `mi(x)::shares(total, rows)` take it under any law: their
# coordinates are standard normal draws, and each group observes the rest of
# its density (`_rk_emit_missing_totals!`). The density is the centred one in
# every case; only its split into prior and pointwise terms differs.
function _rk_plan_completions(plan, mode)
    mode === :centered && return plan
    mode === :noncentered && return _rk_noncentered_completions(plan, nothing)
    mode isa Symbol && throw(ArgumentError("completions must be :centered, " *
        ":noncentered or a collection of `mi` sources to non-center, got :$mode"))
    sources = Set{Symbol}()
    for source in mode
        source isa Symbol || throw(ArgumentError("completions: a collection names " *
            "`mi` sources as Symbols, got $(repr(source))"))
        push!(sources, source)
    end
    isempty(sources) && return plan
    _rk_noncentered_completions(plan, sources)
end
# `sources === nothing` non-centers every completion.
function _rk_noncentered_completions(plan::_RKValuePlan, sources)
    _rk_check_completion_sources(sources, _rk_completion_sources(plan))
    _RKValuePlan(plan.regression, plan.assignments, plan.observations, plan.columns,
        map(c -> sources === nothing || c.source in sources ?
            _rk_noncentered_completion(c) : c, plan.completions))
end
_rk_noncentered_completions(plan::_RKHeldOutPlan, sources) =
    _RKHeldOutPlan(_rk_noncentered_completions(plan.parent, sources), plan.held_out)
# The regression plan's `mi(y)` responses feed nothing downstream: their
# missing rows are integrated out of the likelihood, which reads the observed
# rows only, and sample no coordinate. Nothing is left to non-center.
function _rk_noncentered_completions(plan::_RKStructuralPlan, sources)
    _rk_check_completion_sources(sources, _rk_completion_sources(plan))
    plan
end
# The `mi` sources a `completions` collection may name in a plan.
_rk_completion_sources(plan::_RKValuePlan) = (c.source for c in plan.completions)
_rk_completion_sources(plan::_RKHeldOutPlan) = _rk_completion_sources(plan.parent)
_rk_completion_sources(plan::_RKStructuralPlan) =
    (r.response for r in plan.responses if r.mi_jobs !== nothing)
_rk_completion_sources(plan) = ()
function _rk_check_completion_sources(sources, completions)
    sources === nothing && return nothing
    known = Set{Symbol}(completions)
    unknown = sort!(collect(setdiff(sources, known)))
    isempty(unknown) || throw(ArgumentError("completions names " *
        join(("`$s`" for s in unknown), ", ") * ", which " *
        (length(unknown) == 1 ? "is not a completion" : "are not completions") *
        " with sampled missing entries in this model" * (isempty(known) ? "" :
            "; those are " * join(("`$s`" for s in sort!(collect(known))), ", "))))
    nothing
end
const _RK_NONCENTERED_LAWS = (Normal, LogNormal)
_rk_location_scale_law(d) =
    d.callable in _RK_NONCENTERED_LAWS && isempty(d.kwargs) && length(d.args) == 2
function _rk_noncentered_completion(c::_RKMissingValueSpec)
    d = c.distribution
    F = d.callable
    # Free entries (all of a plain completion's; those of no group with a
    # total) keep the law's support transform and observe its density
    # remainder, which need a continuous univariate law of positional
    # arguments. Grouped entries alone take any shares law (their group terms).
    free = c.totals === nothing || c.free_weights !== nothing
    !free || _rk_location_scale_law(d) || _brm_lower_family(F) ||
        (!(F isa Type) || F <: ContinuousUnivariateDistribution) && isempty(d.kwargs) ||
        error("RK backend: completions=:noncentered draws `mi($(c.source))`'s " *
            "missing entries as standard normal draws carried to the law's support, " *
            "which needs a continuous univariate law of positional arguments; " *
            "its law is `$F`" * (isempty(d.kwargs) ? "" :
                " with keywords $(join(keys(d.kwargs), ", "))"))
    _RKMissingValueSpec(c.source, c.observed, c.observed_rows, c.missing_rows,
        c.observed_component, c.order, c.nmissing, d, true,
        c.totals, c.free_weights)
end
# The completion parametrization of an existing plan, which a replay keeps:
# `:centered`, `:noncentered`, or the non-centered sources when only some are.
function _rk_completion_mode(plan::_RKValuePlan)
    noncentered = Symbol[c.source for c in plan.completions if c.noncentered]
    isempty(noncentered) ? :centered :
        length(noncentered) == length(plan.completions) ? :noncentered : Tuple(noncentered)
end
_rk_completion_mode(plan::_RKHeldOutPlan) = _rk_completion_mode(plan.parent)
_rk_completion_mode(plan) = :centered

Base.@nospecializeinfer function _rk_held_out_selection(@nospecialize(brmi::BRMI), held_out)
    request = _brm_held_out_request(held_out; prefix="RK backend")
    isempty(request.names) && return Set{Symbol}()
    aliases = Dict{Symbol,Set{Symbol}}()
    sources = Set{Symbol}()
    unbound = Symbol[]
    function record(target, column)
        source = _brm_observation_name(column)
        if source === nothing
            push!(unbound, target)
            return
        end
        push!(sources, source)
        for alias in (target, source)
            push!(get!(() -> Set{Symbol}(), aliases, alias), source)
        end
    end
    for (key, node) in _brm_operation_entries(brmi)
        node isa NamedColumn || continue
        operation = parent(node)
        operation isa ExprColumn{typeof(~)} || continue
        lhs, rhs = getargs(operation, 2)
        if rhs isa ExprColumn && getf(rhs) === kernel
            args = getargs(rhs)
            isempty(args) && continue
            lam = first(args)
            parts = _sb_kernel_lambda_parts(lam)
            parts === nothing && continue
            params, body = parts
            for (param, column) in zip(params, args[2:end])
                _sb_cell_param_observed(body, param) && record(param, column)
            end
        elseif _brm_observation_name(lhs) !== nothing
            record(key, lhs)
        end
    end
    _brm_resolve_held_out(request, aliases, sources, unbound; prefix="RK backend")
end

_rk_emitted_observation_name(lhs::Symbol) = lhs
function _rk_emitted_observation_name(lhs::Expr)
    # A joint `[y1, y2]` target retains the identity of its lead original
    # response column; it must root the complete fitted joint likelihood.
    lhs.head in (:ref, :vect) && !isempty(lhs.args) &&
        return _rk_emitted_observation_name(first(lhs.args))
    # A matrix declared by its slices, `eachrow(G[1:m, 1:n]) .~ law`.
    Meta.isexpr(lhs, :call, 2) && lhs.args[1] in (:eachrow, :eachcol) &&
        return _rk_emitted_observation_name(lhs.args[2])
    nothing
end
_rk_emitted_observation_name(_) = nothing

function _rk_source_symbols!(names, value)
    value = _brm_undoc(value)
    value isa Symbol && push!(names, value)
    value isa Expr && foreach(arg -> _rk_source_symbols!(names, arg), value.args)
    names
end
_rk_source_symbols!(names, values::AbstractVector) =
    (foreach(value -> _rk_source_symbols!(names, value), values); names)
function _rk_source_lhs!(names, lhs)
    name = _rk_emitted_observation_name(lhs)
    name === nothing || push!(names, name)
    Meta.isexpr(lhs, :tuple) && foreach(arg -> _rk_source_lhs!(names, arg), lhs.args)
    names
end
# A `@plate for` cell's bare locals belong to its plate (rkppl-use §13); its
# model outputs are its indexed writes, declarations and observations.
function _rk_source_outputs!(names, statement)
    statement = _brm_undoc(statement)
    statement isa Expr || return names
    if statement.head === :(=)
        _rk_source_lhs!(names, first(statement.args))
    elseif statement.head === :call && length(statement.args) == 3 &&
            first(statement.args) in (:~, :.~)
        _rk_source_lhs!(names, statement.args[2])
    elseif Meta.isexpr(statement, :macrocall) && first(statement.args) === Symbol("@plate") &&
            Meta.isexpr(last(statement.args), :for)
        for cell in last(last(statement.args).args).args
            cell = _brm_undoc(cell)
            Meta.isexpr(cell, :(=)) && !Meta.isexpr(first(cell.args), :ref) && continue
            _rk_source_outputs!(names, cell)
        end
    elseif statement.head in (:macrocall, :for, :block)
        foreach(arg -> _rk_source_outputs!(names, arg), statement.args)
    end
    names
end
function _rk_source_observes(statement, conditioned)
    statement isa Expr || return false
    statement.head === :call && length(statement.args) == 3 &&
        first(statement.args) in (:~, :.~) &&
        _rk_emitted_observation_name(statement.args[2]) in conditioned && return true
    statement.head in (:macrocall, :for, :block) &&
        return any(arg -> _rk_source_observes(arg, conditioned), statement.args)
    false
end
function _rk_source_samples(statement)
    statement isa Expr || return false
    statement.head === :call && length(statement.args) == 3 &&
        first(statement.args) in (:~, :.~) && return true
    statement.head in (:macrocall, :for, :block) &&
        return any(_rk_source_samples, statement.args)
    false
end

# Fitted density follows the original observation graph. Independent priors
# never used by any response become generated draws on the Stan route. Resolve
# this before withholding, so a parameter used by a held-out response stays.
# A compound plate remains atomic; its declared block geometry is preserved.
#
# An authored value no response reads (`retained`: a `@plate` output, an `=`
# assignment, a formula predictor) stays a named graph value when everything
# it needs is deterministic over what the fitted program already computes. It
# adds no density, coordinate or Jacobian, a query plans it only when asked,
# and a caller can request it by the name it was written under.
#
# The coordinate records the emission filed for a removed declaration leave
# with it (`_rk_drop_unfitted_records!`), so the semantic inventory describes
# the fitted program's packed coordinates and nothing else.
#
# A statement defining one of `densities` stays whether or not a response
# reads it: the grouped entries of `mi(x)::shares(total, rows)`, whose density
# carries what their residuals say about the law, or a component call whose
# own observations contribute to the likelihood.
function _rk_fitted_source(emitted, conditioned; retained=(), coordinates=nothing,
        densities=())
    isempty(conditioned) && return emitted
    statements = emitted.main.args
    outputs = [_rk_source_outputs!(Set{Symbol}(), statement) for statement in statements]
    references = [_rk_source_symbols!(Set{Symbol}(), statement) for statement in statements]
    block_reads = _rk_block_free_reads(emitted.defs)
    for names in references
        pending = collect(names)
        for name in pending
            for dependency in get(block_reads, name, ())
                dependency in names && continue
                push!(names, dependency)
                push!(pending, dependency)
            end
        end
    end
    kept = [_rk_source_observes(statement, conditioned) ||
        !isdisjoint(outputs[i], densities) for (i, statement) in enumerate(statements)]
    needed = Set{Symbol}()
    for i in eachindex(statements)
        kept[i] && union!(needed, references[i])
    end
    changed = true
    while changed
        changed = false
        for i in eachindex(statements)
            (kept[i] || isempty(intersect(outputs[i], needed))) && continue
            kept[i] = true
            union!(needed, references[i])
            changed = true
        end
    end
    for i in eachindex(statements)
        (kept[i] || isdisjoint(outputs[i], retained)) && continue
        closure, reads = Set([i]), copy(references[i])
        changed = true
        while changed
            changed = false
            for j in eachindex(statements)
                (kept[j] || j in closure || isdisjoint(outputs[j], reads)) && continue
                push!(closure, j)
                union!(reads, references[j])
                changed = true
            end
        end
        any(j -> _rk_source_samples(statements[j]), closure) && continue
        foreach(j -> kept[j] = true, closure)
    end
    _rk_drop_unfitted_records!(coordinates, outputs, kept)
    _RKEmittedProgram(emitted.defs, Expr(:block, statements[kept]...), emitted.bindings)
end
# Withholding changes the observation likelihood. Authored priors and their
# coordinates stay in the program, including priors used only by that response.
function _rk_active_observation_source(emitted, conditioned, held_out)
    statements = emitted.main.args
    names = Set(name for name in conditioned if name in held_out)
    withheld = [_rk_withheld_observation(statement, names) for statement in statements]
    Expr(:block, statements[.!withheld]...)
end
function _rk_withheld_observation(statement, names)
    statement = _brm_undoc(statement)
    statement isa Expr || return false
    statement.head === :call && length(statement.args) == 3 &&
        first(statement.args) in (:~, :.~) &&
        return _rk_emitted_observation_name(statement.args[2]) in names
    # A per-subject observation is a `@plate` holding only that observation.
    body = _rk_plate_body(statement)
    body !== nothing && !isempty(body) && all(s -> _rk_withheld_observation(s, names), body)
end
function _rk_emit_ast(plan::_RKHeldOutPlan; coordinates=nothing, documentation=())
    emitted = _rk_emit_ast(plan.parent; coordinates, documentation)
    conditioned = _rk_observed_names(plan.parent)
    main = _rk_active_observation_source(emitted, conditioned, plan.held_out)
    _RKEmittedProgram(emitted.defs, main, emitted.bindings)
end

_rk_observed_names(plan::_RKStructuralPlan) = (_rk_structural_responses(plan)...,
    _rk_car_term_placeholders(plan)...)
# The responses alone: what the fitted source keeps the statements of
# (`_rk_fitted_source`). A regression inside a value plan has none, and is
# then kept whole for the value plan to prune.
_rk_structural_responses(plan::_RKStructuralPlan) = Tuple(unique(
    [name for response in plan.responses for name in
        (response.response, response.extra_responses...)]))
# With the placeholders whose observation carries a non-centered
# `mi(x)::shares(total, rows)` completion's group terms (`_rk_mi_grouped_arguments!`).
_rk_observed_names(plan::_RKValuePlan) = (Tuple(o.name for o in plan.observations)...,
    (name for assignment in plan.assignments if assignment isa _RKPreparedSubmodelAssignment
        for name in assignment.observed)...,
    (c.totals.weights for c in plan.completions if c.noncentered && c.totals !== nothing)...,
    (_rk_mi_free_weight_name(c) for c in plan.completions if _rk_mi_free_weight_name(c) !== nothing)...,
    _rk_car_term_placeholders(plan)...)

function _rk_mi_free_weight_name(c::_RKMissingValueSpec)
    c.noncentered ? c.free_weights : nothing
end
# The placeholders a `car_units` field's unit terms observe (`_rk_ast_car_separated!`).
_rk_car_term_placeholders(plan) = Tuple(term.options.separation.terms
    for predictor in _rk_replay_predictors(plan) for term in predictor.terms
    if term.kind === :car && get(term.options, :separation, nothing) !== nothing &&
        term.options.separation.terms !== nothing)
_rk_observed_names(plan::_RKKernelPlan) = (plan.kernel.data_columns[
    findfirst(==(plan.kernel.obs_response), plan.kernel.slice_params)],)
_rk_observed_names(plan::_RKHeldOutPlan) = Tuple(
    name for name in _rk_observed_names(plan.parent) if !(name in plan.held_out))
