# Term starts use the prior-address grammar, without evaluating Julia syntax.
# Resolution (including wildcards, qualifiers and specificity) is the same
# operation as prior resolution; only the payload is different.
function _brm_start_term_targets(brmi::BRMI, term_inits)
    specs = NamedTuple[]
    for pair in term_inits
        pair isa Pair && first(pair) isa Expr || error(
            "brm_start: term_inits takes quoted addresses => values, " *
            "for example :(sd(mu, cdar(year))) => 0.02")
        ex, value = pair
        Meta.isexpr(ex, :call) && first(ex.args) isa Symbol || error(
            "brm_start: invalid term address $(repr(ex))")
        head = first(ex.args)
        (head === :sd || haskey(_TERM_PRIOR_HEAD_CLASS, head)) || error(
            "brm_start: `$ex` is not a term-parameter address")
        address = _prior_address(head, Symbol[_prior_slot(a) for a in ex.args[2:end]])
        first(address) in _TERM_PRIOR_CLASSES || error(
            "brm_start: `$ex` addresses a random-effect block; use inits for that block")
        term, qualifiers = _brm_split_term_key(address[2])
        push!(specs, (; class=address[1], term, qualifiers,
            predictor=address[3] === _EFFECT_COLON ? nothing : address[3],
            component=length(address) == 4 ? address[4] : nothing, value))
    end
    isempty(specs) && return NamedTuple[]
    resolved = _brm_resolve_term_priors(brmi; specs, resolve_prior=false, prefix="brm_start")
    priors = _brm_resolve_term_priors(brmi)
    targets = NamedTuple[]
    for (predictor, terms) in resolved, (_, slots) in terms, (key, entry) in slots
        (; spec, term) = entry
        slot = only(s for s in _brm_term_prior_slots(getf(term)) if s.name === key)
        spelling = _brm_term_prior_spelling(merge(spec, (; predictor)))
        prior = get(something(_brm_term_override(get(priors, predictor, Dict()), term),
            Dict()), key, nothing)
        prior !== nothing && _brm_is_derived_term_value(prior.spec.expression) && error(
            "brm_start: `$spelling` is set to `$(prior.spec.expression)`, a value " *
            "computed from model values, not a parameter; start the parameters it reads")
        if prior !== nothing && _brm_is_known_term_value(prior.spec.expression)
            known = prior.spec.expression.value
            value = _brm_known_term_shape(getf(term), term, slot, spec.value, spelling;
                prefix="brm_start")
            isequal(size(value), size(known)) && isapprox(value, known; atol=1e-12, rtol=1e-12) || error(
                "brm_start: `$spelling` is known at $known; it may only be restated consistently")
            continue
        end
        value = _brm_start_term_value(spec.value, spelling, slot)
        push!(targets, (; kind=:term, predictor, term, slot, spelling, value))
    end
    targets
end

function _brm_start_term_value(value, spelling, slot)
    xs = value isa Real ? [_brm_start_real(spelling, value)] :
        value isa AbstractVector{<:Real} ? [_brm_start_real(spelling, v) for v in value] :
        error("brm_start: `$spelling` takes a real value or vector on its natural scale")
    isempty(xs) && error("brm_start: `$spelling` must not be empty")
    slot.support === :positive && !all(>(0), xs) && error(
        "brm_start: `$spelling` must be strictly positive")
    slot.support === :unit_interval && !all(x -> 0 < x < 1, xs) && error(
        "brm_start: `$spelling` must lie strictly between 0 and 1")
    if slot.support === :simplex
        value isa AbstractVector && all(>(0), xs) && isapprox(sum(xs), 1; atol=1e-12, rtol=1e-12) ||
            error("brm_start: `$spelling` must be a positive vector summing to one")
    end
    value isa Real ? only(xs) : xs
end

# Shape is checked against the actual emitted carrier. A scalar broadcasts to
# a vector-valued scale/latent; a supplied vector must have the exact size.
function _brm_start_term_values(t, n)
    xs = t.value isa Real ? fill(t.value, n) : t.value
    length(xs) == n || throw(DimensionMismatch(
        "brm_start: `$(t.spelling)` has $n values, got $(length(xs))"))
    xs
end

_brm_start_term_role(slot) = isnothing(slot.component) ?
    Symbol(_brm_term_prior_label(Val(slot.class))) : slot.name

# These are emitter-owned bindings, never consumer addresses. Most families
# use the descriptor's existing roles; these complete the prior slot inventory.
_brm_start_sb_binding(::typeof(s), slot) = (:sd_pen, nothing)
_brm_start_sb_binding(::typeof(t2), slot) = (:sd_pen, findfirst(==(slot.component), (:rr, :rn, :nr)))
_brm_start_sb_binding(::typeof(gp), slot) = (slot.class === :term_sd ? :sigma : :rho, nothing)
_brm_start_sb_binding(::typeof(me), slot) = (:x_true, nothing)
_brm_start_sb_binding(::typeof(interval_censored), slot) = (:x_interval, nothing)
_brm_start_sb_binding(_f, _slot) = nothing

function _brm_sb_start_term_coordinates(t, d, names)
    key = _brm_term_full_key(t.term)
    entry = _brm_term_entry(d, t.predictor,
        e -> _brm_term_full_key(e.value) === key, "term `$key`")
    binding = _brm_start_sb_binding(getf(t.term), t.slot)
    if isnothing(binding)
        output = _brm_term_parameter_output(d, t.predictor, entry, _brm_start_term_role(t.slot))
        return _brm_element_coordinates(output, names)
    end
    local_name, index = binding
    owner = _brm_term_owner_output(d, t.predictor, entry).declaration
    if local_name === :sd_pen && haskey(owner.keywords, _sb_known_input(:sd_pen_free_idx))
        raw = owner.keywords[_sb_known_input(:sd_pen_free_idx)]
        free = raw isa Symbol ? d.plan.data[raw] : raw
        index = findfirst(==(something(index, 1)), free)
        index === nothing && error("brm_start: `$(t.spelling)` has no sampled smoothing scale")
        local_name = :sd_pen_free
    end
    expected = Symbol(_brm_stan_name(owner), :_, local_name)
    candidates = [o for o in d.outputs if o.kind === :parameter &&
        o.declaration !== nothing && o.declaration.target === owner.target && o.name === expected]
    length(candidates) == 1 || error(
        "brm_start: `$(t.spelling)` has $(length(candidates)) sampled carriers in this Stan model")
    coordinates = _brm_element_coordinates(only(candidates), names)
    isnothing(index) ? coordinates : [coordinates[index]]
end

function _brm_sb_start_terms!(writes, targets, d, names)
    for t in targets
        indices = _brm_sb_start_term_coordinates(t, d, names)
        append!(writes, zip(indices, _brm_start_term_values(t, length(indices))))
    end
    writes
end

# RK records each block's actual local symbols while emitting that block. The
# ordinary coordinate inventory excludes these auxiliary records, so they do
# not claim coordinates twice in transport or in an NLME partition.
function _brm_start_prepared_term(prepared)
    # Keep laws and bounds, not a second copy of fitted design/basis arrays.
    keys = (:sd_prior, :sd_priors, :ar_prior, :sigma_prior, :rho_prior,
        :spatial_prior, :temporal_prior,
        :mixing_prior, :latent_prior, :simplex_prior, :rho_lower, :type,
        :x_lower, :x_upper)
    state = (; (k => prepared.state[k] for k in keys if haskey(prepared.state, k))...)
    _BRMPreparedTerm(prepared.callable, prepared.source, state, prepared.dependencies)
end

function _rk_start_term_record!(records, predictor, term, declaration, bindings; column=nothing)
    records === nothing && return nothing
    haskey(term.options, :start_term) || return nothing
    _rk_coordinate_record!(records, (; kind=:term_start, predictor, declaration,
        term=term.options.start_term, prepared=term.options.start_prepared,
        bindings, column))
end

function _brm_rk_start_term_record(t, records)
    matches = [r for r in records if r.kind === :term_start &&
        r.predictor === t.predictor && _brm_term_full_key(r.term) === _brm_term_full_key(t.term)]
    length(matches) == 1 || error(
        "brm_start: `$(t.spelling)` has $(length(matches)) emitted RK term records")
    only(matches)
end

function _brm_rk_start_terms!(writes, targets, records, values)
    for t in targets
        r = _brm_rk_start_term_record(t, records)
        binding = get(r.bindings, t.slot.name, nothing)
        binding === nothing && error(
            "brm_start: `$(t.spelling)` is predicted or has no sampled carrier; initialize its predictor instead")
        _brm_rk_start_term_writes!(writes, r, t, values, binding)
    end
    writes
end

function _brm_rk_start_term_writes!(writes, r, t, values, binding::Symbol)
    declaration = isempty(string(binding)) ? r.declaration : Symbol(r.declaration, ".", binding)
    current = _brm_rk_declaration_value(values, declaration)
    if r.column !== nothing
        xs = _brm_start_term_values(t, size(current, 1))
        append!(writes, [(declaration, (i, r.column), x) for (i, x) in enumerate(xs)])
    elseif current isa Real
        push!(writes, (declaration, (), only(_brm_start_term_values(t, 1))))
    else
        xs = _brm_start_term_values(t, length(current))
        append!(writes, [(declaration, Tuple(i), x) for (i, x) in zip(CartesianIndices(current), xs)])
    end
    writes
end

function _brm_rk_start_term_writes!(writes, r, t, _values, bindings::Tuple)
    xs = _brm_start_term_values(t, length(bindings))
    append!(writes, [(Symbol(r.declaration, ".", binding), (), x)
                     for (binding, x) in zip(bindings, xs)])
    writes
end
