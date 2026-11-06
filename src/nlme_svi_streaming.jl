# Sequential variational Bayes: the last Gaussian approximation is the next
# window's prior. No previous observations enter the new target.

"""
    BRMSVIState
    brm_svi_checkpoint(fit)

Data-only streaming state: the global Gaussian, unit Gaussians keyed by subject
label, semantic coordinate identities and frozen `BRMFittedState` preprocessing.
It holds neither observations nor a compiled model or AD workspace. Serialize
this state to resume with `brm_svi_fit(state, builder(new_data); ...)` in another
process. Recreate custom model functions in that process before loading it.
Format 1 dense-unit states remain supported. Format 2 also stores local
diagonal-plus-low-rank factors directly, without dense matrices or optimizer
moments. Coordinate reordering preserves their covariance exactly.
This carries a fitted approximation into a new, disjoint window; it holds no
optimizer state. An interrupted optimization of the same model and data
continues from [`brm_svi_optimizer_checkpoint`](@ref) instead.

The checkpoint retains `unit_partition`. Connected continuation accepts complete
unchanged units and wholly new units; a retained subject whose component or
member order changes is refused before optimization. New checkpoints use format 3
and require the same package versions when serialized.
"""
struct BRMSVIState
    format::Int
    preprocessing::BRMFittedState
    source_id::String
    globals::Vector{Symbol}
    q::NamedTuple
    units::Dict{Any,NamedTuple}
    group::Symbol
    nested::Tuple{Vararg{Symbol}}
    shared::Tuple{Vararg{Symbol}}
    fixed::Dict{Symbol,Float64}
    unit_partition::Symbol
    layout::NamedTuple
    updates::Int
end

# Construct a legacy dense/structured subject carry in the current process.
# Serialized objects still require matching Julia and package versions.
BRMSVIState(format, preprocessing, source_id, globals, q, units, group, nested,
    shared, fixed, updates) = BRMSVIState(format, preprocessing, source_id,
        globals, q, units, group, nested, shared, fixed, :subjects, updates)

BRMSVIState(format, preprocessing, source_id, globals, q, units, group, nested,
    shared, fixed, unit_partition, updates) = BRMSVIState(format, preprocessing,
        source_id, globals, q, units, group, nested, shared, fixed, unit_partition,
        NamedTuple(), updates)

include("nlme_svi_unit_streaming.jl")

_brm_svi_global_factor(q) = deepcopy((; q.family, q.μ, q.ρ, q.Β))

function _brm_svi_source_id(rk)
    emitted = _rk_retained_program(rk)
    io = IOBuffer()
    serialize(io, (_brm_declaration_text(brm_source(rk)), _rk_identity_statements(rk),
        _brm_stable_bindings(emitted.bindings)))
    bytes2hex(sha256(take!(io)))
end

function _brm_svi_unit_keys(m, i)
    keys = m.view.subject_margins[i]
    length(Set(keys)) == length(keys) || throw(ArgumentError(
        "brm_svi_fit: unit $(repr(m.view.levels[i])) has ambiguous latent coordinate " *
        "identities; streaming requires identifiable persistent unit coordinates"))
    keys
end

function _brm_svi_scientific_keys(m)
    result = Dict{Int,NamedTuple}()
    for i in eachindex(m.view.levels)
        for (j, key) in zip(m.view.subject_coordinates[i], _brm_svi_unit_keys(m, i))
            result[j] = key
        end
    end
    result
end
_brm_svi_global_keys(m, globals) = let scientific = _brm_svi_scientific_keys(m)
    [get(scientific, j, (; coordinate=m.view.coordinates[j])) for j in globals]
end

function _brm_svi_owned_unit_keys(m, blocks, i; scientific=_brm_svi_scientific_keys(m))
    result = NamedTuple[get(scientific, j, (; coordinate=m.view.coordinates[j]))
        for j in blocks[i]]
    allunique(result) || throw(ArgumentError(
        "brm_svi_fit: inference unit $i has ambiguous coordinate identities"))
    result
end

function _brm_svi_save_units!(units, m, q, blocks=m.view.subject_coordinates;
        levels=m.view.levels)
    scientific = _brm_svi_scientific_keys(m)
    for (i, label) in enumerate(levels)
        units[label] = _brm_svi_saved_unit(
            _brm_svi_owned_unit_keys(m, blocks, i; scientific), q.M[i], q.Λ[i], q.U[i])
    end
    units
end

function brm_svi_checkpoint(fit)
    hasproperty(fit, :diagnostics) && hasproperty(fit.diagnostics, :q) ||
        throw(ArgumentError("brm_svi_checkpoint: expected a brm_svi_fit result"))
    d = fit.diagnostics
    d.streaming === nothing || return deepcopy(d.streaming)
    m = d.model
    _brm_svi_source_id(m.backend) == d.source_id || throw(ArgumentError(
        "brm_svi_checkpoint: model source changed since the fit"))
    state = brm_fitted_state(m.backend)
    blocks = hasproperty(d, :units) ? d.units : m.view.subject_coordinates
    estimated = Set(vcat(d.globals, reduce(vcat, blocks; init=Int[])))
    fixed = Dict(m.view.coordinates[i] => m.unit[i] for i in eachindex(m.unit)
        if !(i in estimated))
    levels = hasproperty(d, :unit_levels) ? d.unit_levels : m.view.levels
    units = _brm_svi_save_units!(Dict{Any,NamedTuple}(), m, d.q, blocks; levels)
    options = d.separator === nothing ? (; separator=:none) : _brm_svi_policy_options(d.separator)
    layout = (; options, global_keys=_brm_svi_global_keys(m, d.globals))
    BRMSVIState(3, deepcopy(state), _brm_svi_source_id(m.backend),
        copy(m.view.coordinates[d.globals]), _brm_svi_global_factor(d.q),
        units, m.view.group,
        Tuple(keys(m.view.nested_groups)), m.view.shared_groups, fixed,
        m.view.unit_partition, layout, 0)
end
brm_svi_checkpoint(state::BRMSVIState) = deepcopy(state)

# Restore semantic ownership before rediscovering the new window's local
# coordinates. An absent ordinary block remains in the carried checkpoint;
# it must not become a new global merely because this window has no reader.
function _brm_svi_previous_ownership!(estimated, seeds, promoted, m, previous::BRMSVIState; levels=m.view.levels)
    position = Dict(name => i for (i, name) in enumerate(m.view.coordinates))
    labels = Dict(label => i for (i, label) in enumerate(levels))
    for name in previous.globals
        haskey(position, name) && push!(promoted, position[name])
    end
    absent = Int[]
    for (label, unit) in previous.units, key in unit.keys
        hasproperty(key, :coordinate) || continue
        j = get(position, key.coordinate, 0)
        j == 0 && continue
        i = get(labels, label, 0)
        if i == 0
            estimated[j] = false
            push!(absent, j)
        else
            push!(seeds[i], j)
        end
    end
    sort!(unique!(promoted))
    absent
end

function _brm_svi_carried_state(previous::BRMSVIState, t, q)
    m = t.m
    BRMSVIState(3, deepcopy(previous.preprocessing), previous.source_id,
        copy(previous.globals), _brm_svi_global_factor(q),
        _brm_svi_save_units!(deepcopy(previous.units), m, q, t.units;
            levels=_brm_svi_unit_levels(t)), previous.group,
        previous.nested, previous.shared, copy(previous.fixed),
        previous.unit_partition, (; options=_brm_svi_separator_options(t),
            global_keys=_brm_svi_global_keys(m, t.globals)), previous.updates + 1)
end

# Model-scope arrays can retain coordinates of absent units. The target
# integrates those carried factors out, but packed output still represents
# their marginal uncertainty rather than filling those slots with constants.
function _brm_svi_fill_absent_draws!(rng, draws, t, previous::BRMSVIState)
    isempty(t.ownership.absent) && return draws
    positions = Dict(t.m.view.coordinates[j] => j for j in t.ownership.absent)
    for unit in values(previous.units)
        source, destination = Int[], Int[]
        for (k, key) in enumerate(unit.keys)
            hasproperty(key, :coordinate) || continue
            j = get(positions, key.coordinate, 0)
            j == 0 && continue
            push!(source, k)
            push!(destination, j)
        end
        isempty(source) && continue
        for draw in axes(draws, 2)
            z = _brm_svi_saved_unit_draw(rng, unit)
            draws[destination, draw] .= z[source]
        end
    end
    draws
end

# Only grouping level fits may change. A transform of the grouping column
# itself (e.g. standardize(subject)) must remain frozen too.
function _brm_svi_level_values(rk, groups)
    emitted = _rk_retained_program(rk)
    twins = Dict(_rk_plan_fit_inputs(rk.plan))
    inputs = Set(input for (input, column) in twins if column in groups)
    names = Set{Symbol}()
    for stmt in emitted.main.args
        Meta.isexpr(stmt, :(=), 2) && stmt.args[1] isa Symbol || continue
        rhs = stmt.args[2]
        Meta.isexpr(rhs, :call) && rhs.args[1] in (:brm_fit_levels, :brm_joint_fit_levels) || continue
        all(x -> x in inputs, rhs.args[2:end]) && push!(names, stmt.args[1])
    end
    names
end

function _brm_svi_check_preprocessing(rk, previous)
    old = previous.preprocessing
    isequal(_rk_fitted_constants(rk), old.constants) || throw(ArgumentError(
        "brm_svi_fit: preprocessing changed: constant(...) values differ"))
    names, _ = _rk_fitted_value_names(rk; prefix="brm_svi_fit")
    values = _rk_frozen_values(rk, names).values
    groups = (previous.group, previous.nested...)
    allowed = _brm_svi_level_values(rk, groups)
    for name in union(keys(old.fitted.values), keys(values))
        name in allowed && continue
        haskey(old.fitted.values, name) && haskey(values, name) &&
            isequal(old.fitted.values[name], values[name]) || throw(ArgumentError(
                "brm_svi_fit: preprocessing changed: fitted value `$name` differs; " *
                "replay with the previous frozen preprocessing"))
    end
    # A data-free replay uses its original planner fits. Exporting its current
    # graph values is useful above, but recording fits from its current rows
    # would refit Julia-side transforms that the bound plan has kept frozen.
    fits = rk.fitted === nothing ? _rk_recorded_fits(rk) : rk.fitted.state.fits
    for key in union(keys(old.fits), keys(fits))
        first(key) === :levels && last(key) in groups && continue
        haskey(old.fits, key) && haskey(fits, key) &&
            isequal(old.fits[key], fits[key]) || throw(ArgumentError(
                "brm_svi_fit: preprocessing changed: fitted transform $(repr(key)) differs"))
    end
    nothing
end

_brm_svi_check_partition(view, previous; caller="brm_svi_fit") =
    view.unit_partition === previous.unit_partition || throw(ArgumentError(
        "$caller: unit_partition changed since the checkpoint"))

function _brm_svi_check_units(view, previous)
    _brm_svi_check_partition(view, previous)
    view.unit_partition === :connected || return nothing
    retained = Dict(subject => label for label in keys(previous.units) for subject in label)
    for label in view.levels, subject in label
        haskey(retained, subject) || continue
        isequal(retained[subject], label) || throw(ArgumentError(
            "brm_svi_fit: connected membership changed for subject $(repr(subject)); " *
            "streaming requires complete unchanged connected units or wholly new units"))
    end
    nothing
end

function _brm_svi_check_model(t, previous)
    m = t.m
    previous.format in (1, 2, 3) || throw(ArgumentError("brm_svi_fit: unsupported streaming state format"))
    m.view.group === previous.group && Tuple(keys(m.view.nested_groups)) == previous.nested &&
        m.view.shared_groups == previous.shared || throw(ArgumentError(
            "brm_svi_fit: subject/nested/shared grouping partition changed"))
    _brm_svi_check_units((; unit_partition=m.view.unit_partition,
        levels=_brm_svi_unit_levels(t)), previous)
    _brm_svi_check_preprocessing(m.backend, previous)
    _brm_svi_source_id(m.backend) == previous.source_id || throw(ArgumentError(
        "brm_svi_fit: model source changed (declaration, emitted program or callable bindings)"))
    globals = t.globals
    m.view.coordinates[globals] == previous.globals || throw(ArgumentError(
        "brm_svi_fit: global coordinate layout changed"))
    if hasproperty(previous.layout, :global_keys)
        isequal(_brm_svi_global_keys(m, globals), previous.layout.global_keys) ||
            throw(ArgumentError("brm_svi_fit: promoted global subject identities changed"))
    end
    estimated = Set(vcat(globals, reduce(vcat, t.units; init=Int[]), t.ownership.absent))
    fixed = Dict(m.view.coordinates[i] => m.unit[i] for i in eachindex(m.unit)
        if !(i in estimated))
    isequal(fixed, previous.fixed) || throw(ArgumentError(
        "brm_svi_fit: fixed shared coordinates changed"))
    nothing
end

function _brm_svi_family(previous::BRMSVIState, family)
    family === nothing && return previous.q.family
    typeof(family) === typeof(previous.q.family) &&
        _brm_svi_rank(family) == _brm_svi_rank(previous.q.family) || throw(ArgumentError(
            "brm_svi_fit: streaming keeps the previous global Gaussian family and rank"))
    family
end

struct _BRMSVIStreamingPrior{G,H,Q,P}
    global_factor::G
    geometry::H
    units::Q
    carried::BitVector
    authored::P
end

function _brm_svi_initialize(t, previous::BRMSVIState, family;
        start, inits, global_scale, unit_scale, unit_dense_limit=64, unit_rank=8)
    _brm_svi_check_model(t, previous)
    q0 = previous.q
    q = _brm_svi_params(t, q0.μ, family; global_scale=1.0, unit_scale=1.0,
        unit_dense_limit, unit_rank)
    q.ρ .= q0.ρ
    q.Β .= q0.Β
    median = first(_brm_nlme_prior_median_point(t.m))
    for (i, block) in enumerate(t.units)
        q.M[i] .= median[block]
    end
    carried = falses(length(t.units))
    scientific = _brm_svi_scientific_keys(t.m)
    for (i, label) in enumerate(_brm_svi_unit_levels(t))
        keys = _brm_svi_owned_unit_keys(t.m, t.units, i; scientific)
        haskey(previous.units, label) || continue
        carried[i] = true
        old = previous.units[label]
        index = Dict(key => j for (j, key) in enumerate(old.keys))
        length(keys) == length(old.keys) && all(haskey(index, k) for k in keys) ||
            throw(ArgumentError("brm_svi_fit: unit $(repr(label)) changed its latent " *
                "coordinate layout; streaming carries persistent unit statics"))
        perm = [index[k] for k in keys]
        _brm_svi_restore_unit!(q, i, old, perm)
    end
    prior = _BRMSVIStreamingPrior(_brm_svi_global_factor(q),
        _brm_svi_global_geometry(q), deepcopy((; q.M, q.Λ, q.U)), carried, t.prior)
    merge(t, (; prior)), q
end

function _brm_svi_global_term(prior::_BRMSVIStreamingPrior, t, g)
    δ = g - prior.global_factor.μ
    gradient = -_brm_svi_global_precision(prior.geometry, δ)
    (dot(δ, gradient) - _brm_svi_global_logdet(prior.geometry) - length(g) * log(2π)) / 2,
        gradient
end

function _brm_svi_unit_prior!(gg, gz, prior::_BRMSVIStreamingPrior, t, i, g, z)
    prior.carried[i] || return _brm_svi_unit_prior!(gg, gz, prior.authored, t, i, g, z)
    δ = z - prior.units.M[i]
    gradient = -(prior.units.Λ[i] * δ)
    gz .+= gradient
    dot(δ, gradient) / 2 + _brm_svi_unit_logdet_root(prior.units.U[i]) - length(z) * log(2π) / 2
end

"""
    brm_svi_fit([save,] previous, x_new; kwargs...)

Continue on a disjoint new observation window. `previous` is an SVI result or
a data-only [`BRMSVIState`](@ref). To continue an interrupted optimization of
the same model and data instead, use [`brm_svi_resume`](@ref); `save`,
`stop_after` and `checkpoint_every` interrupt this window's fit as they do
[`brm_svi_fit`](@ref)'s. `x_new` is the same model on the new window:
a `BRMI` is lowered with the state's frozen preprocessing; an `RKBRMI` or
prepared `BRMNLMEModel` must already have that preprocessing. A previous fit
also accepts raw new data, rebinding its model declaration. A state resumed in
another process takes `builder(new_data)` instead.

The target is `q_previous(globals, units) * p(new_data | globals, units)`.
Both the optimizer start and the prior use the previous factors. The history
is carried through a Gaussian family, never re-evaluated. This sequential
approximation can depend on window order and is not a cumulative-data refit.
The family drops global/unit and between-unit dependence; increasing the
global rank alone does not preserve those dependencies.
Do not supply overlapping windows: their likelihood would be counted again.

Units are matched by label and semantic coordinate identity. New labels get
their authored unit priors under the carried global factor;
absent labels retain their factors in the checkpoint and recover them on
return. An existing unit's latent coordinate set must stay fixed (persistent
statics, not growing time-state blocks). Global coordinates and preprocessing
must stay fixed. For nested units, pass the new window's `nested_groups` map.
Only new-window units are minibatched; diagnostics count their gradient calls.
Checkpoint size grows with distinct units, not the number of observations.
"""
brm_svi_fit(previous, x_new; kwargs...) = _brm_svi_window_fit(nothing, previous, x_new; kwargs...)
brm_svi_fit(save::Function, previous, x_new; kwargs...) =
    _brm_svi_window_fit(save, previous, x_new; kwargs...)

function _brm_svi_window_fit(save, previous, x_new; laplace=nothing, kwargs...)
    state = brm_svi_checkpoint(previous)
    collapsed = state isa BRMSVILaplaceState
    collapsed || laplace===nothing || throw(ArgumentError(
        "brm_svi_fit: GaussianBlockLaplace continuation requires a BRMSVILaplaceState; " *
        "an ordinary SVI checkpoint has no common-factor carry"))
    x = if x_new isa Union{BRMI,RKBRMI,BRMNLMEModel}
        x_new
    elseif previous isa Union{BRMSVIState,BRMSVILaplaceState}
        throw(ArgumentError("brm_svi_fit: pass builder(new_data) with a saved SVI checkpoint"))
    else
        _brm_rebind_brmi(brm_source(previous.diagnostics.model.backend), x_new)
    end
    x, options = _brm_svi_window_input(state, x, (; kwargs...); caller="brm_svi_fit")
    collapsed && return _brm_svi_laplace_dispatch(save, x, state,
        laplace===nothing ? state.method : laplace; options...)
    _brm_svi_fit(save, x, state; options...)
end

# The new window's model under the carried state: a BRMI is lowered with the
# state's frozen preprocessing, and the carried subject and shared groupings
# are the defaults of a model not yet prepared.
function _brm_svi_window_input(state, x, options; caller)
    if x isa BRMI
        _brm_declaration_text(x) == _brm_stable_declaration(state.preprocessing.declaration) || throw(ArgumentError(
            "$caller: model source changed"))
        x = RKBRMI(x, state.preprocessing; resample_groups=(state.group, state.nested...))
    end
    # Before the carried options apply: they are meaningless on another partition.
    partition = x isa BRMNLMEModel ? x.view.unit_partition :
        something(get(options, :unit_partition, nothing), state.unit_partition)
    _brm_svi_check_partition((; unit_partition=partition), state; caller)
    defaults = hasproperty(state, :layout) && hasproperty(state.layout, :options) ?
        merge((; cell_units=:first_unit), state.layout.options) : NamedTuple()
    x, x isa BRMNLMEModel ? merge(defaults, options) :
        merge(defaults, (; subject_group=state.group, shared_groups=state.shared,
            unit_partition=state.unit_partition), options)
end
