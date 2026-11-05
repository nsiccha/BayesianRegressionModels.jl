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
This carries a fitted approximation into a new, disjoint window; it holds no
optimizer state. An interrupted optimization of the same model and data
continues from [`brm_svi_optimizer_checkpoint`](@ref) instead.
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
    updates::Int
end

_brm_svi_global_factor(q) = deepcopy((; q.family, q.μ, q.ρ, q.Β))

function _brm_svi_source_id(rk)
    emitted = _rk_retained_program(rk)
    io = IOBuffer()
    serialize(io, (sprint(show, brm_source(rk); context=:brm_documentation => false), _rk_program_statements(emitted),
        emitted.bindings))
    bytes2hex(sha256(take!(io)))
end

function _brm_svi_unit_keys(m, i)
    keys = m.view.subject_margins[i]
    length(Set(keys)) == length(keys) || throw(ArgumentError(
        "brm_svi_fit: unit $(repr(m.view.levels[i])) has ambiguous latent coordinate " *
        "identities; streaming requires identifiable persistent unit coordinates"))
    keys
end

function _brm_svi_save_units!(units, m, q)
    for (i, label) in enumerate(m.view.levels)
        units[label] = deepcopy((; keys=_brm_svi_unit_keys(m, i),
            M=q.M[i], Λ=q.Λ[i], U=q.U[i]))
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
    estimated = Set(vcat(d.globals, reduce(vcat, m.view.subject_coordinates; init=Int[])))
    fixed = Dict(m.view.coordinates[i] => m.unit[i] for i in eachindex(m.unit)
        if !(i in estimated))
    BRMSVIState(1, deepcopy(state), _brm_svi_source_id(m.backend),
        copy(m.view.coordinates[d.globals]), _brm_svi_global_factor(d.q),
        _brm_svi_save_units!(Dict{Any,NamedTuple}(), m, d.q), m.view.group,
        Tuple(keys(m.view.nested_groups)), m.view.shared_groups, fixed, 0)
end
brm_svi_checkpoint(state::BRMSVIState) = deepcopy(state)

function _brm_svi_carried_state(previous::BRMSVIState, m, q)
    BRMSVIState(1, deepcopy(previous.preprocessing), previous.source_id,
        copy(previous.globals), _brm_svi_global_factor(q),
        _brm_svi_save_units!(deepcopy(previous.units), m, q), previous.group,
        previous.nested, previous.shared, copy(previous.fixed), previous.updates + 1)
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

function _brm_svi_check_model(m, previous)
    previous.format == 1 || throw(ArgumentError("brm_svi_fit: unsupported streaming state format"))
    m.view.group === previous.group && Tuple(keys(m.view.nested_groups)) == previous.nested &&
        m.view.shared_groups == previous.shared || throw(ArgumentError(
            "brm_svi_fit: subject/nested/shared grouping partition changed"))
    _brm_svi_check_preprocessing(m.backend, previous)
    _brm_svi_source_id(m.backend) == previous.source_id || throw(ArgumentError(
        "brm_svi_fit: model source changed (declaration, emitted program or callable bindings)"))
    globals = _brm_svi_globals(m)
    m.view.coordinates[globals] == previous.globals || throw(ArgumentError(
        "brm_svi_fit: global coordinate layout changed"))
    estimated = Set(vcat(globals, reduce(vcat, m.view.subject_coordinates; init=Int[])))
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

struct _BRMSVIStreamingPrior{G,H,Q}
    global_factor::G
    geometry::H
    units::Q
end

function _brm_svi_initialize(t, previous::BRMSVIState, family;
        start, inits, global_scale, unit_scale)
    _brm_svi_check_model(t.m, previous)
    q0 = previous.q
    q = _brm_svi_params(t, q0.μ, family; global_scale=1.0, unit_scale=1.0)
    q.ρ .= q0.ρ
    q.Β .= q0.Β
    for (i, label) in enumerate(t.m.view.levels)
        keys = _brm_svi_unit_keys(t.m, i)
        haskey(previous.units, label) || continue
        old = previous.units[label]
        index = Dict(key => j for (j, key) in enumerate(old.keys))
        length(keys) == length(old.keys) && all(haskey(index, k) for k in keys) ||
            throw(ArgumentError("brm_svi_fit: unit $(repr(label)) changed its latent " *
                "coordinate layout; streaming carries persistent unit statics"))
        perm = [index[k] for k in keys]
        q.M[i] .= old.M[perm]
        q.Λ[i] .= old.Λ[perm, perm]
        q.U[i] .= cholesky(Symmetric(q.Λ[i])).L
    end
    prior = _BRMSVIStreamingPrior(_brm_svi_global_factor(q),
        _brm_svi_global_geometry(q), deepcopy((; q.M, q.Λ, q.U)))
    merge(t, (; prior)), q
end

function _brm_svi_global_term(prior::_BRMSVIStreamingPrior, t, g)
    δ = g - prior.global_factor.μ
    gradient = -_brm_svi_global_precision(prior.geometry, δ)
    (dot(δ, gradient) - _brm_svi_global_logdet(prior.geometry) - length(g) * log(2π)) / 2,
        gradient
end

function _brm_svi_unit_prior!(gz, prior::_BRMSVIStreamingPrior, i, z)
    δ = z - prior.units.M[i]
    gradient = -(prior.units.Λ[i] * δ)
    gz .+= gradient
    dot(δ, gradient) / 2 + sum(log, diag(prior.units.U[i])) - length(z) * log(2π) / 2
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
the model's standard-normal unit prior under the carried global factor;
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
        sprint(show, x; context=:brm_documentation => false) == state.preprocessing.declaration || throw(ArgumentError(
            "$caller: model source changed"))
        x = RKBRMI(x, state.preprocessing; resample_groups=(state.group, state.nested...))
    end
    x, x isa BRMNLMEModel ? options :
        merge((; subject_group=state.group, shared_groups=state.shared), options)
end
