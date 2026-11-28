# StanBlocks' answers to the vocabulary questions `stan_vocabulary.jl`
# declares. `StanBlocks.stan` is StanBlocks itself, so `StanBlocks.stan.<name>`
# and `StanBlocks.<name>` name one binding.

_brm_stan_token(name::Symbol) =
    isdefined(StanBlocks, name) ? getfield(StanBlocks, name) : nothing

_brm_stan_builtin(name::Symbol) =
    isdefined(StanBlocks.builtin, name) ? getfield(StanBlocks.builtin, name) : nothing

_brm_stan_builtin_ref(name::Symbol) = GlobalRef(StanBlocks.builtin, name)

# The tokens are compared by identity. Their types are whatever StanBlocks
# makes them (singleton function types, or a shared value-family type), so one
# method per distinct type keeps two tokens of one type from overwriting each
# other's method.
const _BRM_STAN_FAMILY_TOKENS = (normal=StanBlocks.normal,
    student_t=StanBlocks.student_t, weighted=StanBlocks.weighted,
    censored=StanBlocks.censored, truncated=StanBlocks.truncated,
    interval_censored=StanBlocks.interval_censored)
for T in unique(map(typeof, values(_BRM_STAN_FAMILY_TOKENS)))
    @eval function _brm_stan_family(callable::$T)
        for (name, token) in pairs(_BRM_STAN_FAMILY_TOKENS)
            callable === token && return name
        end
        nothing
    end
end

_brm_stan_value_family_label(callable::StanBlocks.ValueFamily) = callable.label

_brm_stan_family_support(f::Function, typed_args...) =
    StanBlocks.family_support(f, typed_args...)

Base.@nospecializeinfer function _brm_stan_unconditioned_generative(
        @nospecialize(brmi::BRMI); kwargs...)
    sb = SBBRMI(brmi; kwargs...)
    _sb_observation_triage(sb).bound > 0 ? nothing : generative_plan(sb)
end

Base.@nospecializeinfer function _brm_stan_unbound_observations(
        @nospecialize(brmi::BRMI); kwargs...)
    _sb_observation_triage(SBBRMI(brmi; kwargs...)).unbound
end

_brm_stan_sampled_parameters(model::StanBlocks.SlicModel) =
    [o.name for o in StanBlocks.stan_descriptor(model).outputs if o.kind === :parameter]
