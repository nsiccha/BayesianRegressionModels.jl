# The Stan backend's vocabulary, as backend-neutral and RK code meets it.
#
# StanBlocks is the Stan backend's package, and BRM is making it a weak
# dependency (snag `stanblocks-and-b-0e9cf921`, user pick `079l8hv`): an
# environment that only uses RKBRMI should not carry it. Code outside the
# Stan backend therefore never names StanBlocks. It asks the functions below.
# Each fallback here gives the answer for a session without StanBlocks, and
# that answer is exact: with StanBlocks absent, no value is a StanBlocks
# object and no SLIC token is defined. The Stan backend adds the specific
# methods (`sb_vocabulary.jl`), which take over whenever StanBlocks is loaded.

"""
    _brm_stan_token(name::Symbol)

The SLIC binding `StanBlocks.<name>` that an unqualified name in a cell body
falls back to (a family token such as `normal`, a combinator such as
`weighted`, a builtin such as `to_vector`), or `nothing` when StanBlocks
defines no such name or is not loaded.
"""
_brm_stan_token(name) = nothing

"""
    _brm_stan_builtin(name::Symbol)

The SLIC builtin `StanBlocks.builtin.<name>` (a Stan function such as
`ode_rk45`), or `nothing`.
"""
_brm_stan_builtin(name) = nothing

"""
    _brm_stan_builtin_ref(name::Symbol)

A `GlobalRef` to the SLIC builtin `name`, for an emitted program that keeps a
legacy Stan spelling. Without StanBlocks there is no such builtin, and the
fallback says what to write instead.
"""
_brm_stan_builtin_ref(name) = throw(ArgumentError(
    "BRM: `$name` here is Stan's builtin spelling, which needs StanBlocks " *
    "(`using StanBlocks`). Use the Julia spelling instead."))

"""
    _brm_stan_family(callable)

`:normal`, `:student_t`, `:weighted`, `:censored`, `:truncated` or
`:interval_censored` when `callable` is the SLIC family token or combinator of
that name, and `nothing` for any other value.
"""
_brm_stan_family(callable) = nothing

"""
    _brm_stan_value_family_label(callable)

The label of a SLIC value family (`StanBlocks.ValueFamily`, a family
declared with `@lpxf`), or `nothing` for any other value.
"""
_brm_stan_value_family_label(callable) = nothing

"""
    _brm_stan_family_support(f, typed_args...)

StanBlocks' support record (`kind`, `type`, `size`, `support`) for a SLIC
family function `f` called with `typed_args`. Only a SLIC family has one,
so without StanBlocks the fallback refuses and names the alternative.
"""
_brm_stan_family_support(f, typed_args...) = throw(ArgumentError(
    "BRM: the support of family `$f` is declared by StanBlocks (a SLIC family " *
    "token or an `@lpxf` family). Load StanBlocks (`using StanBlocks`), or " *
    "write the component as a Distributions.jl family."))

# An RK program with no bound observation (a prior-only program) and the
# omitted responses of a reprocessed RK program are read, for now, from the
# Stan backend's declaration inventory (`generative_plan`,
# `_sb_observation_triage`), which sees synthetic observations such as a
# modeled baseline. Until BRM derives them RK-natively these need StanBlocks.
_brm_stan_needed_for(what) = throw(ArgumentError(
    "RKBRMI: $what is read from the Stan backend's declaration inventory, " *
    "which needs StanBlocks (`using StanBlocks`)."))

"""
    _brm_stan_unconditioned_generative(brmi; mod)

The Stan backend's `GenerativePlan` of `brmi` when its emitted program binds
no observation to data, and `nothing` when it binds one.
"""
_brm_stan_unconditioned_generative(brmi; kwargs...) =
    _brm_stan_needed_for("a program with no bound observation")

"""
    _brm_stan_unbound_observations(brmi; mod)

The targets of the observations `brmi`'s emitted Stan program declares
without binding them to data.
"""
_brm_stan_unbound_observations(brmi; kwargs...) =
    _brm_stan_needed_for("an omitted response")

"""
    _brm_stan_sampled_parameters(model)

The parameters a prior-only Stan program still samples (a restricted prior and
its ancestors, which forward simulation would mis-draw).
"""
_brm_stan_sampled_parameters(model) = _brm_stan_needed_for("a prior-only program's sampled parameters")
