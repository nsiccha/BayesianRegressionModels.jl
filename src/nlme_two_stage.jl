# src/nlme_two_stage.jl — one unit's conditional posterior, and the shared
# result shape of BRM's approximate-inference methods (two-stage fits, minibatch
# SVI, per-unit Laplace). A unit is one subject of `brm_nlme_view`: given fixed
# population values its random effects are conditionally independent of every
# other unit's, so its posterior is a small problem of its own, evaluated on
# its own observation cells only (`brm_nlme_loglikelihood`).
#
# Coordinates. A unit problem works in BRM's own standardized subject
# coordinates `z`, with `η = C z` and `C` the lower Cholesky factor of the
# unit's Ω block. Those are exactly the RK packed subject coordinates once Ω's
# scale and correlation coordinates are set from Ω, so a unit's draws drop into
# a full RK packed draw (`brm_nlme_draws`) that every BRM reader accepts.

"""
    BRMNLMEUnitProblem

One unit's conditional posterior given fixed population values, returned by
[`brm_nlme_unit_problem`](@ref). Its coordinates are the unit's standardized
random effects `z` (`η = C z`, `C` the lower Cholesky factor of the unit's Ω
block), and its log density is `-|z|²/2 - d/2 log 2π + loglik_i(θ, σ, C z)`,
which is the RK model's density in those coordinates up to the population
terms, which are held fixed. It implements `LogDensityProblems` with a
gradient (first order), so any sampler can run it, one unit at a time or in
parallel.

Fields: `model`, `subject` (the unit's position in `model.view.levels`),
`theta`, `sigma` and `factor` (`C`).
"""
struct BRMNLMEUnitProblem{M<:BRMNLMEModel}
    model::M
    subject::Int
    theta::Vector{Float64}
    sigma::Vector{Float64}
    factor::Matrix{Float64}
end

Base.show(io::IO, p::BRMNLMEUnitProblem) = print(io, "BRMNLMEUnitProblem(subject ",
    repr(p.model.view.levels[p.subject]), ", dimension ", size(p.factor, 1), ")")

"""
    brm_nlme_unit_problem(m::BRMNLMEModel, i, θ, σ, Ω) -> BRMNLMEUnitProblem

Unit `i`'s conditional posterior at fixed population values `θ`, `σ` and `Ω`:
Ω is the natural-effect covariance as [`brm_nlme_estimates`](@ref) takes it, one
matrix for a flat model or one per covariance family for a nested one. The unit
is evaluated on its own observation cells (see [`brm_nlme_model`](@ref)), so
its cost does not grow with the number of units. A flat model whose
observations have no subject cells is refused with the reason. A nested
model's unit gradient evaluates the whole model
(`brm_nlme_loglikelihoods_and_gradients`), at a cost that grows with the
number of units.
"""
function brm_nlme_unit_problem(m::BRMNLMEModel, i::Integer, θ, σ, Ω)
    _brm_nlme_check_point(m, θ, σ, Ω)
    1 <= i <= length(m.view.levels) || throw(BoundsError(m.view.levels, i))
    m.cell_query === nothing && !_brm_nlme_ragged(m.view) && _brm_nlme_error(
        "brm_nlme_unit_problem: a unit is evaluated on its own observation " *
        "cells, but this model's observations have no cells that each belong " *
        "to one subject")
    block = _brm_nlme_unit_omega(m, Ω, i)
    factor = Matrix{Float64}(cholesky(Symmetric(block)).L)
    BRMNLMEUnitProblem(m, Int(i), Vector{Float64}(θ), Vector{Float64}(σ), factor)
end

# A unit's Ω block: the flat model's whole covariance, or the block diagonal of
# the unit's families in `subject_families[i]` order.
_brm_nlme_unit_omega(_m, Ω::AbstractMatrix, _i) = Matrix{Float64}(Ω)
function _brm_nlme_unit_omega(m, Ω::AbstractVector{<:AbstractMatrix}, i)
    families = m.view.subject_families[i]
    n = sum(f -> size(Ω[f], 1), families; init=0)
    block = zeros(n, n)
    at = 0
    for f in families
        K = size(Ω[f], 1)
        block[(at + 1):(at + K), (at + 1):(at + K)] .= Ω[f]
        at += K
    end
    block
end

LogDensityProblems.dimension(p::BRMNLMEUnitProblem) = size(p.factor, 1)
LogDensityProblems.capabilities(::Type{<:BRMNLMEUnitProblem}) =
    LogDensityProblems.LogDensityOrder{1}()

_brm_nlme_unit_prior(z) = -0.5 * (dot(z, z) + length(z) * log(2π))

LogDensityProblems.logdensity(p::BRMNLMEUnitProblem, z) =
    _brm_nlme_unit_prior(z) +
        brm_nlme_loglikelihood(p.model, p.subject, p.theta, p.sigma, p.factor * z)

function LogDensityProblems.logdensity_and_gradient(p::BRMNLMEUnitProblem, z)
    value, gradient = brm_nlme_loglikelihood_and_gradient(p.model, p.subject,
        p.theta, p.sigma, p.factor * z)
    _brm_nlme_unit_prior(z) + value, transpose(p.factor) * gradient .- z
end

# A copy of `m` with its own prepared subject-cell queries and compact gradient
# samplers, so units evaluate concurrently on another thread; everything else
# is shared read-only. A nested model evaluates subject gradients through the
# whole-model sampler and has no per-cell workspace to copy.
function _brm_nlme_worker_model(m::BRMNLMEModel)
    m.cell_gradient === nothing && _brm_nlme_error(
        "brm_two_stage_fit: units of a nested model evaluate through the " *
        "whole-model sampler, which one thread owns; use `threads = 1`")
    cell_query = _brm_nlme_cell_queries(m.backend, m.cell_query)
    cell_gradient = (; _brm_nlme_cell_samplers(m.backend, m.cell_query, m.unit,
        sort!(vcat(m.theta, m.sigma)), m.view.subject_coordinates)...,
        lock=ReentrantLock())
    BRMNLMEModel(m.view, m.backend, m.sampler, m.pointwise, m.eta_hvp, m.attribution,
        m.theta, m.sigma, m.eta_blocks, m.unit, m.observations_query,
        m.observations_pushforward, m.observed, m.observation_attribution,
        cell_query, cell_gradient, m.population_prior, m.prior_structure)
end

"""
    brm_nlme_unit_eta(p::BRMNLMEUnitProblem, z) -> Vector

The unit's natural random effects `η = C z` at its standardized point `z`.
"""
brm_nlme_unit_eta(p::BRMNLMEUnitProblem, z) = p.factor * z

"""
    brm_nlme_draws(m::BRMNLMEModel, θ, σ, ω, unit_draws) -> Matrix{Float64}

Full RK packed unconstrained draws, one column per draw (`m.view.coordinates`
× draws, rows in the order of `coordinate_names(rk.model.layout)`, as
[`rk_restore_draws`](@ref) and `rk_draw_values` take them), at fixed
population values: `θ`, `σ`, and Ω's covariance coordinates from
NLMEEstimation's `ω`, with unit `i`'s coordinates taken from `unit_draws[i]`
(the unit's dimension × draws, in its [`BRMNLMEUnitProblem`](@ref)
coordinates). Every unit gives the same number of draws. This is the result shape BRM's approximate-inference
methods share, so their output reaches the usual readers and replay paths.
Implemented by the NLMEEstimation extension.
"""
function brm_nlme_draws end

"""
    brm_two_stage_fit(sampler, x; minibatch_size=nothing, options=nothing,
        threads=Threads.nthreads(), stage1=nothing, kwargs...) -> NamedTuple

A two-stage approximate posterior for a model with conditionally independent
units (the subjects of [`brm_nlme_view`](@ref)):

1. **Population:** SAEM ([`brm_nlme_fit`](@ref) with `NLMEEstimation.saem`)
   estimates `θ`, `σ` and Ω. With `minibatch_size = b`, each iteration's E-step
   samples a random batch of `b` units (`NLMEEstimation.SAEMOptions(minibatch_size=b)`);
   `options` passes a whole `SAEMOptions` instead. Other keywords go to
   [`brm_nlme_fit`](@ref) (`start`, `inits`, `progress`, `rng`, …); pass
   `rng` for a reproducible stage 1, since SAEM otherwise draws from the
   global RNG. `stage1` reuses an earlier `brm_nlme_fit(x, saem; …)` result
   of the same `x` instead (its model is the one sampled), for example to
   resample the units with another sampler.
2. **Units:** every unit's conditional posterior at those estimates,
   [`brm_nlme_unit_problem`](@ref), is sampled by `sampler(problem)`. It
   returns a dimension × draws matrix in the problem's coordinates (WarmupHMC's
   `posterior_position` orientation), and every
   unit returns the same number of draws. Units are sampled on `threads` tasks,
   each with its own prepared copy of the model's cells, so `sampler` must be
   safe to call concurrently (an independent RNG per unit, as below). A nested
   model samples on one thread: its units evaluate through the whole-model
   sampler.

Each unit's posterior is exact given the population estimates; the population
uncertainty is not propagated (empirical Bayes). That is negligible with many
units and is not negligible with few.

Returns `(; draws, coordinates, units, unit_draws, stage1)`: RK packed draws as
[`brm_nlme_draws`](@ref) assembles them, their coordinate names, the unit
problems, each unit's draws, and the stage-1 `brm_nlme_fit` result.
Implemented by the NLMEEstimation extension.

```julia
using NLMEEstimation, WarmupHMC, Random
fit = brm_two_stage_fit(RKBRMI(model); minibatch_size=64) do unit
    run = adaptive_warmup_mcmc(Xoshiro(unit.subject), unit)
    Matrix(run.posterior_position)
end
```
"""
function brm_two_stage_fit end
