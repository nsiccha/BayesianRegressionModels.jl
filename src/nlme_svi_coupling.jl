# src/nlme_svi_coupling.jl — each unit's factor conditional on the globals.
#
# With independent global and unit factors, q(g) learns a global's uncertainty
# given every unit's coordinates. Where the data pin a combination of a global
# and the units (a population step mean beside per-unit steps the data pin), that
# conditional sd is about the noise over all rows, while the posterior's is the
# units' spread over √N: the global's interval comes out a fraction of the
# posterior's. Learned centering (src/nlme_svi_centering.jl) repairs this for
# the cells whose location and scale it reads from the model; a unit coordinate
# the population enters only through what the model computes from it (an authored
# `@plate` cell latent `z ~ Normal(0, 1)` read as `m .+ s .* z`) stays independent.
#
# LinearCoupling makes every unit's factor conditional on the globals,
#
#     g   = μ + exp.(ρ) .* ε + Β η,
#     z_i = M_i + V_i ζ_i + U_i⁻ᵀ ε_i,      ζ_i = (ε[C_i]; η),
#
# with C_i the globals unit i's term reads (and its centering locations), so
# Cov(z_i, g) = V_i (diag(exp.(ρ[C_i])) ; Β[C_i, :]ᵀ) and z_i | g is Gaussian
# with a mean linear in g: the "branch" Gaussian of Agrawal and Domke (2021),
# "Amortized variational inference for simple hierarchical models", which holds
# the exact posterior of a Gaussian hierarchical model with known scales. Under
# learned centering z_i is the unit's learned coordinates u_i.
#
# V_i couples the unit to the global factor's standard-normal draws, not to g:
# z_i does not depend on (μ, ρ, Β) given (ε, η), so the ELBO's global gradients
# are unchanged (as are the sparse globals' sites and the sd bound), and so is
# the entropy, since the map from (ε, η, ε_i) is block-triangular. M_i and the
# unit's precision keep their natural-gradient step at the coupled draw, and
#
#     ∂ELBO/∂V_i = E[∇_{z_i} log p ζ_iᵀ].
#
# Its natural gradient, with Fisher Λ_i ⊗ I, is Λ_i⁻¹ E[∇ log p ζᵀ]. One visit
# sees one ζ, so V takes a regularized normalized least-mean-squares step
#
#     V_i += β Λ_i⁻¹ ∇̄ ζ_iᵀ / (‖ζ_i‖² + k_i),     k_i = dim(ζ_i),
#
# with ∇̄ the unit's gradient averaged over its antithetic draws (its pair, or
# its orthogonal design). On a target quadratic in (g, z_i) that gradient is
# -Λ_i (V_i - V_i*) ζ_i plus a part independent of this visit's ζ_i: the error
# the unit's mean took up at its previous visit, (V_i - V_i*) ζ_prev, since a
# whole natural-gradient step puts M_i + V_i ζ_prev at that draw's conditional
# mode. So V_i* is the fixed point at any number of globals, the step moves V_i
# about a fraction β / 2k_i of the way to it per visit, and its noise vanishes
# with V_i - V_i*. Dividing by ‖ζ_i‖² alone (the plain normalized step) turns that
# carried error into a ratio of normals, ζ_prev / ζ, whose variance is infinite
# for one coupled global: on the known-scale random intercept in
# test/nlme_svi_linear_coupling.jl, whose marginal sd is a small difference of
# large precisions, it ended at 0.89 of the exact sd. A step that would move the
# unit's conditional mean at a typical ζ by more than one sd of its factor is
# shortened, as the mean's step is (`_BRM_SVI_UNIT_MEAN_TRUST`).
#
# The unit factor's (M_i, Λ_i, U_i) is the conditional of z_i at g = μ; its
# marginal covariance is Λ_i⁻¹ + V_i V_iᵀ. Composition (streaming carry, factor
# starts, `brm_svi_combine`) reads (M_i, Λ_i): the conditional at the saved global
# mean, the projection learned centering's maps take too. A streaming update's
# coupling starts from zero.

"""
    LinearCoupling(; step=0.5)

The default on a fresh [`brm_svi_fit`](@ref) and streaming update: every unit's
variational factor is conditional on the globals, with a mean linear in the
global factor's draw, `z_i = M_i + V_i ζ + U_i⁻ᵀ ε_i` for ζ the global draw's
standard normals at the globals the unit's term reads (and the global factor's
rank part). `V_i` is learned with the fit by a normalized natural-gradient step
of size `step` (about `step / 2k` of the way to its fixed point per visit, for
`k` coupled globals) whenever the unit is in the batch, from zero, so a coupling that
the data do not need stays near zero. It keeps the dependence between a
population value and the units that the data pin only together, whatever the
model's authoring, which independent factors lose. The global factor's and the
units' entropies are unchanged by it. Pass `coupling=nothing` for independent
global and unit factors. Optimizer continuation keeps the saved policy.
"""
struct LinearCoupling
    step::Float64
    function LinearCoupling(; step::Real=0.5)
        isfinite(step) && 0 < step <= 1 || throw(ArgumentError(
            "LinearCoupling: step must lie in (0, 1], got $step"))
        new(Float64(step))
    end
end

# Omission selects the coupling. A Gaussian-block Laplace fit has no unit factors.
function _brm_svi_resolve_coupling(coupling)
    ismissing(coupling) && return LinearCoupling()
    coupling === nothing || coupling isa LinearCoupling || throw(ArgumentError(
        "brm_svi_fit: coupling must be LinearCoupling() or nothing, got $(repr(coupling))"))
    coupling
end

_brm_svi_coupling(q) = hasproperty(q, :coupling) ? q.coupling : nothing

# The globals (positions in `t.globals`) unit i's factor is coupled to: those its
# term reads, and those its learned-centering locations read.
function _brm_svi_coupling_columns(t, i)
    columns = collect(Int, _brm_svi_unit_touched(t, i))
    centering = _brm_svi_centering(t)
    if centering !== nothing
        for inst in centering.instances[i]
            inst isa _BRMSVICenteringBlock || continue
            for location in inst.location, (k, _) in location
                k > 0 && push!(columns, k)
            end
        end
    end
    sort!(unique!(columns))
end

function _brm_svi_with_coupling(t, q, coupling::LinearCoupling)
    r = size(q.Β, 2)
    columns = [_brm_svi_coupling_columns(t, i) for i in eachindex(t.units)]
    V = [zeros(length(t.units[i]), length(columns[i]) + r) for i in eachindex(t.units)]
    t, merge(q, (; coupling=(; coupling.step, columns, V)))
end

# Unit i's ζ at the global draw's standard normals (ε, η).
_brm_svi_coupling_noise(coupling, i, ε, η) = vcat(ε[coupling.columns[i]], η)

# The coupled mean of unit i at ζ: M_i + V_i ζ (M_i itself without coupling).
_brm_svi_coupled_mean(::Nothing, q, i, ζ) = q.M[i]
_brm_svi_coupled_mean(coupling, q, i, ζ) =
    isempty(q.M[i]) ? q.M[i] : q.M[i] .+ coupling.V[i] * ζ

# Λ_i⁻¹ x for unit i's factor: two triangular solves for a dense one, the
# diagonal-plus-low-rank covariance for a structured one.
_brm_svi_unit_covariance_mul(q, i, x) = _brm_svi_unit_covariance_mul(q.U[i], x)
_brm_svi_unit_covariance_mul(U::AbstractMatrix, x) =
    (L = LowerTriangular(U); L' \ (L \ x))
function _brm_svi_unit_covariance_mul(U::_BRMSVIUnitRoot, x)
    f = U.factor
    exp.(2 .* f.ρ) .* x .+ f.Β * (f.Β' * x)
end

# The step of unit i's coupling from ∇̄, its gradient averaged over the visit's
# antithetic draws at ζ, with its factor before this visit's step.
function _brm_svi_coupling_step!(coupling, q, i, ∇̄, ζ, β)
    V = coupling.V[i]
    (isempty(V) || β <= 0) && return q
    n2 = sum(abs2, ζ)
    n2 > 0 || return q
    a = _brm_svi_unit_covariance_mul(q, i, ∇̄)
    all(isfinite, a) || throw(ArgumentError(
        "brm_svi: non-finite unit gradient in its coupling step"))
    den = n2 + length(ζ)
    # At a fresh ζ' the conditional mean moves by ΔV ζ', whose rms in the
    # factor's sds is β ‖∇̄‖_{Λ⁻¹} ‖ζ‖ / den.
    reach = sqrt(max(dot(∇̄, a), 0.0)) * sqrt(n2) / den
    reach > 0 || return q
    β = min(β, _BRM_SVI_UNIT_MEAN_TRUST / reach)
    BLAS.ger!(β / den, a, ζ, V)
    q
end

# Polishing's running average of the coupling, as of a unit's mean.
function _brm_svi_coupling_average!(average, q, i, weight)
    a, x = _brm_svi_coupling(average), _brm_svi_coupling(q)
    (a === nothing || x === nothing) && return average
    @. a.V[i] += weight * (x.V[i] - a.V[i])
    average
end

# What `diagnostics.coupling` reports: the step, each unit's columns (positions
# in `diagnostics.globals`) and coupling matrix, and each unit's marginal sds
# next to its conditional ones, the widening the coupling carries.
function _brm_svi_coupling_report(q)
    coupling = _brm_svi_coupling(q)
    coupling === nothing && return nothing
    units = map(eachindex(q.M)) do i
        V = coupling.V[i]
        conditional = sqrt.(_brm_svi_unit_variances(q.U[i]))
        (; columns=copy(coupling.columns[i]), V=copy(V), conditional_sd=conditional,
           marginal_sd=sqrt.(conditional .^ 2 .+ vec(sum(abs2, V; dims=2))))
    end
    (; coupling.step, rank=size(q.Β, 2), units)
end

# The diagonal of a unit factor's covariance Λ⁻¹.
_brm_svi_unit_variances(U::AbstractMatrix) =
    isempty(U) ? Float64[] : vec(sum(abs2, inv(LowerTriangular(U)); dims=1))
_brm_svi_unit_variances(U::_BRMSVIUnitRoot) =
    exp.(2 .* U.factor.ρ) .+ vec(sum(abs2, U.factor.Β; dims=2))

# Plain data for a snapshot.
_brm_svi_coupling_plain(q) = (c = _brm_svi_coupling(q); c === nothing ? nothing :
    (; c.step, columns=copy.(c.columns), V=copy.(c.V)))

# A run's coupling fits its target: one matrix per unit, rows its coordinates,
# columns its coupled globals plus the global rank.
function _brm_svi_coupling_consistent(q, t)
    coupling = _brm_svi_coupling(q)
    coupling === nothing && return true
    r = size(q.Β, 2)
    length(coupling.V) == length(coupling.columns) == length(t.units) &&
        all(i -> size(coupling.V[i]) == (length(t.units[i]), length(coupling.columns[i]) + r) &&
            all(k -> 1 <= k <= length(t.globals), coupling.columns[i]), eachindex(t.units))
end
