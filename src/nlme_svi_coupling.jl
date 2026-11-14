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
# sees one ζ, so V takes a dimension-scaled stochastic natural-gradient step
#
#     V_i += β Λ_i⁻¹ (∇̄ - b_i) ζ_iᵀ / (2k_i),     k_i = dim(ζ_i),
#
# with ∇̄ the unit's gradient averaged over its antithetic draws (its pair, or
# its orthogonal design). The denominator is deterministic: dividing by a norm
# of this same ζ weights the ELBO gradient by its own draw and changes its fixed
# point on nonlinear targets. For example, z|g ~ Normal(a*g^3, 1), g ~ Normal(0, 1)
# has optimal linear slope 3a, but division by (g^2 + 1) gives about 1.904a.
# Scaling by 2k keeps a typical step comparable without that weighting. The
# baseline b_i is the unit gradient at the current global and unit means,
# independent of this visit's ζ, so E[b_i ζ_iᵀ] = 0. On a quadratic conditional
# of precision H_i it removes the mean's error exactly, leaving
# -H_i (V_i - V_i*) ζ_i. Thus V_i* is the fixed point at any number of globals.
# When Λ_i = H_i, the expected unclipped step moves V_i a fraction β / 2k_i
# of the way to it per visit, and its noise
# vanishes with V_i - V_i*, even while the unit mean remains far from its mode.
# A step that would move the
# unit's conditional mean at a typical ζ by more than one sd of its factor is
# shortened, as the mean's step is (`_BRM_SVI_UNIT_MEAN_TRUST`).
#
# Fresh runs hold V_i at zero during exploration, establishing the independent
# factors before learning their dependence in settling and polishing. On the
# event/Kalman regression, coupling from the start sent the globals toward a
# poor region even with the baseline; independent exploration avoids that path.
# The run records both policies so older optimizer checkpoints keep theirs.
#
# The coupling is linear: first order in the globals. Where the data pin units
# so tightly that a population value's posterior sd is a small difference of large
# precisions, its error in the scales' nonlinearity matters: on the strong random
# intercept of test/nlme_svi_default_centering.jl (sampled scales) the coupling
# alone gave the population mean 0.76 of NUTS's sd and learned centering 0.96, whose
# map follows the scale exactly. So V_i moves only the coordinates no learned-
# centering cell holds (`_brm_svi_coupling_rows`).
#
# The unit factor's (M_i, Λ_i, U_i) is the conditional of z_i at g = μ; its
# marginal covariance is Λ_i⁻¹ + V_i V_iᵀ on the coupled rows. Factor starts and
# `brm_svi_combine` read (M_i, Λ_i): the conditional at the saved global mean, the
# projection learned centering's maps take too. A streaming carry keeps the
# coupling as the unit prior's regression on the globals (below); the update's own
# coupling starts from zero.

"""
    LinearCoupling(; step=0.5)

The default on a fresh [`brm_svi_fit`](@ref) and streaming update: every unit's
variational factor is conditional on the globals, with a mean linear in the
global factor's draw, `z_i = M_i + V_i ζ + U_i⁻ᵀ ε_i` for ζ the global draw's
standard normals at the globals the unit's term reads (and the global factor's
rank part), on the unit's coordinates that learned centering does not centre
(`@plate` cell latents among them; a centred cell follows the globals through its
own map). Fresh fits hold `V_i` at zero during exploration, then learn it when
the unit is in the batch during settling and polishing. Its scaled
natural-gradient step has size `step` (about `step / 2k` of the way to its
quadratic fixed point per visit, for `k` coupling noise components), with a
mean-point gradient baseline that removes noise from unsettled unit means.
It keeps the dependence between a
population value and the units that the data pin only together, whatever the
model's authoring, which independent factors lose. The global factor's and the
units' entropies are unchanged by it. A streaming checkpoint carries each unit's
prior conditional on the globals it was coupled to. Pass `coupling=nothing` for
independent global and unit factors. Optimizer continuation keeps the saved
activation and baseline policies; checkpoints predating these controls retain
immediate updates without the baseline.
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

# Older optimizer checkpoints retain their baseline policy and call count.
_brm_svi_coupling_baseline(c) = get(c, :coupling_baseline, false)
_brm_svi_coupling_baseline_calls(q, i, c) =
    _brm_svi_coupling_baseline(c) && _brm_svi_coupling(q) !== nothing &&
        !isempty(q.coupling.V[i]) ? 1 : 0

# The field term reads neighboring units drawn conditional on this iteration's
# global noise. It cannot be part of a baseline independent of that noise. The
# observation and ordinary prior terms at the variational mean still can.
_brm_svi_coupling_baseline_target(t) = t
_brm_svi_coupling_baseline_target(t::NamedTuple) = merge(t, (; fields=nothing))

function _brm_svi_coupling_baseline_frame(centering, t, q)
    centering === nothing && return nothing
    try
        _brm_svi_centering_frame(centering, t, q.μ)
    catch e
        (e isa DomainError || e isa OverflowError) || rethrow()
        missing
    end
end

# E[b ζ'] = 0 for a baseline b fixed before this global draw. Removing the
# mean-point gradient prevents an unsettled unit mean from creating arbitrary
# global-unit dependence. The auxiliary point need not be in the density's
# domain: a zero baseline is also valid. Actual draw evaluations retain their
# ordinary non-finite checks and errors.
function _brm_svi_coupling_baseline!(b, t, q, i, centering, frame, plan, work)
    fill!(b, 0.0)
    frame === missing && return b
    gg, _ = _brm_svi_global_buffers(work, length(t.globals))
    try
        value = if centering === nothing
            _brm_svi_unit_local!(gg, b, t, i, q.μ, q.M[i], plan, work)
        else
            first(_brm_svi_centered_unit_term!(gg, b, centering, frame, t, q, i,
                q.μ, q.M[i], plan, work))
        end
        isfinite(value) && all(isfinite, b) || fill!(b, 0.0)
    catch e
        (e isa DomainError || e isa OverflowError) || rethrow()
        fill!(b, 0.0)
    finally
        # Reuse the worker's sparse global buffer without changing the visit's
        # actual gradients, including when the auxiliary point is unusable.
        for p in _brm_svi_unit_touched(t, i)
            gg[p] = 0.0
        end
    end
    b
end

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

# The coordinates (positions in unit i's block) the coupling moves: those no
# learned-centering cell holds. A centred cell's coordinate already follows its
# population location and scale through its learned map, and coupling it as well
# makes two parametrizations of one dependence: on the strong random-intercept
# toy of test/nlme_svi_default_centering.jl the population sd fell from 0.96 of
# NUTS's (centering alone) to 0.87 with both on the intercepts.
function _brm_svi_coupling_rows(t, i)
    d = length(t.units[i])
    centering = _brm_svi_centering(t)
    centering === nothing && return collect(1:d)
    setdiff(1:d, _brm_svi_cell_positions(centering, i))
end

function _brm_svi_with_coupling(t, q, coupling::LinearCoupling)
    r = size(q.Β, 2)
    columns = [_brm_svi_coupling_columns(t, i) for i in eachindex(t.units)]
    rows = [_brm_svi_coupling_rows(t, i) for i in eachindex(t.units)]
    V = [zeros(length(rows[i]), length(columns[i]) + r) for i in eachindex(t.units)]
    t, merge(q, (; coupling=(; coupling.step, columns, rows, V)))
end

# Unit i's ζ at the global draw's standard normals (ε, η).
_brm_svi_coupling_noise(coupling, i, ε, η) = vcat(ε[coupling.columns[i]], η)

# The coupled mean of unit i at ζ: M_i + V_i ζ on its coupled rows (M_i itself
# without coupling).
_brm_svi_coupled_mean(::Nothing, q, i, ζ) = q.M[i]
function _brm_svi_coupled_mean(coupling, q, i, ζ)
    V = coupling.V[i]
    isempty(V) && return q.M[i]
    m = copy(q.M[i])
    view(m, coupling.rows[i]) .+= V * ζ
    m
end

# Λ_i⁻¹ x for unit i's factor: two triangular solves for a dense one, the
# diagonal-plus-low-rank covariance for a structured one.
_brm_svi_unit_covariance_mul(q, i, x) = _brm_svi_unit_covariance_mul(q.U[i], x)
_brm_svi_unit_covariance_mul(U::AbstractMatrix, x) =
    (L = LowerTriangular(U); L' \ (L \ x))
function _brm_svi_unit_covariance_mul(U::_BRMSVIUnitRoot, x)
    f = U.factor
    exp.(2 .* f.ρ) .* x .+ f.Β * (f.Β' * x)
end

# The natural-gradient direction of the coupled rows R: (Λ_RR)⁻¹ x_R, the inverse
# of the precision's block (the Fisher of a mean shift on R alone), or Λ⁻¹ x when R
# is every row. A structured unit uses its covariance's block, Σ_RR x_R.
function _brm_svi_coupling_direction(q, i, rows, x)
    length(rows) == length(x) && return _brm_svi_unit_covariance_mul(q, i, x)
    _brm_svi_coupling_direction(q.Λ[i], q.U[i], rows, x[rows])
end
_brm_svi_coupling_direction(Λ::AbstractMatrix, U, rows, x) =
    cholesky(Symmetric(Λ[rows, rows])) \ x
function _brm_svi_coupling_direction(::_BRMSVIUnitPrecision, U::_BRMSVIUnitRoot, rows, x)
    f = U.factor
    B = f.Β[rows, :]
    exp.(2 .* f.ρ[rows]) .* x .+ B * (B' * x)
end

# The step of unit i's coupling from ∇̄, its gradient averaged over the visit's
# antithetic draws at ζ, with its factor before this visit's step.
function _brm_svi_coupling_step!(coupling, q, i, ∇̄, ζ, β)
    V = coupling.V[i]
    (isempty(V) || β <= 0) && return q
    n2 = sum(abs2, ζ)
    n2 > 0 || return q
    rows = coupling.rows[i]
    a = _brm_svi_coupling_direction(q, i, rows, ∇̄)
    all(isfinite, a) || throw(ArgumentError(
        "brm_svi: non-finite unit gradient in its coupling step"))
    den = 2length(ζ)
    # At a fresh ζ' the conditional mean moves by ΔV ζ', whose rms in the
    # factor's sds is β sqrt(∇̄_R' a) ‖ζ‖ / den.
    reach = sqrt(max(dot(view(∇̄, rows), a), 0.0)) * sqrt(n2) / den
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
        V, rows = coupling.V[i], coupling.rows[i]
        conditional = sqrt.(_brm_svi_unit_variances(q.U[i]))
        widening = zeros(length(conditional))
        widening[rows] .= vec(sum(abs2, V; dims=2))
        (; rows=copy(rows), columns=copy(coupling.columns[i]), V=copy(V),
           conditional_sd=conditional, marginal_sd=sqrt.(conditional .^ 2 .+ widening))
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
    (; c.step, columns=copy.(c.columns), rows=copy.(c.rows), V=copy.(c.V)))

# A run's coupling fits its target: one matrix per unit, rows its coordinates,
# columns its coupled globals plus the global rank.
function _brm_svi_coupling_consistent(q, t)
    coupling = _brm_svi_coupling(q)
    coupling === nothing && return true
    r = size(q.Β, 2)
    length(coupling.V) == length(coupling.columns) == length(coupling.rows) ==
        length(t.units) &&
        all(i -> size(coupling.V[i]) == (length(coupling.rows[i]), length(coupling.columns[i]) + r) &&
            all(j -> 1 <= j <= length(t.units[i]), coupling.rows[i]) &&
            all(k -> 1 <= k <= length(t.globals), coupling.columns[i]), eachindex(t.units))
end

# ── Streaming: a carried unit's prior conditional on the globals ─────────────
#
# A window's factors are its next window's prior. With independent factors the
# carried prior is q_old(g) Π q_old(z_i); a coupled fit's joint is q_old(g)
# Π q_old(z_i | g), and carrying only the unit factor at the global mean would
# drop exactly the dependence the coupling learned: on the panel stream of
# test/nlme_svi_streaming.jl the streamed globals' sds then ended at 0.24 and
# 0.58 of a coupled batch fit's. So a coupled unit saves the regression of its
# coupled rows on its coupled globals at the saved global factor,
#
#     W = Cov(z_R, g_C) Σ_CC⁻¹,  Cov(z_R, g_C) = V_ε diag(s_C) + V_η Β_Cᵀ,
#     Σ_CC = diag(s_C²) + Β_C Β_Cᵀ,
#
# (V ./ s_C for a mean-field global factor), and its carried prior is
# N(z; M + W (g_C - μ_C), Λ⁻¹) on those rows. For a mean-field global factor
# that is q_old(z_i | g) exactly; with a rank part, z_i | g_C, whose covariance
# Λ⁻¹ leaves out the rank part's residual (V_η's, given g_C).

function _brm_svi_saved_coupling(q, i)
    coupling = _brm_svi_coupling(q)
    (coupling === nothing || isempty(coupling.V[i])) && return nothing
    C, R, V = coupling.columns[i], coupling.rows[i], coupling.V[i]
    k = length(C)
    s, B = exp.(q.ρ[C]), q.Β[C, :]
    cov = V[:, 1:k] .* s' .+ V[:, k+1:end] * B'
    W = cov / cholesky(Symmetric(Matrix(Diagonal(s .^ 2) + B * B')))
    (; rows=copy(R), columns=copy(C), W, center=q.μ[C])
end

# A carried unit's coupling in the new window's block order: `perm[j]` is the saved
# position of the new j-th coordinate.
function _brm_svi_restored_coupling(old, perm)
    hasproperty(old, :coupling) && old.coupling !== nothing || return nothing
    position = invperm(perm)
    (; rows=position[old.coupling.rows], old.coupling.columns, old.coupling.W,
       old.coupling.center)
end

# The carried prior's mean shift on its rows at the globals g, M + W (g_C - μ_C).
_brm_svi_carried_shift!(δ, ::Nothing, g) = δ
function _brm_svi_carried_shift!(δ, carried, g)
    view(δ, carried.rows) .-= carried.W * (g[carried.columns] .- carried.center)
    δ
end

# Its gradient in the globals from the prior's gradient in the unit, -Λ δ = `gradient`.
_brm_svi_carried_gradient!(gg, ::Nothing, gradient) = gg
function _brm_svi_carried_gradient!(gg, carried, gradient)
    view(gg, carried.columns) .-= carried.W' * gradient[carried.rows]
    gg
end

# The carried units' coupled globals join their touched sets, which the batch reads
# their global gradients at. New arrays: the prepared target keeps its own.
function _brm_svi_carried_touched(t, couplings)
    all(isnothing, couplings) && return t
    touched = map(eachindex(t.touched)) do i
        c = couplings[i]
        c === nothing ? t.touched[i] : sort!(union(t.touched[i], c.columns))
    end
    merge(t, (; touched))
end
