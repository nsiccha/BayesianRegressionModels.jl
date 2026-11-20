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
# At a constant β the coupling does not settle at V_i*: its noise keeps it
# wandering around V_i*. A row's squared wander is about β m / 4 of its
# conditional variance, m being the mean square of its updates in its conditional
# sds (a_r² / σ_r², a = Λ⁻¹ (∇̄ - b)). Polishing averages the wander out, but
# settling runs at constant β. On a Gaussian row m is of order one or less, and
# the wander is small. A row the data barely inform and whose conditional is a
# funnel, such as a population spread that one unit alone reads with its
# non-centred step, is different: its updates' rms reached 4 to 5 conditional sds
# on snag default-linearco-49897cce's 32-unit event panel. The wander also feeds
# itself, because a wider coupled draw lands deeper in the funnel, where the
# gradients are larger (rms about 1 at step 0.1). The row's coupling sd then
# reached 1.4 to 2.2 times its conditional sd within a few hundred visits, with
# entries that were noise (median |t| of their updates 0.5), and the unit's
# over-dispersed draws pulled the population AR(1) towards 1. So a fresh coupling
# keeps each coupled row's mean square m over all its updates so far and divides
# that row's step by m wherever m exceeds one, which holds its squared wander near
# β/4. The scale is read before the visit's own update enters it, so it does not
# depend on the current draw and every fixed point stays. Such a row's updates
# are heavy-tailed: a rare draw deep in the funnel gives one far larger than the
# rest. The whole-history mean keeps those in m, where a window of the last 50
# visits forgot them between spikes. A fresh coupling's step also moves the
# conditional mean at a typical ζ by at most `_BRM_SVI_COUPLING_TRUST` (a quarter)
# of the factor's sd rather than a whole one: a typical step moves it by
# β |a| ‖ζ‖ / 2k, a tenth or less, so the bound binds only on such a spike. A
# whole-sd spike, reverted at the scaled step's slow rate, otherwise held the row
# near one sd. On that panel the 50-visit window with the one-sd bound let a
# row reach 1.4 sds once (SVI seed 1, iteration 64,000).
# A row whose updates stay within one conditional sd steps as before unless a step
# would move the mean by more than the quarter sd. A noisier row approaches its
# fixed point more slowly: far from it, its distance is part of m. Optimizer
# checkpoints without this state keep the unscaled step and the one-sd bound.
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
# map follows the scale exactly. The fitted model therefore uses centering alone
# for that block. Unit-owned GMRFs retain coupling after their direct-parent
# centering pass (`_brm_svi_coupling_rows`).
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
mean-point gradient baseline that removes noise from unsettled unit means. A
coupled row whose updates are noisier than one conditional sd (a population spread
that one unit alone reads, in a funnel with its non-centred step) has its step
divided by their mean square, and a fresh coupling's step moves a unit's
conditional mean by at most a quarter of its sd, so the coupling's noise cannot
widen that row.
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
_brm_svi_coupling_baseline_target(t::NamedTuple) =
    merge(Base.structdiff(t, NamedTuple{(:field_sweep,)}), (; fields=nothing))

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

# The coordinates (positions in unit i's block) the coupling moves. An ordinary
# centred cell's coordinate already follows its
# population location and scale through its learned map, and coupling it as well
# makes two parametrizations of one dependence: on the strong random-intercept
# toy of test/nlme_svi_default_centering.jl the population sd fell from 0.96 of
# NUTS's (centering alone) to 0.87 with both on the intercepts.
function _brm_svi_coupling_rows(t, i)
    d = length(t.units[i])
    centering = _brm_svi_centering(t)
    centering === nothing && return collect(1:d)
    # One Vecchia pass leaves a dependence on lengths and amplitude in the
    # learned field coordinates; these rows retain their linear coupling.
    setdiff(1:d, Iterators.flatten(inst.positions for inst in centering.instances[i]))
end

function _brm_svi_with_coupling(t, q, coupling::LinearCoupling)
    r = size(q.Β, 2)
    columns = [_brm_svi_coupling_columns(t, i) for i in eachindex(t.units)]
    rows = [_brm_svi_coupling_rows(t, i) for i in eachindex(t.units)]
    V = [zeros(length(rows[i]), length(columns[i]) + r) for i in eachindex(t.units)]
    # Each coupled row's running mean square of its update in conditional sds, and
    # the visits it has been updated on (the noise scale, above).
    noise = [zeros(length(rows[i])) for i in eachindex(t.units)]
    updates = zeros(Int, length(t.units))
    t, merge(q, (; coupling=(; coupling.step, columns, rows, V, noise, updates)))
end

# How far one step of a fresh coupling may move a unit's conditional mean at a
# typical draw of the globals, in sds of its factor (the noise scale, above).
const _BRM_SVI_COUPLING_TRUST = 0.25

# Whether a run's coupling scales its rows' steps by their noise: fresh couplings
# since snag default-linearco-49897cce; older optimizer checkpoints lack the state.
_brm_svi_coupling_scaled(coupling) = hasproperty(coupling, :noise)

# A sparse global steps on its readers' sites (src/nlme_svi_sites.jl), and a site's
# curvature is Stein's estimate along that global's own draw ε, the reader's
# factor held fixed. A reader coupled to the same ε moves with it, so its site
# measures the curvature along the coupled path, the marginal one, which pulls the
# global's mean back too weakly. On the window toy of
# test/nlme_svi_sparse_globals.jl (boundaries read by two windows each, the
# windows' `e .~ Normal(0, 1)` innovations coupled, batch 2 so N/B = 32) the
# boundaries' means drifted 60 to 220 conditional sds off through settling and
# the fit hit its iteration cap; the same coupled fit with no sparse global
# (batch 32) reached the exact means (0.03 of their own sds) and the boundaries'
# exact marginal sd (2.13 against 2.16 times the conditional). So a fresh
# coupling leaves a unit's sparse globals out of its columns: they keep their
# sites, and the unit stays coupled to the globals stepped on the minibatch
# estimate. Called before the first iteration, while every V_i is still zero.
_brm_svi_uncouple_sparse!(coupling, sites) = coupling    # no coupling or no sparse global
function _brm_svi_uncouple_sparse!(coupling::NamedTuple, sites::_BRMSVISites)
    for i in eachindex(coupling.columns)
        all(iszero, coupling.V[i]) || error("brm_svi_fit: internal: a sparse global " *
            "left a unit's coupling after its first step")
        keep = [k for (k, p) in enumerate(coupling.columns[i]) if sites.slot[p] == 0]
        length(keep) == length(coupling.columns[i]) && continue
        r = size(coupling.V[i], 2) - length(coupling.columns[i])
        coupling.columns[i] = coupling.columns[i][keep]
        coupling.V[i] = zeros(size(coupling.V[i], 1), length(keep) + r)
    end
    coupling
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

# The diagonal of the coupled rows' direction operator, (Λ_RR)⁻¹: their
# conditional variances. A structured unit's is that of its covariance block Σ_RR,
# the operator its direction uses.
function _brm_svi_coupling_variances(q, i, rows)
    length(rows) == length(q.M[i]) && return _brm_svi_unit_variances(q.U[i])
    _brm_svi_coupling_variances(q.Λ[i], q.U[i], rows)
end
_brm_svi_coupling_variances(Λ::AbstractMatrix, U, rows) =
    vec(sum(abs2, inv(cholesky(Symmetric(Λ[rows, rows])).L); dims=1))
_brm_svi_coupling_variances(::_BRMSVIUnitPrecision, U::_BRMSVIUnitRoot, rows) =
    (f = U.factor; exp.(2 .* f.ρ[rows]) .+ vec(sum(abs2, view(f.Β, rows, :); dims=2)))

# y' A y for a step y of the coupled rows, A the inverse of their direction
# operator: Λ_RR, or for a structured unit Σ_RR⁻¹ (by Woodbury's identity).
_brm_svi_coupling_norm2(q, i, rows, y) = _brm_svi_coupling_norm2(q.Λ[i], q.U[i], rows, y)
_brm_svi_coupling_norm2(Λ::AbstractMatrix, U, rows, y) = dot(y, Symmetric(Λ[rows, rows]) * y)
function _brm_svi_coupling_norm2(::_BRMSVIUnitPrecision, U::_BRMSVIUnitRoot, rows, y)
    f = U.factor
    d = exp.(2 .* f.ρ[rows])
    B = f.Β[rows, :]
    w = y ./ d
    C = cholesky(Symmetric(I + B' * (B ./ d)))
    dot(y, w) - sum(abs2, C.L \ (B' * w))
end

# Each coupled row's step scale: one over the mean square of its updates so far in
# conditional sds where that exceeds one, otherwise one (also before its first
# update). A row stepped at s β with updates of mean square m wanders by about
# s β m / 4 squared sds, so s = 1/m holds that at β/4 whatever the noise (1/rms
# would leave it growing like rms).
_brm_svi_coupling_row_scales(coupling, i) = 1 ./ max.(1.0, coupling.noise[i])

# This visit's update a joins unit i's rows' mean squares, in their conditional sds.
function _brm_svi_coupling_noise!(coupling, q, i, rows, a)
    n = coupling.updates[i] += 1
    σ2 = _brm_svi_coupling_variances(q, i, rows)
    m = coupling.noise[i]
    @. m += (a^2 / σ2 - m) / n
    m
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
    # The rows' scales from their earlier updates, read before this one joins them.
    s, trust = nothing, _BRM_SVI_UNIT_MEAN_TRUST
    if _brm_svi_coupling_scaled(coupling)
        s, trust = _brm_svi_coupling_row_scales(coupling, i), _BRM_SVI_COUPLING_TRUST
        _brm_svi_coupling_noise!(coupling, q, i, rows, a)
    end
    # At a fresh ζ' the conditional mean moves by ΔV ζ', whose rms in the
    # factor's sds is β sqrt(∇̄_R' a) ‖ζ‖ / den for the unscaled step.
    if s === nothing || all(isone, s)
        reach = sqrt(max(dot(view(∇̄, rows), a), 0.0)) * sqrt(n2) / den
    else
        a = a .* s
        reach = sqrt(max(_brm_svi_coupling_norm2(q, i, rows, a), 0.0)) * sqrt(n2) / den
    end
    reach > 0 || return q
    β = min(β, trust / reach)
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
    scaled = _brm_svi_coupling_scaled(coupling)
    length(coupling.V) == length(coupling.columns) == length(coupling.rows) ==
        length(t.units) &&
        (!scaled || length(coupling.noise) == length(coupling.updates) == length(t.units)) &&
        all(i -> size(coupling.V[i]) == (length(coupling.rows[i]), length(coupling.columns[i]) + r) &&
            (!scaled || length(coupling.noise[i]) == length(coupling.rows[i])) &&
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
