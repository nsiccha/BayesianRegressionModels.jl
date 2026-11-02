# Prior/likelihood power-scaling sensitivity for BRM models.
# PowerScaling.jl owns CJS distances, PSIS weights and the sensitivity summary;
# BRM assembles the joint densities and selects model parameter columns.

"""
    BRMPowerscaleSensitivity

Alias for `PowerScaling.PowerscaleSensitivity`, returned by
[`brm_powerscale_sensitivity`](@ref). The result contains `variables`, `prior`,
`likelihood`, `diagnosis`, the four `pareto_k` values, and component-wise
`reliable` flags. Check reliability before interpreting the sensitivities.
"""
const BRMPowerscaleSensitivity = PowerScaling.PowerscaleSensitivity

"""
    brm_powerscale_sensitivity(draws, log_prior, log_lik, names;
        variables=nothing, lower_alpha=0.99, upper_alpha=1.01,
        threshold=0.05, reff=1.0)

Prior/likelihood power-scaling sensitivity over fitted posterior draws,
computed by `PowerScaling.powerscale_sensitivity` using PSIS.jl. `draws` is
draws × coordinates, `names` contains its column names, and `log_prior` /
`log_lik` contain one joint log-density per draw. `variables` selects columns
by name (default: all). Use [`brm_powerscale_inputs`](@ref) to assemble these
inputs from an SBBRMI fit with saved generated quantities.

The returned [`BRMPowerscaleSensitivity`](@ref) reports CJS-distance gradients
and diagnoses prior sensitivity or potential prior-data conflict at `threshold`.
`reff` is the relative efficiency of the inverse importance ratios (1 assumes
independent draws). Check the result's `reliable` flags and `pareto_k` values
before interpreting it; these diagnostics do not establish MCMC convergence.

The summary scales exactly the joint densities supplied. Hierarchical models
may need densities restricted to selected prior levels; model-specific density
assembly remains the caller's responsibility.
"""
function brm_powerscale_sensitivity(draws::AbstractMatrix,
                                    log_prior::AbstractVector,
                                    log_lik::AbstractVector,
                                    names; kwargs...)
    _brm_check_draw_names(draws, names)
    PowerScaling.powerscale_sensitivity(draws, log_prior, log_lik, names; kwargs...)
end

# ---- SBBRMI assembly --------------------------------------------------------

# Covered surface (v1). Scalar priors admit exactly these families; likelihoods
# admit exactly these response families (both introduce no latent parameters of
# their own, so the Julia-side joint prior is COMPLETE for an accepted model —
# up to an additive constant, which importance weights normalise away). Term
# heads that own sampled parameters outside this evaluator are rejected by
# name; deterministic transforms (log/exp/*, offset, center/zscale/...) pass.
const _BRM_POWERSCALE_PRIOR_FAMILIES = (Normal, Exponential)
const _BRM_POWERSCALE_LIKELIHOODS = (Normal, BernoulliLogit)
const _BRM_POWERSCALE_REJECT_HEADS =
    ((|), doublepipe, gr, mm, gp, hsgp, s, t2, ar, dar, rw, cdar, car, mo, mo1,
     kernel, ragged, mi, me, interval_censored, LKJCovarianceFactor,
     MvNormalCholesky, effect, r2d2)

function _brm_powerscale_collect_heads!(heads::Set, node)
    if node isa ExprColumn
        push!(heads, getf(node))
        for a in getargs(node)
            _brm_powerscale_collect_heads!(heads, a)
        end
        for v in values(getkwargs(node))
            _brm_powerscale_collect_heads!(heads, v)
        end
    elseif node isa MultiMembershipTerm
        push!(heads, mm)
    elseif node isa JointResponseColumn
        push!(heads, JointResponseColumn)
    end
    heads
end

_brm_powerscale_head_name(h) =
    h isa Type ? string(nameof(h)) :
    h isa Function ? string(nameof(h)) : string(h)

# Fail-closed coverage inventory over the whole BRMI. Every `~` operation is
# classified (likelihood / prior address / scalar prior / linear predictor)
# and anything the Julia evaluator below cannot price errors by NAME. An
# accepted model has a provably complete joint prior; the BridgeStan identity
# battery in `test/powerscale.jl` proves it per admitted shape.
function _brm_powerscale_gate!(brmi::BRMI)
    for spec in r2d2_priors(brmi)
        error("brm_powerscale_inputs: `$(spec.predictor)` carries an `r2d2` " *
              "variance decomposition, whose joint prior the Julia evaluator " *
              "does not cover yet. Drop the decomposition or use the refit " *
              "comparison from brm-use until selective scaling ships.")
    end
    for spec in ranef_effect_priors(brmi)
        error("brm_powerscale_inputs: `$(spec.class)(:, $(spec.id))` " *
              "addresses a shared random-effect covariance the Julia " *
              "evaluator does not cover yet (and hierarchical priors need " *
              "selective top-level-only scaling — see the EABM sensitivity " *
              "chapter). Use the refit comparison from brm-use instead.")
    end
    isempty(term_priors(brmi)) || error(
        "brm_powerscale_inputs: term-addressed priors " *
        "(`sd`/`ar`/`simplex`/`latent`/`length_scale` on s/t2/gp/hsgp/...) " *
        "are outside the covered surface. Use the refit comparison instead.")
    for spec in effect_priors(brmi)
        if spec.family !== Normal || !isempty(spec.keywords)
            error("brm_powerscale_inputs: `effect($(spec.predictor), " *
                  "$(spec.coefficient))` carries a non-Normal (or keyword) " *
                  "prior the Julia evaluator does not cover. Only " *
                  "`effect(...) ~ Normal(location, scale)` is supported.")
        end
    end
    for (key, op_nc) in pairs(brmi.operations)
        op = _named_op(op_nc)
        isnothing(op) && continue
        getf(op) === (~) || continue
        lhs, rhs = getargs(op, 2)
        if !isnothing(_observed_lhs_or_nothing(lhs))
            _brm_powerscale_gate_likelihood!(key, lhs, rhs)
        elseif lhs isa ExprColumn && getf(lhs) === effect
            continue
        elseif _brm_is_prior_declaration(brmi, key)
            _brm_powerscale_gate_scalar!(key, lhs, rhs)
        else
            _brm_powerscale_gate_predictor!(key, rhs)
        end
    end
    nothing
end

function _brm_powerscale_gate_likelihood!(key::Symbol, lhs, rhs)
    plain = lhs isa NamedColumn && parent(lhs) isa DataColumn
    plain || error("brm_powerscale_inputs: response `$key` has a decorated " *
                   "LHS (link / `mi` / joint), outside the " *
                   "covered surface. Use the refit comparison instead.")
    rhs isa ExprColumn && getf(rhs) in _BRM_POWERSCALE_LIKELIHOODS || error(
        "brm_powerscale_inputs: response `$key` uses " *
        "`$(_brm_powerscale_head_name(rhs isa ExprColumn ? getf(rhs) : rhs))`, " *
        "outside the covered likelihoods (`Normal`, `BernoulliLogit`). " *
        "Use the refit comparison instead.")
    nothing
end

function _brm_powerscale_gate_scalar!(key::Symbol, lhs, rhs)
    lhs isa NamedColumn || error(
        "brm_powerscale_inputs: prior `$key` has a non-bare LHS, outside " *
        "the covered surface.")
    rhs isa ExprColumn && getf(rhs) in _BRM_POWERSCALE_PRIOR_FAMILIES || error(
        "brm_powerscale_inputs: prior `$key` uses " *
        "`$(_brm_powerscale_head_name(rhs isa ExprColumn ? getf(rhs) : rhs))`, " *
        "outside the covered scalar families (`Normal`, `Exponential`).")
    for k in keys(getkwargs(rhs))
        k in (:lower, :upper) || error(
            "brm_powerscale_inputs: prior `$key` carries keyword `$k`; only " *
            "support bounds (`lower`/`upper`) are covered.")
        isnothing(_brm_numeric_constant(getkwargs(rhs)[k])) && error(
            "brm_powerscale_inputs: prior `$key` has a non-constant `$k` " *
            "bound; only constant support bounds are covered.")
    end
    for a in getargs(rhs)
        isnothing(_brm_numeric_constant(a)) && error(
            "brm_powerscale_inputs: prior `$key` has a non-constant argument " *
            "`$a` (data- or parameter-dependent). Hierarchical priors need " *
            "selective scaling; data-dependent scales are not resolved yet. " *
            "Use the refit comparison instead.")
    end
    nothing
end

function _brm_powerscale_gate_predictor!(key::Symbol, rhs)
    heads = _brm_powerscale_collect_heads!(Set{Any}(), rhs)
    for h in _BRM_POWERSCALE_REJECT_HEADS
        h in heads && error(
            "brm_powerscale_inputs: predictor `$key` contains " *
            "`$(_brm_powerscale_head_name(h))`, whose parameters the Julia " *
            "evaluator does not cover yet. Use the refit comparison from " *
            "brm-use instead.")
    end
    JointResponseColumn in heads && error(
        "brm_powerscale_inputs: predictor `$key` contains a joint response, " *
        "outside the covered surface.")
    nothing
end

# ---- Part 4b: the Julia-side joint prior -------------------------------------

# Numeric `(location, scale)` from a Normal effect/scalar RHS, mirroring the
# emission-side arg handling (`_brm_normal_effect_args` defaults, constant
# folding). Anything non-constant or non-positive-scale fails closed.
function _brm_powerscale_normal_args(rhs::ExprColumn, what::String)
    _sb_is_normal_effect_prior(rhs) || error(
        "brm_powerscale_inputs: $what is not a `Normal` prior " *
        "(internal coverage error — the gate should have rejected it)")
    raw = _brm_normal_effect_args(rhs; prefix="brm_powerscale_inputs")
    loc = _brm_numeric_constant(raw[1])
    scale = _brm_numeric_constant(raw[2])
    (isnothing(loc) || isnothing(scale)) && error(
        "brm_powerscale_inputs: $what has a non-constant location/scale; " *
        "only constant `Normal(location, scale)` is covered.")
    isfinite(loc) && isfinite(scale) && scale > 0 || error(
        "brm_powerscale_inputs: $what has invalid location/scale " *
        "($loc, $scale)")
    Float64(loc), Float64(scale)
end

function _brm_powerscale_scalar_dist(rhs::ExprColumn, key::Symbol)
    F = getf(rhs)
    args = getargs(rhs)
    if F === Normal
        length(args) <= 2 || error(
            "brm_powerscale_inputs: prior `$key` has $(length(args)) " *
            "arguments; `Normal` takes at most location and scale.")
        loc = isempty(args) ? 0.0 : _brm_numeric_constant(args[1])
        scale = length(args) < 2 ? 1.0 : _brm_numeric_constant(args[2])
        (isnothing(loc) || isnothing(scale)) && error(
            "brm_powerscale_inputs: prior `$key` has a non-constant " *
            "location/scale (internal coverage error)")
        isfinite(loc) && isfinite(scale) && scale > 0 || error(
            "brm_powerscale_inputs: prior `$key` has invalid location/scale " *
            "($loc, $scale)")
        dist = Normal(Float64(loc), Float64(scale))
    elseif F === Exponential
        length(args) <= 1 || error(
            "brm_powerscale_inputs: prior `$key` has $(length(args)) " *
            "arguments; `Exponential` takes at most a scale.")
        scale = isempty(args) ? 1.0 : _brm_numeric_constant(only(args))
        isnothing(scale) && error(
            "brm_powerscale_inputs: prior `$key` has a non-constant scale " *
            "(internal coverage error)")
        isfinite(scale) && scale > 0 || error(
            "brm_powerscale_inputs: prior `$key` has invalid scale ($scale)")
        # Julia parameterisation (mean `scale`); Stan lowering translates to
        # rate, so this matches the fitted density exactly.
        dist = Exponential(Float64(scale))
    else
        error("brm_powerscale_inputs: prior `$key` uses `$F` (internal " *
              "coverage error — the gate should have rejected it)")
    end
    kw = getkwargs(rhs)
    for k in (:lower, :upper)
        haskey(kw, k) || continue
        v = _brm_numeric_constant(kw[k])
        (isnothing(v) || !isfinite(v)) && error(
            "brm_powerscale_inputs: prior `$key` has a non-constant `$k` " *
            "bound (internal coverage error)")
    end
    lower = haskey(kw, :lower) ? Float64(_brm_numeric_constant(kw[:lower])) : -Inf
    upper = haskey(kw, :upper) ? Float64(_brm_numeric_constant(kw[:upper])) : Inf
    dist, lower, upper
end

# Per-contrast `(location, scale)` for one categorical block: the shared
# override (treatment), the per-level vector (cell means), or the `Normal(0,1)`
# default. `coords` order is the fitted level order on both paths.
function _brm_powerscale_cat_priors(override, n::Int, col::Symbol, lp::Symbol)
    if isnothing(override)
        return fill((0.0, 1.0), n)
    elseif override isa ExprColumn
        loc, scale = _brm_powerscale_normal_args(
            override, "categorical block `$col` on `$lp`")
        return fill((loc, scale), n)
    elseif override isa AbstractVector
        length(override) == n || error(
            "brm_powerscale_inputs: cell-mean block `$col` on `$lp` has $n " *
            "coordinates but $(length(override)) prior cells — descriptor " *
            "drift; re-reflect the model that produced the draws.")
        return map(enumerate(override)) do (level, cell)
            isnothing(cell) ? (0.0, 1.0) :
                _brm_powerscale_normal_args(
                    cell, "cell-mean level $level of `$col` on `$lp`")
        end
    end
    error("brm_powerscale_inputs: categorical block `$col` on `$lp` has an " *
          "unrecognised prior value (internal coverage error)")
end

function _brm_powerscale_log_prior(brmi::BRMI, d::BRMDescriptor,
                                   draws::AbstractMatrix, names)
    S = size(draws, 1)
    acc = zeros(Float64, S)
    for (key, op_nc) in pairs(brmi.operations)
        _brm_is_prior_declaration(brmi, key) || continue
        op = _named_op(op_nc)
        lhs, rhs = getargs(op, 2)
        lhs isa NamedColumn || error(
            "brm_powerscale_inputs: prior `$key` has a non-bare LHS " *
            "(internal coverage error — the gate should have rejected it)")
        target = name(lhs)
        coords = brm_output_coordinates(d, target, names)
        length(coords) == 1 || error(
            "brm_powerscale_inputs: prior `$target` resolves to " *
            "$(length(coords)) coordinates; only scalar priors are covered.")
        dist, lower, upper = _brm_powerscale_scalar_dist(rhs, key)
        col = @view draws[:, only(coords)]
        for i in 1:S
            x = col[i]
            acc[i] += (x < lower || x > upper) ? -Inf : logpdf(dist, x)
        end
    end
    effect_overrides = _sb_effect_prior_overrides(brmi)
    for entry in _brm_population_effect_entries(brmi)
        lp = entry.logical
        labels = try
            popcoefnames(brmi, lp)
        catch err
            error("brm_powerscale_inputs: cannot name the population " *
                  "columns of `$lp`: $(sprint(showerror, err))")
        end
        isnothing(labels) && error(
            "brm_powerscale_inputs: `$lp` has no population columns " *
            "(internal coverage error)")
        col_overrides = _sb_pop_effect_overrides(effect_overrides, lp)
        if !isnothing(col_overrides) && length(col_overrides) != length(labels)
            error("brm_powerscale_inputs: `$lp` has $(length(labels)) " *
                  "population labels but $(length(col_overrides)) prior " *
                  "cells (internal coverage error)")
        end
        for (i, label) in enumerate(labels)
            cell = isnothing(col_overrides) ? nothing : col_overrides[i]
            loc, scale = isnothing(cell) ? (0.0, 1.0) :
                _brm_powerscale_normal_args(cell, "`effect($lp, $label)`")
            res = brm_population_effect_coordinates(d, lp, names;
                                                   coefficient=label)
            length(res.coordinates) == 1 || error(
                "brm_powerscale_inputs: coefficient `$label` on `$lp` " *
                "resolves to $(length(res.coordinates)) coordinates; " *
                "re-reflect the model that produced the draws.")
            acc .+= logpdf.(Normal(loc, scale),
                            @view draws[:, only(res.coordinates)])
        end
        cat_map = try
            _sb_cat_address_map(brmi, lp)
        catch err
            error("brm_powerscale_inputs: cannot map the categorical blocks " *
                  "of `$lp`: $(sprint(showerror, err))")
        end
        block_overrides = _sb_cat_effect_overrides(effect_overrides, lp)
        for (col, block) in cat_map
            res = brm_population_effect_coordinates(d, lp, names; coefficient=col)
            priors = _brm_powerscale_cat_priors(
                get(block_overrides, block, nothing),
                length(res.coordinates), col, lp)
            for (j, c) in enumerate(res.coordinates)
                loc, scale = priors[j]
                acc .+= logpdf.(Normal(loc, scale), @view draws[:, c])
            end
        end
    end
    acc
end

# ---- Part 4c: joint log-likelihood from the pointwise twins ------------------

function _brm_powerscale_log_lik(d::BRMDescriptor, draws::AbstractMatrix, names)
    S = size(draws, 1)
    twins = brm_outputs(d; role=:pointwise_loglik)
    isempty(twins) && error(
        "brm_powerscale_inputs: this model emits no pointwise log-likelihood " *
        "twins, so the joint log-likelihood cannot be assembled. Models " *
        "whose likelihood was fully held out have nothing to scale.")
    acc = zeros(Float64, S)
    seen = Set{Symbol}()
    for t in twins
        logical = isnothing(t.logical) ? t.source : t.logical
        isnothing(logical) && error(
            "brm_powerscale_inputs: pointwise twin `$(t.name)` has no " *
            "logical response; re-reflect the model.")
        logical in seen && continue
        push!(seen, logical)
        coords = try
            brm_output_coordinates(d, logical, names; role=:pointwise_loglik)
        catch err
            error("brm_powerscale_inputs: cannot resolve the pointwise " *
                  "log-likelihood twin for `$logical` in the draw columns " *
                  "— the fit must SAVE generated quantities " *
                  "(`BridgeStan.param_names(...; include_gq=true)` plus the " *
                  "matching draws). Underlying error: $(sprint(showerror, err))")
        end
        for c in coords
            acc .+= @view draws[:, c]
        end
    end
    acc
end

# Default sensitivity variables: every draw column except the predictive and
# pointwise twins (they are per-observation quantities, not parameters) and
# Stan sampler diagnostics (`lp__`, ...). Transformed parameters stay in —
# priorsense likewise scores every variable except its log-density groups.
function _brm_powerscale_default_variables(d::BRMDescriptor, names)
    twin_idx = Set{Int}()
    for role in (:posterior_predictive, :pointwise_loglik)
        for t in brm_outputs(d; role)
            logical = isnothing(t.logical) ? t.source : t.logical
            isnothing(logical) && continue
            coords = try
                brm_output_coordinates(d, logical, names; role)
            catch
                continue
            end
            union!(twin_idx, coords)
        end
    end
    map(Symbol, (string(names[i]) for i in eachindex(names)
                if !(i in twin_idx) && !endswith(string(names[i]), "__")))
end

"""
    brm_powerscale_inputs(sb::SBBRMI, constrained_draws, constrained_names)

Assemble the `(draws, log_prior, log_lik)` triple
[`brm_powerscale_sensitivity`](@ref) needs from a fitted SBBRMI. Returns
`(; log_prior, log_lik, variables, descriptor)` where `log_prior` /
`log_lik` are per-draw joint densities, `variables` the default sensitivity
selection (every parameter column — twins and `__` diagnostics excluded),
and `descriptor` the `brm_descriptor(sb)` the assembly ran through.

`constrained_draws` is draws × coordinates with `constrained_names` from the
SAME transformed-parameter / generated-quantity settings, and the fit MUST
have saved generated quantities (the pointwise log-likelihood twins live
there). Models outside the covered surface in `src/powerscale.jl` (random
effects, GPs, smooths, R2D2, non-Normal/non-Exponential priors, …) error by
name — a silently incomplete joint prior would be a wrong sensitivity, so
there is no partial credit. Pair with the refit comparison from brm-use for
those models.
"""
function brm_powerscale_inputs(sb::SBBRMI, constrained_draws::AbstractMatrix,
                               constrained_names)
    _brm_check_draw_names(constrained_draws, constrained_names)
    brmi = sb.parent
    _brm_powerscale_gate!(brmi)
    d = brm_descriptor(sb)
    log_prior = _brm_powerscale_log_prior(brmi, d, constrained_draws,
                                          constrained_names)
    log_lik = _brm_powerscale_log_lik(d, constrained_draws, constrained_names)
    variables =
        _brm_powerscale_default_variables(d, constrained_names)
    (; log_prior, log_lik, variables, descriptor=d)
end
