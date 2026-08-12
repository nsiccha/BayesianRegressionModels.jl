#!/usr/bin/env julia
# test/probe_pkcore_skeleton.jl — varyingsource2_pkcore BRM skeleton transpile probe
#
# Verifies that the full PK-core @brm skeleton (6 covariate-laden predictors +
# stubbed twocmt_superposition term + addprop obs family) transpiles green.
# The stub matches the real term's signature (5 data + 6 kwargs) so swap-in
# requires only replacing the _sb_submodel_rhs! emitter body, not the @brm formula.
#
# Covariate sets (from correlated_std_normal_health_covariates_no_diet, pkpd_models.jl:2515):
#   log_Vc, log_k10: 1 + male + zage + zweight + diseased + (1|p|subject)
#   log_k12, log_k21: 1 + diseased + (1|p|subject)
#   log_abs_rate, log_abs_mode: 1 + (1|p|subject)
#
# obs_scale prior: Exponential(1) proxy (deployed uses matrix_exponential(obs_scale_rate,...);
# faithful-port follow-up, not this probe's concern).

using BayesianRegressionModels, StanBlocks
using Distributions: Exponential, Normal

import BayesianRegressionModels: _sb_submodel_rhs!, getkwargs, name

# ── Stub term ─────────────────────────────────────────────────────────────────
# Same function name as the real term -> clean swap when Bruno:qt lands.

function twocmt_superposition end

# Emitter: reads all 6 kwargs (proves all 6 LP names resolve at seam-time).
# Direct assignment `amount_pk = log_Vc` avoids a slic sub-model -- SLIC would
# inline log_Vc's ranef computation into the sub-model scope, causing a name
# collision with the already-emitted parent assignments. Direct = bypasses that.
function _sb_submodel_rhs!(stmts, data, target::Symbol,
                            ::typeof(twocmt_superposition), rhs)
    kw = getkwargs(rhs)
    log_Vc_sym       = name(kw[:log_Vc])
    log_k10_sym      = name(kw[:log_k10])
    log_k12_sym      = name(kw[:log_k12])
    log_k21_sym      = name(kw[:log_k21])
    log_abs_rate_sym = name(kw[:log_abs_rate])
    log_abs_mode_sym = name(kw[:log_abs_mode])
    println("  [stub emitter] kwargs resolved:")
    println("    log_Vc       -> :$log_Vc_sym")
    println("    log_k10      -> :$log_k10_sym")
    println("    log_k12      -> :$log_k12_sym")
    println("    log_k21      -> :$log_k21_sym")
    println("    log_abs_rate -> :$log_abs_rate_sym")
    println("    log_abs_mode -> :$log_abs_mode_sym")
    # Proxy: assign amount_pk = log_Vc (correct shape: per-obs vector).
    # Real swap-in replaces only this statement with the ODE-kernel call.
    push!(stmts, :($target = $log_Vc_sym))
    return :done
end

# ── Mock data ─────────────────────────────────────────────────────────────────
# Ragged dose/dtimes match real term shapes (zero edits at swap-in).
# 4 observations, 2 subjects; subject-level covariates repeated per obs.
# male/diseased are Float64 (not Int) to match the deployed model's numeric
# matrix-multiply treatment of these binary columns. Integer input would instead
# select BRM's predictor-scoped categorical treatment-contrast path.
df = (;
    subject  = [1, 1, 2, 2],
    male     = [1.0, 1.0, 0.0, 0.0],
    zage     = [0.5, 0.5, -0.3, -0.3],
    zweight  = [1.0, 1.0, -0.5, -0.5],
    diseased = [0.0, 0.0, 1.0, 1.0],
    subj     = [1, 1, 2, 2],
    t        = [1.0, 2.0, 1.0, 2.0],
    dtimes   = [[1.0], [1.0]],     # ragged: 2 subjects x 1 dose time
    dsubj    = [1, 2],
    dose     = [[100.0], [100.0]], # ragged: 2 subjects x 1 dose
    pk_conc  = [0.1, 0.05, 0.08, 0.04],
)

# ── Build skeleton ────────────────────────────────────────────────────────────
println("=== Building varyingsource2_pkcore BRMI ===")
varyingsource2_pkcore = @brm df begin
    log_Vc       ~ 1 + male + zage + zweight + diseased + (1|p|subject)
    log_k10      ~ 1 + male + zage + zweight + diseased + (1|p|subject)
    log_k12      ~ 1 + diseased + (1|p|subject)
    log_k21      ~ 1 + diseased + (1|p|subject)
    log_abs_rate ~ 1 + (1|p|subject)
    log_abs_mode ~ 1 + (1|p|subject)
    sigma_add    ~ Exponential(1)
    sigma_prop   ~ Exponential(1)
    amount_pk    ~ twocmt_superposition(subj, t, dtimes, dsubj, dose;
                       log_Vc=log_Vc, log_k10=log_k10, log_k12=log_k12,
                       log_k21=log_k21, log_abs_rate=log_abs_rate, log_abs_mode=log_abs_mode)
    pk_conc      ~ Normal(amount_pk, addprop(amount_pk, sigma_add, sigma_prop))
end

println("=== Building SBBRMI ===")
sbbrmi = SBBRMI(varyingsource2_pkcore)

println("=== Transpile check ===")
transpile_ok = StanBlocks.stan.transpiles(sbbrmi.model)
@assert transpile_ok "SKELETON FAIL: model does not transpile"
println("Transpile: PASSED")

code = BayesianRegressionModels.stan_code(sbbrmi)

println("\n=== Generated Stan code ===")
println(code)

# ── Checks ────────────────────────────────────────────────────────────────────
println("\n=== Check: all 6 predictors present ===")
for sym in ("log_Vc", "log_k10", "log_k12", "log_k21", "log_abs_rate", "log_abs_mode")
    @assert occursin(sym, code) "FAIL: $sym absent from Stan code"
    println("  $sym: present ✓")
end

println("\n=== Check: predictor composition lines (fixed+ranef from shared |p| block) ===")
lines = split(code, '\n')
for sym in ("log_Vc", "log_k10", "log_k12", "log_k21", "log_abs_rate", "log_abs_mode")
    matching = filter(l -> occursin(sym, l) && occursin("=", l), lines)
    println("\n  Stan assignment lines for '$sym':")
    for l in matching
        println("    ", strip(l))
    end
end

println("\n=== Check: addprop obs line ===")
addprop_lines = filter(l -> occursin("addprop", l) || (occursin("pk_conc", l) && occursin("normal", l)), lines)
println("  Stan lines for addprop/pk_conc:")
for l in addprop_lines
    println("    ", strip(l))
end
@assert any(l -> occursin("addprop", l), lines) "FAIL: addprop absent from Stan code"
println("  addprop: present ✓")

println("\n=== Check: amount_pk assignment (stub proxy) ===")
amount_pk_lines = filter(l -> occursin("amount_pk", l) && occursin("=", l), lines)
println("  Stan lines for amount_pk:")
for l in amount_pk_lines
    println("    ", strip(l))
end

println("\n=== SKELETON TRANSPILE COMPLETE ===")
