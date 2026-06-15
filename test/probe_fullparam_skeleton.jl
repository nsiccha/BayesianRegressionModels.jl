#!/usr/bin/env julia
# test/probe_fullparam_skeleton.jl — varyingsource2 full 13-param correlated block probe
#
# Extends probe_pkcore_skeleton.jl (#6, 1f97ce9) from 6 PK predictors to the full
# 13-param roster verified from source (pkpd_models.jl):
#   n_unit_reparams3(twocmt)    = 4  (PK: Vc, k10, k12, k21)
#   n_unit_params(pd23_approx)  = 7  (PD: baseline_pbmc, kout, theta1_pbmc,
#                                       theta2_pbmc, baseline_csf, theta1_csf, theta2_csf)
#   n_unit_params(transit_source) = 2 (Source: abs_rate, abs_mode)
#   TOTAL = 13
#
# Covariate structure (correlated_std_normal_health_covariates_no_diet, lines 2515-2543):
#   params 1-2  (log_Vc, log_k10):       1 + male + zage + zweight + diseased + (1|p|subject)
#   params 3-11 (log_k12..log_theta2_csf): 1 + diseased + (1|p|subject)
#   params 12-13 (log_abs_rate, log_abs_mode): 1 + (1|p|subject)
#
# Stub term kept from #6: reads 6 PK kwargs, emits amount_pk = log_Vc.
# PD predictors (params 5-11) declared but not fed to stub — probe verifies
# whether all 13 stay in the Stan `parameters` block (LKJ coupling) or any
# drop to generated_quantities (StanBlocks §9 unconsumed-param rule).
#
# Faithfulness notes (structural deltas, NOT errors):
# (F1) BRM emits independent per-predictor fixed-effect coefs; deployed uses shared
#      covariate_effects_scale×2×3 matrix. Prior-shape approximation, transpile-irrelevant.
# (F2) BRM emits standard lkj_corr_cholesky; deployed uses stable_lkj_corr_cholesky_model.
#      Transpile-identical; sampling-time stability deferred (#23).

using BayesianRegressionModels, StanBlocks
using Distributions: Exponential, Normal

import BayesianRegressionModels: _sb_submodel_rhs!, getkwargs, name

# ── Stub term (same as #6) ────────────────────────────────────────────────────
function twocmt_superposition end

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
    push!(stmts, :($target = $log_Vc_sym))
    return :done
end

# ── Mock data (same as #6) ────────────────────────────────────────────────────
# Binary covariates as Float64 (decision 1x6dtzq: avoids _sb_emit_cat! collision).
# 4 observations, 2 subjects; all 13 predictors share these columns.
df = (;
    subject  = [1, 1, 2, 2],
    male     = [1.0, 1.0, 0.0, 0.0],
    zage     = [0.5, 0.5, -0.3, -0.3],
    zweight  = [1.0, 1.0, -0.5, -0.5],
    diseased = [0.0, 0.0, 1.0, 1.0],
    subj     = [1, 1, 2, 2],
    t        = [1.0, 2.0, 1.0, 2.0],
    dtimes   = [[1.0], [1.0]],
    dsubj    = [1, 2],
    dose     = [[100.0], [100.0]],
    pk_conc  = [0.1, 0.05, 0.08, 0.04],
)

# ── Build BRMI ────────────────────────────────────────────────────────────────
println("=== Building varyingsource2_fullparam BRMI (13-param block) ===")
varyingsource2_fullparam = @brm df begin
    # PK params (1-2): full covariates (male, zage, zweight, diseased)
    log_Vc          ~ 1 + male + zage + zweight + diseased + (1|p|subject)
    log_k10         ~ 1 + male + zage + zweight + diseased + (1|p|subject)
    # PK params (3-4): disease only
    log_k12         ~ 1 + diseased + (1|p|subject)
    log_k21         ~ 1 + diseased + (1|p|subject)
    # PD params (5-11): disease only
    log_baseline_pbmc ~ 1 + diseased + (1|p|subject)
    log_kout          ~ 1 + diseased + (1|p|subject)
    log_theta1_pbmc   ~ 1 + diseased + (1|p|subject)
    log_theta2_pbmc   ~ 1 + diseased + (1|p|subject)
    log_baseline_csf  ~ 1 + diseased + (1|p|subject)
    log_theta1_csf    ~ 1 + diseased + (1|p|subject)
    log_theta2_csf    ~ 1 + diseased + (1|p|subject)
    # Source params (12-13): intercept + ranef only
    log_abs_rate    ~ 1 + (1|p|subject)
    log_abs_mode    ~ 1 + (1|p|subject)
    # Obs-error params
    sigma_add       ~ Exponential(1)
    sigma_prop      ~ Exponential(1)
    # Stub term (PD predictors declared above but not consumed here)
    amount_pk       ~ twocmt_superposition(subj, t, dtimes, dsubj, dose;
                          log_Vc=log_Vc, log_k10=log_k10, log_k12=log_k12,
                          log_k21=log_k21, log_abs_rate=log_abs_rate, log_abs_mode=log_abs_mode)
    # Obs likelihood
    pk_conc         ~ Normal(amount_pk, addprop(amount_pk, sigma_add, sigma_prop))
end

println("=== Building SBBRMI ===")
sbbrmi = SBBRMI(varyingsource2_fullparam)

println("=== Transpile check ===")
transpile_ok = StanBlocks.stan.transpiles(sbbrmi.model)
@assert transpile_ok "FULL-PARAM FAIL: model does not transpile"
println("Transpile: PASSED")

code = BayesianRegressionModels.stan_code(sbbrmi)

println("\n=== Generated Stan code ===")
println(code)

lines = split(code, '\n')

# ── Check 1: all 13 predictors present ────────────────────────────────────────
println("\n=== Check 1: all 13 predictor names in Stan code ===")
all_params = [
    "log_Vc", "log_k10", "log_k12", "log_k21",
    "log_baseline_pbmc", "log_kout", "log_theta1_pbmc", "log_theta2_pbmc",
    "log_baseline_csf", "log_theta1_csf", "log_theta2_csf",
    "log_abs_rate", "log_abs_mode",
]
for sym in all_params
    @assert occursin(sym, code) "FAIL: $sym absent from Stan code"
    println("  $sym: present ✓")
end

# ── Check 2: PD-param disposition (parameters vs generated_quantities) ────────
println("\n=== Check 2: PD-param disposition (parameters block vs generated_quantities) ===")
params_block_start = findfirst(l -> occursin("parameters {", l), lines)
params_block_end   = params_block_start === nothing ? nothing :
    findnext(l -> strip(l) == "}" && !occursin("//", l), lines, params_block_start+1)
gq_block_start = findfirst(l -> occursin("generated quantities {", l), lines)

println("  parameters block: lines $(params_block_start)–$(params_block_end)")
println("  generated_quantities block starts: line $(gq_block_start)")

pd_params = ["log_baseline_pbmc", "log_kout", "log_theta1_pbmc", "log_theta2_pbmc",
             "log_baseline_csf", "log_theta1_csf", "log_theta2_csf"]
println("\n  PD-param block disposition:")
for sym in pd_params
    in_params = false
    in_gq     = false
    for (i, l) in enumerate(lines)
        if occursin(sym, l)
            if params_block_start !== nothing && params_block_end !== nothing &&
               params_block_start <= i <= params_block_end
                in_params = true
            elseif gq_block_start !== nothing && i >= gq_block_start
                in_gq = true
            end
        end
    end
    status = in_params ? "PARAMETERS ✓" : (in_gq ? "GENERATED_QUANTITIES (escalate!)" : "not found in either block")
    println("    $sym: $status")
end

# ── Check 3: b_p_subject block width ──────────────────────────────────────────
println("\n=== Check 3: b_p_subject / shared ranef block width ===")
block_lines = filter(l -> occursin("b_p_subject", l), lines)
println("  Stan lines mentioning b_p_subject:")
for l in block_lines
    println("    ", strip(l))
end
has_13_col = any(l -> occursin("13", l) && occursin("b_p_subject", l), lines)
println("  13-column signature detected: $(has_13_col ? "YES ✓" : "NOT EXPLICITLY (check declaration above)")")

# ── Check 4: differential covariate structure in fixed-effect declarations ────
println("\n=== Check 4: differential covariate structure (fixed-effect coefs) ===")
println("  Params 1-2 (log_Vc, log_k10) should carry male/zage/zweight/diseased coefs:")
for sym in ("log_Vc", "log_k10")
    coef_lines = filter(l -> occursin("pop_$sym", l) || (occursin(sym, l) && occursin("beta", l)), lines)
    if isempty(coef_lines)
        coef_lines = filter(l -> occursin(sym, l) && occursin("=", l), lines)
    end
    println("\n  Stan assignment lines for '$sym':")
    for l in coef_lines[1:min(5, end)]
        println("    ", strip(l))
    end
end

println("\n  Params 3-11 (disease only) — sample: log_k12, log_baseline_pbmc:")
for sym in ("log_k12", "log_baseline_pbmc")
    coef_lines = filter(l -> occursin(sym, l) && occursin("=", l), lines)
    println("\n  Stan assignment lines for '$sym':")
    for l in coef_lines[1:min(5, end)]
        println("    ", strip(l))
    end
end

println("\n  Params 12-13 (intercept-only) — log_abs_rate, log_abs_mode:")
for sym in ("log_abs_rate", "log_abs_mode")
    coef_lines = filter(l -> occursin(sym, l) && occursin("=", l), lines)
    println("\n  Stan assignment lines for '$sym':")
    for l in coef_lines[1:min(5, end)]
        println("    ", strip(l))
    end
end

# ── Check 5: LKJ prior ────────────────────────────────────────────────────────
println("\n=== Check 5: LKJ correlation prior ===")
lkj_lines = filter(l -> occursin("lkj", lowercase(l)), lines)
println("  Stan LKJ lines:")
for l in lkj_lines
    println("    ", strip(l))
end

# ── Check 6: addprop obs line ──────────────────────────────────────────────────
println("\n=== Check 6: addprop obs line ===")
obs_lines = filter(l -> occursin("addprop", l) || (occursin("pk_conc", l) && occursin("normal", l)), lines)
for l in obs_lines
    println("    ", strip(l))
end
@assert any(l -> occursin("addprop", l), lines) "FAIL: addprop absent from Stan code"
println("  addprop: present ✓")

println("\n=== FULL-PARAM SKELETON TRANSPILE COMPLETE ===")
