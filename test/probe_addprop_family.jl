#!/usr/bin/env julia
# test/probe_addprop_family.jl — add+prop obs-family seam probe
#
# GOAL: confirm (with Stan evidence) whether the NATURAL add+prop form:
#   sigma = sqrt(add^2 + (mu_pk * prop)^2)
#   y ~ Normal(mu_pk, sigma)
# already transpiles, and if not, identify the exact owning-layer gap(s).
#
# THREE FORMULATIONS (A = intended end-state; B/C = diagnostics):
#
#   A (NATURAL, intended end-state):
#     sigma = sqrt(add^2 + (mu_pk * prop)^2)
#     y ~ Normal(mu_pk, sigma)
#
#   B (diagnostic — localize ^ gap by replacing ^ with *):
#     add_sq   = add * add
#     prop_eta = mu_pk * prop
#     sigma    = sqrt(add_sq + prop_eta * prop_eta)
#     y ~ Normal(mu_pk, sigma)
#
#   C (diagnostic — sigma inline in Normal arg, using * only):
#     y ~ Normal(mu_pk, sqrt(add*add + mu_pk*prop*mu_pk*prop))
#
# SCOPE: mechanism only (computed-sigma Normal + bare scalar params).
# Does NOT port addpropnormal_lpdfs (LLOQ/cens/multi-assay — separate concern).
# matrix_exponential prior: noted as follow-up; Exponential(1) used here.

using BayesianRegressionModels, StanBlocks
using Distributions: Exponential, Normal

# ── Mock data ──────────────────────────────────────────────────────────────
# mu_pk already computed (simulating amount_pk .* exp(-log_Vc) upstream)
df = (;
    mu_pk = [1.0, 2.0, 1.5, 3.0],
    y     = [1.1, 2.1, 1.6, 2.9],
)

# ── Helper: try to build SBBRMI and transpile, catching errors ────────────
function try_probe(label, brmi)
    println("\n" * "="^60)
    println("=== Formulation $label ===")
    println("="^60)
    local sbbrmi, code
    try
        sbbrmi = SBBRMI(brmi)
    catch e
        println("  SBBRMI BUILD ERROR: $e")
        return nothing
    end
    try
        transpiles = StanBlocks.stan.transpiles(sbbrmi.model)
        if transpiles
            code = BayesianRegressionModels.stan_code(sbbrmi)
            println("  Transpile: PASSED ✓")
            println("\n  --- Generated Stan (parameters block) ---")
            for l in split(code, '\n')
                if occursin("parameters", l) || occursin("transformed parameters", l) || occursin("model", l)
                    println("  ", strip(l))
                end
            end
            println("\n  --- Stan lines mentioning add, prop, sigma ---")
            for l in split(code, '\n')
                s = strip(l)
                if any(t -> occursin(t, s), ["add", "prop", "sigma", "sqrt"])
                    println("  ", s)
                end
            end
            return code
        else
            println("  Transpile: FAILED (transpiles returned false, no error text)")
            return nothing
        end
    catch e
        println("  Transpile ERROR: $(sprint(showerror, e))")
        # Try to get Stan code even if it doesn't transpile cleanly
        try
            code = BayesianRegressionModels.stan_code(sbbrmi)
            println("\n  --- Generated Stan (for diagnosis) ---")
            println(code)
        catch e2
            println("  (stan_code also failed: $e2)")
        end
        return nothing
    end
end

# ── Formulation A: NATURAL form — intended end-state ──────────────────────
# add^2 uses ^ operator; (mu_pk*prop)^2 applies ^ to a vector expression.
# _sb_scalar_expr translates * → .* but NOT ^ → .^.
# If ^ on vectors errors in Stan, this is the gap we need to fix.
brmi_A = @brm df begin
    add  ~ Exponential(1)
    prop ~ Exponential(1)
    sigma = sqrt(add^2 + (mu_pk * prop)^2)
    y ~ Normal(mu_pk, sigma)
end

code_A = try_probe("A (NATURAL: sqrt(add^2 + (mu_pk*prop)^2))", brmi_A)

# ── Formulation B: diagnostic — replace ^ with * ──────────────────────────
# Avoids ^ operator entirely. If this passes while A fails, the gap is ^ only.
# (NOT the intended end-state — diagnostic to localize the ^ issue)
brmi_B = @brm df begin
    add  ~ Exponential(1)
    prop ~ Exponential(1)
    add_sq   = add * add
    prop_eta = mu_pk * prop
    sigma    = sqrt(add_sq + prop_eta * prop_eta)
    y ~ Normal(mu_pk, sigma)
end

code_B = try_probe("B (diagnostic: ^ replaced with *)", brmi_B)

# ── Formulation C: diagnostic — sigma inline in Normal arg ────────────────
# Tests whether sigma-as-computed-expression works directly as Normal arg.
# Also uses * only (avoids ^). If B passes but C fails, the sigma-assignment
# intermediate is necessary. If C also passes, inline is fine.
brmi_C = @brm df begin
    add  ~ Exponential(1)
    prop ~ Exponential(1)
    y ~ Normal(mu_pk, sqrt(add*add + mu_pk*prop*mu_pk*prop))
end

code_C = try_probe("C (diagnostic: sigma inline in Normal)", brmi_C)

# ── Summary ─────────────────────────────────────────────────────────────────
println("\n" * "="^60)
println("=== SUMMARY ===")
println("="^60)
println("  A (natural ^ form):           $(isnothing(code_A) ? "FAIL" : "PASS")")
println("  B (diagnostic * form):        $(isnothing(code_B) ? "FAIL" : "PASS")")
println("  C (diagnostic inline sigma):  $(isnothing(code_C) ? "FAIL" : "PASS")")
println()
if isnothing(code_A) && !isnothing(code_B)
    println("DIAGNOSIS: ^ on vectors is a real gap in BRM._sb_scalar_expr.")
    println("  Natural form (A) fails; diagnostic (B, using * only) passes.")
    println("  Fix: translate ^ -> .^ in _sb_scalar_expr (sbimpl.jl:2229).")
elseif !isnothing(code_A)
    println("Natural form A PASSES — no BRM-core gap for add+prop mechanism.")
else
    println("Both A and B FAIL — deeper issue (possibly sqrt function-obj).")
end
