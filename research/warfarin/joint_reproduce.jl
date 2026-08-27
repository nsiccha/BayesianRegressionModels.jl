# Joint Warfarin model: one posterior over shared PK and PD effects.
# `reproduce.jl` remains the exact public two-stage program; this file is a
# distinct model that propagates PK uncertainty into the PD likelihood.
include(joinpath(@__DIR__, "reproduce.jl"))

"""
    warfarin_joint_brmi([data])

Fit the public Warfarin PK and PD likelihoods in one posterior. Subject-level
PK parameters are latent inputs to both the concentration likelihood and the
turnover ODE, so PK uncertainty propagates into PD and PD data can update PK.

PK and PD retain independent observation distributions. Their observations
occur at different times and have different units, so a rowwise residual
covariance would require an additional alignment and measurement model rather
than merely joining the two public likelihoods.
"""
function warfarin_joint_brmi(data = warfarin_fixture())
    return @brm data begin
        sigma_pk ~ Normal(0.0, 2.0; lower=0.0)
        kappa_pk ~ Gamma(0.2, 5.0)
        sigma_pd ~ Normal(0.0, 10.0; lower=0.0)
        kappa_pd ~ Gamma(0.2, 5.0)

        lag_logit ~ 1 + (1 | tlag_bsv | subject)
        log_ka ~ 1 + (1 | ka_bsv | subject)
        # Allometric weight scaling as a formula OFFSET (fixed exponents 0.75 / 1.0),
        # shared by both the PK and PD cells below — not hand-wired in either cell.
        log_cl0 ~ 1 + offset(0.75 * log_weight_ratio) + (1 | cl_bsv | subject)
        log_v0 ~ 1 + offset(log_weight_ratio) + (1 | v_bsv | subject)
        log_r0 ~ 1 + (1 | r0_bsv | subject)
        log_inv_kout ~ 1 + (1 | kout_bsv | subject)
        log_ec50 ~ 1 + (1 | ec50_bsv | subject)

        effect(lag_logit, :) ~ Normal(0.0, 2.0)
        effect(log_ka, :) ~ Normal(log(1.0), log(2.0) / 1.96)
        effect(log_cl0, :) ~ Normal(log(0.1), log(10.0) / 1.96)
        effect(log_v0, :) ~ Normal(log(10.0), log(10.0) / 1.96)
        effect(log_r0, :) ~ Normal(log(80.0), log(10.0) / 1.96)
        effect(log_inv_kout, :) ~ Normal(log(30.0), log(10.0) / 1.96)
        effect(log_ec50, :) ~ Normal(log(2.5), log(10.0) / 1.96)
        sd(:, tlag_bsv) ~ Normal(0.0, 0.5)
        sd(:, ka_bsv) ~ Normal(0.0, 0.5)
        sd(:, cl_bsv) ~ Normal(0.0, 0.5)
        sd(:, v_bsv) ~ Normal(0.0, 0.5)
        sd(:, r0_bsv) ~ Normal(0.0, 0.5)
        sd(:, kout_bsv) ~ Normal(0.0, 0.5)
        sd(:, ec50_bsv) ~ Normal(0.0, 0.5)

        pk_pred ~ kernel(
            pk_time, dose, pk_dv,
            lag_logit, log_ka, log_cl0, log_v0,
        ) do times, dose_i, observed,
             lag_i, lka_i, lcl_i, lv_i
            log_tlag_i = log_inv_logit(lag_i)
            prediction = exp(warfarin_pk_logconcentration(
                times, log(dose_i), lka_i, lcl_i, lv_i, log_tlag_i,
            )) + 1e-5
            pk_pointwise_loglik = warfarin_gamma2_overdisp_lpdfs(
                observed, prediction, sigma_pk, kappa_pk * 25.0,
            )
            observed ~ warfarin_gamma2_overdisp(
                prediction, sigma_pk, kappa_pk * 25.0,
            )
            prediction
        end

        pd_pred ~ kernel(
            pd_time, dose, pd_dv,
            lag_logit, log_ka, log_cl0, log_v0,
            log_r0, log_inv_kout, log_ec50,
        ) do times, dose_i, observed,
             lag_i, lka_i, lcl_i, lv_i,
             lr0_i, linvkout_i, lec50_i
            log_tlag_i = log_inv_logit(lag_i)
            prediction = warfarin_turnover_prediction(
                times, log(dose_i), lka_i, lcl_i, lv_i,
                log_tlag_i, lr0_i, linvkout_i, lec50_i,
            )
            pd_pointwise_loglik = warfarin_gamma2_overdisp_lpdfs(
                observed, prediction, sigma_pd, kappa_pd * 625.0,
            )
            observed ~ warfarin_gamma2_overdisp(
                prediction, sigma_pd, kappa_pd * 625.0,
            )
            prediction
        end
    end
end

warfarin_joint_sbbrmi(data = warfarin_fixture()) =
    SBBRMI(warfarin_joint_brmi(data); mod=@__MODULE__)

if abspath(PROGRAM_FILE) == @__FILE__
    @testset "joint Warfarin PK/PD model" begin
        sb = warfarin_joint_sbbrmi()
        code = StanBlocks.stan_code(sb.model)
        @test StanBlocks.stan.transpiles(sb.model)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test occursin("warfarin_gamma2_overdisp_lpdf", code)
        @test occursin("ode_rk45_tol", code)
        @test occursin(r"pk_pred_log_tlag_i.*lag_logit", code)
        @test occursin(r"pd_pred_log_tlag_i.*lag_logit", code)
        @test occursin("pk_pred_pk_pointwise_loglik", code)
        @test occursin("pd_pred_pd_pointwise_loglik", code)
        @test length(collect(eachmatch(
            r"vector\[num_elements\(subject_idx\)\] lag_logit =", code,
        ))) == 1
        # The two-stage fixture carries these fixed PK summaries for the exact
        # sequential model. A genuinely joint model must not consume them.
        @test !occursin("pk_log_tlag", code)
        @test !occursin("pk_log_ka", code)
        @test !occursin("pk_log_cl", code)
        @test !occursin("pk_log_v", code)

        if get(ENV, "BRM_WARFARIN_JOINT_RUNTIME", "1") != "0"
            problem = warfarin_problem(sb, :joint)
            q = zeros(LogDensityProblems.dimension(problem))
            lp, grad = LogDensityProblems.logdensity_and_gradient(problem, q)
            @test isfinite(lp)
            @test all(isfinite, grad)

            descriptor = brm_descriptor(sb; name=:warfarin_joint)
            operations = Symbol[operation.name for operation in descriptor.operations]
            @test all(in(operations), (:fit, :predict, :pointwise_loglik))
            predictive = brm_execute(
                descriptor, :predict; problem, draws=q, seed=20260821,
            )
            @test length(predictive.pk_dv_gen) == sum(length, warfarin_fixture().pk_dv)
            @test length(predictive.pd_dv_gen) == sum(length, warfarin_fixture().pd_dv)
            aggregate_loglik = brm_execute(
                descriptor, :pointwise_loglik;
                problem, draws=q, seed=20260821,
            )
            @test length(aggregate_loglik.pk_dv_likelihood) ==
                  length(warfarin_fixture().pk_dv)
            @test length(aggregate_loglik.pd_dv_likelihood) ==
                  length(warfarin_fixture().pd_dv)

            names = StanBlocks.BridgeStan.param_names(
                problem.model; include_tp=true, include_gq=true,
            )
            @test length(brm_output_coordinates(
                descriptor, :pk_pointwise_loglik, names,
            )) == sum(length, warfarin_fixture().pk_dv)
            @test length(brm_output_coordinates(
                descriptor, :pd_pointwise_loglik, names,
            )) == sum(length, warfarin_fixture().pd_dv)
        end
    end
end
