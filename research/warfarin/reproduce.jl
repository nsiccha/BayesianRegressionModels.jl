using BayesianRegressionModels
using Distributions: Gamma, Normal
using LogDensityProblems
using StanBlocks
using Test

# Exact helper functions from Weber's public StanCon 2018 Warfarin programs,
# translated from legacy Stan signatures to StanBlocks' current variadic surface.
StanBlocks.@deffun begin
    @stanonly begin
        warfarin_log_diff_exp_abs(log_a::real, log_b::real)::real = begin
            return 0.5 * log_diff_exp(
                log_sum_exp(2.0 * log_a, 2.0 * log_b),
                log(2.0) + log_a + log_b,
            )
        end

        warfarin_pk_logconcentration(
            tad::vector[n], log_dose::real, log_ka::real, log_cl::real,
            log_v::real, log_tlag::real,
        )::vector[n] = begin
            log_ke = log_cl - log_v
            log_delta = warfarin_log_diff_exp_abs(log_ka, log_ke)
            log_scale = log_dose - log_v + log_ka - log_delta
            value::vector[n]
            for i in 1:n
                if tad[i] <= exp(log_tlag)
                    value[i] = negative_infinity()
                else
                    log_tad = log(tad[i] - exp(log_tlag))
                    a = -exp(log_ke + log_tad)
                    b = -exp(log_ka + log_tad)
                    value[i] = log_scale + warfarin_log_diff_exp_abs(a, b)
                end
            end
            return value
        end

        warfarin_pk_logconcentration_one(
            time::real, log_dose::real, log_ka::real, log_cl::real,
            log_v::real, log_tlag::real,
        )::real = begin
            if time < exp(log_tlag)
                return -25.0
            end
            log_ke = log_cl - log_v
            log_delta = warfarin_log_diff_exp_abs(log_ka, log_ke)
            log_scale = log_dose - log_v + log_ka - log_delta
            log_tad = log(time - exp(log_tlag))
            a = -exp(log_ke + log_tad)
            b = -exp(log_ka + log_tad)
            return log_scale + warfarin_log_diff_exp_abs(a, b)
        end

        warfarin_turnover_rhs(
            time::real, state::vector[m], log_dose::real, log_ka::real,
            log_cl::real, log_v::real, log_tlag::real, log_r0::real,
            log_inv_kout::real, log_ec50::real,
        )::vector[m] = begin
            log_conc = warfarin_pk_logconcentration_one(
                time, log_dose, log_ka, log_cl, log_v, log_tlag,
            )
            log_kout = -log_inv_kout
            log_kin = log_r0 + log_kout
            log_inhibition = log_inv_logit(log_conc - log_ec50)
            derivative::vector[m]
            derivative[1] = exp(log_kin + log1m_exp(log_inhibition)) -
                            state[1] * exp(log_kout)
            return derivative
        end

        # Keep the ODE's `array[] vector` trajectory inside a typed function.
        # A kernel plate cell can then return the ragged vector it owns without
        # asking the plate hoister to materialize a two-dimensional cell-local.
        warfarin_turnover_prediction(
            times::vector[n], log_dose::real, log_ka::real, log_cl::real,
            log_v::real, log_tlag::real, log_r0::real,
            log_inv_kout::real, log_ec50::real,
        )::vector[n] = begin
            initial_state = rep_vector(exp(log_r0), 1)
            trajectory = ode_rk45_tol(
                warfarin_turnover_rhs, initial_state, -1e-4,
                to_array_1d(times), 1e-5, 1e-3, 500,
                log_dose, log_ka, log_cl, log_v, log_tlag,
                log_r0, log_inv_kout, log_ec50,
            )
            return to_vector(trajectory[:, 1])
        end

        @lhs @lpxf warfarin_gamma2_overdisp_lpdf(
            y::vector[n], mu::vector[n], sigma::real, kappa::real,
        )::real = begin
            lp = 0.0
            for i in 1:n
                variance = square(sigma) + square(mu[i]) / kappa
                shape = square(mu[i]) / variance
                rate = mu[i] / variance
                lp += gamma_lpdf(y[i], shape, rate)
            end
            return lp
        end

        warfarin_gamma2_overdisp_lpdfs(
            y::vector[n], mu::vector[n], sigma::real, kappa::real,
        )::vector[n] = begin
            lp::vector[n]
            for i in 1:n
                variance = square(sigma) + square(mu[i]) / kappa
                shape = square(mu[i]) / variance
                rate = mu[i] / variance
                lp[i] = gamma_lpdf(y[i], shape, rate)
            end
            return lp
        end

        warfarin_gamma2_overdisp_rng(
            mu::real, sigma::real, kappa::real,
        )::real = begin
            variance = square(sigma) + square(mu) / kappa
            shape = square(mu) / variance
            rate = mu / variance
            return gamma_rng(shape, rate)
        end

        warfarin_gamma2_overdisp_rng(
            vector[n], mu::vector[n], sigma::real, kappa::real,
        )::vector[n] = begin
            draw::vector[n]
            for i in 1:n
                draw[i] = warfarin_gamma2_overdisp_rng(mu[i], sigma, kappa)
            end
            return draw
        end
    end
end

# The public 32-subject data, and the PD stage's data with the first stage's
# per-subject PK medians (written by fit.jl).
include(joinpath(@__DIR__, "data.jl"))

"""Exact first-stage population PK model from the public Weber program."""
function warfarin_pk_brmi(data = warfarin_data())
    return @brm data begin
        sigma_pk ~ Normal(0.0, 2.0; lower=0.0)
        kappa_pk ~ Gamma(0.2, 5.0)

        lag_logit ~ 1 + (1 | tlag_bsv | subject)
        log_ka ~ 1 + (1 | ka_bsv | subject)
        # Allometric weight scaling as a formula OFFSET (fixed exponents 0.75 / 1.0),
        # not hand-wired in the kernel cell: clearance ~ weight^0.75, volume ~ weight^1.
        log_cl0 ~ 1 + offset(0.75 * log_weight_ratio) + (1 | cl_bsv | subject)
        log_v0 ~ 1 + offset(log_weight_ratio) + (1 | v_bsv | subject)

        effect(lag_logit, :) ~ Normal(0.0, 2.0)
        effect(log_ka, :) ~ Normal(log(1.0), log(2.0) / 1.96)
        effect(log_cl0, :) ~ Normal(log(0.1), log(10.0) / 1.96)
        effect(log_v0, :) ~ Normal(log(10.0), log(10.0) / 1.96)
        sd(:, tlag_bsv) ~ Normal(0.0, 0.5)
        sd(:, ka_bsv) ~ Normal(0.0, 0.5)
        sd(:, cl_bsv) ~ Normal(0.0, 0.5)
        sd(:, v_bsv) ~ Normal(0.0, 0.5)

        # One cell per subject: its sampling times, dose and observed
        # concentrations, and its own values of the four linear predictors.
        # log_cl0 / log_v0 already carry the allometric offset (formula term above).
        @plate for i in eachindex(lag_logit)
            log_tlag = -log1pexp(-lag_logit[i])    # log(tlag), tlag = logistic(lag_logit) < tlagMax = 1
            pk_pred[i] = exp.(warfarin_pk_logconcentration(
                pk_time[i], log(dose[i]), log_ka[i], log_cl0[i], log_v0[i], log_tlag,
            )) .+ 1e-5
            pk_dv[i] ~ warfarin_gamma2_overdisp(pk_pred[i], sigma_pk, kappa_pk * 25.0)
        end
    end
end

"""Exact second-stage PD model, conditioned on the public PK posterior medians."""
function warfarin_pd_brmi(data = warfarin_pd_data())
    return @brm data begin
        sigma_pd ~ Normal(0.0, 10.0; lower=0.0)
        kappa_pd ~ Gamma(0.2, 5.0)

        log_r0 ~ 1 + (1 | r0_bsv | subject)
        log_inv_kout ~ 1 + (1 | kout_bsv | subject)
        log_ec50 ~ 1 + (1 | ec50_bsv | subject)

        effect(log_r0, :) ~ Normal(log(80.0), log(10.0) / 1.96)
        effect(log_inv_kout, :) ~ Normal(log(30.0), log(10.0) / 1.96)
        effect(log_ec50, :) ~ Normal(log(2.5), log(10.0) / 1.96)
        sd(:, r0_bsv) ~ Normal(0.0, 0.5)
        sd(:, kout_bsv) ~ Normal(0.0, 0.5)
        sd(:, ec50_bsv) ~ Normal(0.0, 0.5)

        # One cell per subject. pk_log_* are that subject's fixed PK posterior
        # medians from the first stage: data columns, one value per subject.
        @plate for i in eachindex(log_r0)
            pd_pred[i] = warfarin_turnover_prediction(
                pd_time[i], log(dose[i]), pk_log_ka[i], pk_log_cl[i], pk_log_v[i],
                pk_log_tlag[i], log_r0[i], log_inv_kout[i], log_ec50[i],
            )
            pd_dv[i] ~ warfarin_gamma2_overdisp(pd_pred[i], sigma_pd, kappa_pd * 625.0)
        end
    end
end

function warfarin_sbbrmis()
    return (
        pk = SBBRMI(warfarin_pk_brmi(); mod=@__MODULE__),
        pd = SBBRMI(warfarin_pd_brmi(); mod=@__MODULE__),
    )
end

function warfarin_problem(sb, tag)
    cache = joinpath(tempdir(), "brm-warfarin")
    isdir(cache) || mkpath(cache)
    code = StanBlocks.stan_code(sb.model)
    return StanBlocks.stan_instantiate(
        sb.model; path=joinpath(cache, "$(tag)_$(hash(code)).stan"),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    @testset "exact public two-stage Warfarin PK/PD reproduction" begin
        models = warfarin_sbbrmis()
        for (tag, sb) in pairs(models)
            code = StanBlocks.stan_code(sb.model)
            @test StanBlocks.stan.transpiles(sb.model)
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
            @test occursin("warfarin_gamma2_overdisp_lpdf", code)
            @test occursin("brm_ranef_sd", code)
            @test occursin("[2]'", code)
            @test !occursin("vector[\"", code)

            if get(ENV, "BRM_WARFARIN_RUNTIME", "1") != "0"
                problem = warfarin_problem(sb, tag)
                q = zeros(LogDensityProblems.dimension(problem))
                lp, grad = LogDensityProblems.logdensity_and_gradient(problem, q)
                @test isfinite(lp)
                @test all(isfinite, grad)
            end
        end
        @test occursin("ode_rk45_tol", StanBlocks.stan_code(models.pd.model))
        @test occursin("0.75", StanBlocks.stan_code(models.pk.model))
    end
end
