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

"""Two-subject slice copied from the public Warfarin data and PD Stan dump."""
function warfarin_fixture()
    return (
        # Dense grouping indices preserve the row order required by kernel;
        # source_subject records the original public-data labels.
        subject = [1, 2],
        source_subject = [100, 2],
        dose = [100.0, 100.0],
        log_weight_ratio = log.([66.7, 66.7] ./ 70.0),
        pk_time = [
            [1.0, 2.0, 3.0, 6.0, 9.0, 12.0, 24.0, 36.0, 48.0, 72.0],
            [2.0, 3.0, 6.0, 12.0, 24.0, 36.0, 48.0, 72.0, 96.0, 120.0],
        ],
        pk_dv = [
            [1.9, 3.3, 6.6, 9.1, 10.8, 8.6, 5.6, 4.0, 2.7, 0.8],
            [8.4, 9.7, 9.8, 11.0, 8.3, 7.7, 6.3, 4.1, 3.0, 1.4],
        ],
        pd_time = [
            [24.0, 36.0, 48.0, 72.0, 96.0, 120.0, 144.0],
            [0.0, 24.0, 36.0, 48.0, 72.0, 96.0, 120.0, 144.0],
        ],
        pd_dv = [
            [44.0, 27.0, 28.0, 31.0, 60.0, 65.0, 71.0],
            [100.0, 46.0, 22.0, 19.0, 20.0, 42.0, 49.0, 54.0],
        ],
        # Fixed subject-specific parameter medians from the public PD model's
        # stan_data.R dump. Columns 1 and 3 correspond to original IDs 100 and
        # 2, matching this fixture. These are already log(tlag), log(ka),
        # allometric log(CL), and allometric log(V), not standardized effects.
        pk_log_tlag = [-0.161353257334516, -0.125956102587045],
        pk_log_ka = [-1.12518262665871, -0.00274334970440031],
        pk_log_cl = [-1.27221940132543, -2.04962147951756],
        pk_log_v = [2.05396166702831, 2.10767454410602],
    )
end

"""Exact first-stage population PK model from the public Weber program."""
function warfarin_pk_brmi(data = warfarin_fixture())
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

        pk_pred ~ kernel(
            pk_time, dose, pk_dv,
            lag_logit, log_ka, log_cl0, log_v0,
        ) do times, dose_i, observed,
             lag_i, lka_i, lcl_i, lv_i
            # log_cl0 / log_v0 already carry the allometric offset (formula term above),
            # so the cell uses them directly — no weight scaling hand-wired here.
            log_tlag_i = log_inv_logit(lag_i)
            prediction = exp(warfarin_pk_logconcentration(
                times, log(dose_i), lka_i, lcl_i, lv_i, log_tlag_i,
            )) + 1e-5
            observed ~ warfarin_gamma2_overdisp(
                prediction, sigma_pk, kappa_pk * 25.0,
            )
            prediction
        end
    end
end

"""Exact second-stage PD model, conditioned on the public PK posterior medians."""
function warfarin_pd_brmi(data = warfarin_fixture())
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

        pd_pred ~ kernel(
            pd_time, dose, pd_dv,
            pk_log_tlag, pk_log_ka, pk_log_cl, pk_log_v,
            log_r0, log_inv_kout, log_ec50,
        ) do times, dose_i, observed,
             log_tlag_i, log_ka_i, log_cl_i, log_v_i,
             lr0_i, linvkout_i, lec50_i
            prediction = warfarin_turnover_prediction(
                times, log(dose_i), log_ka_i, log_cl_i, log_v_i,
                log_tlag_i, lr0_i, linvkout_i, lec50_i,
            )
            observed ~ warfarin_gamma2_overdisp(
                prediction, sigma_pd, kappa_pd * 625.0,
            )
            prediction
        end
    end
end

function warfarin_sbbrmis(data = warfarin_fixture())
    return (
        pk = SBBRMI(warfarin_pk_brmi(data); mod=@__MODULE__),
        pd = SBBRMI(warfarin_pd_brmi(data); mod=@__MODULE__),
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
