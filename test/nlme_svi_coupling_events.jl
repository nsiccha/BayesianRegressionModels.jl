# Public 32-unit event/Kalman reproduction of snag brm-svi-fit-defa-afacf161.
# The model and data generator below are copied unchanged from the reporter's
# toy_events.jl, SHA256 05f376543fbc89a2e0058445374384c94107dbea00016b67066d6f51ae4234d3.
# Both policies get the same data, fit seed, batch size and total iteration cap.
# RUN: julia --project=test test/nlme_svi_coupling_events.jl
using Test, Treebars
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
import Random, Statistics

const TOY3 = @brm begin
    level ~ 1 + (1 | unit)
    effect(level, Intercept) ~ Normal(level_loc, level_scale)
    sd(level, unit) ~ Exponential(level_sd_prior)
    trend ~ 1 + (1 | unit)
    effect(trend, Intercept) ~ Normal(0.0, trend_scale)
    sd(trend, unit) ~ Exponential(trend_sd_prior)
    cyc_cos ~ 1 + (1 | unit)
    effect(cyc_cos, Intercept) ~ Normal(0.0, cyc_scale)
    sd(cyc_cos, unit) ~ Exponential(cyc_sd_prior)
    cyc_sin ~ 1 + (1 | unit)
    effect(cyc_sin, Intercept) ~ Normal(0.0, cyc_scale)
    sd(cyc_sin, unit) ~ Exponential(cyc_sd_prior)
    atanh(phi) ~ 1 + (1 | unit)
    effect(phi, Intercept) ~ Normal(phi_loc, phi_scale)
    sd(phi, unit) ~ Exponential(small_sd_prior)
    log(sigma) ~ 1 + (1 | unit)
    effect(sigma, Intercept) ~ Normal(log_sigma_loc, log_scale)
    sd(sigma, unit) ~ Exponential(small_sd_prior)
    log(noise) ~ 1 + (1 | unit)
    effect(noise, Intercept) ~ Normal(log_noise_loc, log_scale)
    sd(noise, unit) ~ Exponential(small_sd_prior)
    sigma_dev_spread ~ Exponential(small_sd_prior)
    gain_mean[1:K] .~ Normal.(0.0, step_scale)
    gain_spread[1:K] .~ Exponential.(step_sd_prior)
    offset_mean[1:K] .~ Normal.(0.0, offset_scale)
    offset_spread[1:K] .~ Exponential.(offset_sd_prior)
    rows = groups(row_unit => unit)
    @plate for i in eachindex(level)
        z_gain ~ Normal(event_zero[i], 1)
        z_offset ~ Normal(event_zero[i], 1)
        z_sigma ~ Normal(zero_loc[i], 1)
        gain_steps = gain_mean[event_kind[i]] .+ gain_spread[event_kind[i]] .* z_gain
        offset_steps = offset_mean[event_kind[i]] .+ offset_spread[event_kind[i]] .* z_offset
        @scan begin
            epoch_log_gain[1] = 0.0
            for j in 2:n_epochs[i]
                epoch_log_gain[j] = epoch_log_gain[j - 1] + gain_steps[j - 1]
            end
        end
        @scan begin
            epoch_offset[1] = 0.0
            for j in 2:n_epochs[i]
                epoch_offset[j] = epoch_offset[j - 1] + offset_steps[j - 1]
            end
        end
        days = t[rows[i]]
        n_days = length(days)
        angle = (2 * pi) .* days ./ period
        base = level[i] .+ trend[i] .* days ./ period .+ cyc_cos[i] .* cos.(angle) .+ cyc_sin[i] .* sin.(angle)
        epochs = epoch[rows[i]]
        expected = exp.(epoch_log_gain[epochs]) .* base .+ epoch_offset[epochs]
        noise_var = noise[i] * noise[i]
        scale = sigma[i] * exp(sigma_dev_spread * z_sigma)
        state_var = scale * scale / (1 - phi[i] * phi[i])
        gaps = days[2:n_days] .- days[1:(n_days - 1)]
        decays = phi[i] .^ gaps
        fresh = state_var .* (1 .- decays .* decays)
        first_var = state_var + noise_var
        @scan begin
            filt_var[1] = state_var * (1 - state_var / first_var)
            for k in 2:n_days
                pv = decays[k - 1] * decays[k - 1] * filt_var[k - 1] + fresh[k - 1]
                filt_var[k] = pv * (1 - pv / (pv + noise_var))
            end
        end
        pred_var = vcat(state_var, decays .* decays .* filt_var[1:(n_days - 1)] .+ fresh)
        innovation_var = pred_var .+ noise_var
        gain = pred_var ./ innovation_var
        obs = y[rows[i]]
        @scan begin
            filt[1] = gain[1] * (obs[1] - expected[1])
            for k in 2:n_days
                filt[k] = decays[k - 1] * filt[k - 1] + gain[k] * (obs[k] - expected[k] - decays[k - 1] * filt[k - 1])
            end
        end
        state_pred = vcat(0.0, decays .* filt[1:(n_days - 1)])
        y[rows[i]] ~ Normal(expected .+ state_pred, sqrt.(innovation_var))
    end
end

function toy3_data(; seed = 1, S = 32, T = 110, change = 60, phi = 0.6, sigma = 0.5, noise = 0.2,
                   ratio = 1.18, extra_rate = 0.5, period = 365.0, missing_rate = 0.06)
    rng = Random.Xoshiro(seed)
    row_unit = Int[]; t = Float64[]; y = Float64[]; epoch = Int[]
    event_kind = Vector{Int}[]; event_zero = Vector{Float64}[]; n_epochs = Int[]
    for s in 1:S
        L = 80 + 15 * randn(rng); tr = 2 * randn(rng); cc = 3 + randn(rng); cs = randn(rng)
        ev = [(change, 1)]
        rand(rng) < extra_rate && push!(ev, (rand(rng, 5:(T - 5)), 2))
        sort!(ev)
        lg = cumsum(vcat(0.0, [k == 1 ? log(ratio) + 0.01 * randn(rng) : 0.05 * randn(rng) for (_, k) in ev]))
        of = cumsum(vcat(0.0, [k == 1 ? 0.0 + 0.1 * randn(rng) : 1.0 * randn(rng) for (_, k) in ev]))
        push!(event_kind, [k for (_, k) in ev]); push!(event_zero, zeros(length(ev))); push!(n_epochs, length(ev) + 1)
        sg = sigma * exp(0.1 * randn(rng))
        x = sg / sqrt(1 - phi^2) * randn(rng)
        for d in 1:T
            d > 1 && (x = phi * x + sg * randn(rng))
            rand(rng) < missing_rate && continue
            e = 1 + count(((dd, _),) -> dd <= d, ev)
            a = 2pi * d / period
            m = exp(lg[e]) * (L + tr * d / period + cc * cos(a) + cs * sin(a)) + of[e]
            push!(row_unit, s); push!(t, d); push!(epoch, e); push!(y, m + x + noise * randn(rng))
        end
    end
    (; unit = collect(1:S), row_unit, t, y, epoch, event_kind, event_zero, n_epochs, zero_loc = zeros(S), K = 2,
     period, level_loc = 80.0, level_scale = 30.0, level_sd_prior = 20.0, trend_scale = 5.0, trend_sd_prior = 2.0,
     cyc_scale = 5.0, cyc_sd_prior = 2.0, phi_loc = 0.5, phi_scale = 1.0, small_sd_prior = 0.2,
     log_sigma_loc = log(0.5), log_noise_loc = log(0.2), log_scale = 1.0,
     step_scale = 0.2, step_sd_prior = 0.05, offset_scale = 2.0, offset_sd_prior = 1.0)
end

function event_coupling_fit(rk, label; options...)
    Treebars.with_progress(:state; description=label) do root
        path = get(ENV, "KB_PROGRESS_FILE", tempname())
        Treebars.snapshot_progress(root, path) do
            brm_svi_fit(rk; subject_group=:unit, batch_size=4,
                rng=Random.Xoshiro(1), ndraws=1000, max_iterations=100_000,
                progress=root, options...)
        end
    end
end

function event_coupling_summary(rk, fit)
    draws = rk_restore_draws(rk, fit.draws)
    ratio = Statistics.quantile(exp.(draws.gain_mean[1, :]), [0.05, 0.5, 0.95])
    phi = Statistics.quantile(tanh.(collect(Float64, vec(draws.phi_Intercept))), [0.05, 0.5, 0.95])
    (; ratio, phi, elbo=Statistics.mean(fit.diagnostics.elbo[end-999:end]))
end

@testset "default coupling retains the event posterior region" begin
    rk = RKBRMI(TOY3(toy3_data(; S=32, seed=1)))
    coupled = event_coupling_fit(rk, "event model: default coupling")
    independent = event_coupling_fit(rk, "event model: independent factors"; coupling=nothing)
    c, p = event_coupling_summary(rk, coupled), event_coupling_summary(rk, independent)
    println("EVENT_COUPLING coupled=", c, " independent=", p,
        " coupled_converged=", coupled.diagnostics.converged,
        " independent_converged=", independent.diagnostics.converged)
    # Truth is gain ratio 1.18 and phi 0.6. Immediate coupling previously sent
    # the gain ratio far below 1 or phi toward 1; the independent fit recovered
    # the intended region but substantially understated global uncertainty.
    @test 1.13 < c.ratio[2] < 1.23
    @test 0.4 < c.phi[2] < 0.85
    @test abs(c.ratio[2] - p.ratio[2]) < 0.03
    @test abs(c.phi[2] - p.phi[2]) < 0.1
    @test c.ratio[3] - c.ratio[1] > 1.5 * (p.ratio[3] - p.ratio[1])
    @test c.elbo > p.elbo - 50
    @test any(V -> any(!iszero, V), coupled.diagnostics.q.coupling.V)
    @test coupled.diagnostics.finished && independent.diagnostics.finished
    @test !coupled.diagnostics.converged || coupled.diagnostics.stationarity.current.passed
end
