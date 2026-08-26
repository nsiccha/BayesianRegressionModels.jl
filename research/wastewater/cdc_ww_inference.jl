# Full port of the CDC `ww-inference-model` (github.com/CDCgov/ww-inference-model)
# — a joint wastewater + hospital-admissions renewal model — onto BRM's StanBlocks
# backend. Authorized by decision `2026-08-26T16-44-33-638-1yz7ybe` (option B).
#
# The model (faithful to `model_definition.md`) has four coupled pieces:
#
#  1. INFECTION PROCESS. Reference-subpopulation weekly unadjusted log-Rᵘ follows a
#     differenced autoregression, `log Rᵘ(tᵢ)=log Rᵘ(tᵢ₋₁)+β·Δ+σ_r ε` (`dar_logru`).
#     The realized reproduction number carries INFECTION FEEDBACK,
#     `R(t)=Rᵘ(t)·exp(−γ Σ_τ I(t−τ)g(τ))`, and incidence follows the renewal
#     equation with a 50-day exponential seeding phase (`renewal_feedback`).
#
#  2. MULTI-SUBPOPULATION HIERARCHY. K catchment subpopulations (+ a reference)
#     each carry `log Rᵘ_k = log Rᵘ_0 + m + δ_k(t)` with δ_k an AR(1) deviation
#     (`ar1_dev`); each runs its own renewal, and jurisdiction incidence is the
#     population-weighted aggregate `I(t)=Σ_k w_k I_k(t)`.
#
#  3. WASTEWATER OBSERVATION (per subpopulation/site). Genome concentration is the
#     shedding-kinetics convolution of that subpopulation's infections scaled by
#     genomes-per-infection / per-person volume (`shed_convolve`, `log_shed_scale`);
#     site-lab measurements are log-normal with a site scaling `M_k` and a
#     hierarchical observation SD `σ_c,k`, with below-limit-of-detection
#     left-censoring.
#
#  4. HOSPITAL-ADMISSIONS OBSERVATION (jurisdiction). Expected admissions convolve
#     aggregate infections with an infection-to-hospitalization delay
#     (`delay_convolve`), scaled by a time-varying AR(1)-logit IHR and a day-of-week
#     Dirichlet effect; counts are negative-binomial.
#
# The carried-state parts (renewal, AR processes, all convolutions) live in
# `@deffun`s and are called loop-free from the `@slic` body — `plate` broadcasts the
# per-subpopulation renewal+wastewater cell over independent subpopulations, and the
# aggregate + hospitalizations are ordinary top-level statements on the collected
# infection matrix.
#
# Verified: transpile + stanc + finite BridgeStan density/gradient (dim=379 on the
# default fixture); see `test/cdc_ww_inference.jl`. NOT sampled — the deliverable is
# the model. The generation-interval / shedding / delay PMFs and priors are
# illustrative discretizations, not calibrated values.

using BayesianRegressionModels
using StanBlocks

StanBlocks.@deffun begin
    # Differenced-AR on log scale: x[i]=x[i-1]+beta*(x[i-1]-x[i-2])+sigma*eps[i].
    dar_logru(logru1::real, eps::vector[nw], beta::real, sigma::real)::vector[nw] = begin
        x::vector[nw]
        x[1] = logru1
        if nw >= 2
            x[2] = x[1] + sigma * eps[2]
        end
        for i in 3:nw
            x[i] = x[i - 1] + beta * (x[i - 1] - x[i - 2]) + sigma * eps[i]
        end
        x
    end
    # AR(1) deviation: d[t]=phi*d[t-1]+sigma*eps[t].
    ar1_dev(eps::vector[nt], phi::real, sigma::real)::vector[nt] = begin
        d::vector[nt]
        d[1] = sigma * eps[1]
        for t in 2:nt
            d[t] = phi * d[t - 1] + sigma * eps[t]
        end
        d
    end
    # Renewal with infection feedback + exponential seeding.
    renewal_feedback(logRt::vector[nt], g::vector[ng], gamma::real, I0::real,
                     r::real, n_seed::int)::vector[nt] = begin
        I::vector[nt]
        for t in 1:nt
            if t <= n_seed
                I[t] = I0 * exp(r * (t - 1))
            else
                conv = 0.0
                for s in 1:ng
                    prev = t - s >= 1 ? I[t - s] : 0.0
                    conv = conv + g[s] * prev
                end
                Rt = exp(logRt[t]) * exp(-gamma * conv)
                I[t] = Rt * conv
            end
        end
        I
    end
    # Shedding-load convolution: C(t)=Σ_{k=1}^{nsh} s(k) I(t-k+1).
    shed_convolve(I::vector[nt], s::vector[nsh])::vector[nt] = begin
        c::vector[nt]
        for t in 1:nt
            acc = 0.0
            for k in 1:nsh
                idx = t - k + 1
                acc = acc + (idx >= 1 ? s[k] * I[idx] : 0.0)
            end
            c[t] = acc
        end
        c
    end
    # Infection->outcome delay convolution: L(t)=Σ_{k=1}^{nd} d(k) I(t-k+1) (d[1]=lag 0).
    delay_convolve(I::vector[nt], d::vector[nd])::vector[nt] = begin
        l::vector[nt]
        for t in 1:nt
            acc = 0.0
            for k in 1:nd
                idx = t - k + 1
                acc = acc + (idx >= 1 ? d[k] * I[idx] : 0.0)
            end
            l[t] = acc
        end
        l
    end
end

"""
    cdc_ww_inference_fixture(; K = 3, nt = 84, n_weeks = 12, n_seed = 15)

A minimal jurisdiction dataset: `K` catchment subpopulations over `nt` daily
timepoints. `ww` is a ragged per-subpopulation log-concentration series; `h` is the
daily hospital-admissions count. `g`/`s`/`d` are the (fixed) generation-interval,
shedding-kinetics and infection-to-hospitalization delay PMFs; `w` are the
population weights; `lloq` the log limit of detection; `n_pop` the jurisdiction
population; `week_idx`/`dow` map days to weeks / day-of-week.
"""
function cdc_ww_inference_fixture(; K = 3, nt = 84, n_weeks = 12, n_seed = 15)
    g = [0.02, 0.06, 0.12, 0.16, 0.16, 0.14, 0.11, 0.08, 0.06, 0.04, 0.03, 0.02]
    s = [0.01, 0.05, 0.12, 0.17, 0.17, 0.14, 0.11, 0.08, 0.06, 0.05, 0.03, 0.01]
    d = [0.01, 0.03, 0.06, 0.11, 0.14, 0.13, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.01]
    (;
        ww = [[log(30.0) + 0.7 * exp(-((t - 45.0) / 15)^2) + 0.05 * sin(t / 4) for t in 1:nt]
              for _ in 1:K],
        h = [max(0, round(Int, 20 + 40 * exp(-((t - 50.0) / 14)^2))) for t in 1:nt],
        g = g, s = s, d = d,
        week_idx = [min(n_weeks, div(t - 1, 7) + 1) for t in 1:nt],
        dow = [((t - 1) % 7) + 1 for t in 1:nt],
        w = (ws = [K - i + 1.0 for i in 1:K]; ws ./ sum(ws)),   # normalized population weights
        lloq = log(5.0), n_pop = 500000.0,
        nt = nt, n_weeks = n_weeks, ng = length(g), nsh = length(s), nd = length(d),
        K = K, n_seed = n_seed,
    )
end

"""
    cdc_ww_inference_model([data])

The full CDC `ww-inference-model` as a re-bindable `@slic` model: multi-subpopulation
differenced-AR Rt with infection feedback, per-subpopulation censored wastewater
measurement, and a joint negative-binomial hospital-admissions stream. Swap data with
`cdc_ww_inference_model()(; h = new_h, …)`.
"""
function cdc_ww_inference_model(data = cdc_ww_inference_fixture())
    @slic data begin
        # reference-subpopulation weekly differenced-AR log-Rᵘ
        logru1 ~ normal(0.0, 0.5)
        beta   ~ normal(0.5, 0.2; lower = 0.0, upper = 1.0)
        sigma_r ~ normal(0.0, 0.2; lower = 0.0)
        eps_r::vector[n_weeks] ~ std_normal()
        logru_daily = dar_logru(logru1, eps_r, beta, sigma_r)[week_idx]
        # shared subpopulation-hierarchy hyperparameters
        m_int ~ normal(0.0, 0.5)
        gamma ~ lognormal(-4.0, 0.5)                    # infection feedback
        phi_delta ~ normal(0.1, 0.05; lower = 0.0, upper = 1.0)
        sigma_delta ~ normal(0.0, 0.2; lower = 0.0)
        r_growth ~ normal(0.0, 0.05)                    # seeding growth rate
        # wastewater measurement hyperparameters
        log_shed_scale ~ normal(0.0, 1.0)               # log(genomes / per-person volume)
        sigma_m ~ normal(0.0, 0.25; lower = 0.0)
        log_sigma_c_hat ~ normal(0.0, 1.0)
        sigma_log_sigma_c ~ normal(0.0, 0.7; lower = 0.0)
        # per-subpopulation renewal + wastewater observation; collect I_k trajectories
        I_mat ~ plate(ww; outer = (K,)) do wwi
            log_I0 ~ normal(-6.0, 1.0)
            eps_d::vector[nt] ~ std_normal()
            delta_k = ar1_dev(eps_d, phi_delta, sigma_delta)
            logRt_k = logru_daily + (m_int + delta_k)
            I_k = renewal_feedback(logRt_k, g, gamma, exp(log_I0), r_growth, n_seed)
            logM_k ~ normal(0.0, sigma_m)
            log_sigma_c_k ~ normal(log_sigma_c_hat, sigma_log_sigma_c)
            C_k = shed_convolve(I_k, s) * exp(log_shed_scale)
            wwi ~ censored(normal, log(C_k) + logM_k, exp(log_sigma_c_k); lower = lloq)
            I_k
        end
        # jurisdiction aggregate infections
        I_agg = I_mat * w
        # hospital admissions: delay convolution + AR(1) logit IHR + day-of-week + NB
        logit_ihr0 ~ normal(-4.6, 0.3)
        phi_H ~ normal(0.01, 0.01; lower = 0.0, upper = 1.0)
        sigma_H ~ normal(0.0, 0.01; lower = 0.0)
        eps_H::vector[nt] ~ std_normal()
        logit_ph = logit_ihr0 + ar1_dev(eps_H, phi_H, sigma_H)
        p_hosp = inv_logit(logit_ph)
        omega_raw::simplex[7] ~ dirichlet(rep_vector(5.0, 7))
        omega = 7.0 * omega_raw                         # day-of-week effect, mean 1
        lat = delay_convolve(I_agg, d)
        H = (omega[dow] .* p_hosp) .* lat
        phi_h ~ normal(0.0, 1.0; lower = 0.0)
        h ~ neg_binomial_2(n_pop * H, phi_h)
    end
end
