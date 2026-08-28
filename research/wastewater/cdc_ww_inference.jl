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
using Distributions: Beta, Dirichlet, LogNormal

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
    # Jurisdiction aggregate: population-weighted sum across subpopulation columns
    # of the collected infection matrix. `I_mat` is [nt x K] (one column per subpop),
    # `w` the population weights; `I_mat * w` is Stan matrix-vector product -> vector[nt].
    # (Used by the @brm surface form below, where `*` at the formula level is the
    # Wilkinson interaction operator, not matmul — so the matmul lives here.)
    wsum(I_mat::matrix[m, K], w::vector[K])::vector[m] = I_mat * w
    # Expand one weekly latent value onto a daily row axis through a data index.
    weekly_expand(x::vector[nw], week_idx::int[nt])::vector[nt] = x[week_idx]
    # Build sparse wastewater-record means from the latent subpopulation matrix.
    # Records own independent time, subpopulation and lab-site mappings, so a
    # latent reference/uncovered population need not have a wastewater series.
    ww_expected_log(I_mat::matrix[nt, K], sh::vector[nsh],
                    sample_time::int[n], sample_subpop::int[n],
                    sample_lab::int[n], log_lab_mod::vector[nlab],
                    log10_g::real, mwpd::real)::vector[n] = begin
        out::vector[n]
        for i in 1:n
            shed = 0.0
            for lag in 1:nsh
                t = sample_time[i] - lag + 1
                shed = shed + (t >= 1 ? sh[lag] * I_mat[t, sample_subpop[i]] : 0.0)
            end
            out[i] = log(10.0) * log10_g + log(shed + 1e-8) - log(mwpd) +
                     log_lab_mod[sample_lab[i]]
        end
        out
    end
    gather_exp(x::vector[nlab], idx::int[n])::vector[n] = exp(x[idx])
    take_window(x::vector[n], start_idx::int, width::int)::vector[width] =
        x[start_idx:(start_idx + width - 1)]
    scale_simplex(x::vector[K], scale::real)::vector[K] = scale * x
    hospital_daily_mean(lat::vector[nt], logit_p::vector[nt], dow::int[nt],
                        dow_effect::vector[7], npop::real)::vector[nt] =
        npop * (dow_effect[dow] .* inv_logit(logit_p) .* lat)
    # Expected admissions from the aggregate latent infections: day-of-week multiplier
    # exp(log_dow), AR(1)-logit IHR inv_logit(logit_p), and jurisdiction population.
    ihr_scale(lat::vector[nt], logit_p::vector[nt], log_dow::vector[nt],
              npop::real)::vector[nt] = npop * (exp(log_dow) .* inv_logit(logit_p) .* lat)
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

# ---------------------------------------------------------------------------
# The SAME full CDC model expressed on the `@brm` FORMULA surface.
#
# This is the direct refutation of "the CDC model cannot be a @brm model": every
# coupled piece is a formula-level statement. The one faithful re-parameterization
# vs. the @slic form is the reference log-Rᵘ prior — the @slic version uses a
# differenced-AR (`dar_logru`, which needs a top-level innovation vector the formula
# surface does not yet declare); the @brm form uses the built-in `ar(time; p=1)`
# autoregressive term (AR(1) around a mean), a faithful autoregressive-Rt prior in
# the same spirit. The day-of-week effect is a log-additive `factor(dow)` (the
# regression-idiomatic form of the @slic Dirichlet mean-1 multiplier).
#
# How each formula seam carries the model:
#   - shared daily log-Rᵘ .............. `log_ru ~ 1 + ar(time_grid; p=1)`, CLOSED
#                                        OVER inside the kernel cell (top-level
#                                        latents/params close over a plate cell);
#   - infection feedback + renewal ..... `renewal_feedback` @deffun in the cell;
#   - multi-subpopulation hierarchy .... `kernel(...) do ... end` broadcasts the
#                                        per-subpop renewal+WW cell; the per-subpop
#                                        AR(1) deviation is drawn IN the cell
#                                        (`eps_d ~ std_normal()` + `ar1_dev`); the
#                                        `(1|site)` seeding intercept derives the
#                                        kernel grouping; cells collect to a matrix;
#   - per-subpop wastewater ............ censored-lognormal `~` INSIDE the cell;
#   - jurisdiction aggregate ........... `I_agg = wsum(I_mat, w)` (population-weighted);
#   - hospital admissions .............. delay convolution + `ar(...)`-logit IHR +
#                                        `factor(dow)` day-of-week + `NegativeBinomial2`
#                                        on the dense aggregate.
#
# Verified: transpile + stanc + finite BridgeStan density/gradient; see
# `test/cdc_ww_inference.jl`. NOT sampled — the deliverable is the model.
"""
    cdc_ww_brm_fixture(; K = 3, nt = 84, uot = 50, ht = 14)

Multi-frame fixture for [`cdc_ww_brm_model`](@ref). `K` is the number of latent
subpopulations and includes an uncovered reference population at index one. The
latent kernel has one row per subpopulation, while wastewater measurements occupy
their own sparse record frame with explicit time, subpopulation, lab-site and LOD
mappings. Several lab sites may map to one catchment and repeated sampling times are
valid. Hospital observations remain a dense daily jurisdiction stream in this first
modular fixture.
"""
function cdc_ww_brm_fixture(; K = 3, nt = 84, uot = 50, ht = 14)
    K >= 2 || error("cdc_ww_brm_fixture needs at least a reference and one sampled subpopulation")
    nt >= 14 || error("cdc_ww_brm_fixture needs at least 14 observed days")
    uot >= 1 || error("cdc_ww_brm_fixture needs a positive unobserved seeding window")
    ht >= 0 || error("cdc_ww_brm_fixture needs a nonnegative forecast horizon")
    g  = [0.02, 0.06, 0.12, 0.16, 0.16, 0.14, 0.11, 0.08, 0.06, 0.04, 0.03, 0.02]
    sh = [0.01, 0.05, 0.12, 0.17, 0.17, 0.14, 0.11, 0.08, 0.06, 0.05, 0.03, 0.01]
    dl = [0.01, 0.03, 0.06, 0.11, 0.14, 0.13, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.01]
    n_total = uot + nt + ht
    n_weeks = cld(n_total, 7)
    subpopulation = [i == 1 ? "reference-uncovered" : "catchment-$(i - 1)" for i in 1:K]
    subpop_size = collect(range(2.0, 1.0; length=K))
    subpop_weight = subpop_size ./ sum(subpop_size)

    # At least two lab sites map to the first sampled catchment. This creates
    # repeated (subpopulation, time) records without duplicating latent states.
    n_labs = K
    ww_lab = ["lab-$i" for i in 1:n_labs]
    lab_to_subpop = [i <= 2 ? 2 : min(K, i) for i in 1:n_labs]
    ww_time = Int[]
    ww_subpop = Int[]
    ww_lab_idx = Int[]
    ww_lod = Float64[]
    ww_log_conc = Float64[]
    for lab in 1:n_labs, observed_day in 1:7:nt
        push!(ww_time, uot + observed_day)
        push!(ww_subpop, lab_to_subpop[lab])
        push!(ww_lab_idx, lab)
        lod = log(2.0 + 0.25 * lab + 0.01 * observed_day)
        latent_signal = log(15.0 + 8.0 * exp(-((observed_day - nt / 2) / 15)^2)) +
                        0.08 * lab
        push!(ww_lod, lod)
        # A value exactly at its row-specific LOD selects the censored branch.
        push!(ww_log_conc, (observed_day + lab) % 4 == 0 ? lod : max(lod + 0.05, latent_signal))
    end

    (;
        subpopulation,
        is_reference = [i == 1 ? 1.0 : 0.0 for i in 1:K],
        t_grid = [collect(1.0:n_total) for _ in 1:K],
        time_grid = collect(1.0:n_total),
        week_grid = collect(1.0:n_weeks),
        week_idx = [min(n_weeks, div(t - 1, 7) + 1) for t in 1:n_total],
        dow = [mod(t - uot - 1, 7) + 1 for t in 1:n_total],
        ww_lab, lab_to_subpop, ww_time, ww_subpop, ww_lab_idx, ww_lod, ww_log_conc,
        hosp = [max(0, round(Int, 20 + 40 * exp(-((t - 50.0) / 14)^2))) for t in 1:nt],
        g = g, sh = sh, dl = dl,
        w = subpop_weight,
        uot, nt, ht, n_total, n_weeks, K, n_labs,
        mwpd = 757.0,
        n_pop = 500000.0,
    )
end

"""
    cdc_ww_brm_model([df]) -> BRMI

The full CDC `ww-inference-model` on the `@brm` formula surface, returned as a
`BRMI`; lower with `SBBRMI(cdc_ww_brm_model())`. See the block comment above for how
each of the four coupled pieces maps onto a formula seam. Re-bind data with
`cdc_ww_brm_model(cdc_ww_brm_fixture(; K = 2, nt = 56))`.
"""
function cdc_ww_brm_model(df = cdc_ww_brm_fixture())
    @brm df begin
        # Shared epidemic parameters. The public formula surface currently has an
        # ordinary AR term rather than CDC's differenced-AR vector process; the
        # documentation states that approximation explicitly.
        gamma       ~ LogNormal(-4.0, 0.5)
        phi_delta   ~ Beta(2.0, 8.0)
        sigma_delta ~ Exponential(4.0)
        log10_g     ~ Normal(12.0, 1.0)

        log_ru_week ~ 1 + ar(week_grid; p = 1)
        effect(log_ru_week, Intercept) ~ Normal(0.0, 0.5)
        log_ru = weekly_expand(log_ru_week, week_idx)

        # Hierarchical incidence and initial growth are defined over the latent
        # subpopulation axis, including the uncovered reference population.
        logit_I0      ~ 1 + (1 | initial | subpopulation)
        initial_growth ~ 1 + (1 | growth | subpopulation)
        effect(logit_I0, Intercept) ~ Normal(-9.0, 1.0)
        effect(initial_growth, Intercept) ~ Normal(0.0, 0.05)
        sd(:, initial) ~ Normal(0.0, 0.75)
        sd(:, growth) ~ Normal(0.0, 0.05)

        # The kernel produces latent trajectories only. Observation frames are
        # mapped onto this matrix after the cell has been collected.
        I_mat ~ kernel(t_grid, is_reference, logit_I0, initial_growth) do ts, isref, lI0, growth_i
            eps_d::vector[n_total] ~ std_normal()
            delta_k = ar1_dev(eps_d, phi_delta, sigma_delta)
            logRt_k = log_ru + (1.0 - isref) * delta_k
            renewal_feedback(logRt_k, g, gamma, inv_logit(lI0), growth_i, uot)
        end

        # Sparse wastewater records have independent time, subpopulation, lab
        # and record-specific LOD mappings. Multiple labs may observe the same
        # catchment/time pair; the reference population may have no records.
        log_lab_mod ~ 0 + (1 | ww_scale | ww_lab)
        log_sigma_ww ~ 1 + (1 | ww_noise | ww_lab)
        effect(log_sigma_ww, Intercept) ~ Normal(log(0.35), 0.5)
        sd(:, ww_scale) ~ Normal(0.0, 0.5)
        sd(:, ww_noise) ~ Normal(0.0, 0.35)
        ww_mu = ww_expected_log(I_mat, sh, ww_time, ww_subpop, ww_lab_idx,
                                log_lab_mod, log10_g, mwpd)
        ww_sigma = gather_exp(log_sigma_ww, ww_lab_idx)
        ww_log_conc ~ censored(Normal(ww_mu, ww_sigma); lower = ww_lod)

        # jurisdiction aggregate infections (population-weighted across subpops)
        I_agg = wsum(I_mat, w)

        # Dense daily hospital stream for the first checkpoint. The next module
        # replaces this fixed frame with explicit stream and interval mappings.
        logit_ihr_week ~ 1 + ar(week_grid; p = 1)
        effect(logit_ihr_week, Intercept) ~ Normal(-4.6, 0.3)
        logit_ihr = weekly_expand(logit_ihr_week, week_idx)
        dow_share ~ Dirichlet(7, 5.0)
        dow_effect = scale_simplex(dow_share, 7.0)
        phi_h ~ Exponential(1.0)
        lat = delay_convolve(I_agg, dl)
        hosp_mu_full = hospital_daily_mean(lat, logit_ihr, dow, dow_effect, n_pop)
        hosp_mu = take_window(hosp_mu_full, uot + 1, nt)
        hosp ~ NegativeBinomial2(hosp_mu, phi_h)
    end
end
