# Grey-seal integrated population model (IPM) — a VERBATIM port of the SlicTranspiler
# spotlight example (github.com/nsiccha/SlicTranspiler, `src/grey_seal_ipm.jl`) onto
# BRM's StanBlocks backend, per decision `2026-08-26T19-58-24-844-0yiqrzn` (option A:
# port a real, canonical model, not a sketch).
#
# The model is a Baltic grey-seal IPM: a YEAR-RECURSIVE, age/sex-structured state
# process (Leslie aging matrix, size-structured mortality, density-dependent birth
# rates, within-year hunting solved by an ODE, and multinomial fate allocation of
# each cohort into survived / bycatch / hunted-SW / hunted-FI), observed through EIGHT
# data streams: aerial pup counts (negative binomial), Swedish/Finnish harvest bags
# (normal), Swedish/Finnish hunting age-composition (multinomial), bycatch composition
# (multinomial), pregnancy counts (binomial), and reproductive-signs composition
# (multinomial).
#
# It is authored as inert SLIC source — one parent `@slic` body (`GREY_SEAL_IPM_SOURCE`)
# + 31 `@deffun` UDF cards + 2 anonymous observation-stream submodels
# (`GREY_SEAL_IPM_PARTS`) — and assembled with StanBlocks' `compile_slic_bundle`. The
# source below is copied VERBATIM from SlicTranspiler; only the fixture + builder are
# BRM-side. It exceeds the `@brm` formula DSL (multi-stream state-space model), so —
# like the CDC model — it lives at BRM's `@slic`/`@deffun` backend layer.
#
# Verified: `compile_slic_bundle` + `stanc` clean (~52 KB of Stan) on the example data;
# see `test/seal_ipm.jl`. Model artifact only (not sampled).

using BayesianRegressionModels
using StanBlocks
using Distributions
import StanBlocks: compile_slic_bundle

# Shim mirroring SlicTranspiler's `_bundle_part` so the verbatim source loads here.
_bundle_part(slug, title, source; kind = :submodel, data = nothing) =
    (; slug, title, kind, source = strip(source), data)

# ============================ VERBATIM SlicTranspiler source ============================
# Structured grey-seal integrated population model.
#
# The main model, 31 Stan-only UDF cards, and two anonymous observation-stream
# cards are kept as inert strings. The public worker parses and validates each
# part before StanBlocks compiles the isolated workspace.

const GREY_SEAL_IPM_SOURCE = raw"""
# ---- STATE-PROCESS parameters (base — the scan needs them) ----
    phi_a_sc                   ~ uniform(0.0, 1.0)
    phi_sc                     ~ uniform(0.0, 1.0)
    survival_shape             ~ uniform(0.0, 1.0)
    male_pup_survival_offset   ~ cauchy(0.0, 1.0)
    male_adult_survival_offset ~ cauchy(0.0, 1.0)
    carrying_capacity          ~ lognormal(11.0, 1.0)
    max_baseline_birth_rate    ~ uniform(0.0, 1.0)
    min_baseline_birth_rate_sc ~ uniform(0.0, 1.0)
    herring_intercept_scaled   ~ normal(0.0, 1.0)
    herring_slope              ~ normal(0.0, 1.0)
    herring_weight             ~ uniform(0.0, 1.0)
    hunting_selectivity_sweden  :: vector[n_demo] ~ normal(0.0, 1.0)
    hunting_selectivity_finland :: vector[n_demo] ~ normal(0.0, 1.0)
    hunting_effort_sd_sweden   ~ cauchy(0.0, 1.0; lower=0.0)
    hunting_effort_sd_finland  ~ cauchy(0.0, 1.0; lower=0.0)
    population_init_size       ~ lognormal(11.0, 1.0)
    # harvest-bag CV: an OBSERVATION param, but shared by SW+FI, so it can't live
    # in a single Form-A stream (two obs would share it) — it stays in the base.
    harvest_bag_cv             ~ lognormal(0.0, 1.0; lower=0.0)
    epsilon_birth        :: vector[n_state_years] ~ std_normal()
    epsilon_sex          :: vector[n_state_years] ~ std_normal()
    epsilon_h_sw         :: vector[n_state_years] ~ std_normal()
    epsilon_h_fi         :: vector[n_state_years] ~ std_normal()
    transition_noise_vec :: vector[3 * n_demo * n_state_years] ~ std_normal()
    # reproductive-signs reporting params (feed pi_s/pi_c into the scan — FLAG h)
    report_ca_mean        ~ uniform(0.0, 1.0)
    report_placental_mean ~ uniform(0.0, 1.0)
    prob_of_ca            ~ uniform(0.0, 1.0)
    report_placental_sd   ~ normal(0.0, 0.1; lower=0.0)
    report_ca_sd          ~ normal(0.0, 0.1; lower=0.0)
    epsilon_ca        :: vector[n_state_years] ~ std_normal(; lower=0.0)
    epsilon_placental :: vector[n_state_years] ~ std_normal(; lower=0.0)

    # ---- transformed quantities feeding the scan ----
    phi_a   = phi_a_sc
    phi_pup = phi_sc * phi_a
    mu_m    = mortality_rates(phi_pup, phi_a, survival_shape, n_age,
                              male_pup_survival_offset, male_adult_survival_offset)
    S_diag  = exp(-mu_m)
    aging   = create_aging_matrix(n_demo, n_age)
    baseline_bbr = compute_baseline_birth_rate(
        max_baseline_birth_rate * min_baseline_birth_rate_sc, max_baseline_birth_rate,
        herring_intercept_scaled, herring_slope, herring_weight,
        herring_index_1, herring_index_2)
    dd_scaled    = birth_rate_at_carrying_capacity(phi_a, mu_m, n_age)
    dd_intercept = compute_density_dependence_intercept(max_baseline_birth_rate, dd_scaled)
    dd_slope     = -log(carrying_capacity)
    pop_first    = initialize_population(population_init, population_init_size,
                                         population_burn_in, baseline_bbr[1], aging, S_diag, n_age)
    pi_s = report_placental_mean * exp(-epsilon_placental * report_placental_sd)
    pi_c = report_ca_mean        * exp(-epsilon_ca        * report_ca_sd)
    transition_noise_raw = to_matrix(
        transition_noise_vec, 3 * n_demo, n_state_years)

    # ---- THE state process: computed ONCE (state carriers) ----
    state = run_state_process(
        n_state_years, n_age, pop_first, baseline_bbr[1], sum(pop_first),
        baseline_bbr, dd_intercept, dd_slope, aging, S_diag, mu_m,
        hunting_selectivity_sweden, hunting_selectivity_finland,
        hunting_quota_sweden, hunting_quota_finland,
        hunting_effort_sd_sweden, hunting_effort_sd_finland,
        epsilon_h_sw, epsilon_h_fi, t_mate_to_preg, t_birth_to_end_hunt,
        epsilon_birth, epsilon_sex, transition_noise_raw,
        pi_s, pi_c, prob_of_ca, ode_init_state, ode_times)

    # =========================================================================
    #  OBSERVATION NODES — Form A: the REAL observed data is on the LHS. DELETE a
    #  line to deactivate that stream (a submodel node also drops its own params).
    #  The 2 submodels take EVERY name they read as a KWARG (no scope-flow —
    #  StanBlocks snag data-submodel-li-75dc835a); the 6 direct family calls read
    #  the model's own `state`/data/params in scope. Embed only via `~`.
    # =========================================================================
    obs_aerial_count ~ aerial_stream(;
        aerial_year, population_total = state.population_total)
    obs_bycatch_comp ~ bycatch_stream(;
        bycatch_comp_year, bycatch_comp_sample_size, n_demo,
        bycatch_expected = state.bycatch_expected)

    obs_hunting_bag_sweden  ~ harvest_bags(hunting_bag_year_sweden,
        state.hunting_bag_total_sweden,  harvest_bag_cv)
    obs_hunting_bag_finland ~ harvest_bags(hunting_bag_year_finland,
        state.hunting_bag_total_finland, harvest_bag_cv)
    obs_hunting_comp_sweden  ~ hunting_comp(hunting_comp_year_sweden,
        state.hunted_sweden, state.hunting_bag_total_sweden,
        hunting_comp_sample_size_sweden)
    obs_hunting_comp_finland ~ hunting_comp(hunting_comp_year_finland,
        state.hunted_finland, state.hunting_bag_total_finland,
        hunting_comp_sample_size_finland)
    obs_pregnancy_count ~ pregnancy(pregnancy_count_year,
        pregnancy_sample_size, state.pregnancy_rate)
    obs_reproductive_signs_finland ~ reproductive_signs(reproductive_signs_year,
        state.reproductive_probs, reproductive_signs_sample_size)
"""

const GREY_SEAL_IPM_PARTS = (
    _bundle_part("dH_dt", "DH dH/dt", raw"""
dH_dt(tau::real, H::vector[ny], n0::real, k::real,
      E_1::real, E_2::real, mu::real)::vector[ny] = begin
    surv = exp(-(E_1 + E_2) * (k * tau - tau * tau / 2) - mu * tau)
    rep_vector(n0 * E_1 * surv * (k - tau), 1)
end
"""; kind = :udf),
    _bundle_part("create_aging_matrix", "Create Aging Matrix", raw"""
create_aging_matrix(n_demo::int, n_age::int)::matrix[n_demo, n_demo] = begin
    A = rep_matrix(0.0, n_demo, n_demo)
    for i in 2:n_age
        A[i, i - 1] = 1.0
    end
    A[n_age, n_age] = 1.0
    for i in (n_age + 2):n_demo
        A[i, i - 1] = 1.0
    end
    A[n_demo, n_demo] = 1.0
    A
end
"""; kind = :udf),
    _bundle_part("mortality_rates", "Mortality Rates", raw"""
mortality_rates(phi_pup::real, phi_adult::real, c::real, n_age::int,
                male_pup_offset::real, male_adult_offset::real) = begin
    n_demo = 2 * n_age
    mu_m::vector[n_demo]
    mu_pup_f = -log(phi_pup)
    mu_ad_f  = -log(phi_adult)
    mu_pup_m = exp(log(mu_pup_f) + male_pup_offset)
    mu_ad_m  = exp(log(mu_ad_f)  + male_adult_offset)
    mu_m[1]     = mu_pup_f
    mu_m[n_age] = mu_ad_f
    for j in 2:(n_age - 1)
        w = exp(c * log((j - 1.0) / (n_age - 1.0)))
        mu_m[j] = exp(log(mu_pup_f) + w * (log(mu_ad_f) - log(mu_pup_f)))
    end
    mu_m[n_age + 1] = mu_pup_m
    mu_m[n_demo]    = mu_ad_m
    for j in 2:(n_age - 1)
        w_male = exp(c * log((j - 1.0) / (n_age - 1.0)))
        mu_m[n_age + j] = exp(log(mu_pup_m) + w_male * (log(mu_ad_m) - log(mu_pup_m)))
    end
    mu_m
end
"""; kind = :udf),
    _bundle_part("compute_density_dependence_intercept", "Compute Density Dependence Intercept", raw"""
compute_density_dependence_intercept(max_bbr::real, dd_scaled::real)::real =
    -log(max_bbr + (1.0 - max_bbr) * dd_scaled)
"""; kind = :udf),
    _bundle_part("update_birth_rate", "Update Birth Rate", raw"""
update_birth_rate(b0::real, theta0::real, theta1::real, N_prev::real)::real =
    b0 * exp(-theta0 * (exp(theta1 * N_prev) - 1.0))
"""; kind = :udf),
    _bundle_part("birth_rate_at_carrying_capacity", "Birth Rate At Carrying Capacity", raw"""
birth_rate_at_carrying_capacity(phi_a::real, mu_m::vector[n_demo], n_age::int)::real =
    2.0 * (1.0 - phi_a) / exp(sum(-mu_m[1:(n_age - 1)]))
"""; kind = :udf),
    _bundle_part("update_pregnancy_rate", "Update Pregnancy Rate", raw"""
update_pregnancy_rate(baseline_birth_rate::real, theta0::real, theta1::real,
                      N_tot::real, tau_s::real)::real =
    baseline_birth_rate * exp(theta0 * (1.0 - tau_s * exp(theta1 * N_tot)))
"""; kind = :udf),
    _bundle_part("compute_baseline_birth_rate", "Compute Baseline Birth Rate", raw"""
compute_baseline_birth_rate(min_bbr::real, max_bbr::real,
                            h_int::real, h_slope::real, h_weight::real,
                            h1::vector[T], h2::vector[T2]) = begin
    weighted_h = h_weight * h1 + (1.0 - h_weight) * h2
    min_bbr + (max_bbr - min_bbr) * inv_logit(h_slope * (h_int + weighted_h))
end
"""; kind = :udf),
    _bundle_part("update_population_from_survivors", "Update Population From Survivors", raw"""
update_population_from_survivors(prev_surv::vector[n_demo], aging::matrix[n_demo, n_demo],
                                 birth_rate::real, eps_birth::real, eps_sex::real,
                                 n_age::int) = begin
    N = aging * prev_surv
    N_temp = N[n_age] * birth_rate +
             sqrt(N[n_age] * birth_rate * (1.0 - birth_rate)) * eps_birth
    N[1]         = N_temp / 2.0 + sqrt(N_temp / 4.0) * eps_sex
    N[n_age + 1] = N_temp - N[1]
    N
end
"""; kind = :udf),
    _bundle_part("create_transition_matrix", "Create Transition Matrix", raw"""
create_transition_matrix(hc_sw::vector[n], hc_fi::vector[n], pop::vector[n],
                         hp_sw::vector[n], hp_fi::vector[n], tau_h::real,
                         S_diag::vector[n])::matrix[4 * n, n] = begin
    M_hunted_sw = hc_sw ./ pop
    M_hunted_fi = hc_fi ./ pop
    M_survived  = exp(-(hp_sw + hp_fi) * (tau_h * tau_h) / 2.0) .* S_diag
    M_died      = 1.0 - M_survived - M_hunted_sw - M_hunted_fi
    append_row(diag_matrix(M_survived),
        append_row(diag_matrix(M_died),
            append_row(diag_matrix(M_hunted_sw), diag_matrix(M_hunted_fi))))
end
"""; kind = :udf),
    _bundle_part("multinomial_allocation", "Multinomial Allocation", raw"""
multinomial_allocation(eta_row::row_vector[4], u_row::row_vector[3], N::real)::row_vector[4] = begin
    eta     = eta_row'
    eta_adj = eta * (1.0 + 1.0 / min(eta))
    mean_logratio = (digamma(eta_adj[2:4]) - digamma(eta_adj[1]))'
    Sigma = rep_matrix(trigamma(eta_adj[1]), 3, 3) + diag_matrix(trigamma(eta_adj[2:4]))
    L = cholesky_decompose(Sigma)
    logits = append_col(rep_row_vector(0.0, 1), mean_logratio + u_row * L')
    allocation = softmax(logits')'
    allocation * N
end
"""; kind = :udf),
    _bundle_part("initialize_population", "Initialize Population", raw"""
initialize_population(pop_init::vector[n_demo], pop_init_size::real,
                      burn_in::int, birth_rate_year::real,
                      aging::matrix[n_demo, n_demo], S_diag::vector[n_demo],
                      n_age::int) = begin
    pop = pop_init
    for k in 1:burn_in
        pop = aging * diag_matrix(S_diag) * pop
        pop[1]         = birth_rate_year / 2.0 * pop[n_age]
        pop[n_age + 1] = birth_rate_year / 2.0 * pop[n_age]
    end
    pop * pop_init_size / sum(pop)
end
"""; kind = :udf),
    _bundle_part("run_state_process", "Year-recursive state process", raw"""
run_state_process(n_state_years::int, n_age::int,
                  pop_first::vector[n_demo], birth_rate_first::real, pop_total_first::real,
                  baseline_bbr::vector[Tb], dd_intercept::real, dd_slope::real,
                  aging::matrix[n_demo, n_demo], S_diag::vector[n_demo], mu_m::vector[n_demo],
                  hs_sw::vector[n_demo], hs_fi::vector[n_demo],
                  hq_sw::int[n_state_years], hq_fi::int[n_state_years],
                  he_sd_sw::real, he_sd_fi::real,
                  eps_h_sw::vector[n_state_years], eps_h_fi::vector[n_state_years],
                  t_mate_to_preg::real, t_birth_to_end_hunt::real,
                  eps_birth::vector[n_state_years], eps_sex::vector[n_state_years],
                  transition_noise_raw::matrix[Tn, n_state_years],
                  pi_s::vector[n_state_years], pi_c::vector[n_state_years], prob_of_ca::real,
                  ode_init_state::vector[1], ode_times::vector[1]) = begin

    n_demo = 2 * n_age
    ode_ts = to_array_1d(ode_times)

    birth_rate::vector[n_state_years]
    pregnancy_rate::vector[n_state_years]
    population_total::vector[n_state_years]
    hunted_sweden::matrix[n_demo, n_state_years]
    hunted_finland::matrix[n_demo, n_state_years]
    bycatch_expected::matrix[n_demo, n_state_years]
    hunting_bag_total_sweden::vector[n_state_years]
    hunting_bag_total_finland::vector[n_state_years]
    reproductive_probs::matrix[4, n_state_years]
    population_comp::matrix[n_demo, n_state_years]
    survivors::matrix[n_demo, n_state_years]

    for year in 1:n_state_years
        if year == 1
            birth_rate[year]         = birth_rate_first
            population_comp[:, year] = pop_first
            population_total[year]   = pop_total_first
        else
            birth_rate[year] = update_birth_rate(
                baseline_bbr[year], dd_intercept, dd_slope, population_total[year - 1])
            population_comp[:, year] = update_population_from_survivors(
                survivors[:, year - 1], aging, birth_rate[year],
                eps_birth[year], eps_sex[year], n_age)
            population_total[year] = sum(population_comp[:, year])
        end

        pregnancy_rate[year] = update_pregnancy_rate(
            baseline_bbr[year + 1], dd_intercept, dd_slope,
            population_total[year], t_mate_to_preg)

        hp_sw::vector[n_demo]
        hp_fi::vector[n_demo]
        log_N = log(population_comp[:, year])
        log_denom_sw = log_sum_exp(hs_sw + log_N)
        log_denom_fi = log_sum_exp(hs_fi + log_N)
        if hq_sw[year] == 0
            hp_sw = rep_vector(0.0, n_demo)
        else
            hp_sw = exp(hs_sw + log(hq_sw[year]) + log(2.0)
                        - 2.0 * log(t_birth_to_end_hunt)
                        - eps_h_sw[year] * he_sd_sw - log_denom_sw)
        end
        if hq_fi[year] == 0
            hp_fi = rep_vector(0.0, n_demo)
        else
            hp_fi = exp(hs_fi + log(hq_fi[year]) + log(2.0)
                        - 2.0 * log(t_birth_to_end_hunt)
                        - eps_h_fi[year] * he_sd_fi - log_denom_fi)
        end

        exp_hunted_sw::vector[n_demo]
        exp_hunted_fi::vector[n_demo]
        for demo in 1:n_demo
            # Reference package defaults are literal here because Stan requires
            # solver controls to be data-only and @deffun has no such qualifier yet.
            sol_sw = ode_rk45_tol(dH_dt, ode_init_state, 0.0, ode_ts, 1.0e-6, 1.0e-6, 1000,
                population_comp[demo, year], t_birth_to_end_hunt,
                hp_sw[demo], hp_fi[demo], mu_m[demo])
            exp_hunted_sw[demo] = sol_sw[1][1]
            sol_fi = ode_rk45_tol(dH_dt, ode_init_state, 0.0, ode_ts, 1.0e-6, 1.0e-6, 1000,
                population_comp[demo, year], t_birth_to_end_hunt,
                hp_fi[demo], hp_sw[demo], mu_m[demo])
            exp_hunted_fi[demo] = sol_fi[1][1]
        end

        transition_matrix = create_transition_matrix(
            exp_hunted_sw, exp_hunted_fi, population_comp[:, year],
            hp_sw, hp_fi, t_birth_to_end_hunt, S_diag)
        expected_fate = to_matrix(transition_matrix * population_comp[:, year], n_demo, 4)
        noise_year    = to_matrix(transition_noise_raw[:, year], n_demo, 3)

        realized_fate::matrix[n_demo, 4]
        for demo in 1:n_demo
            realized_fate[demo, :] = multinomial_allocation(
                expected_fate[demo, :], noise_year[demo, :], population_comp[demo, year])
        end

        survivors[:, year]        = realized_fate[:, 1]
        bycatch_expected[:, year] = realized_fate[:, 2]
        hunted_sweden[:, year]    = realized_fate[:, 3]
        hunted_finland[:, year]   = realized_fate[:, 4]
        hunting_bag_total_sweden[year]  = sum(hunted_sweden[:, year])
        hunting_bag_total_finland[year] = sum(hunted_finland[:, year])

        reproductive_probs[2, year] = birth_rate[year] * pi_s[year] * (1.0 - pi_c[year])
        reproductive_probs[3, year] = birth_rate[year] * (1.0 - pi_s[year]) * pi_c[year] +
                                      (1.0 - birth_rate[year]) * prob_of_ca * pi_c[year]
        reproductive_probs[4, year] = birth_rate[year] * pi_s[year] * pi_c[year]
        reproductive_probs[1, year] = 1.0 - sum(reproductive_probs[2:4, year])
    end

    (; birth_rate, pregnancy_rate, population_total,
       hunted_sweden, hunted_finland, bycatch_expected,
       hunting_bag_total_sweden, hunting_bag_total_finland, reproductive_probs)
end
"""; kind = :udf),
    _bundle_part("aerial_count_lpmf", "Aerial Count Lpmf", raw"""
# @slic markers: stanonly, lhs, lpxf
aerial_count_lpmf(obs::int[n_obs], year::int[n_obs],
                             population_total::vector[T], mu::real, phi::real)::real = begin
    neg_binomial_2_lpmf(obs, mu * population_total[year], phi)
end
"""; kind = :udf),
    _bundle_part("aerial_count_lpmfs", "Aerial Count Lpmfs", raw"""
aerial_count_lpmfs(obs::int[n_obs], year::int[n_obs],
                   population_total::vector[T], mu::real, phi::real)::vector[n_obs] = begin
    neg_binomial_2_lpmfs(obs, mu * population_total[year], phi)
end
"""; kind = :udf),
    _bundle_part("aerial_count_rng", "Aerial Count Rng", raw"""
aerial_count_rng(int[n_obs], year::int[n_obs],
                 population_total::vector[T], mu::real, phi::real)::int[n_obs] = begin
    neg_binomial_2_rng(mu * population_total[year], phi)
end
"""; kind = :udf),
    _bundle_part("harvest_bags_lpdf", "Harvest Bags Lpdf", raw"""
# @slic markers: stanonly, lhs, lpxf
harvest_bags_lpdf(obs::vector[n_obs], year::int[n_obs],
                             hunted_total::vector[T], cv::real)::real = begin
    expected = hunted_total[year]
    normal_lpdf(obs, expected, cv * expected)
end
"""; kind = :udf),
    _bundle_part("harvest_bags_lpdfs", "Harvest Bags Lpdfs", raw"""
harvest_bags_lpdfs(obs::vector[n_obs], year::int[n_obs],
                   hunted_total::vector[T], cv::real)::vector[n_obs] = begin
    expected = hunted_total[year]
    normal_lpdfs(obs, expected, cv * expected)
end
"""; kind = :udf),
    _bundle_part("harvest_bags_rng", "Harvest Bags Rng", raw"""
harvest_bags_rng(vector[n_obs], year::int[n_obs],
                 hunted_total::vector[T], cv::real)::vector[n_obs] = begin
    expected = hunted_total[year]
    to_vector(normal_rng(expected, cv * expected))
end
"""; kind = :udf),
    _bundle_part("hunting_comp_lpmf", "Hunting Comp Lpmf", raw"""
# @slic markers: stanonly, lhs, lpxf
hunting_comp_lpmf(obs::int[n_obs, K], year::int[n_obs],
                             hunted::matrix[K, T], hunted_total::vector[T],
                             row_N::int[n_obs])::real = begin
    lp = 0.0
    for i in 1:n_obs
        t = year[i]
        lp += multinomial_lpmf(obs[i, :], hunted[:, t] ./ hunted_total[t])
    end
    lp
end
"""; kind = :udf),
    _bundle_part("hunting_comp_lpmfs", "Hunting Comp Lpmfs", raw"""
hunting_comp_lpmfs(obs::int[n_obs, K], year::int[n_obs],
                   hunted::matrix[K, T], hunted_total::vector[T],
                   row_N::int[n_obs])::vector[n_obs] = begin
    ll::vector[n_obs]
    for i in 1:n_obs
        t = year[i]
        ll[i] = multinomial_lpmf(obs[i, :], hunted[:, t] ./ hunted_total[t])
    end
    ll
end
"""; kind = :udf),
    _bundle_part("hunting_comp_rng", "Hunting Comp Rng", raw"""
hunting_comp_rng(int[n_obs, K], year::int[n_obs],
                 hunted::matrix[K, T], hunted_total::vector[T], row_N::int[n_obs])::int[n_obs, K] = begin
    y::int[n_obs, K]
    for i in 1:n_obs
        t = year[i]
        y[i, :] = multinomial_rng(hunted[:, t] ./ hunted_total[t], row_N[i])
    end
    y
end
"""; kind = :udf),
    _bundle_part("bycatch_comp_lpmf", "Bycatch Comp Lpmf", raw"""
# @slic markers: stanonly, lhs, lpxf
bycatch_comp_lpmf(obs::int[n_obs, K], year::int[n_obs],
                             bycatch_expected::matrix[K, T], bias::vector[K],
                             row_N::int[n_obs])::real = begin
    w = exp(bias)
    lp = 0.0
    for i in 1:n_obs
        t = year[i]
        lp += multinomial_lpmf(obs[i, :],
                (w .* bycatch_expected[:, t]) ./ dot_product(w, bycatch_expected[:, t]))
    end
    lp
end
"""; kind = :udf),
    _bundle_part("bycatch_comp_lpmfs", "Bycatch Comp Lpmfs", raw"""
bycatch_comp_lpmfs(obs::int[n_obs, K], year::int[n_obs],
                   bycatch_expected::matrix[K, T], bias::vector[K],
                   row_N::int[n_obs])::vector[n_obs] = begin
    w = exp(bias)
    ll::vector[n_obs]
    for i in 1:n_obs
        t = year[i]
        ll[i] = multinomial_lpmf(obs[i, :],
                (w .* bycatch_expected[:, t]) ./ dot_product(w, bycatch_expected[:, t]))
    end
    ll
end
"""; kind = :udf),
    _bundle_part("bycatch_comp_rng", "Bycatch Comp Rng", raw"""
bycatch_comp_rng(int[n_obs, K], year::int[n_obs],
                 bycatch_expected::matrix[K, T], bias::vector[K], row_N::int[n_obs])::int[n_obs, K] = begin
    w = exp(bias)
    y::int[n_obs, K]
    for i in 1:n_obs
        t = year[i]
        y[i, :] = multinomial_rng(
            (w .* bycatch_expected[:, t]) ./ dot_product(w, bycatch_expected[:, t]), row_N[i])
    end
    y
end
"""; kind = :udf),
    _bundle_part("pregnancy_lpmf", "Pregnancy Lpmf", raw"""
# @slic markers: stanonly, lhs, lpxf
pregnancy_lpmf(obs::int[n_obs], year::int[n_obs],
                          sample_size::int[n_obs], pregnancy_rate::vector[T])::real = begin
    binomial_lpmf(obs, sample_size, pregnancy_rate[year])
end
"""; kind = :udf),
    _bundle_part("pregnancy_lpmfs", "Pregnancy Lpmfs", raw"""
pregnancy_lpmfs(obs::int[n_obs], year::int[n_obs],
                sample_size::int[n_obs], pregnancy_rate::vector[T])::vector[n_obs] = begin
    binomial_lpmfs(obs, sample_size, pregnancy_rate[year])
end
"""; kind = :udf),
    _bundle_part("pregnancy_rng", "Pregnancy Rng", raw"""
pregnancy_rng(int[n_obs], year::int[n_obs],
              sample_size::int[n_obs], pregnancy_rate::vector[T])::int[n_obs] = begin
    binomial_rng(sample_size, pregnancy_rate[year])
end
"""; kind = :udf),
    _bundle_part("reproductive_signs_lpmf", "Reproductive Signs Lpmf", raw"""
# @slic markers: stanonly, lhs, lpxf
reproductive_signs_lpmf(obs::int[n_obs, 4], year::int[n_obs],
                                   reproductive_probs::matrix[4, T],
                                   row_N::int[n_obs])::real = begin
    lp = 0.0
    for i in 1:n_obs
        lp += multinomial_lpmf(obs[i, :], reproductive_probs[:, year[i]])
    end
    lp
end
"""; kind = :udf),
    _bundle_part("reproductive_signs_lpmfs", "Reproductive Signs Lpmfs", raw"""
reproductive_signs_lpmfs(obs::int[n_obs, 4], year::int[n_obs],
                         reproductive_probs::matrix[4, T],
                         row_N::int[n_obs])::vector[n_obs] = begin
    ll::vector[n_obs]
    for i in 1:n_obs
        ll[i] = multinomial_lpmf(obs[i, :], reproductive_probs[:, year[i]])
    end
    ll
end
"""; kind = :udf),
    _bundle_part("reproductive_signs_rng", "Reproductive Signs Rng", raw"""
reproductive_signs_rng(int[n_obs, 4], year::int[n_obs],
                       reproductive_probs::matrix[4, T], row_N::int[n_obs])::int[n_obs, 4] = begin
    y::int[n_obs, 4]
    for i in 1:n_obs
        y[i, :] = multinomial_rng(reproductive_probs[:, year[i]], row_N[i])
    end
    y
end
"""; kind = :udf),
    _bundle_part("aerial_stream", "Aerial observation stream", raw"""
aerial_stream = begin
    aerial_count_mu             ~ beta(2.0, 2.0)
    aerial_count_overdispersion ~ lognormal(0.0, 1.0)
    obs ~ aerial_count(aerial_year, population_total,
                       aerial_count_mu, aerial_count_overdispersion)
    return obs
end
"""; kind = :anonymous),
    _bundle_part("bycatch_stream", "Bycatch observation stream", raw"""
bycatch_stream = begin
    bycatch_bias :: vector[n_demo] ~ normal(0.0, 1.0)
    obs ~ bycatch_comp(bycatch_comp_year, bycatch_expected, bycatch_bias,
                       bycatch_comp_sample_size)
    return obs
end
"""; kind = :anonymous),
)
# ========================== end verbatim SlicTranspiler source ==========================

"""
    grey_seal_ipm_fixture()

The SlicTranspiler example dataset (3 age classes × 2 sexes × 3 state years) — small,
so the ODE-and-multinomial model transpiles quickly. Verbatim from SlicTranspiler's
`seal_ipm` DataBlock.
"""
grey_seal_ipm_fixture() = (;
    n_age = 3, n_state_years = 3, n_demo = 6, population_burn_in = 2,
    population_init = [0.08, 0.17, 0.25, 0.08, 0.17, 0.25],
    herring_index_1 = [-0.2, 0.0, 0.1, 0.2], herring_index_2 = [0.1, 0.0, -0.1, -0.2],
    hunting_quota_sweden = [40, 45, 50], hunting_quota_finland = [20, 22, 25],
    t_mate_to_preg = 0.4, t_birth_to_end_hunt = 0.75,
    ode_init_state = [0.0], ode_times = [0.75],
    obs_aerial_count = [850, 910, 980], aerial_year = [1, 2, 3],
    obs_hunting_bag_sweden = [38.0, 44.0, 48.0], hunting_bag_year_sweden = [1, 2, 3],
    obs_hunting_bag_finland = [18.0, 21.0, 24.0], hunting_bag_year_finland = [1, 2, 3],
    obs_hunting_comp_sweden = [3 4 5 2 3 1; 4 4 5 2 3 2; 4 5 5 2 3 2],
    hunting_comp_year_sweden = [1, 2, 3], hunting_comp_sample_size_sweden = [18, 20, 21],
    obs_hunting_comp_finland = [2 2 3 1 2 1; 2 3 3 1 2 1; 3 3 4 1 2 1],
    hunting_comp_year_finland = [1, 2, 3], hunting_comp_sample_size_finland = [11, 12, 14],
    obs_bycatch_comp = [1 2 3 1 2 1; 1 2 2 2 2 1; 2 2 3 2 2 1],
    bycatch_comp_year = [1, 2, 3], bycatch_comp_sample_size = [10, 10, 12],
    obs_pregnancy_count = [22, 24, 25], pregnancy_count_year = [1, 2, 3],
    pregnancy_sample_size = [30, 32, 34],
    obs_reproductive_signs_finland = [8 5 4 3; 9 5 4 4; 9 6 5 4],
    reproductive_signs_year = [1, 2, 3], reproductive_signs_sample_size = [20, 22, 24],
)

# Parse a `_bundle_part` source string to its definition AST (comments are dropped).
_grey_seal_part_expr(source) = begin
    ex = Meta.parseall(source)
    ex.head === :toplevel ? only(a for a in ex.args if !(a isa LineNumberNode)) : ex
end

"""
    grey_seal_ipm_bundle([data])

Assemble the grey-seal IPM via StanBlocks' `compile_slic_bundle`, mirroring the
SlicTranspiler worker: each `@deffun` UDF card becomes a `@stanonly` definition (the
six observation-family density cards additionally carry `@lhs @lpxf`, flagged by their
`# @slic markers:` comment), and the two `aerial_stream` / `bycatch_stream` cards are
anonymous Form-A observation submodels. Returns the bundle result — `.model` (traced),
`.code` (generated Stan), `.descriptor`.
"""
function grey_seal_ipm_bundle(data = grey_seal_ipm_fixture())
    udf_defs = Any[]
    anon_defs = Any[]
    for p in GREY_SEAL_IPM_PARTS
        if p.kind == :udf
            markers = occursin("markers: stanonly, lhs, lpxf", p.source) ?
                (:stanonly, :lhs, :lpxf) : :stanonly
            push!(udf_defs, p.slug => (; definition = _grey_seal_part_expr(p.source), markers))
        elseif p.kind == :anonymous
            name, body = _grey_seal_part_expr(p.source).args   # `name = begin … end`
            push!(anon_defs, p.slug => (name => body))
        end
    end
    body_expr = Meta.parse("begin\n" * GREY_SEAL_IPM_SOURCE * "\nend")
    compile_slic_bundle(data, Pair{String,Any}[], "body" => body_expr;
        udf_definitions = udf_defs, anonymous_submodels = anon_defs, name = :grey_seal_ipm)
end
