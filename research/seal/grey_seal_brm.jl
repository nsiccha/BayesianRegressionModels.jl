# The FULL grey-seal IPM on the `@brm` FORMULA surface — a faithful port of the
# verbatim SlicTranspiler research model (`grey_seal_ipm.jl`), hoisting every
# regression / random-effect / covariate piece onto brms-style formulas while the
# mechanistic core stays a `@deffun` scan.
#
# What is a brms FORMULA here (hoisted from the @slic parent):
#   - birth-rate covariate REGRESSION ... `bbr_logit ~ 1 + herring_index_1 + herring_index_2`
#     (compute_baseline_birth_rate's logistic-on-herring), bounded in the scan;
#   - per-year process/effort RANDOM EFFECTS with ESTIMATED group SDs (the @slic
#     `epsilon_*` fixed-unit innovations become proper REs): `eps_birth ~ 0 + (1|year)`,
#     `eps_sex`, `eps_h_sw`, `eps_h_fi`, `eps_ca`, `eps_placental`;
#   - per-demographic-class hunting SELECTIVITY + bycatch bias: `hs_sw ~ 0 + (1|demo)`,
#     `hs_fi`, `bycatch_bias`;
#   - per-cell transition process NOISE: `tnoise ~ 0 + (1|noise_cell)` (crossed cell RE).
#
# What stays a `@deffun` scan (the carried-state recurrence — correct home, as CDC's
# renewal is): the VERBATIM `run_state_process` (Leslie aging, size-structured
# mortality, density-dependent births, a within-year hunting ODE `ode_rk45_tol` on
# `dH_dt`, and a `multinomial_allocation` logistic-normal survived/bycatch/hunted split)
# and its 12 verbatim UDFs. `seal_state` wraps the parent's transformed quantities +
# the scan; it returns a NamedTuple of carriers CONSUMED VIA FIELD ACCESS.
#
# All EIGHT observation streams are `@brm` family likelihoods reading `state.<field>`
# indexed to each stream's observation years — on their own frames (multi-frame is
# fine on the formula surface): aerial pup counts (`NegativeBinomial2`), Swedish/Finnish
# harvest bags (`Normal`, cv-scaled sd), Swedish/Finnish hunting age-composition and
# bycatch composition and reproductive-signs composition (per-row `Multinomial`), and
# pregnancy counts (`Binomial`).
#
# Verified (test/seal_brm.jl): transpile + `stanc` + `compiles`, matching the `@slic`
# reference's verification level (`grey_seal_ipm.jl`). Like the reference it is NOT
# sampled — the mechanistic scan's simplex constraints make a naive init degenerate for
# both forms; the deliverable is the model. Illustrative dataset (3 age classes x 2
# sexes x 3 state years), verbatim from the SlicTranspiler `seal_ipm` fixture.

using BayesianRegressionModels, StanBlocks, LogDensityProblems
using Distributions: Uniform, Cauchy, LogNormal, Binomial, Multinomial
import StanBlocks.stan: transpiles

StanBlocks.@deffun begin
dH_dt(tau::real, H::vector[ny], n0::real, k::real,
      E_1::real, E_2::real, mu::real)::vector[ny] = begin
    surv = exp(-(E_1 + E_2) * (k * tau - tau * tau / 2) - mu * tau)
    rep_vector(n0 * E_1 * surv * (k - tau), 1)
end
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
compute_density_dependence_intercept(max_bbr::real, dd_scaled::real)::real =
    -log(max_bbr + (1.0 - max_bbr) * dd_scaled)
update_birth_rate(b0::real, theta0::real, theta1::real, N_prev::real)::real =
    b0 * exp(-theta0 * (exp(theta1 * N_prev) - 1.0))
birth_rate_at_carrying_capacity(phi_a::real, mu_m::vector[n_demo], n_age::int)::real =
    2.0 * (1.0 - phi_a) / exp(sum(-mu_m[1:(n_age - 1)]))
update_pregnancy_rate(baseline_birth_rate::real, theta0::real, theta1::real,
                      N_tot::real, tau_s::real)::real =
    baseline_birth_rate * exp(theta0 * (1.0 - tau_s * exp(theta1 * N_tot)))
compute_baseline_birth_rate(min_bbr::real, max_bbr::real,
                            h_int::real, h_slope::real, h_weight::real,
                            h1::vector[T], h2::vector[T2]) = begin
    weighted_h = h_weight * h1 + (1.0 - h_weight) * h2
    min_bbr + (max_bbr - min_bbr) * inv_logit(h_slope * (h_int + weighted_h))
end
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
    # ---- wrapper: transformed quantities (verbatim from the @slic parent) + the scan.
    # Consumes the brms-formula-built LPs/REs (bbr_logit, eps_*, hs_*, tnoise) + scalars.
    seal_state(bbr_logit::vector[Tb], eps_birth::vector[T], eps_sex::vector[T],
               eps_h_sw::vector[T], eps_h_fi::vector[T], eps_ca::vector[T], eps_placental::vector[T],
               hs_sw::vector[nd], hs_fi::vector[nd], tnoise::vector[Tn],
               phi_a_sc::real, phi_sc::real, survival_shape::real,
               male_pup_offset::real, male_adult_offset::real, carrying_capacity::real,
               max_bbr::real, min_bbr_sc::real, he_sd_sw::real, he_sd_fi::real, pop_init_size::real,
               report_ca_mean::real, report_placental_mean::real, report_ca_sd::real,
               report_placental_sd::real, prob_of_ca::real,
               population_init::vector[nd], hq_sw::int[T], hq_fi::int[T],
               t_mate::real, t_hunt::real, burn_in::int, n_age::int,
               ode_init::vector[1], ode_times::vector[1]) = begin
        n_demo = nd
        phi_a = phi_a_sc
        phi_pup = phi_sc * phi_a
        mu_m = mortality_rates(phi_pup, phi_a, survival_shape, n_age, male_pup_offset, male_adult_offset)
        S_diag = exp(-mu_m)
        aging = create_aging_matrix(n_demo, n_age)
        min_bbr = min_bbr_sc * max_bbr
        baseline_bbr = min_bbr + (max_bbr - min_bbr) * inv_logit(bbr_logit)
        dd_scaled = birth_rate_at_carrying_capacity(phi_a, mu_m, n_age)
        dd_intercept = compute_density_dependence_intercept(max_bbr, dd_scaled)
        dd_slope = -log(carrying_capacity)
        pop_first = initialize_population(population_init, pop_init_size, burn_in,
                                          baseline_bbr[1], aging, S_diag, n_age)
        pi_s = report_placental_mean * exp(-eps_placental * report_placental_sd)
        pi_c = report_ca_mean * exp(-eps_ca * report_ca_sd)
        transition_noise_raw = to_matrix(tnoise, 3 * n_demo, T)
        run_state_process(T, n_age, pop_first, baseline_bbr[1], sum(pop_first),
            baseline_bbr, dd_intercept, dd_slope, aging, S_diag, mu_m,
            hs_sw, hs_fi, hq_sw, hq_fi, he_sd_sw, he_sd_fi, eps_h_sw, eps_h_fi,
            t_mate, t_hunt, eps_birth, eps_sex, transition_noise_raw,
            pi_s, pi_c, prob_of_ca, ode_init, ode_times)
    end

    # observation-mean / composition-probs helpers (read state carriers at obs years).
    aerial_mean(mu::real, poptot::vector[T], years::int[na])::vector[na] = mu * poptot[years]
    harvest_mean(total::vector[T], years::int[nb])::vector[nb] = total[years]
    harvest_sd(total::vector[T], years::int[nb], cv::real)::vector[nb] = cv * total[years]
    comp_hunted(hunted::matrix[K, T], total::vector[T], years::int[nc])::matrix[nc, K] = begin
        p::matrix[nc, K]
        for i in 1:nc; p[i, :] = (hunted[:, years[i]] ./ total[years[i]])'; end
        p
    end
    comp_bycatch(bycatch::matrix[K, T], bias::vector[K], years::int[nc])::matrix[nc, K] = begin
        w = exp(bias)
        p::matrix[nc, K]
        for i in 1:nc; p[i, :] = ((w .* bycatch[:, years[i]]) ./ dot_product(w, bycatch[:, years[i]]))'; end
        p
    end
    comp_repro(rp::matrix[4, T], years::int[nc])::matrix[nc, 4] = begin
        p::matrix[nc, 4]
        for i in 1:nc; p[i, :] = rp[:, years[i]]'; end
        p
    end
end

function grey_seal_brm_fixture()
    n_state_years = 3; n_demo = 6
    base = (; n_age = 3, n_state_years, n_demo, population_burn_in = 2,
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
        reproductive_signs_year = [1, 2, 3], reproductive_signs_sample_size = [20, 22, 24])
    # grouping columns for the brms REs (year / demo / transition-noise frames)
    merge(base, (; year = collect(1:n_state_years), demo = collect(1:n_demo),
        noise_cell = collect(1:(3 * n_demo * n_state_years))))
end

"""
    grey_seal_brm_model([df]) -> BRMI

The full grey-seal IPM on the `@brm` formula surface (see the block comment above for
the seam-by-seam mapping), returned as a `BRMI`; lower with
`SBBRMI(grey_seal_brm_model())`.
"""
function grey_seal_brm_model(df = grey_seal_brm_fixture())
    @brm df begin
    phi_a_sc ~ Uniform(0.0, 1.0); phi_sc ~ Uniform(0.0, 1.0); survival_shape ~ Uniform(0.0, 1.0)
    male_pup_survival_offset ~ Cauchy(0.0, 1.0); male_adult_survival_offset ~ Cauchy(0.0, 1.0)
    carrying_capacity ~ LogNormal(11.0, 1.0)
    max_baseline_birth_rate ~ Uniform(0.0, 1.0); min_baseline_birth_rate_sc ~ Uniform(0.0, 1.0)
    hunting_effort_sd_sweden ~ Cauchy(0.0, 1.0; lower = 0.0); hunting_effort_sd_finland ~ Cauchy(0.0, 1.0; lower = 0.0)
    population_init_size ~ LogNormal(11.0, 1.0)
    report_ca_mean ~ Uniform(0.0, 1.0); report_placental_mean ~ Uniform(0.0, 1.0); prob_of_ca ~ Uniform(0.0, 1.0)
    report_placental_sd ~ Normal(0.0, 0.1; lower = 0.0); report_ca_sd ~ Normal(0.0, 0.1; lower = 0.0)
    aerial_mu ~ Uniform(0.0, 1.0); phi_aerial ~ LogNormal(0.0, 1.0)
    harvest_bag_cv ~ LogNormal(0.0, 1.0; lower = 0.0)
    # brms formulas
    bbr_logit ~ 1 + herring_index_1 + herring_index_2
    eps_birth ~ 0 + (1 | year); eps_sex ~ 0 + (1 | year)
    eps_h_sw ~ 0 + (1 | year); eps_h_fi ~ 0 + (1 | year)
    eps_ca ~ 0 + (1 | year); eps_placental ~ 0 + (1 | year)
    hs_sw ~ 0 + (1 | demo); hs_fi ~ 0 + (1 | demo)
    bycatch_bias ~ 0 + (1 | demo)
    tnoise ~ 0 + (1 | noise_cell)
    state = seal_state(bbr_logit, eps_birth, eps_sex, eps_h_sw, eps_h_fi, eps_ca, eps_placental,
        hs_sw, hs_fi, tnoise, phi_a_sc, phi_sc, survival_shape,
        male_pup_survival_offset, male_adult_survival_offset, carrying_capacity,
        max_baseline_birth_rate, min_baseline_birth_rate_sc,
        hunting_effort_sd_sweden, hunting_effort_sd_finland, population_init_size,
        report_ca_mean, report_placental_mean, report_ca_sd, report_placental_sd,
        prob_of_ca, population_init, hunting_quota_sweden, hunting_quota_finland,
        t_mate_to_preg, t_birth_to_end_hunt, population_burn_in, n_age, ode_init_state, ode_times)
    obs_aerial_count ~ NegativeBinomial2(aerial_mean(aerial_mu, state.population_total, aerial_year), phi_aerial)
    obs_hunting_bag_sweden ~ Normal(harvest_mean(state.hunting_bag_total_sweden, hunting_bag_year_sweden), harvest_sd(state.hunting_bag_total_sweden, hunting_bag_year_sweden, harvest_bag_cv))
    obs_hunting_bag_finland ~ Normal(harvest_mean(state.hunting_bag_total_finland, hunting_bag_year_finland), harvest_sd(state.hunting_bag_total_finland, hunting_bag_year_finland, harvest_bag_cv))
    obs_hunting_comp_sweden ~ Multinomial(hunting_comp_sample_size_sweden, comp_hunted(state.hunted_sweden, state.hunting_bag_total_sweden, hunting_comp_year_sweden))
    obs_hunting_comp_finland ~ Multinomial(hunting_comp_sample_size_finland, comp_hunted(state.hunted_finland, state.hunting_bag_total_finland, hunting_comp_year_finland))
    obs_bycatch_comp ~ Multinomial(bycatch_comp_sample_size, comp_bycatch(state.bycatch_expected, bycatch_bias, bycatch_comp_year))
    obs_pregnancy_count ~ Binomial(pregnancy_sample_size, state.pregnancy_rate[pregnancy_count_year])
    obs_reproductive_signs_finland ~ Multinomial(reproductive_signs_sample_size, comp_repro(state.reproductive_probs, reproductive_signs_year))
end
end
