# Wastewater-based Rt inference — a faithful port of the EpiSewer / CDC
# `ww-inference-model` semi-mechanistic renewal model, expressed with BRM's
# StanBlocks backend (`@slic` + `@deffun`), authorized by decision
# `2026-08-26T10-40-17-350-1kyw2vk` / `2026-08-26T11-06-02-112-12s1pu8`.
#
# The model has three EpiSewer modules:
#
#   1. INFECTION MODEL — a renewal process. Latent incidence follows
#          I(t) = Rt(t) · Σ_{s=1}^{ng} g(s) · I(t-s),
#      seeded by a per-site initial level `I0` over the first window. `g` is the
#      (fixed, known) generation-interval PMF, passed as DATA. The reproduction
#      number is a Gaussian random walk on the log scale (EpiSewer's smoothing
#      prior family): logRt = logR0 + cumsum(eps)·sigma_rw.
#
#   2. SHEDDING LOAD MODEL — infections are convolved with a fecal shedding-load
#      PMF `sh` (also DATA):
#          load(t) = Σ_{s=1}^{nsh} sh(s) · I(t-s+1).
#
#   3. MEASUREMENT MODEL — observed log-concentration is Normal on the log load
#      plus a load→concentration scale (per-case shedding / flow):
#          log C(t) ~ Normal(log load(t) + log_scale, sigma_obs).
#      `wastewater_censored_model` swaps in left-censoring below a log limit of
#      detection (`lloq`) for non-detects, which real wastewater series carry.
#
# Multi-site: sites share sigma_rw / sigma_obs / log_scale / logR0; each site has
# its own random-walk Rt trajectory and seeding, broadcast by `plate`.
#
# The two convolutions (the renewal recurrence and the shedding kernel) are the
# CARRIED-STATE part that `plate` deliberately cannot express — they live in
# `@deffun` Stan functions, called loop-free from the `@slic`/plate body. The
# generation interval and shedding PMF are data captured into the plate cell, so
# they are re-bindable without editing the model (no baked-in constants).
#
# Verified: transpile + stanc + BridgeStan finite density/gradient; see
# `test/wastewater_model.jl`. NOT sampled — the deliverable is the model, per the
# resolving user comment on `2026-08-26T11-06-02-112-12s1pu8`.

using BayesianRegressionModels
using StanBlocks

StanBlocks.@deffun begin
    # Renewal recurrence. `g` is the generation-interval PMF (length ng); `I0`
    # seeds the pre-series window. Carried-state sequential scan: I[t] reads the
    # ng previously-computed infections, so it is a @deffun, not a plate cell.
    ww_renewal(logRt::vector[nt], g::vector[ng], I0::real)::vector[nt] = begin
        I::vector[nt]
        for t in 1:nt
            acc = 0.0
            for s in 1:ng
                prev = t - s >= 1 ? I[t - s] : I0
                acc = acc + g[s] * prev
            end
            I[t] = exp(logRt[t]) * acc
        end
        I
    end

    # Shedding-load convolution. `sh` is the shedding-load PMF (length nsh);
    # load(t) sums shedding from infections on days t, t-1, …, t-nsh+1.
    ww_shed(I::vector[nt], sh::vector[nsh])::vector[nt] = begin
        load::vector[nt]
        for t in 1:nt
            acc = 0.0
            for s in 1:nsh
                idx = t - s + 1
                contrib = idx >= 1 ? I[idx] : 0.0
                acc = acc + sh[s] * contrib
            end
            load[t] = acc
        end
        load
    end
end

"""
    wastewater_fixture(; nsites = 3, nt = 28)

A minimal multi-site wastewater dataset for exercising the model: `nsites`
sites, `nt` daily observations each. `ww` is a ragged per-site vector of observed
**log**-concentrations (one synthetic epidemic bump). `g` / `sh` are the fixed
generation-interval and shedding-load PMFs (data). `lloq` is a log limit of
detection for the censored variant.
"""
function wastewater_fixture(; nsites = 3, nt = 28)
    g  = [0.05, 0.15, 0.25, 0.22, 0.16, 0.10, 0.07]
    sh = [0.02, 0.08, 0.15, 0.18, 0.17, 0.13, 0.10, 0.08, 0.06, 0.03]
    ww = [[log(50.0) + 1.4 * exp(-((t - 14.0) / 6)^2) + 0.1 * sin(t / 3) for t in 1:nt]
          for _ in 1:nsites]
    (; ww = ww, g = g, sh = sh, nt = nt, ng = length(g), nsh = length(sh),
       nsites = nsites, lloq = log(20.0))
end

"""
    wastewater_model([data])

Core EpiSewer-style wastewater→Rt model: per-site random-walk log-Rt, renewal
infection process with a data generation interval, shedding-load convolution,
and a lognormal wastewater-concentration likelihood. Returns a re-bindable
`@slic` model; swap data with `wastewater_model()(; ww = new_ww, …)`.
"""
function wastewater_model(data = wastewater_fixture())
    @slic data begin
        sigma_rw  ~ normal(0.0, 0.2; lower = 0.0)   # RW step SD on log-Rt
        sigma_obs ~ normal(0.0, 1.0; lower = 0.0)   # measurement noise (log scale)
        log_scale ~ normal(0.0, 1.0)                # load -> concentration scaling
        logR0     ~ normal(0.0, 0.5)                # shared initial log-Rt
        conc ~ plate(ww; outer = (nsites,)) do wi
            log_I0 ~ normal(2.0, 1.0)               # per-site seeding (log infections)
            eps::vector[nt] ~ std_normal()          # RW innovations
            logRt = logR0 + cumulative_sum(eps) * sigma_rw
            infections = ww_renewal(logRt, g, exp(log_I0))
            load = ww_shed(infections, sh)
            wi ~ normal(log(load) + log_scale, sigma_obs)
            load
        end
    end
end

"""
    wastewater_censored_model([data])

As `wastewater_model`, but observations below the log limit of detection `lloq`
are left-censored (`censored(normal, …; lower = lloq)`) — the non-detect handling
real wastewater series need. `data.lloq` sets the threshold.
"""
function wastewater_censored_model(data = wastewater_fixture())
    @slic data begin
        sigma_rw  ~ normal(0.0, 0.2; lower = 0.0)
        sigma_obs ~ normal(0.0, 1.0; lower = 0.0)
        log_scale ~ normal(0.0, 1.0)
        logR0     ~ normal(0.0, 0.5)
        conc ~ plate(ww; outer = (nsites,)) do wi
            log_I0 ~ normal(2.0, 1.0)
            eps::vector[nt] ~ std_normal()
            logRt = logR0 + cumulative_sum(eps) * sigma_rw
            infections = ww_renewal(logRt, g, exp(log_I0))
            load = ww_shed(infections, sh)
            wi ~ censored(normal, log(load) + log_scale, sigma_obs; lower = lloq)
            load
        end
    end
end

# --- the same model on the `@brm` formula surface -----------------------------
# Since the kernel-cell data-vector capture landed, the generation-interval `g`
# and shedding PMF `sh` reach the cell as DATA (referenced by name), so the model
# reads as an ordinary `@brm` formula: a smooth log-Rt (`hsgp`), a per-site
# seeding ranef, and the renewal/shedding `@deffun`s in the kernel cell.

"""
    wastewater_brm_fixture(; nsites = 3, nt = 28)

Dataset for the `@brm` form: per-site ragged observed-log-concentration series
(`dv`) and a matching time grid (`t_grid`), a flat time frame (`time_x` with
`time_site` naming each row's site) for the smooth log-Rt, and the fixed
generation-interval / shedding PMFs `g` / `sh` as shared data captured into the
kernel cell.
"""
function wastewater_brm_fixture(; nsites = 3, nt = 28)
    sites = ["site$(i)" for i in 1:nsites]
    g  = [0.05, 0.15, 0.25, 0.22, 0.16, 0.10, 0.07]
    sh = [0.02, 0.08, 0.15, 0.18, 0.17, 0.13, 0.10, 0.08, 0.06, 0.03]
    (;
        site = sites,
        t_grid = [collect(1.0:nt) for _ in sites],
        dv = [[log(50.0) + 1.4 * exp(-((t - 14.0) / 6)^2) + 0.1 * sin(t / 3) for t in 1:nt]
              for _ in sites],
        time_x = reduce(vcat, [collect(1.0:nt) for _ in sites]),
        time_site = reduce(vcat, [fill(s, nt) for s in sites]),
        g = g, sh = sh,
    )
end

"""
    wastewater_brm_model([data])

The EpiSewer-style renewal model as a single `@brm` formula. `g` and `sh` are
referenced directly in the kernel do-block and captured as shared data (no baked
`@deffun` constants). Returns a `BRMI`; lower with `SBBRMI(wastewater_brm_model())`.
"""
function wastewater_brm_model(data = wastewater_brm_fixture())
    @brm data begin
        sigma_obs ~ Exponential(1.0)                 # measurement noise (log scale)
        log_scale ~ Normal(0.0, 1.0)                 # load -> concentration scaling
        log_rt ~ 1 + hsgp(time_x; k = 10)            # smooth log-Rt over time
        log_I0 ~ 1 + (1 | site)                      # per-site seeding
        conc ~ kernel(t_grid, dv, ragged(log_rt, time_site), log_I0) do ts, yy, logRt_i, lI0
            infections = ww_renewal(logRt_i, g, exp(lI0))   # g captured as data
            load = ww_shed(infections, sh)                   # sh captured as data
            yy ~ normal(log(load) + log_scale, sigma_obs)
            load
        end
    end
end
