# epidist reference: simulate the "approximate inference" article's data,
# prepare epidist's marginal model, export its brms-generated Stan program,
# the Stan data and the aggregated marginal rows, and time epidist's own fit.
#
#   Rscript research/epidist/epidist_reference.R <outdir> <sample_size> <seed>
#
# Library: epidist 1.0.0, brms and cmdstanr, optionally from an extra R library
# named by R_EPIDIST_LIB; CmdStan from CMDSTAN (default ~/.cmdstan/cmdstan-2.40.0).
args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[[1]] else "research/epidist/results/n200"
sample_size <- if (length(args) >= 2) as.integer(args[[2]]) else 200L
seed <- if (length(args) >= 3) as.integer(args[[3]]) else 101L
lib <- Sys.getenv("R_EPIDIST_LIB", "")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({
  library(epidist); library(dplyr); library(cmdstanr); library(jsonlite)
})
cmdstan <- Sys.getenv("CMDSTAN", path.expand("~/.cmdstan/cmdstan-2.40.0"))
set_cmdstan_path(cmdstan)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# --- simulation, as in epidist's "Approximate Bayesian inference" article ---
meanlog <- 1.8; sdlog <- 0.5; obs_time <- 25
outbreak <- simulate_gillespie(seed = seed)
obs <- outbreak |>
  simulate_secondary(meanlog = meanlog, sdlog = sdlog) |>
  mutate(ptime_lwr = floor(.data$ptime), ptime_upr = .data$ptime_lwr + 1,
         stime_lwr = floor(.data$stime), stime_upr = .data$stime_lwr + 1,
         obs_time = obs_time)
truncated_obs <- obs |> filter(.data$stime_upr <= .data$obs_time)
set.seed(seed)
n_take <- min(sample_size, nrow(truncated_obs))
sampled <- truncated_obs |> slice_sample(n = n_take, replace = FALSE)
linelist <- as_epidist_linelist_data(
  sampled$ptime_lwr, sampled$ptime_upr, sampled$stime_lwr, sampled$stime_upr,
  obs_time = sampled$obs_time)
marginal <- as_epidist_marginal_model(linelist)

# --- the exact program and data epidist hands to Stan ---
code <- epidist(data = marginal, fn = brms::make_stancode)
sdata <- epidist(data = marginal, fn = brms::make_standata)
writeLines(code, file.path(outdir, "epidist_marginal.stan"))
cmdstanr::write_stan_json(unclass(sdata)[names(sdata)], file.path(outdir, "epidist_marginal.data.json"))
cols <- c("delay_lwr", "delay_upr", "pwindow", "swindow", "relative_obs_time", "delay_min", "n")
rows <- as.data.frame(marginal)[, intersect(cols, names(marginal))]
write_json(list(rows = rows, n_linelist = nrow(linelist), n_rows = nrow(rows),
                n_available = nrow(truncated_obs), truth = list(meanlog = meanlog, sdlog = sdlog),
                seed = seed, sample_size = n_take,
                versions = list(epidist = as.character(packageVersion("epidist")),
                                brms = as.character(packageVersion("brms")),
                                primarycensored = as.character(packageVersion("primarycensored")),
                                cmdstan = cmdstanr::cmdstan_version())),
           file.path(outdir, "marginal_rows.json"), digits = NA, auto_unbox = TRUE)
cat("rows", nrow(rows), "from", nrow(linelist), "linelist rows\n")

# --- epidist's own fit (brms + cmdstanr, 4 chains x (1000 warmup + 1000 draws), one core: chains run in sequence) ---
# The wall time includes brms's Stan compilation; Stan's per-chain times do not.
t_fit <- system.time(fit <- epidist(data = marginal, chains = 4, cores = 1, iter = 2000,
                                    seed = 1, backend = "cmdstanr", refresh = 0))[["elapsed"]]
draws <- posterior::as_draws_df(fit)
summ <- posterior::summarise_draws(posterior::subset_draws(draws, c("b_Intercept", "b_sigma_Intercept")))
times <- rstan::get_elapsed_time(fit$fit)
write_json(list(wall_s = t_fit,
                stan_warmup_s = as.numeric(times[, "warmup"]), stan_sample_s = as.numeric(times[, "sample"]),
                summary = as.data.frame(summ),
                divergences = rstan::get_num_divergent(fit$fit)),
           file.path(outdir, "epidist_fit.json"), digits = NA, auto_unbox = TRUE)
print(summ)
cat("wall (incl. compile)", t_fit, "s; chain totals", rowSums(times), "\n")
