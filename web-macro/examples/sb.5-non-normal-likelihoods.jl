# label: sb.5 non-Normal likelihoods — needs impl
# tier: 2
# status: open
# stages_pass: brmi,parse,slic_model,stan_code,stan_compile,stan_eval,stan_fit_pathfinder,stan_fit_warmup,stan_generate,stan_instantiate,stan_shapes,transform,wrap

log_odds_b ~ 1 + a
bin_y ~ BernoulliLogit(log_odds_b)
