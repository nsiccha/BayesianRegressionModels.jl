# Faithful two-stage Warfarin PK → PD workflow

This example reproduces Sebastian Weber's public StanCon 2018 Warfarin program
as two BRM models. It is a **sequential PK → PD workflow**:

- **PK fit.** Fit the population PK model.
- **Fixed-median handoff.** Reduce each subject's PK posterior to fixed medians
  for `log(tlag)`, `log(ka)`, allometric `log(CL)`, and allometric `log(V)`.
- **PD fit.** Fit the turnover-PD model conditional on those fixed PK values.

!!! warning "This is not the separately specified joint model"

    The PD likelihood does not update the PK posterior. The fixed PK posterior
    medians are inputs to the second fit, exactly as in the public two-stage
    source. A joint PK/PD model is a different model and is not presented here.

The checked-in reproduction includes the complete StanBlocks helper functions,
custom `gamma2_overdisp` likelihood, public two-subject fixture, and executable
stanc/BridgeStan acceptance harness. Read the
[executable source](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/warfarin/reproduce.jl)
alongside its
[provenance and capability report](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/warfarin/README.md).
The reviewed source revision is `9fa8f77`.

## Stage 1: population PK

The PK stage is a one-compartment oral model with first-order absorption and a
bounded lag. Clearance and volume use fixed allometric weight effects. Four
independent subject effects modify lag, absorption, clearance, and volume.

The block below is reproduced verbatim from `warfarin_pk_brmi` in the reviewed
source. Its `kernel(...)` cell evaluates one ragged concentration-time course
per subject.

```@eval
Main.BRMDocsComparisons.source_excerpt(raw"""
function warfarin_pk_brmi(data = warfarin_fixture())
    return @brm data begin
        sigma_pk ~ Normal(0.0, 2.0; lower=0.0)
        kappa_pk ~ Gamma(0.2, 5.0)

        lag_logit ~ 1 + (1 | tlag_bsv | subject)
        log_ka ~ 1 + (1 | ka_bsv | subject)
        log_cl0 ~ 1 + (1 | cl_bsv | subject)
        log_v0 ~ 1 + (1 | v_bsv | subject)

        effect(lag_logit, :) ~ Normal(0.0, 2.0)
        effect(log_ka, :) ~ Normal(log(1.0), log(2.0) / 1.96)
        effect(log_cl0, :) ~ Normal(log(0.1), log(10.0) / 1.96)
        effect(log_v0, :) ~ Normal(log(10.0), log(10.0) / 1.96)
        sd(:, tlag_bsv) ~ Normal(0.0, 0.5)
        sd(:, ka_bsv) ~ Normal(0.0, 0.5)
        sd(:, cl_bsv) ~ Normal(0.0, 0.5)
        sd(:, v_bsv) ~ Normal(0.0, 0.5)

        pk_pred ~ kernel(
            pk_time, dose, log_weight_ratio, pk_dv,
            lag_logit, log_ka, log_cl0, log_v0,
        ) do times, dose_i, log_weight_i, observed,
             lag_i, lka_i, lcl_i, lv_i
            log_tlag_i = log_inv_logit(lag_i)
            log_cl_i = lcl_i + 0.75 * log_weight_i
            log_v_i = lv_i + log_weight_i
            prediction = exp(warfarin_pk_logconcentration(
                times, log(dose_i), lka_i, log_cl_i, log_v_i, log_tlag_i,
            )) + 1e-5
            observed ~ warfarin_gamma2_overdisp(
                prediction, sigma_pk, kappa_pk * 25.0,
            )
            prediction
        end
    end
end
""")
```

The model preserves the source priors, `tlagMax = 1`, the `0.75` clearance and
`1.0` volume allometric exponents, and the PK overdispersion multiplier
`5² = 25`.

## The fixed-median handoff

The second fit receives four values per subject:

- `pk_log_tlag`
- `pk_log_ka`
- `pk_log_cl`
- `pk_log_v`

They are posterior medians from the PK fit, already transformed and—with
clearance and volume—already including the weight effects. They are ordinary
fixed data columns in the PD model, not sampled parameters shared between the
two likelihoods. The small executable fixture uses the corresponding medians
from the public PD model's Stan data dump.

## Stage 2: turnover PD

The PD stage solves a turnover ODE conditional on those fixed PK medians. Three
independent subject effects modify baseline response, inverse turnover rate,
and EC50.

```@eval
Main.BRMDocsComparisons.source_excerpt(raw"""
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
""")
```

The helper called by the kernel uses RK45 with `t₀ = -1e-4`, relative
tolerance `1e-5`, absolute tolerance `1e-3`, and at most 500 steps. The PD
overdispersion multiplier is `25² = 625`.

Both stages retain the source observation variance
`sigma² + mu² / kappa`. The reviewed executable passed lowering, stanc, and
finite BridgeStan density/gradient checks for both models.

## Provenance boundary

The public [brms issue](https://github.com/paul-buerkner/brms/issues/1509#issuecomment-1598639613)
mentions a Warfarin PK/PD model but supplies no equations, source, data, or
citation. It therefore cannot establish identity with an unavailable private
model.

This page faithfully reproduces the strongest identifiable public match:
Weber's [StanCon 2018 Warfarin model directory](https://github.com/stan-dev/stancon_talks/tree/master/2018-helsinki/Contributed-Talks/weber/stancon18-master),
including its separate
[PK](https://github.com/stan-dev/stancon_talks/blob/master/2018-helsinki/Contributed-Talks/weber/stancon18-master/warfarin_pk_tlagMax.stan)
and [PD](https://github.com/stan-dev/stancon_talks/blob/master/2018-helsinki/Contributed-Talks/weber/stancon18-master/warfarin_pd_tlagMax_2.stan)
programs. “Faithful” refers to that public two-stage program; it is not a claim
about unspecified private source.

## Run the complete reproduction

After bootstrapping the repository's test environment, run:

```sh
BRM_WARFARIN_RUNTIME=1 julia --startup-file=no --project=test \
  research/warfarin/reproduce.jl
```

Set `BRM_WARFARIN_RUNTIME=0` to run lowering and stanc without BridgeStan
instantiation.
