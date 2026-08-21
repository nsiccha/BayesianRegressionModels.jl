# Warfarin PK/PD: two-stage and joint models

This page presents two distinct contracts built from Sebastian Weber's public
StanCon 2018 Warfarin program:

- the [faithful public two-stage workflow](#public-two-stage-workflow), where
  the PK posterior is reduced to fixed subject medians before the PD fit; and
- a [joint PK/PD model](#joint-pkpd-one-posterior), where both likelihoods
  contribute to one posterior over shared latent PK effects.

Every displayed declaration uses the standard four-pane documentation view:
the exact BRM authoring source, the StanBlocks model it emits, generated Stan,
and the selected Turing model. The docs build reads the declarations directly
from the checked-in reproduction scripts, so the displayed BRM source cannot
drift from the executable source. Unsupported backends remain visible and show
their current construction error.

The hidden setup below evaluates the shared public-data fixture, custom
`gamma2_overdisp` distribution, analytical PK helpers, and turnover ODE helper
before the three declarations are rendered.

```@eval
Main.BRMDocsComparisons.evaluate_source_prelude(
    Main.BRMDocsComparisons.example_module(:warfarin),
    "research/warfarin/reproduce.jl";
    before=:warfarin_pk_brmi,
    starting_at="StanBlocks.@deffun begin",
)
nothing
```

## Public two-stage workflow

The public reproduction is a **sequential PK → PD workflow**:

1. Fit the population PK model.
2. Reduce each subject's PK posterior to fixed medians for `log(tlag)`,
   `log(ka)`, allometric `log(CL)`, and allometric `log(V)`.
3. Fit the turnover-PD model conditional on those fixed PK values.

!!! warning "This is not the joint model"

    The PD likelihood does not update the PK posterior. The fixed PK posterior
    medians are inputs to the second fit, exactly as in the public two-stage
    source.

The checked-in reproduction includes the complete StanBlocks helper functions,
custom observation family, public two-subject fixture, and executable
stanc/BridgeStan acceptance harness. Read the
[executable source](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/9fa8f77/research/warfarin/reproduce.jl)
alongside its
[provenance and capability report](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/9fa8f77/research/warfarin/README.md).
The reviewed two-stage source revision is `9fa8f77`.

### Stage 1: population PK

The PK stage is a one-compartment oral model with first-order absorption and a
bounded lag. Clearance and volume use fixed allometric weight effects. Four
independent subject effects modify lag, absorption, clearance, and volume.

The declaration below is extracted verbatim from `warfarin_pk_brmi` in the
reviewed source. Its `kernel(...)` cell evaluates one ragged
concentration-time course per subject.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:warfarin),
    Main.BRMDocsComparisons.source_function(
        "research/warfarin/reproduce.jl", :warfarin_pk_brmi,
    ),
    :warfarin_pk_brmi;
    title="Public Warfarin population PK",
)
```

The model preserves the source priors, `tlagMax = 1`, the `0.75` clearance and
`1.0` volume allometric exponents, and the PK overdispersion multiplier
`5² = 25`.

### The fixed-median handoff

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

### Stage 2: turnover PD

The PD stage solves a turnover ODE conditional on those fixed PK medians. Three
independent subject effects modify baseline response, inverse turnover rate,
and EC50.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:warfarin),
    Main.BRMDocsComparisons.source_function(
        "research/warfarin/reproduce.jl", :warfarin_pd_brmi,
    ),
    :warfarin_pd_brmi;
    title="Public Warfarin turnover PD",
)
```

The helper called by the kernel uses RK45 with `t₀ = -1e-4`, relative
tolerance `1e-5`, absolute tolerance `1e-3`, and at most 500 steps. The PD
overdispersion multiplier is `25² = 625`.

Both stages retain the source observation variance
`sigma² + mu² / kappa`. The reviewed executable passed lowering, stanc, and
finite BridgeStan density/gradient checks for both models.

## Joint PK/PD: one posterior

The joint model combines the two public likelihoods in a single BRM
declaration. It samples each subject's latent lag, absorption, clearance, and
volume effects once and feeds those same quantities into both the PK
concentration kernel and the concentration-driven PD turnover ODE.

That shared latent path creates bidirectional updating: PK uncertainty
propagates into the PD model, while PD observations can update the posterior
for the PK quantities. There is no fixed-median handoff and the joint model
does not consume `pk_log_tlag`, `pk_log_ka`, `pk_log_cl`, or `pk_log_v`.

The PK and PD observations still use independent residual distributions. They
are measured on different time grids and in different units; adding residual
covariance would require a new alignment and measurement model, not merely a
shared posterior. The coupling here is through the shared latent PK effects.

The joint declaration and acceptance harness were reviewed at canonical commit
`f64cf6d291bf40565a6e73299623ea61ead34aa3`. See the
[executable joint source](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/f64cf6d291bf40565a6e73299623ea61ead34aa3/research/warfarin/joint_reproduce.jl)
and its
[joint-model contract](https://github.com/nsiccha/BayesianRegressionModels.jl/blob/f64cf6d291bf40565a6e73299623ea61ead34aa3/research/warfarin/JOINT.md).

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:warfarin),
    Main.BRMDocsComparisons.source_function(
        "research/warfarin/joint_reproduce.jl", :warfarin_joint_brmi,
    ),
    :warfarin_joint_brmi;
    title="Joint Warfarin PK/PD",
)
```

The Turing pane is intentionally retained even though this structural kernel
is outside the current Turing executor. Its build-time construction error is
part of the comparison rather than being hidden.

## Provenance boundary

The public [brms issue](https://github.com/paul-buerkner/brms/issues/1509#issuecomment-1598639613)
mentions a Warfarin PK/PD model but supplies no equations, source, data, or
citation. It therefore cannot establish identity with an unavailable private
model.

The public two-stage sections faithfully reproduce Weber's
[StanCon 2018 Warfarin model directory](https://github.com/stan-dev/stancon_talks/tree/master/2018-helsinki/Contributed-Talks/weber/stancon18-master),
including its separate
[PK](https://github.com/stan-dev/stancon_talks/blob/master/2018-helsinki/Contributed-Talks/weber/stancon18-master/warfarin_pk_tlagMax.stan)
and [PD](https://github.com/stan-dev/stancon_talks/blob/master/2018-helsinki/Contributed-Talks/weber/stancon18-master/warfarin_pd_tlagMax_2.stan)
programs. “Faithful” refers to that public two-stage program; it is not a claim
about unspecified private source. The joint section is explicitly a new model
built from those public likelihoods, not a claim that the public program was
joint.

## Run the complete reproductions

After bootstrapping the repository's test environment, run the public
two-stage checks with:

```sh
BRM_WARFARIN_RUNTIME=1 julia --startup-file=no --project=test \
  research/warfarin/reproduce.jl
```

Run the joint checks with:

```sh
BRM_WARFARIN_JOINT_RUNTIME=1 julia --startup-file=no --project=test \
  research/warfarin/joint_reproduce.jl
```

Set the corresponding runtime variable to `0` to run lowering and `stanc`
without BridgeStan instantiation.
