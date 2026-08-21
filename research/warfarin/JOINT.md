# Joint Warfarin PK/PD model

`joint_reproduce.jl` is a new joint model built from the two public likelihoods
in `reproduce.jl`. It is intentionally separate from the exact public
two-stage reproduction.

## What “joint” means here

The concentration and response data contribute to one posterior. The four
subject-level PK quantities—lag, absorption, clearance, and volume—are sampled
once and feed both:

1. the PK concentration likelihood; and
2. the concentration-driven PD turnover ODE.

Consequently, PK uncertainty propagates into PD, and PD observations may
update the PK parameters. This is the inferential coupling that the public
two-stage program omits when it fixes the PK posterior medians before fitting
PD.

The model retains the source priors and independent Gamma observation models.
PK and PD measurements occur at different times and have different units, so a
residual covariance cannot be added without specifying an observation-time
alignment and a new measurement model. Such a covariance would be an optional
extension, not part of this minimal joint contract.

## Relationship to the public reproduction

- `reproduce.jl`: exact public sequential PK then PD workflow.
- `joint_reproduce.jl`: new single-posterior BRM model with shared latent PK
  quantities.

The joint model does not consume the public PD program's fixed
`pk_log_tlag`, `pk_log_ka`, `pk_log_cl`, or `pk_log_v` inputs. Tests assert
that none of those names enters the emitted Stan program.

## Run

```sh
BRM_WARFARIN_JOINT_RUNTIME=1 julia --startup-file=no --project=test \
  research/warfarin/joint_reproduce.jl
```

Set `BRM_WARFARIN_JOINT_RUNTIME=0` for lowering and `stanc` only.
