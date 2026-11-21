# Spatial and space-time fields

Share information between nearby observations, neighboring areas or related
measurement channels. Choose a representation from the observation geometry
and the predictions you need.

| Data | Term | Representation |
| :--- | :--- | :--- |
| Areas with known neighbors | `car(area; edges=(from,to), type=:bym2)` | Graph field plus independent area effects |
| Smooth continuous coordinates | `hsgp(east,north; k=(kx,ky))` | Fixed Hilbert basis with `kx*ky` weights |
| Points without a time axis | `gmrf((east,north); mesh=...)` | Triangular-mesh Matérn field |
| Points evolving over time | `gmrf((east,north),time; mesh=...)` | Triangular-mesh Matérn field with OU evolution |
| Sparse irregular position–time reads | `gmrf((east,north),time; neighbors=m)` | Local Gaussian conditionals on the observed states |
| Irregular reads needing exact OU time covariance | `gmrf((east,north),time; multiresolution=m)` | Sparse spatial basis of shared OU paths |

Use a common planar distance unit. Longitude and latitude in degrees need
a suitable projection before they represent distances in these terms.

## Areal effects

ICAR shares information through adjacency. BYM2 adds independent area effects
and estimates the structured variance share. The intercept carries the
overall level; an exposure offset keeps the field on a relative-risk scale.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:ArealGuide), raw"""
model = (@brm begin
    log(rate) ~ 1 + offset(log_expected) + car(area; edges=(from, to), type=:bym2)
    cases ~ Poisson(rate)

    effect(rate, Intercept) ~ Normal(0, 1)
    sd(rate, car(area)) ~ Normal(0, 0.5)
    mixing(rate, car(area)) ~ Beta(2, 2)
end)((; area=["a", "b", "c", "d"], cases=[14, 23, 9, 18],
    log_expected=log.([12., 20., 12., 20.]),
    from=["a", "a", "b", "c"], to=["b", "c", "d", "d"]))
""", :model; title="Area counts with a BYM2 field", require_stan=true)
```

List each undirected edge once. `nodes=all_areas` can preserve unobserved
areas and islands. An island has an independent effect under BYM2 and zero
spatial effect under ICAR. Replay keeps the fitted graph. Read `rate` directly:
the declared log link already maps the predictor onto the count scale.

## Smooth surfaces and channels

HSGP uses a squared-exponential covariance, smoother than the Matérn field
below. `iso=true` shares a length across axes; `iso=false` gives each axis
its own. `by=channel` allocates weights per channel, and `cor=Omega`
correlates matching weights between channels.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:SurfaceGuide), raw"""
model = (@brm begin
    mu ~ 0 + hsgp(east, north; k=(6, 6),
        domain=((-25., 125.), (-25., 125.)), by=channel, cor=Omega)
    y ~ Normal(mu, sigma)

    Omega ~ LKJCholesky(2, 2)
    sigma ~ Exponential(0.5)
    sd(mu, hsgp(east, north)) ~ Exponential(1)
    length_scale(mu, hsgp(east, north)) ~ truncated(LogNormal(log(25), 0.5);
        lower=length_scale_floor, upper=300)
end)((; east=[10., 35., 70., 90., 15., 75.], north=[20., 80., 30., 70., 25., 65.],
    channel=["a", "a", "a", "a", "b", "b"], y=[0.3, 0.8, -0.2, 0.4, 0.2, 0.5]))
""", :model; title="Correlated channel surfaces", require_stan=true)
```

The lower bound keeps the length inside the fitted basis's approximation
range. Increase `k` when shorter scales matter. For a grid, replay `east`,
`north` and `channel` without `y`, then read the latent `mu`. Keep the grid
inside the fitted domain; replay retains its basis and channel labels.

## Matérn space and OU time

A mesh is a two-column vertex matrix and an integer cell matrix with three
one-based vertex indices per triangle. The following small mesh illustrates
the declaration; scientific fits need an appropriate boundary buffer and
resolution.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:MeshGuide), raw"""
model = (@brm begin
    field ~ 0 + gmrf((east, north), time, channel;
        mesh=(vertices=xy, cells=triangles), cor=Omega)
    y ~ Normal(field, 0.2)

    Omega ~ LKJCholesky(2, 2)
    sd(field, gmrf((east, north), time, channel)) ~ Exponential(0.7)
    length_scale(field, gmrf((east, north), time, channel), space) ~ LogNormal(log(0.4), 0.4)
    length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(log(1.7), 0.4)
end)((; east=[0.1, 0.9, 0.7, 0.3], north=[0.2, 0.4, 0.8, 0.5],
    time=[0., 0.3, 0.3, 1.7], channel=["b", "a", "b", "a"], y=[0.2, -0.1, 0.3, 0.1],
    xy=[0. 0.; 1. 0.; 1. 1.; 0. 1.; 0.5 0.5],
    triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5]))
""", :model; title="Matérn space, OU time and correlated channels", require_stan=true)
```

The term directly contributes the field, without an extra population
coefficient. Omit `channel` for a single field; omit `cor` for independent
channels. `cor` may also name a known correlation matrix. Replace a supported
term prior by `= value` to fix that quantity.

| Quantity | Interpretation | Units |
| :--- | :--- | :--- |
| `sd` | Nominal continuum marginal SD | Predictor units |
| Spatial `length_scale` | Matérn ν=1 length `ell_s`; practical range is `2ell_s` | Coordinate units |
| Temporal `length_scale` | OU correlation time `ell_t`; a gap of `ell_t` has correlation `exp(-1)` | Time units |

All three default to `LogNormal(0,1)`. Finite-mesh marginal variances depend
on boundary and resolution, so its SD is a continuum parameter, not a
promise of identical vertex variances. The normalized field density includes
the hyperparameter-dependent determinants.

## Matérn space without time

Leave out the time column for a purely spatial field, as in an SPDE model
of a single survey. The field has one state per mesh vertex, an SD and a
spatial length, and no temporal length, channels or `cor`. A
`length_scale` address needs no `space` component, since the field has
only one length.

```@eval
Main.BRMDocsComparisons.comparison(
    Main.BRMDocsComparisons.example_module(:SpatialMeshGuide), raw"""
model = (@brm begin
    mu ~ 1 + gmrf((east, north); mesh=(vertices=xy, cells=triangles))
    y ~ Normal(mu, 0.2)

    sd(mu, gmrf((east, north))) ~ Exponential(0.7)
    length_scale(mu, gmrf((east, north))) ~ LogNormal(log(0.4), 0.4)
end)((; east=[0.1, 0.9, 0.7, 0.3, 0.55], north=[0.2, 0.4, 0.8, 0.5, 0.35],
    y=[0.4, -0.3, 0.2, 0.1, -0.2],
    xy=[0. 0.; 1. 0.; 1. 1.; 0. 1.; 0.5 0.5],
    triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5]))
""", :model; title="A Matérn field on a mesh, without time", require_stan=true)
```

The law is the space-time field's on a single time, where the OU factor
plays no part. `neighbors=m` and `multiresolution=m` work without time as
well, and embedded Laplace integrates the mesh form.

## Representation and prediction

| Representation | State and replay boundary |
| :--- | :--- |
| Mesh | Vertices × fitted times × channels; three barycentric reads per observation. New coordinates may lie inside the mesh, but frozen replay requires fitted times and channels. |
| `neighbors=m` | Local conditionals on distinct observed position–time states. Spatial and temporal covariance are approximated. Frozen replay keeps that support. |
| `multiresolution=m` | Sparse normalized spatial basis with exact OU temporal covariance and marginal SD. Basis paths use their readers' time unions; frozen replay keeps positions and times. |

For `neighbors`, `brm_gmrf_predict` can evaluate new points. Without
an RNG it returns conditional means; pass an RNG to include conditional
new-point variation. These draws have pointwise conditional laws and do
not supply the joint covariance needed for uncertainty in spatial totals.
Observation noise is added only when predicting measurements.

For a model-scope `gmrf(...; neighbors=m)`, the fitted RK graph exposes
`<id>.conditional_mean` and `<id>.conditional_scale`. Entry `k` in each
vector gives the mean and standard deviation of the prior on packed field
coordinate `k`, given the model's length scales, channel correlation and
preceding field coordinates. Channels vary fastest. With correlated channels,
each coordinate also conditions on earlier channels at that state. Multiply
both quantities by the term's marginal field SD to express them in the
predictor's units.

For a single state, `<id>.conditional_layout.inputs[k]` lists the packed parent
indices. The row `<id>.conditional_factor[k]` holds their coefficients in that
order and ends with the conditional standard deviation. The dot product of the
coefficients and the parents' standardized field values gives the conditional mean.

The figure isolates the OU temporal factor at one location and channel,
computed directly as `exp(-gap / ell_t)`. It shows how a fixed prior
correlation time changes dependence; it is not a posterior reconstruction.

```@eval
isdefined(Main, :BRMGuideFigures) || Base.include(Main, joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "docs", "guide_figures.jl"))
Main.BRMDocsComparisons.figure(Main.BRMGuideFigures.temporal_correlation();
    caption="Prior OU temporal correlation at one location and channel, with fixed correlation times 0.5 (blue) and 2 (orange). At a gap equal to its correlation time, each curve equals exp(-1). These curves are computed during the build, without fitting data.")
```

All representations support full-model MCMC on RK and Stan. RK mesh fields
may live in independent plate cells; the cell predictor must be `0 + gmrf`,
and shared channel correlation belongs outside the cell. Neighbor fields
do not use that plate or embedded-Laplace surface. `field_states=:units`
lets supported field states become SVI units; see the [SVI example](variational-inference.md).
An SVI checkpoint with unit-owned states can take new observations at the
fitted positions, times and channels. Retained coordinates keep their owners.
The new window uses the saved joint approximation as its prior. With
`LearnedCentering(; fields=true)` it retains each state's fitted control. A station absent from the window
still contributes draws from its saved factor conditional on the globals when
other stations read its field. Adding field states requires rebuilding the fit.
Embedded Laplace is an explicit mesh integration choice. Large operator
solves use a fixed-probe determinant estimate, so even a Gaussian integral
can be approximate. Check the solver and probe settings against the
accuracy needed for the fit. The
[representation notes](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/research/field_representations)
retain the precision derivations and synthetic benchmark protocols.

Continue to the [spatial-fields example](spatial-fields.md) for a fitted
station field with its diagnostics and maps. For a continuous-time latent
process fitted to discrete observations, continue to the
[state-space example](state-space.md).
