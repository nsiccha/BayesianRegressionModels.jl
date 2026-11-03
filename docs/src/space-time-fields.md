# Space-time Gaussian Markov fields

`gmrf((east, north), time, channel; mesh, cor=Omega)` contributes a spatial
Matérn field with stationary Ornstein–Uhlenbeck (OU) evolution in time. Each
observation reads three vertices of a triangular mesh at its own time and
channel. The latent field has a normalized joint Gaussian prior, including
the determinants that depend on its spatial length, temporal length and
channel correlation.

## Declare the field and its priors

Supply planar coordinates and a triangular mesh in the same units. The vertex
matrix has two columns; each row of the integer cell matrix contains three
one-based vertex indices. The example uses a tiny mesh to show the syntax.
A mesh for scientific use needs a resolution and boundary buffer appropriate
to the spatial range prior.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__, raw"""
small_field = (@brm begin
    Omega ~ LKJCholesky(2, 2.0)
    field ~ 0 + gmrf((east, north), time, channel;
        mesh=(vertices=xy, cells=triangles), cor=Omega)
    sd(field, gmrf((east, north), time, channel)) ~ Exponential(0.7)
    length_scale(field, gmrf((east, north), time, channel), space) ~ LogNormal(log(0.4), 0.4)
    length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(log(1.7), 0.4)
    y ~ Normal(field, 0.2)
end)((;
    east=[0.1, 0.9, 0.7, 0.3],
    north=[0.2, 0.4, 0.8, 0.5],
    time=[0.0, 0.3, 0.3, 1.7],
    channel=["b", "a", "b", "a"],
    y=[0.2, -0.1, 0.3, 0.1],
    xy=[0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0; 0.5 0.5],
    triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5],
))
""", :small_field; title="Matérn space × OU time × correlated channels", require_stan=true)
```

The StanBlocks and Stan panes are generated from this declaration during the
documentation build. Construct `SBBRMI(small_field)` to prepare the same model.
The field is a direct predictor contribution: its amplitude is the addressed
`sd`, without a second population coefficient multiplying it. Add an
intercept explicitly with `1 + gmrf(...)` when the model needs one.

Omit `channel` for a single field. With a channel column, omitting `cor` gives
independent channels conditional on their shared field hyperparameters.
`cor=Omega` accepts either a known correlation matrix in the data or an
`Omega ~ LKJCholesky(K, eta)` declaration. Its dimension must match the fitted
channel count. A matrix supplied as data contains correlations; a sampled
`LKJCholesky` declaration supplies their lower Cholesky factor.

The three term parameters are positive and default to `LogNormal(0, 1)`:

| Address | Meaning | Units |
| --- | --- | --- |
| `sd(field, gmrf((east, north), time, channel))` | Nominal continuum marginal SD, ``\sigma`` | Predictor units |
| `length_scale(field, gmrf((east, north), time, channel), space)` | Spatial length, ``\ell_s`` | Coordinate units |
| `length_scale(field, gmrf((east, north), time, channel), time)` | OU correlation time, ``\ell_t`` | Time units |

Replace `~ Prior(...)` by `= value` to hold one of these parameters fixed.
The spatial and temporal components must be addressed explicitly. The two
coordinate axes use one isotropic spatial length; this term does not infer
separate east–west and north–south length scales.

## Covariance and length convention

The spatial Matérn smoothness is ``\nu=1``. BRM uses

```math
\kappa = \frac{\sqrt{2}}{\ell_s},
\qquad
\rho = \frac{\sqrt{8}}{\kappa} = 2\ell_s,
```

where ``\rho`` is the conventional practical range. Thus a prior expressed
for ``\rho`` must be converted to one on ``\ell_s=\rho/2``. For example,
``\log\rho\sim\mathcal N(m,s^2)`` corresponds to
``\log\ell_s\sim\mathcal N(m-\log 2,s^2)``. The time length is already the OU
correlation time: a gap of ``\ell_t`` has correlation ``e^{-1}``.

In the infinite-domain continuum, the covariance between positions a distance
``r`` apart, time gap ``d`` apart and channels ``c,c'`` is

```math
\sigma^2\,\Omega_{cc'}\,
  (\kappa r)K_1(\kappa r)\,\exp(-|d|/\ell_t),
```

with the spatial factor defined as one at ``r=0``. ``K_1`` is the modified
Bessel function of the second kind. The finite mesh approximates this spatial
covariance; its boundary and resolution affect both variances and correlations.

For the linear triangular finite elements, let ``C`` be the diagonal lumped
mass matrix and ``G`` the stiffness matrix. The standardized spatial precision
used by the implementation is

```math
K = \kappa^2 C + G,
\qquad
Q_s = \frac{K C^{-1}K}{4\pi\kappa^2}.
```

The mesh uses natural Neumann boundary conditions. Consequently, ``\sigma``
is a nominal continuum SD, not a guarantee of exactly that marginal SD at
every finite-mesh vertex or interpolated observation. Check interior and
boundary covariance over the spatial range prior, refine the mesh where
needed, and extend it beyond the observation hull. BRM does not silently
truncate the range prior to fit the mesh or renormalize each vertex separately.

For fitted times ``t_1<\cdots<t_T``, the standardized vertex-by-channel field
``F_t`` evolves as

```math
\begin{aligned}
\operatorname{vec}(F_1)&\sim\mathcal N(0,\Omega\otimes Q_s^{-1}),\\
F_t&=a_t F_{t-1}+\sqrt{1-a_t^2}\,E_t,
\qquad a_t=\exp(-(t_t-t_{t-1})/\ell_t),\\
\operatorname{vec}(E_t)&\overset{\mathrm{iid}}\sim
    \mathcal N(0,\Omega\otimes Q_s^{-1}).
\end{aligned}
```

This includes the stationary initial distribution and exact OU transitions
over irregular gaps. For barycentric weights ``w_{ij}``, observation ``i``
reads ``\sigma\sum_{j=1}^{3}w_{ij}F_{t_i}[v_{ij},c_i]``. Observations are
not aggregated to mesh vertices.

## Fitted support and replay

The fitted state contains every mesh vertex at every fitted channel and
sorted distinct observed time. Vertex order comes from the supplied mesh;
channel and time labels identify the other axes. Rows may share time and
channel values, and their input order does not determine latent coordinates.
An observation outside the supplied triangulation is rejected.

`reprocess(sb, new_rows)` preserves the fitted mesh, time lattice, channel
axis and priors by default. New observation coordinates inside the mesh may
change the three vertex reads. Reordered rows and subsets of fitted times or
channels retain the same latent field. A previously unseen time or channel
requires rebuilding the model; frozen replay does not append states or
interpolate in time.

## Cost of the representation

With ``N`` vertices, ``C_h`` channels and ``T`` fitted times, there are
``N C_h T`` latent field coordinates. Sparse spatial precision and local
observation reads reduce the cost of working with the prior, but do not
remove this state dimension. The joint prior uses spatial factors and OU
transitions without constructing a dense space-time covariance.

Posterior inference must also account for the observation model and shared
parameters. Sparse posterior factorization can have substantial fill, and a
fast precision application alone does not establish scalable marginalization.

## Independent fields in plate cells

Write the term inside `@plate for p in groups(g)` to give each cell its own
field. Every cell keeps the shared mesh but fits its own times and channels
from its rows, and its term hyperparameters are its own:

```julia
@plate for p in groups(window)
    field ~ 0 + gmrf((east[p], north[p]), time[p], channel[p];
        mesh=(vertices=xy, cells=triangles), cor=Omega)
    sd(field, gmrf((east[p], north[p]), time[p], channel[p])) ~ Exponential(0.7)
    y[p] ~ Normal(field, 0.2)
end
```

A prior may read a model-scope parameter or a cell-local one declared in the
same cell (`tau ~ Exponential(s)`, then `sd(...) ~ Exponential(tau)`). A cell
declaration never takes caller data of the same name: on SBBRMI it is emitted
as a StanBlocks `local` binding. Cells are independent conditional on the
shared hyperparameters; use one model-scope field when cells must be
correlated. On the ReactiveKernels backend the cell formula must be
`0 + gmrf(...)` alone, a correlation declared inside the cell is not lowered,
and replay keeps the fit's cells.

## Embedded Laplace

On `RKBRMI`, `brm_laplace_problem(rk; ad_backend, integrate=(declaration,))`
integrates a model-scope field, where `declaration` is the field's RK
declaration (`gmrf_<predictor>_<axes…>`, as `brm_laplace_plan` lists it), and
`brm_laplace_plan(rk; ad_backend, integrate=:auto)` admits it. With Normal
rows the integrated target is the exact marginal. The field's precision moves
with its lengths and channel factor, and the target's gradient includes their
log-determinant terms. Applying a plan to such a target (which copies its
prepared problem) and integrating per-cell fields wait on two ReactiveKernels
repairs. Measured cost grows superlinearly with the number of field states
under both exact and matrix-free algebra (about 12–15 s per value and
gradient at 51,200 states on one or two CPUs), dominated by space-time
factorization and log-determinant work; large fields are not yet a supported
Laplace workload.
