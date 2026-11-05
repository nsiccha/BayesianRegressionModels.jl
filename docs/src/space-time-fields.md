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
transitions without constructing a dense space-time covariance. When this state
is far larger than the reads, use `neighbors=` below.

Posterior inference must also account for the observation model and shared
parameters. Sparse posterior factorization can have substantial fill, and a
fast precision application alone does not establish scalable marginalization.
The Embedded Laplace section below reports what its operator algebra measures at
large sizes.

## A field too large for a mesh: `neighbors=`

Replace `mesh` by `neighbors=m` when the mesh × time state is too large and the
rows read far fewer position and time pairs than that product:

```julia
field ~ 0 + gmrf((east, north), time, channel; neighbors=20, cor=Omega)
```

The field has the same three parameters, addresses, priors and length
convention. Its latent state is one coordinate per channel at each distinct
`(east, north, time)` the rows read, so it is bounded by the data rather than
by vertices × times. A row reads its own state directly.

Instead of a finite-element precision, the joint law is a product of Gaussian
conditionals under the exact covariance of the section above, the continuum
``(\kappa r)K_1(\kappa r)\exp(-|d|/\ell_t)``:

```math
x_i\mid x_{N(i)}\sim\mathcal N(b_i^{\mathsf T}x_{N(i)},F_i),
\qquad b_i=K_{N(i)N(i)}^{-1}K_{N(i)i},
\qquad F_i=1-K_{iN(i)}b_i.
```

Its precision ``(I-B)^{\mathsf T}F^{-1}(I-B)`` is sparse. Reads are ordered by
time, then by one greedy maximin position order for the whole support. The
conditioning set ``N(i)`` of a read holds:

- its `m` nearest lower-order positions read at its own time;
- the previous read of its own position and of each of those neighbours;
- when its own position was not read at the previous read time, its `m` nearest
  among the latest earlier reads, reaching back until they hold at least `m`.

Selection depends only on the coordinates and `m`, never on the sampled lengths.
Only ``b_i`` and ``F_i`` move with them, and reads with the same geometry share
one solve, so a repeated observing pattern costs one solve per position and
gap rather than one per read.

**What it preserves.** The spatial correlation is the continuum Matérn at the
actual positions: there is no mesh boundary inflation, no resolution floor at
short ranges and no interpolation within triangles. Channels are exactly
separable (``\Omega\otimes K``). On positions read at the same times, the
implied covariance is the spatial approximation times the exact OU correlation
across every gap. No time gap is cut: every read time conditions on earlier
reads however long ago they were.

**What it approximates.** The conditioning set. Implied covariances differ from
the exact covariance by an error that shrinks with `m`, is largest at distances
near the range when the range is much longer than the read spacing, and grows
for pairs linked only through a chain of sparsely read times. Measure it for your
support and your range prior: on a network of about 1,600 positions, a range
about five times the typical spacing has a largest covariance error of about
0.04 at `m=20` and 0.015 at `m=30`. A conditioning set holds at most `2m+1`
reads when its position was read at the previous read time, and `3m+1` otherwise.

Lengths far beyond the read domain make a conditional correlation numerically
singular, which raises a `DomainError` naming them. Positions a metre apart in a
domain of about 1,000 km first fail at a range of about 5,000 km with a correlation
time of ``10^5`` time units, or a range of ``10^6`` km at any correlation time.

Frozen replay reads only fitted `(position, time)` pairs, in any order and any
subset. A new position or time needs a conditional prediction law and is
refused. It is one model-scope field: use `mesh=` for independent fields in
plate cells. Embedded Laplace does not integrate it.

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
`brm_laplace_plan(rk; ad_backend, integrate=:auto)` admits recognized affine
Normal rows, or eligible count rows that pass the local ESS gate. From BRM
0.2.166, arbitrary plate cells and model-value observations require explicit
selection: automatic planning cannot certify their conditional log-concavity.
Concavity does not itself guarantee Gaussian approximation accuracy. With
affine Normal rows and exact algebra the integrated target is the exact marginal.
The field's precision moves
with its lengths and channel factor, and the target's gradient includes their
log-determinant terms. `brm_laplace_problem(rk; ad_backend, integrate=plan)`
applies the plan to the same target, and `deepcopy` gives each chain its own
problem. `integrate=(declaration,)` also integrates one field per
`@plate for p in groups(g)` cell, each cell's rows reading only that cell's
field. Applying an explicitly selected plan and `deepcopy`
also work for these fields. Automatic selection retains the structural
eligibility rule above; a passing local diagnostic does not admit a cell.

Above 1,024 coordinates `solver=:auto` uses operator algebra, which needs no
space-time factorization. Its determinant is an estimate from fixed probes, so
even with Normal rows the target is then approximate. A component holding one
field uses a reference with one sparse spatial factor per channel and time, which
sees the data's curvature and stays fixed. The estimate is the difference from that
reference rebuilt at each point, whose own determinant is exact, and the gradient
is the exact derivative of the fixed-probe value. Fields read by the same rows are
coupled through one of them. Measured per warm value and gradient, one CPU of a
loaded host, one Normal field at its prior medians:

| States | Vertices × times × channels | Probes | Operator algebra | Exact |
|:--|:--|:--|:--|:--|
| 10,000 | 400 × 25 × 1 | 8 | 0.28 s | 0.61 s |
| 51,200 | 1,024 × 50 × 1 | 8 | 3.0 s | 9.4–19 s |
| 202,500 | 2,025 × 100 × 1 | 8 | 13 s | not run |
| 640,000 | 1,600 × 400 × 1 | 8 | 51 s | not run |
| 810,000 | 8,100 × 100 × 1 | 4 | 87 s | not run |
| 1,698,450 | 4,225 × 402 × 1 | 4 | 134 s | not run |
| 1,920,000 | 1,600 × 400 × 3 | 4 | 262 s | not run |

Work grows about linearly in times and channels, and faster than linearly in
vertices, because each spatial factor does. A point far from the prior medians
costs about three to four times as much. At 10,000 states and 8 probes the value and
gradient agreed with exact within their probe error, for Normal and Poisson rows,
and the gradient's error across probe seeds was about 0.3 to 0.6 posterior standard
deviations of each hyperparameter. That ratio does not shrink as a field grows, so
compare with more probes before relying on a fit. Correlated channels read at
different places, and the scales of fields read by the same rows, have several times
that error. Sizes beyond the table have not been measured.
