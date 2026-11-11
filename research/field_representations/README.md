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

```julia
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
```

Construct `SBBRMI(small_field)` to prepare the model. The current
[fields guide](../../docs/src/space-time-fields.md) executes a small declaration.
The field is a direct predictor contribution: its amplitude is the addressed
`sd`, without a second population coefficient multiplying it. Add an
intercept explicitly with `1 + gmrf(...)` when the model needs one.

Omit `channel` for a single field. With a channel column, omitting `cor` gives
independent channels conditional on their shared field hyperparameters. Their
identity factor is read from the fitted channels, so models that differ only in
the number of channels build one Stan program and emit one RK program (BRM
0.2.230). From ReactiveKernelsPPL 0.1.113 they also lower to one program.
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
is far larger than the reads, consider a read-support form below.
`multiresolution=` preserves exact temporal covariance on irregular reads.

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

Selection depends only on the set of read pairs and `m`, never on the sampled
lengths or the order of the rows. Only ``b_i`` and ``F_i`` move with them, and reads with the same geometry share
one solve, so a repeated observing pattern costs one solve per position and
gap rather than one per read.

**What it preserves.** The local conditioning matrices use the continuum
Matérn × OU kernel at actual read positions, without mesh interpolation.
Channels are exactly separable. On a synchronized position grid, the joint
covariance retains the exact OU time factor across every gap.

**What it approximates.** The joint covariance implied by the product of
conditionals, including spatial correlations, marginal variances and the
temporal factor on irregular read support. Exact kernel entries in a local
conditional do not imply an exact joint covariance. With intermittent reads,
parent states occur at different times, and even same-position temporal
covariance can depart from the OU exponential. Increasing `m` does not
guarantee smaller error for every pair. Measure the constructed covariance on
the actual read support across the length priors. Distinguish the covariance
ratio ``K_{ij}/K_{jj}`` from the Pearson correlation
``K_{ij}/\sqrt{K_{ii}K_{jj}}`` when variances differ.

A conditioning set holds at most `2m+1` reads when its position was read at the
previous read time, and `3m+1` otherwise. Reads with the same conditioning
geometry share the factorization.

Lengths far beyond the read domain make a conditional correlation numerically
singular, which raises a `DomainError` naming them. Positions a metre apart in a
domain of about 1,000 km first fail at a range of about 5,000 km with a correlation
time of ``10^5`` time units, or a range of ``10^6`` km at any correlation time.

Frozen replay reads only fitted `(position, time)` pairs, in any order and any
subset, and refuses a new position or time: `brm_gmrf_predict` below gives the
conditional law there. It is one model-scope field: use `mesh=` for independent fields
in plate cells. Embedded Laplace does not integrate it.

### At new positions and in minibatch SVI

`brm_gmrf_predict(rk, draws, :field; east, north, time)` gives a read-pair field at
points the fit need not have read, on the scale the model reads it. A point follows the
covariance above given its nearest read positions at the read times around it. With
`rng` each entry is a conditional draw; without it, a conditional mean. Marginals are the
law's. Points are conditionally independent given the states, so a joint quantity over
several new points, such as an areal total, is not.

`brm_svi_fit` takes a read-pair field, each state with its own global Gaussian factor. On
toys with several rows per state that factor stays close to the exact posterior, while
the same factor on a mesh's states collapses the learned lengths, so `brm_svi_fit`
refuses the mesh form. With few rows per state the read-pair form's spatial length can
also come out short, so check the learned lengths against their prior.

With `field_states=:units`, a state read by one unit joins that unit's Gaussian factor,
next to its effects, and a state read by several units stays global. The field's prior
becomes one conditional per state, owned by the state's owner, so the objective is the
same and a minibatch of units estimates it without bias. A step then costs what its
units read: on a synthetic station field it stayed between 0.034 and 0.046 s from 256 to
16,384 states, while the global form rose to 0.60 s. Memory follows the number of units:
with ReactiveKernelsPPL 0.1.116 or later the prepared target holds about 23 KB per unit
on that toy beyond the model's own, where earlier versions copied the model's graph into
each unit's prior query, about 0.8 MB per unit. On those toys the field's mean came closer
to the exact posterior than with the global form, and σ came out low (0.79 to 0.89 for
a truth of 1). A unit with more coordinates than `unit_dense_limit` gets a low-rank
factor, which made σ and the temporal length lower still, so raise the limit to cover
a station's reads when that is affordable. `:units` has no streaming form and needs the
field's lengths and correlation factor to be global. `multiresolution=` runs only under
the global form, where learned lengths make each unit pay for the whole basis.

## Exact time covariance on irregular reads: `multiresolution=`

Use `multiresolution=m` when the read support is irregular and its temporal
covariance must remain exactly OU. It has the same hyperparameter addresses,
priors and length convention as the other forms. Select exactly one of `mesh=`,
`neighbors=`, and `multiresolution=`; the latter two take an integer from 1
through 100, written as a literal or held by a scalar data field or
`name = constant(expr)`, as in `neighbors = nb` with `nb` in the data. Models
that differ only in the count build one Stan program and emit one RK program.

```julia
shared_ou_field = (@brm begin
    Omega ~ LKJCholesky(2, 2.0)
    field ~ 0 + gmrf((east, north), time, channel; multiresolution=3, cor=Omega)
    sd(field, gmrf((east, north), time, channel)) ~ Exponential(0.7)
    length_scale(field, gmrf((east, north), time, channel), space) ~ LogNormal(log(0.4), 0.4)
    length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(log(1.7), 0.4)
    y ~ Normal(field, 0.2)
end)((;
    east=[0.1, 0.9, 0.1, 0.7, 0.9], north=[0.2, 0.4, 0.2, 0.8, 0.4],
    time=[0.0, 0.3, 1.7, 72.0, 72.0], channel=["b", "a", "a", "b", "a"],
    y=[0.2, -0.1, 0.3, 0.1, 0.2],
))
```

Each position uses a normalized sparse spatial basis of independent,
channel-correlated OU innovation paths:

```math
f(s,t,c)=\sigma\sum_k b_{sk}(\ell_s)z_k(t,c),\qquad\sum_k b_{sk}^2=1,
```

```math
\operatorname{Cov}(f(s,t,c),f(s',t',c'))=
\sigma^2(b_s^{\mathsf T}b_{s'})\exp(-|t-t'|/\ell_t)\Omega_{cc'}.
```

The same coefficients apply at every time. The temporal exponential factor,
channel separability and marginal SD ``\sigma`` are exact at all fitted reads.
Spatial covariance remains an approximation to Matérn-1. Local spatial
conditionals in a deterministic maximin ordering define a triangular factor;
its sparse inverse retains the nearest `m` earlier innovations in each
count-doubling band of that ordering. Each row has at most
``1+m(1+\lfloor\log_2 P\rfloor)`` coefficients for ``P`` positions. Supports
overlap across spatial regions. Normalize only after propagating the inverse
rows. Check spatial accuracy on the actual positions across the range prior;
rank increases do not guarantee monotone pointwise improvement.

Each innovation stores only the union of its readers' observed times. Its
initial state is stationary, with exact transitions across the actual gaps.
There is no reset per observation, region or time block, and no completed
position × time grid. Latent dimension can exceed the number of read pairs
because several innovations contribute to each read. The normalized prior on
these innovation coordinates depends on the temporal length and channel
factor; the spatial length enters the deterministic read weights.

The native backend supplies analytic derivatives of the spatial recurrence
and OU density; Stan differentiates the same computations. Frozen replay keeps
innovation position, time and channel identities under row permutations,
subsets and duplicates, and accepts only fitted position/time pairs with known
channels. Coordinate transport pairs these innovation identities across
backends. Read the term output for field values; raw sampler coordinates are
innovation states. For a nonlinear predictor such as log rain, read the
weighted field first, then exponentiate, then average physical values over
observation support.

This is a model-scope field on `RKBRMI` and `SBBRMI`. Independent cell fields
and embedded Laplace retain the mesh form. Very long spatial lengths can make
a local correlation matrix numerically singular; such a failure is reported,
without imposing a range floor or changing the prior.

### Evaluating units with a learned spatial length

With the lengths fixed by `=`, the read weights are data, and a strict partition
reads only each read's innovations. A learned spatial length makes the weights
one whole value per length. They have no row form: a row is built from its
parents' rows, so one row needs the local solves of 40 %, 31 % and 24 % of the
positions at 400, 1,600 and 6,400 random positions (`m = 30`), a count that keeps
growing with the positions.

A partition that declares nothing therefore refuses, with a
`PartitionLocalityError` naming `weights`. `locality = :warn` evaluates anyway by
recomputing the whole basis for every unit, which is quadratic in the positions
over a sweep. Hold the basis as one shared value instead, by passing `partitioned`
and `u0`. ReactiveKernelsPPL then computes it once per length and refreshes it when
the length changes. Name the coordinates the units own, or declare that they own
none (ReactiveKernelsPPL 0.1.111, which BRM's floor includes):

```julia
plan = rk_bound_plan(rk)
# Units own effects:
partitioned = (:unit_effect => unit_labels,)
# Units own nothing, as in a model of the field alone: every coordinate is shared.
partitioned = :none

query = prepare_partition(rk.model, plan, :y; by=(; y=unit), locality=:error, partitioned, u0=u)
query(u, label)
```

That makes a unit's value local, but not its whole gradient.
`partition_value_and_gradient!` includes the spatial length's derivative, so each
call runs one pullback of the whole basis, again quadratic over a sweep. Choose by
what the step needs:

- **A batch of units:** one call for the batch, with one pullback of the basis:
  `partition_batch_value_and_gradient!(sampler, g, u, labels; weights)` returns the
  units' summed value and their summed packed gradient (`g` has the packed length;
  `weights`, one per label, scale each unit's term). A pass over the batch's cells
  does the same at the level of cells, below.
- **A unit's own coordinates, the shared ones held fixed:**
  `partition_value_and_gradient!(sampler, g, u, label; shared=false)`.
- **Every unit:** the whole-model gradient, which also pays one basis derivative
  and is cheaper than a pass over all the cells.

```julia
backend = AutoEnzyme(; mode=Enzyme.Reverse)
sampler = prepare_partition_sampler(rk.model, plan, :y, u; by=(; y=unit), backend,
    locality=:error, partitioned)
value, g = partition_batch_value_and_gradient!(sampler, zeros(length(u)), u, batch_labels)
```

The pass over the batch's cells:

```julia
compact = prepare_compact_cell_ad(ReactiveKernels.prepare_ad, rk.model, plan, :y, u, Int[], backend;
    partitioned, locality=:error)
accumulator = prepare_cell_accumulator(compact, backend, u; order=0)
pullback = prepare_cell_pullback(compact, backend, u)
gradient = similar(u)
total = Ref(0.0)
cell_pullback!(gradient, pullback, u) do batch
    for cell in batch_rows
        total[] += first(accumulate_cell!(batch, accumulator, nothing, 1.0, Float64[], Int[], cell))
    end
end
```

`Int[]` selects no unit-owned coordinates, the only selection checked on this
field. The gradient is the batch's full packed gradient, shared coordinates
included. Queries and samplers hold the shared value, so prepare one per thread.

| Warm cost, one CPU on a shared host | 400 positions | 1,600 positions |
| --- | --- | --- |
| Unit value, `locality = :warn` ¹ | 6.5 ms | 37 ms |
| Unit value, shared basis | 20 to 26 µs | 33 to 64 µs |
| Unit value and gradient | 11.5 ms | 60 to 62 ms |
| Batch call, one unit | 11.6 ms | 64 ms |
| Batch call, 16 units | 13.2 ms | 67 ms |
| Batch call, every unit | 45 ms | 310 ms |
| Cell pass over one unit ¹ | 18 ms | 98 ms |
| Cell pass over every unit ¹ | 156 ms | 952 ms |
| Whole-model value and gradient | 14.6 ms | 83 ms |

The basis pullback dominates: a batch of 16 units costs about as much as one
unit's value and gradient, so about 14 times less than 16 unit calls. The toy
has one Normal read per row, one unit per position, 6 times and `m = 30`.
¹ Measured on BRM 0.2.217 (ReactiveKernels 0.1.86, ReactiveKernelsPPL 0.1.100).
The other rows were measured on BRM 0.2.241 (ReactiveKernels 0.1.104,
ReactiveKernelsPPL 0.1.114), on a Linux fleet host at load 15 to 18 of 16 cores, with
units owning nothing (`partitioned = :none`).
| Whole-model value and gradient | 29 ms | 125 ms |

These are from a toy: `m = 30`, about 4 reads per position, one unit per position,
one Normal observation per read, ReactiveKernels 0.1.86 and ReactiveKernelsPPL
0.1.100.

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
