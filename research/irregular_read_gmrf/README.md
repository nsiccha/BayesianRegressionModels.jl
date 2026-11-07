# Irregular-read field acceptance

These scripts measure `gmrf(...; multiresolution=30)` on the recovered physical
and rain read supports from the original `neighbors=30` scientific review.
They use the existing range/time quantiles and A2=0.03/A4=1e-6 tolerances.
They do not fit the ODL likelihood or construct an archive covariance matrix.

The model uses the same normalized sparse spatial row at every time. Each
spatial innovation follows a stationary OU path on the union of the actual
times at which it is needed. Thus spatial covariance is approximate and the
temporal factor is exact. The core checks multiply constructed OU transitions;
they do not merely evaluate the target exponential as their answer.

## Input and reproduction

The original input is a Julia `Serialization` file containing `.physical` and
`.rain` named tuples with `east`, `north`, and `time` vectors. It has 685,798
physical and 7,880,793 rain read rows. Coordinates are km and times are hours.
The compressed input is preserved in the private KB:

`/home/niko/.local/state/kb-agents/uploads/b0c9d8de4bd1634b.jlsgz`

Download through `/code/download?path=<that path>` and decompress with gzip.
Keep data and generated receipts outside the repository.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| Compressed input | 25,139,460 | `b36c6e9c59dbc3344cadc7e86070d73006dd712060da60bc8214ea93179063be` |
| Uncompressed input | 205,598,363 | `5d5a162f718b29ae84938c7a31e0a47c7b6153ff7ab05139b623b06a650cc2b0` |

Use the repository's pinned `test/setup_env.jl` environment and the host's
managed compute wrapper. The native controls need ReactiveKernels `c497e3a7`
(RK 0.1.83 / RKPPL 0.1.95) or later for exact reader footprints, and Enzyme
dogfood `7a1f689c` or later: before that repair, static Reverse shadowed the
support's literal dictionary once per reader. The commands below show the Julia payloads; substitute
absolute paths for `REPO`, `SUPPORTS`, and `OUTPUT`. Each cost control runs in a
fresh process with one CPU.

```sh
julia --startup-file=no --project=test research/irregular_read_gmrf/accept_core.jl REPO SUPPORTS OUTPUT.jls
julia --startup-file=no --project=test research/irregular_read_gmrf/paired_support.jl SUPPORTS physical OUTPUT-physical.toml
julia --startup-file=no --project=test research/irregular_read_gmrf/paired_support.jl SUPPORTS rain OUTPUT-rain.toml
julia --startup-file=no --project=test research/irregular_read_gmrf/measure_native_primal.jl SUPPORTS physical 1
julia --startup-file=no --project=test research/irregular_read_gmrf/measure_native_primal.jl SUPPORTS rain 1
julia --startup-file=no --project=test research/irregular_read_gmrf/measure_native_primal.jl SUPPORTS rain 2
julia --startup-file=no --project=test research/irregular_read_gmrf/measure_native_gradient.jl SUPPORTS physical 1
julia --startup-file=no --project=test research/irregular_read_gmrf/measure_native_gradient.jl SUPPORTS rain 1
julia --startup-file=no --project=test research/irregular_read_gmrf/measure_native_gradient.jl SUPPORTS rain 2
```

The core-only drivers can also use a minimal environment containing Bessels
0.2.8; they include the current repository's core source directly. The backend
controls require the full test environment.

## What each check establishes

`accept_core.jl` checks all spatial row variances and sampled spatial pairs at
the 2.5%, median, and 97.5% range quantiles. Its independent OU operator check
uses actual same-position read pairs at one hour, rounded theta, and 72 hours
across all range/time quantiles. Noninteger theta is explicitly labelled as a
rounded lattice lag. Known and sampled two-channel matrices have separate draw
checks. The persisted serialized result contains the spatial and temporal
records; stdout also records channel draws and support sizes.

`paired_support.jl` reproduces the original producer's seed-7 source selection:
60 physical and 45 rain sources, with the same interior/hull split and all six
delivered range/time combinations. It compares sparse covariance columns of
the original representation with the new representation on those same reads.
The TOML receipt retains all source identities, bin counts and errors, and
concrete worst pairs. Each pair includes both endpoint variances, both
source/target-normalized covariance ratios, and the Pearson correlation.
Additional covariance columns compute the endpoint variances exactly. Read
fractions and adjacent gaps describe intermittent support; a missing adjacent
read is encoded as gap -1. Empty interior A2 bins at very large ranges remain
explicitly empty, and are not evidence of zero error. Median A2 bins must be
nonempty. The broad spatial probe supplies additional quantile coverage.

`measure_native_primal.jl` measures construction, query preparation, and density
evaluation for a Normal observation control on every supplied read. It reports
peak process RSS and cumulative allocations per phase. Those two memory
quantities have different meanings. It checks finite values at zero and nonzero
latent vectors and repeatability. It does not measure gradients or fit a target.
Its two-channel control uses independent channels.
`measure_native_gradient.jl` supplies the analogous value/gradient control once
the small native reverse checks pass, with sampled LKJ correlation for multiple
channels. It also compares the native Reverse gradient with central differences
of the native primal along two unit directions: a dense one, and one
concentrated on the hyperparameter and correlation coordinates. Each direction
prints the relative error at steps `1e-2` to `1e-5` and must agree within `1e-4`
at one of them, because rounding in a density of this size divided by the
smallest step can exceed that tolerance. Its existence is not a passed
measurement.

The small backend tests separately cover compiled Stan, sampled LKJ channels,
coordinate transport, nonlinear read locality, multiple fields and replay.
Run `test/gmrf_multiresolution_backends.jl 'Primal coordinate'` for the independent
native-value/compiled-Stan finite-difference control. Passing it does not imply
that the native reverse compiler or locality checks passed.
