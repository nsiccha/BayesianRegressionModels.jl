# Julia and JAX regression

This example compares independent fits of a varying-intercept, varying-slope
Gaussian regression on synthetic data. `varying_slopes_data` in `reproduce.jl`
creates 24 groups using seed 2026. Population coefficients, group scales and
the intercept-slope correlation are known in the generating process.

`reproduce.jl` emits a Python module through ReactiveKernelsPPL's `jax_source`,
fits the native target with WarmupHMC and the exported target with NumPyro,
and compares densities and gradients at eight native posterior points.
The two independent fits have different warmup algorithms and starts. A
completion quorum selects the native chains by completion time; diagnostics
do not exclude a resulting selection effect.

Rerunning the native fit requires the pinned test environment's custom
libEnzyme build. `test/setup_env.jl` currently locates that build through the
fleet resolver, so that installer requires additional infrastructure beyond a
fresh public checkout. The model, synthetic data and Python driver are public;
the recorded native fit's missing provenance remains marked unavailable.

```sh
python3 -m venv research/jax/.venv
research/jax/.venv/bin/python -m pip install jax==0.12.0.1 numpyro==0.22.0 numpy==2.5.4
julia --startup-file=no --project=test test/setup_env.jl
RK_JAX_PYTHON="$PWD/research/jax/.venv/bin/python" julia --startup-file=no --threads=1 --project=test research/jax/reproduce.jl
```

Run this deliberately when the model, data or settings change, at most about
once a month. The generated module and full draws are local outputs. The
committed tables contain summaries; each method has its own `population.tsv`,
additional interval statistics, group effects and receipt. Population tables
follow the DocsFits schema for a later transport migration. The legacy run
predates the receipt contract; unavailable provenance is identified explicitly.

With local draw caches retained, `--reuse-draws` rebuilds summaries after checking
the exported program, inputs and sampling settings against the saved run.
The documentation build reads the summaries, regenerates the downloadable module
and emits AlgebraOfVega specifications. It never samples. To write those figure
specifications separately:

```sh
julia --project=docs research/jax/figures.jl /path/to/figure-cache
```

Exporting a module does not establish complete Python-side data preparation:
inspect `UNTRANSLATED`. This model still exports group-level lookup inputs.
The current backend does not emit periodic HSGP bases, dense `gp` covariance
plates, `gmrf`, or every collected plate output. `JAXEmissionError` names an
unsupported construct. The executable coverage record is
`test/rk_jax_models.jl`; `test/rk_jax_inference.jl` also compares independent
posterior samples. The backend guide owns the general export contract.
