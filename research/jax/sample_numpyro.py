"""Sample a BRM model's JAX module with NumPyro NUTS, and evaluate it at given points.

    python sample_numpyro.py MODEL.py INPUTS.json POINTS.json OUT.json \
        [--chains 4] [--warmup 1000] [--samples 1000] [--seed 7]

MODEL.py is the module `jax_source` wrote. INPUTS.json holds the columns
`prepare_data` takes, by name, each as {"dtype", "shape", "data"} with the data in
column-major order (research/jax/reproduce.jl writes it from `rk_bound_data`).
POINTS.json is a list of unconstrained points. OUT.json gets the module's
`log_density` (eager and jitted) and `jax.grad` at each point, the NUTS draws as
chains x draws x dimension (flattened in column-major order), the divergence
count, timings and versions.

Requirements: Python 3.14, jax 0.12.0.1, numpyro 0.22.0, numpy 2.5.4 (the
versions this page was produced with; see results/run.tsv).
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS


def load_module(path):
    spec = importlib.util.spec_from_file_location("brm_jax_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode(column):
    """A bound column as `prepare_data` takes it: a NumPy array in Julia's
    column-major layout, or a list of strings for labels."""
    dtype, shape, data = column["dtype"], tuple(column["shape"]), column["data"]
    if dtype == "str":
        return list(data)
    if dtype == "object":
        return [None if v is None else v for v in data]
    values = [float(v) if isinstance(v, str) else v for v in data]  # "Inf", "NaN" spelled by Julia
    return np.asarray(values, dtype=np.dtype(dtype)).reshape(shape, order="F")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("inputs")
    parser.add_argument("points")
    parser.add_argument("out")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-accept", type=float, default=0.8)
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        module_bytes = f.read()
    with open(args.inputs, "rb") as f:
        input_bytes = f.read()
    signature = hashlib.sha256(module_bytes + input_bytes).hexdigest()

    m = load_module(args.model)
    with open(args.inputs) as f:
        inputs = {k: decode(v) for k, v in json.load(f).items()}
    prepared = m.prepare_data(**{k: inputs[k] for k in m.DATA})
    dimension = int(m.dimension(prepared))

    log_density = jax.jit(lambda u: m.log_density(u, **prepared))
    log_density_arguments = jax.jit(m.log_density)
    gradient = jax.jit(jax.grad(lambda u: m.log_density(u, **prepared)))

    with open(args.points) as f:
        points = [jnp.asarray(u, dtype=jnp.float64) for u in json.load(f)]
    start = time.time()
    log_density(points[0]).block_until_ready()
    first_call_seconds = time.time() - start   # includes the compile
    parity = []
    for u in points:
        eager = float(m.log_density(u, **prepared))
        closed = float(log_density(u))
        arguments = float(log_density_arguments(u, **prepared))
        np.testing.assert_allclose([closed, arguments], eager, rtol=1e-12, atol=1e-12)
        parity.append({
            "log_density_eager": eager,
            "log_density_jit": closed,
            "log_density_jit_arguments": arguments,
            "gradient": np.asarray(gradient(u)).tolist(),
        })

    potential = jax.jit(lambda u: -m.log_density(u, **prepared))
    kernel = NUTS(potential_fn=potential, target_accept_prob=args.target_accept)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                num_chains=args.chains, chain_method="sequential", progress_bar=False)
    rng = np.random.default_rng(args.seed)
    init = jnp.asarray(0.1 * rng.standard_normal((args.chains, dimension)), dtype=jnp.float64)
    if args.chains == 1:
        init = init[0]
    start = time.time()
    mcmc.run(jax.random.PRNGKey(args.seed), init_params=init, extra_fields=("diverging",))
    draws = np.asarray(mcmc.get_samples(group_by_chain=True))          # chains x draws x dimension
    diverging = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
    elapsed = time.time() - start   # NumPy conversion waits for the JAX work.

    untranslated = getattr(m, "UNTRANSLATED", {})
    if not isinstance(untranslated, dict):
        untranslated = {str(k): "" for k in untranslated}
    out = {
        "signature": signature,
        "settings": {"chains": args.chains, "warmup": args.warmup,
                     "samples": args.samples, "seed": args.seed,
                     "target_accept": args.target_accept},
        "versions": {"python": platform.python_version(), "jax": jax.__version__,
                     "numpyro": numpyro.__version__, "numpy": np.__version__,
                     "devices": [str(d) for d in jax.devices()]},
        "dimension": dimension,
        "data_names": list(m.DATA),
        "prepared_names": list(m.PREPARED),
        "untranslated": {str(k): str(v) for k, v in untranslated.items()},
        "first_call_seconds": first_call_seconds,
        "parity": parity,
        "elapsed_seconds": elapsed,
        "shape": list(draws.shape),
        "draws": draws.flatten(order="F").tolist(),
        "divergences": int(diverging.sum()),
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"sampled {args.chains} chains x {args.samples} draws in {elapsed:.1f} s, "
          f"{int(diverging.sum())} divergent transitions; dimension {dimension}")


if __name__ == "__main__":
    main()
