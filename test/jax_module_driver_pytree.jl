# test/jax_module_driver_pytree.jl — `jax_module_driver.jl` takes prepared
# values that are JAX pytrees, not one array each.
#
# A `jax_source` module's `prepare_data` passes an array of tuples or records
# as the tuple of its columns (ReactiveKernels `_rk_columns_np`, RK 0.1.252 and
# later), and a padded record's lengths as a tuple with `None` for a number
# field. Such columns differ in shape, so `np.shape` of the whole tuple raises
# `ValueError: ... inhomogeneous shape` (BRM snag `jax-module-drive-2c1ee23a`,
# hit by Bruno's `web-pkpd/test/brm_jax_status.jl`). The driver records each
# array's shape in the value's own structure and evaluates the functions. A
# `StaticInt` (RK's shape-setting `int`, a pytree node with no children) is a
# leaf of shape `[]`, not its value.
#
# The module below is hand-written in that shape, so the file needs only JSON
# and a Python with `jax` (`RK_JAX_PYTHON`), not ReactiveKernels:
#
#     RK_JAX_PYTHON=<jaxenv>/bin/python BRM_JAX_REQUIRE=1 julia --project=test test/jax_module_driver_pytree.jl
using Test, JSON
include(joinpath(@__DIR__, "jax_module_driver.jl"))

const PYTREE_MODULE = raw"""
import numpy as np
import jax
import jax.numpy as jnp

# A shape-setting integer, a pytree node with no children, as RK emits it.
class StaticInt(int):
    pass

jax.tree_util.register_pytree_node(StaticInt, lambda s: ((), int(s)), lambda v, _: StaticInt(v))

DATA = ["n"]
PREPARED = ["design", "design_lengths", "pairs", "y", "n_static"]
INPUTS = ["n"]
DIMENSION = 3
UNTRANSLATED = {}

def prepare_data(n):
    i = np.arange(int(n))
    j10, j4 = np.arange(10), np.arange(4)
    # A vector of records: two vector fields padded to widths 10 and 4, and a
    # number field. Their lengths: one array per vector field, None for the
    # number field.
    design = (((i[:, None] * 10 + j10[None, :]) % 7) / 7.0,
              ((i[:, None] * 4 + j4[None, :]) % 5) / 5.0,
              i / float(n))
    design_lengths = (1 + i % 10, 1 + i % 4, None)
    pairs = (i.astype(np.float64), (i ** 2).astype(np.float64))  # same-shape columns
    return dict(design=design, design_lengths=design_lengths, pairs=pairs, y=np.cos(i),
                n_static=StaticInt(n))

def _masked_sum(x, lengths):
    return jnp.sum(x * (jnp.arange(x.shape[1])[None, :] < lengths[:, None]), axis=1)

def _residual(u, design, design_lengths, pairs, y, n_static):
    a, b, c = design
    la, lb, _ = design_lengths
    p0, p1 = pairs
    eta = u[0] * _masked_sum(a, la) + u[1] * _masked_sum(b, lb) + u[2] * c + 1e-3 * (p0 - 0.5 * p1)
    return y[:n_static] - 0.1 * eta   # a static slice under jit

def log_prior(u, design, design_lengths, pairs, y, n_static):
    return -0.5 * jnp.sum(u ** 2)

def log_likelihood(u, design, design_lengths, pairs, y, n_static):
    return -0.5 * jnp.sum(_residual(u, design, design_lengths, pairs, y, n_static) ** 2)

def log_jacobian(u, design, design_lengths, pairs, y, n_static):
    return jnp.asarray(0.0)

def log_density(u, design, design_lengths, pairs, y, n_static):
    return (log_prior(u, design, design_lengths, pairs, y, n_static)
            + log_likelihood(u, design, design_lengths, pairs, y, n_static))

def constrain(u, design, design_lengths, pairs, y, n_static):
    return u

def dimension(prepared):
    return 3
"""

# The same density in Julia, as an independent reference.
function pytree_reference(n, u)
    i = 0:n-1
    a = [((ii * 10 + j) % 7) / 7 for ii in i, j in 0:9]
    b = [((ii * 4 + j) % 5) / 5 for ii in i, j in 0:3]
    sa = [sum(a[k, 1:1+(i[k]%10)]) for k in 1:n]
    sb = [sum(b[k, 1:1+(i[k]%4)]) for k in 1:n]
    c = i ./ n
    X = hcat(sa, sb, c)
    r = cos.(i) .- 0.1 .* (X * u .+ 1e-3 .* (i .- 0.5 .* i .^ 2))
    prior, lik = -0.5 * sum(abs2, u), -0.5 * sum(abs2, r)
    (; prior, lik, density=prior + lik, grad=-u .+ 0.1 .* (X' * r))
end

@testset "jax_module_driver: prepared values that are pytrees" begin
    if brm_jax_require_python()
        n = 12
        us = [[0.1, -0.2, 0.3], [0.0, 0.5, -0.1]]
        run = brm_jax_run(PYTREE_MODULE, [(; n)], [us])
        d = only(run.datasets)
        # Each leaf's shape, in the value's own structure; `None` stays `nothing`.
        @test d["prepared"]["design"] == [[n, 10], [n, 4], [n]]
        @test d["prepared"]["design_lengths"] == [[n], [n], nothing]
        @test d["prepared"]["pairs"] == [[n], [n]]
        @test d["prepared"]["y"] == [n]   # a lone array keeps its plain shape list
        @test d["prepared"]["n_static"] == []   # a StaticInt is a leaf, not its value
        @test d["dimension"] == 3
        for (u, p) in zip(us, d["points"])
            ref = pytree_reference(n, u)
            for (f, want) in (("log_prior", ref.prior), ("log_likelihood", ref.lik),
                              ("log_density", ref.density), ("log_jacobian", 0.0))
                @test p[f]["eager"] ≈ want rtol = 1e-12
                @test p[f]["jit"] ≈ want rtol = 1e-12
                @test p[f]["jit_args"]["ok"]
                p[f]["jit_args"]["ok"] && @test p[f]["jit_args"]["value"] ≈ want rtol = 1e-12
            end
            @test p["grad"] ≈ ref.grad rtol = 1e-12
            @test p["grad_eager"] ≈ ref.grad rtol = 1e-12
            @test p["grad_args"]["ok"]
            p["grad_args"]["ok"] && @test p["grad_args"]["value"] ≈ ref.grad rtol = 1e-12
            @test p["constrain_eager"] ≈ u
        end
    end
end
