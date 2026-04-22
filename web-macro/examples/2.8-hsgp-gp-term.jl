# label: 2.8 HSGP gp(x; k, c) — Hilbert-space approx Gaussian process
# tier: 2
# status: done (vimpl); sb backend open
#=
**Status: done in vimpl.** `gp(x; k=K, c=C)` adds a 1D Hilbert-space approx GP
term to the linear predictor (Riutort-Mayol et al. 2020). Squared-exponential
kernel, population-level, data-x only.

**Semantics.** `loc ~ 1 + a + gp(b; k=20, c=1.5)` adds a smooth function of `t`
to the predictor without committing to a functional form. Parameters (per gp()
call): log_rho (lengthscale), log_sigma (marginal SD), and K latent betas —
total K+2 unconstrained reals. Priors are Normal(0,1) on all three groups
(=> rho, sigma ~ LogNormal(0,1); beta_k ~ Normal(0,1)).

**Implementation sketch** (vimpl.jl):
- `_hsgp_basis(raw, K, c)` precomputes `PHI::(n,K)` and `lambda::K` from
  centered x (L = c * maximum(abs, x_centered); lambda[k] = (k*pi/(2L))^2;
  PHI[i,k] = (1/sqrt(L)) * sin(sqrt(lambda[k]) * (x_centered[i] + L))).
- New Part kind `hsgp` with `(; PHI, lambda, sqrt_spd, beta, contrib)`.
- `lprior!(::Part{hsgp}, x)` reads K+2 params, refreshes
  `sqrt_spd[k] = sigma * (2pi)^(1/4) * sqrt(rho) * exp(-0.25 * rho^2 * lambda[k])`
  and `contrib = PHI * (sqrt_spd .* beta)`, returns Normal(0,1) log-priors.
- Parser returns `Base.broadcasted(identity, contrib)` so the linear predictor
  just adds the per-draw `contrib` vector in.

**Parameter-x extension.** If `x` itself is a Stan parameter (measurement-error
or latent predictor models), PHI must be rebuilt per draw — that lands as a
separate code path, not this TODO. Data-x covers ~all real uses.

**sb backend.** Emit the same `_hsgp_basis` precompute into SB data, plus an
`_sb_hsgp` submodel that does the sqrt_spd matvec inside Stan. Straightforward
once array-indexed basis expansions land.

**Verification.** Synthetic sinusoidal f(t) + noise. Confirm the fitted GP
recovers the curve at reasonable K (>= 15) and that the log-prior / gradient
stay finite across the parameter range.

=#

loc ~ 1 + a + gp(b; k=20, c=1.5)
y1 ~ Normal(loc, 1)
