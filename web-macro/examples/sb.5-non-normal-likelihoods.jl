# label: sb.5 non-Normal likelihoods — needs impl
# tier: 2
# status: open
#=
**Status: not implemented.** `_sb_lik_family` dispatches only on `::Type{<:Normal}`.

**What to implement.** Add a table of supported `Distribution` types -> Stan `_lpdf`/`_lpmf` names:

| Julia                        | Stan                        |
|------------------------------|-----------------------------|
| `Normal(loc, scale)`         | `normal(loc, scale)`        |
| `Bernoulli(p)`               | `bernoulli(p)`              |
| `Binomial(n, p)`             | `binomial(n, p)`            |
| `Poisson(lambda)`            | `poisson(lambda)`           |
| `Gamma(alpha, beta)`         | `gamma(alpha, beta)`        |
| `NegativeBinomial(r, p)`     | `neg_binomial(r, p)`        |
| `Beta(a, b)`                 | `beta(a, b)`                |

Most map 1:1 — the Bernoulli/Binomial/Poisson logistic/exp links already canonicalize on the Julia side (`bin_succ ~ Binomial(bin_n, logistic(eta))`), so sbimpl just has to pass the arguments through `_sb_scalar_expr`.

**Verification.** Reuse `1.1` formula (`bin_succ ~ Binomial(bin_n, logistic(log_odds_bin))` + hierarchical Bernoulli). Needs `sb.3` (ranefs) for the `(1 | g1)` piece.

=#

log_odds_b ~ 1 + a
bin_y ~ BernoulliLogit(log_odds_b)
