# label: 1.5 zerocorr — independent random effects
# tier: 1
# status: open
#=
**What it is.** brms (via lme4 syntax) lets you opt out of the LKJ correlation between multiple random terms in the same group. `(1 + x || group)` (double bar) says "estimate the random intercept and the random slope independently — don't fit a 2×2 Cholesky factor between them". Useful when there isn't enough data to estimate the correlations, or when you have prior reason to believe the terms are uncorrelated.

**Why it matters.** Multi-term random specs are common, and the LKJ correlation often dominates the prior cost without much identifiability. Letting users skip it is a meaningful sampling speedup and prior simplification.

**Implementation.** Our `_x` walker already wraps `||` as `ExprColumn{typeof(doublepipe)}`. Add a `vmeta_sampling_rhs` overload that splits each term inside the `||` LHS into its own block (with a synthetic per-term key like `Symbol(group_name, :__nocor__, term_index)`):

```julia
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(doublepipe)}; kwargs...) = begin
    lhs, rhs = getargs(x, 2)
    terms = lhs isa ExprColumn{typeof(+)} ? getargs(lhs) : (lhs,)
    foldl(enumerate(terms); init=(meta, ())) do (m, args), (i, term)
        nocor_key = NamedColumn(Symbol(name(rhs), :__nocor__, i), parent(rhs))
        m, arg = vmeta_sampling_rhs(m, term; group=nocor_key)
        m, (args..., arg)
    end |> ((m, args),) -> (m, Base.broadcasted(+, args...))
end
```

Each per-term block ends up as 1×1 with one `log_scale` Cholesky parameter — `lprior!`'s existing single-column path handles this with no changes.

**Verification.** Preset: `loc ~ 1 + (1 + a || g1); y1 ~ Normal(loc, 1)`. Compare its dim against the correlated `(1 + a | g1)` version: the correlated version has 3 Cholesky params (1+2/2 for a 2×2), the uncorrelated version has 2 (one log_scale per term). Same direct-parameter count (2 cols × 8 levels = 16) either way.

=#

loc ~ 1 + (1 + a || g1)
y1 ~ Normal(loc, 1)
