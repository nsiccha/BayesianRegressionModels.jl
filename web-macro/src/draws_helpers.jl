# Element-returning counterpart to `Base.findfirst(pred, coll)` (which returns an
# index or `nothing`). Does the index-then-lookup dance once so callers don't.
findfirstelement(pred, coll) = begin
    i = findfirst(pred, coll)
    isnothing(i) ? nothing : coll[i]
end

# Stan draws → DataFrames plumbing, shared between prior-predictive generation
# and posterior fits (pathfinder / warmup). Keep these as plain module-level
# helpers so callsites inside `@include stan = …` don't accidentally become IPs.

# Param-constrain each column of `unc_draws` (dim × n) with `include_tp=true,
# include_gq=true`, returning an (m × n) matrix where `m = length(param_names(instance; include_tp=true, include_gq=true))`.
constrain_draws(unc_draws, instance; rng_seed) = begin
    rng = BridgeStan.StanRNG(instance, rng_seed)
    m = length(BridgeStan.param_names(instance; include_tp=true, include_gq=true))
    n = size(unc_draws, 2)
    mat = Matrix{Float64}(undef, m, n)
    for i in 1:n
        mat[:, i] = BridgeStan.param_constrain(
            instance, collect(view(unc_draws, :, i));
            include_tp=true, include_gq=true, rng=rng,
        )
    end
    mat
end

# Build the (long, wide, summary) DataFrame triple from a constrained draws
# matrix `constrained` (m × n) and its matching parameter `names` (length m).
# Splits indexed names on the first `.` into (:param, :index) with :index as Int
# (0 for scalars); summary groups by (:param, :index) with the bands columns
# expected by `pointinterval(bands=…)` / `lineribbon(bands=…)`.
dfs_from_constrained(constrained, names) = begin
    n = size(constrained, 2)
    splits     = [split(nm, '.', limit=2) for nm in names]
    base_names = [String(first(s)) for s in splits]
    parse_idx(s) = (v = tryparse(Int, s); isnothing(v) ? 0 : v)
    indices    = [length(s) > 1 ? parse_idx(String(s[2])) : 0 for s in splits]
    long = DataFrame(
        param = repeat(base_names, inner=n),
        index = repeat(indices, inner=n),
        draw  = repeat(1:n, outer=length(base_names)),
        value = vec(constrained'),
    )
    wide = DataFrame(
        [Symbol(names[i]) => constrained[i, :] for i in eachindex(names)]
    )
    summary = combine(
        groupby(long, [:param, :index]),
        :value => (v -> quantile(v, 0.025)) => :q025,
        :value => (v -> quantile(v, 0.10))  => :q10,
        :value => (v -> quantile(v, 0.25))  => :q25,
        :value => median                    => :median,
        :value => (v -> quantile(v, 0.75))  => :q75,
        :value => (v -> quantile(v, 0.90))  => :q90,
        :value => (v -> quantile(v, 0.975)) => :q975,
    )
    (; long, wide, summary)
end
