@dynamicstruct struct Formula
    text::String

    raw = Meta.parse("begin\n$text\nend")

    violation = walk(raw)
    is_safe = violation === nothing

    _t = begin
        local alllocals = OrderedDict{Symbol,Symbol}()
        (; ex=parse!(deepcopy(raw); info=(;alllocals)), alllocals)
    end
    transformed = _t.ex
    alllocals   = _t.alllocals
end
@dynamicstruct struct Dataset
    n::Int = 16
    seed::Int = 1

    df = begin
        rng = Xoshiro(seed)
        a = randn(rng, n)
        b = randn(rng, n)
        c = randn(rng, n)
        d = randn(rng, n)
        # Grouping factors with different numbers of levels -- use these on
        # the right-hand side of `(... | gN)` to test multiple random-effects
        # blocks.
        g1 = repeat(1:8, inner=cld(n, 8))[1:n]
        g2 = repeat(1:4, inner=cld(n, 4))[1:n]
        g3 = rand(rng, 1:6, n)
        c1 = rand(rng, 1:3, n)
        c2 = rand(rng, 1:2, n)
        c3 = rand(rng, 1:4, n)
        exposure = 0.5 .+ rand(rng, n)
        eta1 = 0.5 .+ 1.2 .* a .- 0.7 .* b .+ 0.3 .* c .+ 0.1 .* d
        y1 = eta1 .+ 0.3 .* randn(rng, n)
        y2 = -0.2 .+ 0.6 .* a .+ 0.4 .* b .+ 0.2 .* randn(rng, n)
        k1 = rand.(rng, Distributions.Poisson.(exp.(0.5 .* eta1)))
        k2 = rand.(rng, Distributions.Poisson.(exp.(0.3 .+ 0.4 .* a)))
        bin_n = rand(rng, 5:30, n)
        bin_p_true = @. 1 / (1 + exp(-(0.2 + 0.5 * a)))
        bin_succ = [rand(rng, Distributions.Binomial(n_i, p_i))
                    for (n_i, p_i) in zip(bin_n, bin_p_true)]
        bin_y = [rand(rng, Distributions.Bernoulli(p_i)) ? 1 : 0
                 for p_i in bin_p_true]
        # `y1` with ~25% of entries randomly set to `missing` -- exercise the
        # `mi(y_mi) ~ Family(...)` path without disturbing the other presets.
        y_mi = Vector{Union{Missing,Float64}}(y1)
        for i in shuffle(rng, 1:n)[1:max(1, n ÷ 4)]
            y_mi[i] = missing
        end
        DataFrame(; a, b, c, d, g1, g2, g3, c1, c2, c3, exposure,
                    y1, y2, k1, k2, bin_n, bin_succ, bin_y, y_mi)
    end

    # NamedTuple view of `df`, merged with namespace-dispatched extras so
    # extensions (e.g. bruno-ext.jl) can splice in `dose_times` etc. without
    # touching macro.jl. `@brm` only needs `hasproperty`/`getproperty` on
    # its data argument, so the NamedTuple stands in for the DataFrame.
    container(namespace=:default) = begin
        cols = (; (Symbol(c) => df[!, c] for c in names(df))...)
        merge(cols, dataset_extras(Val(namespace), df))
    end
end
