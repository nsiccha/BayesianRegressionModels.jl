# test/plate_gathered_predictor.jl — a predictor on its own rows reading
# another part's values through a model-scope gather, read by @plate cells.
#
# Run: julia --project=test test/plate_gathered_predictor.jl
#
# Part 1 (`mu1`, its response `y1` partly missing) lies on 6 rows; part 2
# (`log_other`) on 5 rows of its own, observed as group totals by a plate over
# `groups(g2 => ky)`. Part 2 reads part 1's values on its rows by position:
# `c2 = c[pos2]` and `z2 = z[pos2]`, where `z` also reads part-1 data columns.
# A gather lies on its index's rows, so `log_other` has 5 rows. Before, both
# backends took `z`'s data columns (6 rows) for `log_other`'s and refused the
# plate's grouping, and RKBRMI also refused `c2` as a term (snag
# rk-backend-predi-9a0c2121).
#
# The reference is the same model with the part-1 columns `z` reads gathered
# onto part 2's rows by hand: equal Stan and RK densities and gradients.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PlateGatheredPredictor
using BayesianRegressionModels, Distributions

const DATA = (;
    x1=[0.1, -0.3, 0.5, 0.2, -0.1, 0.4],
    y1=Union{Missing,Float64}[1.2, 0.8, missing, 1.5, missing, 1.1],
    w=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    v=[0.3, 0.7, 0.2, 0.9, 0.5, 0.6],
    pos2=[1, 2, 2, 4, 6],
    t2=[-1.0, 0.0, 1.0, -0.5, 0.5],
    g2=["a", "a", "b", "b", "c"],
    ky=["a", "b", "c"],
    tot=[9.0, 8.5, 4.0],
    scale=[1.0, 0.5, 2.0])
const BY_HAND = merge(DATA, (; w2=DATA.w[DATA.pos2], v2=DATA.v[DATA.pos2]))

gathered(d) = @brm d begin
    mu1 ~ 1 + x1
    sigma1 ~ Exponential(1)
    mi(y1) ~ LogNormal(mu1, sigma1)
    c = log(y1) - 0.2
    z = w * (v - log(y1) - 0.5) / 2.0
    c2 = c[pos2]
    z2 = z[pos2]
    log_other ~ 1 + t2 + c2 + z2
    sigma2 ~ Exponential(1)
    rows = groups(g2 => ky)
    @plate for k in eachindex(tot)
        tot[k] ~ Normal(sum(exp.(log_other[rows[k]])), sigma2 * scale[k])
    end
end

# `z` on part 2's rows written out: `log(y1)[pos2]` is `c2 + 0.2`.
by_hand(d) = @brm d begin
    mu1 ~ 1 + x1
    sigma1 ~ Exponential(1)
    mi(y1) ~ LogNormal(mu1, sigma1)
    c = log(y1) - 0.2
    c2 = c[pos2]
    z2 = w2 * (v2 - c2 - 0.2 - 0.5) / 2.0
    log_other ~ 1 + t2 + c2 + z2
    sigma2 ~ Exponential(1)
    rows = groups(g2 => ky)
    @plate for k in eachindex(tot)
        tot[k] ~ Normal(sum(exp.(log_other[rows[k]])), sigma2 * scale[k])
    end
end

# A part-1 column read directly by part 2 is still on part 1's rows.
misaligned(d) = @brm d begin
    mu1 ~ 1 + x1
    sigma1 ~ Exponential(1)
    mi(y1) ~ LogNormal(mu1, sigma1)
    c = log(y1) - 0.2
    c2 = c[pos2]
    log_other ~ 1 + t2 + c2 + v
    sigma2 ~ Exponential(1)
    rows = groups(g2 => ky)
    @plate for k in eachindex(tot)
        tot[k] ~ Normal(sum(exp.(log_other[rows[k]])), sigma2 * scale[k])
    end
end
end

const PG = PlateGatheredPredictor
probe_points(n) = (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))

@stestset "a gather lies on its index's rows" begin
    lp = only(node for (key, node) in BRM._brm_operation_entries(PG.gathered(PG.DATA))
        if key === :log_other)
    lengths = BRM._brm_kernel_lp_data_lengths(lp)
    @test (:pos2, 5) in lengths
    @test all(==(5), last.(lengths))
    @test :w ∉ first.(lengths) && :v ∉ first.(lengths)
end

@stestset "a part-1 column read directly is still refused" begin
    @test_throws "has 6 entries (data vector `v`) but `g2` has 5" SBBRMI(
        PG.misaligned(PG.DATA); mod=PG, total_groups=())
    @test_throws "has 6 entries (data vector `v`) but `g2` has 5" RKBRMI(
        PG.misaligned(PG.DATA))
end

@stestset "SB: the gathered model equals the by-hand twin" begin
    names, got = stan_density_points(PG.gathered(PG.DATA), "plate_gathered")
    ref_names, ref = stan_density_points(PG.by_hand(PG.BY_HAND), "plate_gathered_by_hand")
    @test names == ref_names
    @test all(isfinite, got)
    @test got ≈ ref atol=1e-12 rtol=1e-12
end

@stestset "RK: the gathered model equals the by-hand twin" begin
    for completions in (:centered, :noncentered)
        a = RKBRMI(PG.by_hand(PG.BY_HAND); completions)
        b = RKBRMI(PG.gathered(PG.DATA); completions)
        @test coordinate_names(b.model.layout) == coordinate_names(a.model.layout)
        pa = rk_logdensity_problem(a; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        pb = rk_logdensity_problem(b; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        for u in probe_points(LogDensityProblems.dimension(pa))
            va, ga = LogDensityProblems.logdensity_and_gradient(pa, u)
            vb, gb = LogDensityProblems.logdensity_and_gradient(pb, u)
            @test isfinite(va)
            @test vb ≈ va atol=1e-12 rtol=1e-12
            @test gb ≈ ga atol=1e-12 rtol=1e-12
        end
    end
end

@stestset "RK equals Stan on the gathered model; its printed source replays" begin
    backend, problem = consumer_problem(PG.gathered(PG.DATA))
    stan = consumer_stan(PG.gathered(PG.DATA), "plate_gathered_parity")
    stan_names = BridgeStan.param_unc_names(stan.model)
    # RK `pop_mu1.beta_pop.1` is Stan's `pop_mu1_beta_pop.1`.
    rk_names = [replace(string(n), "." => "_"; count=1)
        for n in coordinate_names(backend.model.layout)]
    @test sort(rk_names) == sort(stan_names)
    perm = [findfirst(==(n), stan_names) for n in rk_names]
    for u in probe_points(length(stan_names))
        g = similar(u)
        value, _ = BridgeStan.log_density_gradient!(stan.model, u, g; propto=false)
        rk_value, rk_gradient = LogDensityProblems.logdensity_and_gradient(problem, u[perm])
        @test isfinite(value)
        @test rk_value ≈ value atol=1e-10 rtol=1e-10
        @test rk_gradient ≈ g[perm] atol=1e-10 rtol=1e-10
    end
end
