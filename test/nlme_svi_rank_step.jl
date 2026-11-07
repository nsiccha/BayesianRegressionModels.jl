# LowRankGlobal's global steps (src/nlme_svi.jl). In exploration
# (`_brm_svi_exploration_global_step!`) each row of the global rank part Β moves
# in units of its coordinate's scale exp(ρ_k) while μ and ρ step exactly as
# before; in polishing (`_brm_svi_polish_global_step!`) μ moves in units of
# exp(ρ_k) too, not of a square root of the whole covariance. A mean-field factor
# steps as before in both.
#
# Β's gradient w ηᵀ carries the minibatch noise of w, so Adam's normalized step is
# mostly a random walk of one step size per iteration. Before 0.2.243 that step
# was `lr_global` in absolute units for every coordinate: through a long
# exploration Β's rows grew to several times their coordinates' own sds, the
# factor came out wider than the posterior, and on a real target typical draws
# reached a non-finite density (snag lowrank-slice-sv-baba88d2).
#
# RUN: julia --project=test test/nlme_svi_rank_step.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, Random, Statistics, WarmupHMC
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))

# A global factor of `D` coordinates with scales `s` and a rank-`r` part.
svi_rank_q(s, r) = (; μ=zeros(length(s)), ρ=log.(s), Β=zeros(length(s), r))

@stestset "a low-rank exploration step moves Β rows in units of their scale" begin
    # Two coordinates with scales 1 and 0.01 receive the same gradient on their
    # Β rows at every step, and none on μ or ρ: their rows then move by the same
    # number of their own scales, 1 : 0.01.
    s = [1.0, 0.01]
    q = svi_rank_q(s, 1)
    opt = BRM._BRMSVIAdam(6, 0.01)
    rng = Xoshiro(1)
    for _ in 1:2000
        ξ = randn(rng)
        x = BRM._brm_svi_global_vector(q)
        BRM._brm_svi_exploration_global_step!(opt, q, x, [0, 0, 0, 0, ξ, ξ], 0.01)
        BRM._brm_svi_set_global!(q, x)
    end
    @test abs(q.Β[1]) > 0.1
    @test q.Β[2] / q.Β[1] ≈ 0.01 rtol=1e-4
    # μ and ρ did not move: no gradient reached them.
    @test q.μ == zeros(2) && q.ρ == log.(s)
end

@stestset "μ and ρ step as before, and a mean-field step is unchanged to the last bit" begin
    rng = Xoshiro(2)
    for r in (0, 2)
        D = 5
        q = svi_rank_q(exp.(randn(rng, D)), r)
        q.μ .= randn(rng, D)
        q.Β .= randn(rng, D, r)
        n = 2D + D * r
        a, b = BRM._BRMSVIAdam(n, 0.01), BRM._BRMSVIAdam(n, 0.01)
        xa, xb = BRM._brm_svi_global_vector(q), BRM._brm_svi_global_vector(q)
        for _ in 1:50
            g = 10 .* randn(rng, n)
            BRM._brm_svi_step!(a, xa, g; clip=10.0)
            BRM._brm_svi_exploration_global_step!(b, q, xb, g, b.lr; clip=10.0)
        end
        head = 1:2D
        @test isequal(xa[head], xb[head])
        @test isequal(a.m[head], b.m[head]) && isequal(a.v[head], b.v[head])
        @test a.t == b.t
        r == 0 && @test isequal(xa, xb) && isequal(a.m, b.m) && isequal(a.v, b.v) &&
            a.clipped == b.clipped
    end
end

@stestset "a low-rank polishing step moves μ in units of its diagonal scale" begin
    # A rank column couples the two coordinates. The whitened step moves μ along
    # it, the polishing step in units of each coordinate's own scale only.
    s = [1.0, 0.01]
    q = svi_rank_q(s, 1)
    q.Β .= [0.0, 5.0]
    g = [1.0, 1.0, 0, 0, 0, 0]
    x = BRM._brm_svi_global_vector(q)
    BRM._brm_svi_polish_global_step!(BRM._BRMSVIAdam(6, 0.05), q, x, g, 0.05)
    @test x[2] / x[1] ≈ 0.01 rtol=1e-6
    w = BRM._brm_svi_global_vector(q)
    BRM._brm_svi_whitened_global_step!(BRM._BRMSVIAdam(6, 0.05), q, w, g, 0.05)
    @test abs(w[2]) > 10 * abs(x[2])
    # Without a rank part the polishing step is the whitened step to the last bit.
    rng = Xoshiro(3)
    q = svi_rank_q(exp.(randn(rng, 4)), 0)
    q.μ .= randn(rng, 4)
    a, b = BRM._BRMSVIAdam(8, 0.05), BRM._BRMSVIAdam(8, 0.05)
    xa, xb = BRM._brm_svi_global_vector(q), BRM._brm_svi_global_vector(q)
    for _ in 1:50
        g = 10 .* randn(rng, 8)
        BRM._brm_svi_whitened_global_step!(a, q, xa, g, 0.05; clip=10.0)
        BRM._brm_svi_polish_global_step!(b, q, xb, g, 0.05; clip=10.0)
    end
    @test isequal(xa, xb) && isequal(a.m, b.m) && isequal(a.v, b.v)
end

# The NUTS reference of `rk`'s target: coordinates × draws.
function svi_rank_nuts(rk, m; seed, n_draws=1000, chains=4)
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    u0 = first(BRM._brm_nlme_prior_median_point(m))
    reduce(hcat, map(1:chains) do c
        run = adaptive_warmup_mcmc(Xoshiro(seed + c), problem;
            init=(; position=copy(u0), squared_scale=ones(length(u0))), n_draws)
        convert(Matrix{Float64}, run.posterior_position)
    end)
end

@stestset "a long low-rank exploration stays narrower than the posterior" begin
    # 15,000 iterations at the exploration step size (min_windows keeps polishing
    # away), 200 subjects in batches of 4. A variational factor of this family
    # comes out narrower than the posterior; with the absolute step of 0.2.242 and earlier one global's sd
    # read 1.53 times the NUTS sd here, against 0.67 now.
    rk = RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(1), 200)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    reference = svi_rank_nuts(rk, m; seed=20)
    fit = brm_svi_fit(m; rng=Xoshiro(2), family=LowRankGlobal(2), batch_size=4, window=100,
        min_windows=10_000, max_iterations=10^6, stop_after=15_000, ndraws=4000)
    g = fit.diagnostics.globals
    sd = vec(std(reference[g, :]; dims=2))
    @test all(<(1.2), vec(std(fit.draws[g, :]; dims=2)) ./ sd)
    @test maximum(abs.(vec(mean(fit.draws[g, :]; dims=2)) .- vec(mean(reference[g, :]; dims=2))) ./ sd) < 1.0
end
