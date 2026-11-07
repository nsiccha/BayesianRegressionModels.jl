# test/nlme_completion_laws.jl — `mi(...)` completions whose law reads the
# units' draws, on the NLME paths (snag brm-nlme-loglike-7c4b209b). Centred,
# such a law is a prior term on shared coordinates: each unit's conditional
# (values, one-unit gradients) omitted it while the lockstep gradient and the
# η Hessian, taken from the whole sampler, carried it. `brm_nlme_view` refuses
# that reading by name; the fits' `@brm` entry lowers exactly those
# completions non-centered, which every continuous univariate law takes (its
# support transform on a standard-normal draw, its density remainder in
# `:pointwise`), so every route computes one function.
#
# Covered: the reported panel through the `@brm` entry, an interval (Beta) law,
# a mixed model whose other completion stays centred, a law reading no unit's
# draws (kept centred and accepted), and a regression-plan `mi(y)` response
# (nothing to non-center).
#
# Run: julia --project=test test/nlme_completion_laws.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, LinearAlgebra, Random

include(joinpath(@__DIR__, "nlme_panel_fixture.jl"))
include(joinpath(@__DIR__, "nlme_shares_fixture.jl"))

const PANEL_GROUPS = (; subject_group=:region, nested_groups=PANEL_NESTED)

central(f, x; h=1e-5) = map(eachindex(x)) do j
    a = copy(x); a[j] += h
    b = copy(x); b[j] -= h
    (f(a) - f(b)) / 2h
end

# Every route of `m` at one point: the one-unit value and gradient against the
# lockstep ones and against central differences of the values, and each unit's
# η Hessian block against differences of its one-unit gradient.
function check_routes(m; seed=42)
    rng = Xoshiro(seed)
    θ = 0.1 .* randn(rng, length(m.theta))
    σ = 0.1 .* randn(rng, length(m.sigma))
    H = [0.2 .* randn(rng, length(b)) for b in m.view.subject_coordinates]
    values, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    blocks = BRM._brm_nlme_hessian_blocks(m, θ, σ, H)
    for i in eachindex(H)
        value, g = brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, H[i])
        @test value ≈ values[i] rtol=1e-12
        @test g ≈ G[i] atol=1e-8
        @test g ≈ central(η -> brm_nlme_loglikelihood(m, i, θ, σ, η), H[i]) atol=1e-6
        fd = reduce(hcat, [central(η -> brm_nlme_loglikelihood_and_gradient(m, i, θ, σ,
            η)[2][j], H[i]) for j in eachindex(H[i])])
        @test blocks[i] ≈ fd atol=1e-5
    end
end

# The centred and non-centered lowerings of a law with no location and scale
# sample the same coordinates (the draws replace the entries) under the same
# density. (A `Normal`/`LogNormal(location, scale)` law's draws are
# standardized instead: other coordinates of the same model.)
function check_same_density(centred, noncentered; seed=5)
    names = String.(BRM._rk_layout_coordinate_names(centred))
    @test replace.(names, ".y_mis" => ".z_mis", ".y_free" => ".z_free") ==
        String.(BRM._rk_layout_coordinate_names(noncentered))
    u = 0.3 .* randn(Xoshiro(seed), length(names))
    a, ga = LogDensityProblems.logdensity_and_gradient(
        rk_logdensity_problem(centred; ad_backend=NLME_AD), u)
    b, gb = LogDensityProblems.logdensity_and_gradient(
        rk_logdensity_problem(noncentered; ad_backend=NLME_AD), u)
    @test a ≈ b rtol=1e-12
    @test ga ≈ gb atol=1e-10
end

@stestset "the reported panel: one function on every route through the @brm entry" begin
    # `x1` missing for every row of region 1, so that unit observes none.
    data = panel_data()
    x1 = copy(data.x1)
    x1[data.region .== 1] .= missing
    data = merge(data, (; x1))
    @test_throws "`mi(x1)` reads the units' draws" brm_nlme_view(RKBRMI(panel(data));
        PANEL_GROUPS...)
    rk = BRM._brm_nlme_lowered(panel(data); PANEL_GROUPS...)
    @test BRM._rk_completion_mode(rk.plan) === :noncentered
    check_routes(brm_nlme_model(rk; ad_backend=NLME_AD, PANEL_GROUPS...))
end

beta_data = let rng = Xoshiro(11), n = 36
    region = repeat(1:3, inner=12)
    q = 0.5 .* randn(rng, n)
    p = Vector{Union{Missing,Float64}}(1 ./ (1 .+ exp.(-q)))
    y = q .+ 0.3 .* randn(rng, n)
    p[rand(rng, n) .< 0.3] .= missing
    (; region, p, y)
end
beta_model(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | region)
    mi(p) ~ Beta(exp(mu), 2.0)
    q = log(p) - log(1 - p)
    eta ~ 1 + q + (1 | region)
    y ~ Normal(eta, sigma)
end

@stestset "a Beta completion law is non-centered by its logistic transform" begin
    centred = RKBRMI(beta_model(beta_data))
    rk = RKBRMI(beta_model(beta_data); completions=:noncentered)
    check_same_density(centred, rk)
    @test_throws "`mi(p)` reads the units' draws" brm_nlme_view(centred;
        subject_group=:region)
    @test BRM._rk_completion_mode(BRM._brm_nlme_lowered(beta_model(beta_data);
        subject_group=:region).plan) === :noncentered
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:region)
    @test count(==(:shared_latent), m.view.role) == 0
    check_routes(m)
end

two_data = let rng = Xoshiro(13), n = 36
    region = repeat(1:3, inner=12)
    a = Vector{Union{Missing,Float64}}(randn(rng, n))
    b = Vector{Union{Missing,Float64}}(exp.(0.5 .+ 0.3 .* randn(rng, n)))
    y = coalesce.(a, 0.0) .+ log.(coalesce.(b, 1.6)) .+ 0.3 .* randn(rng, n)
    a[rand(rng, n) .< 0.25] .= missing
    b[rand(rng, n) .< 0.25] .= missing
    (; region, a, b, y)
end
two_completions(d) = @brm d begin
    sigma ~ Exponential(1)
    mu_a ~ 1 + (1 | region)
    mi(a) ~ Normal(mu_a, 1.0)
    mi(b) ~ LogNormal(0.5, 0.3)
    lb = log(b)
    eta ~ 1 + a + lb + (1 | region)
    y ~ Normal(eta, sigma)
end

@stestset "only the completion whose law reads the units is non-centered" begin
    rk = BRM._brm_nlme_lowered(two_completions(two_data); subject_group=:region)
    @test BRM._rk_completion_mode(rk.plan) == (:a,)
    # `a` (a Normal law: standardized draws) takes `a.z_mis`; `b` keeps its
    # centred entries `b.y_mis`.
    names = String.(BRM._rk_layout_coordinate_names(rk))
    @test any(startswith("a.z_mis"), names) && !any(startswith("a.y_mis"), names)
    @test any(startswith("b.y_mis"), names) && !any(startswith("b.z_mis"), names)
    @test replace.(String.(BRM._rk_layout_coordinate_names(RKBRMI(two_completions(two_data)))),
        "a.y_mis" => "a.z_mis") == names
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:region)
    # `b`'s missing entries stay shared, point-estimated coordinates.
    @test count(==(:shared_latent), m.view.role) == count(ismissing, two_data.b)
    check_routes(m)
end

population_law(d) = @brm d begin
    sigma_x ~ Exponential(0.2)
    mu_x ~ 1 + z
    mi(x1) ~ LogNormal(mu_x, sigma_x)
    c_x = log(x1) - c0
    log_w ~ 1 + c_x + year + (1 + c_x || region) + (1 | d | district) +
        (1 | m | muni) + (1 | dy | district_year)
    sd(log_w, region) ~ truncated(Normal(0, 0.5); lower = 0)
    sd(:, m) ~ truncated(Normal(0, 0.5); lower = 0)
    sd(:, dy) ~ LogNormal(-3, 1)
    sd(:, d) ~ LogNormal(-1.5, 0.75)
    sigma_t ~ Exponential(0.03)
    rows_t = groups(district_year => dy_obs)
    @plate for k in eachindex(total)
        total[k] ~ Normal(sum(exp.(log_w[rows_t[k]])), sigma_t * total[k])
    end
end

@stestset "a completion law reading no unit's draws stays centred" begin
    rk = BRM._brm_nlme_lowered(population_law(PANEL); PANEL_GROUPS...)
    @test BRM._rk_completion_mode(rk.plan) === :centered
    v = brm_nlme_view(rk; PANEL_GROUPS...)
    @test count(==(:shared_latent), v.role) == count(ismissing, PANEL.x1)
end

@stestset "the @brm entry non-centers shares completions, refused centred" begin
    data = shares_panel()
    @test_throws "are centred" brm_nlme_view(RKBRMI(shares_model(data)); SHARES_NESTED...)
    rk = BRM._brm_nlme_lowered(shares_model(data); SHARES_NESTED...)
    @test BRM._rk_completion_mode(rk.plan) === :noncentered
    @test count(==(:shared_latent), brm_nlme_view(rk; SHARES_NESTED...).role) == 0
end

@stestset "a regression-plan mi(y) lowers the same under every spelling" begin
    g = repeat(1:4, inner=6)
    x = collect(range(0.0, 1.0; length=length(g)))
    y = Union{Missing,Float64}[i % 5 == 0 ? missing : 0.3 * x[i] + 0.1 * sin(i)
        for i in eachindex(g)]
    model(d) = @brm d begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | g)
        mi(y) ~ Normal(mu, sigma)
    end
    data = (; g, x, y)
    centred = RKBRMI(model(data))
    @test centred.plan isa BRM._RKStructuralPlan
    check_same_density(centred, RKBRMI(model(data); completions=:noncentered))
    check_same_density(centred, RKBRMI(model(data); completions=(:y,)))
    @test_throws "`x`, which is not a completion" RKBRMI(model(data); completions=(:x,))
end
