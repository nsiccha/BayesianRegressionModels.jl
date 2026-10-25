# test/stratified_ranef_priors.jl — `sd` and `cor` priors on stratified
# `gr(g, by=b)` random-effect blocks (snag `a-prior-on-the-s-a96bac26`).
#
# A stratified block samples one scale vector and one correlation factor per
# stratum of `b`. Its `sd(...)` statements put each margin's prior on that
# margin's scale in every stratum, and `cor(...) ~ LKJCholesky(K, eta)` gives
# every stratum's correlation factor `LKJ(eta)`: the statements address the
# block, by its `|ID|` or, without one, by its grouping column.
#
# Run: julia --project=test test/stratified_ranef_priors.jl [filter...]
# Set BRM_STRATIFIED_PRIORS_RUNTIME=0 to skip the BridgeStan and RK densities.
#
# The densities are checked against the block written without statements: the
# two programs share every coordinate and transform, so their log densities
# differ by exactly the configured minus the default prior of every stratum's
# scales and correlation factor, evaluated at the constrained values.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LinearAlgebra
using Distributions: Exponential, LKJCholesky, LogNormal, Normal, TDist, logpdf

const STRATIFIED_RUNTIME = get(ENV, "BRM_STRATIFIED_PRIORS_RUNTIME", "1") != "0"
const STRATIFIED_CACHE = joinpath(tempdir(), "brm-stratified-ranef-priors")

const GROUPS = repeat(1:8; inner=4)
const DF = (; g=GROUPS, b=[g <= 4 ? 1 : 2 for g in GROUPS],
            x=round.(sin.(1:32); digits=3), y=round.(cos.(3 .* (1:32)); digits=3))

code(sb) = BRM.stan_code(sb)
build(b; kwargs...) = SBBRMI(b; mod=@__MODULE__, total_groups=(), kwargs...)
stanc_ok(sb) = StanBlocks.stanc_check(code(sb); warn_pedantic=false).ok
# Whether program `c` samples an element slice of `name` from `law` (a long
# left-hand side wraps across lines).
samples(c, name, law) = occursin(Regex(name * raw"\[[^\]]*\]\s*~ " * law), c)

function instantiate(sb)
    mkpath(STRATIFIED_CACHE)
    BRM.stan_instantiate(sb; path=joinpath(STRATIFIED_CACHE, string(hash(code(sb))) * ".stan"))
end

# The reporter's shape: a lone intercept whose spread differs by stratum.
intercept = @brm DF begin
    mu ~ 1 + x + (1 | gr(g, by=b))
    sd(mu, g) ~ LogNormal(-1, 0.75)
    y ~ Normal(mu, 1.0)
end
intercept_twin = @brm DF begin
    mu ~ 1 + x + (1 | mu__g__by__b | gr(g, by=b))
    sd(mu, mu__g__by__b) ~ LogNormal(-1, 0.75)
    y ~ Normal(mu, 1.0)
end
intercept_default = @brm DF begin
    mu ~ 1 + x + (1 | mu__g__by__b | gr(g, by=b))
    y ~ Normal(mu, 1.0)
end

# Per-margin scale priors and a correlation prior.
slope = @brm DF begin
    mu ~ 1 + x + (1 + x | gr(g, by=b))
    sd(mu, g, Intercept) ~ Exponential(2.0)
    sd(mu, g, x) ~ LogNormal(0, 0.5)
    cor(mu, g) ~ LKJCholesky(2, 3.0)
    y ~ Normal(mu, 1.0)
end
slope_default = @brm DF begin
    mu ~ 1 + x + (1 + x | mu__g__by__b | gr(g, by=b))
    y ~ Normal(mu, 1.0)
end

# One stratified block shared by two predictors.
shared = @brm DF begin
    mu ~ 1 + x + (1 | p | gr(g, by=b))
    log(sigma) ~ 1 + (1 | p | gr(g, by=b))
    sd(:, p) ~ Exponential(1.5)
    cor(:, p) ~ LKJCholesky(2, 2.0)
    y ~ Normal(mu, sigma)
end
shared_default = @brm DF begin
    mu ~ 1 + x + (1 | p | gr(g, by=b))
    log(sigma) ~ 1 + (1 | p | gr(g, by=b))
    y ~ Normal(mu, sigma)
end

@stestset "a grouping-column address reaches every stratum" begin
    sb = build(intercept)
    @test [(s.class, s.id, s.predictor) for s in ranef_effect_priors(parent(sb))] ==
          [(:sd, :mu__g__by__b, :mu)]
    @test ranefcoefnames(parent(sb), :mu__g__by__b) == [(; predictor=:mu, coefficient=:Intercept)]
    # The addressed block lowers as the one-predictor `|ID|` block written by hand.
    @test code(sb) == code(build(intercept_twin))
    c = code(sb)
    @test samples(c, "b_mu__g__by__b_g__by__b_tau_s_tau", raw"lognormal\(-1, 0\.75\);")
    @test only(ranef_blocks(sb)).family === :ranef_correlated_by_draws_generic
    @test stanc_ok(sb)
    # The unconfigured block keeps the default emission.
    default_sb = build(intercept_default)
    @test only(ranef_blocks(default_sb)).family === :ranef_correlated_by_draws
    @test samples(code(default_sb), "b_mu__g__by__b_g__by__b_tau_s_tau", raw"std_normal\(\);")
    @test !samples(code(default_sb), "b_mu__g__by__b_g__by__b_tau_s_tau", "lognormal")
    @test stanc_ok(build(intercept; cv_groups=(:g,)))
end

@stestset "per-margin scales and the correlation of every stratum" begin
    sb = build(slope)
    c = code(sb)
    @test samples(c, "b_mu__g__by__b_g__by__b_L_s_L", raw"lkj_corr_cholesky\(3\.0\);")
    # Mixed families take one generated per-coordinate family in the stratum
    # cell (a plate cell cannot hold StanBlocks' `vcat` prior family yet).
    @test samples(c, "b_mu__g__by__b_g__by__b_tau_s_tau", "brm_vector_prior_")
    @test only(ranef_blocks(sb)).family === :ranef_correlated_by_draws_generic
    @test stanc_ok(sb)

    sb = build(shared)
    @test only(ranef_blocks(sb)).family === :ranef_correlated_by_draws_generic
    @test samples(code(sb), "b_p_g__by__b_tau_s_tau", raw"exponential\(\(1\.0 \./ 1\.5\)\);")
    @test samples(code(sb), "b_p_g__by__b_L_s_L", raw"lkj_corr_cholesky\(2\.0\);")
    @test stanc_ok(sb)
end

@stestset "statements a stratified block cannot take yet" begin
    # Capability gaps, not wrong shapes: known values, Student-t laws and R2D2
    # decompositions have no per-stratum form yet, so each fails closed in the
    # statement's own spelling until it is built.
    known = @brm DF begin
        mu ~ 1 + x + (1 | gr(g, by=b))
        sd(mu, g) = 0.5
        y ~ Normal(mu, 1.0)
    end
    student = @brm DF begin
        mu ~ 1 + x + (1 | gr(g, by=b))
        ranef(mu, g) ~ TDist(4.0)
        y ~ Normal(mu, 1.0)
    end
    decomposed = @brm DF begin
        mu ~ 1 + x + (1 | gr(g, by=b))
        sd(mu, g) ~ r2d2(mean_R2=0.5, prec_R2=2.0, reference_scale=1.0)
        y ~ Normal(mu, 1.0)
    end
    # The same statements on an `|ID|` block.
    known_id = @brm DF begin
        mu ~ 1 + x + (1 | p | gr(g, by=b))
        sd(:, p) = 0.5
        y ~ Normal(mu, 1.0)
    end
    student_id = @brm DF begin
        mu ~ 1 + x + (1 | p | gr(g, by=b))
        ranef(:, p) ~ TDist(4.0)
        y ~ Normal(mu, 1.0)
    end
    for model in (known, student, decomposed, known_id, student_id)
        @test_broken (build(model); true)
    end
    @test_broken (RKBRMI(student); true)
end

# Constrained BridgeStan values of one parameter, keyed by index tuple.
function stan_values(model, u, name)
    names = BridgeStan.param_names(model)
    values = BridgeStan.param_constrain(model, u)
    Dict(Tuple(parse.(Int, split(n[ncodeunits(name)+2:end], '.'))) => v
         for (n, v) in zip(names, values) if startswith(n, name * "."))
end

lkj_delta(L, K, eta) = logpdf(LKJCholesky(K, eta), Cholesky(L, 'L', 0)) -
                       logpdf(LKJCholesky(K, 1.0), Cholesky(L, 'L', 0))

# Configured minus default prior of a block's strata: `tau[s][k]`, `L[s]`.
function prior_delta(tau, L, priors, eta)
    out = sum(logpdf(priors[k], t[k]) - logpdf(Normal(), t[k])
              for t in tau for k in eachindex(priors))
    isnothing(L) ? out : out + sum(lkj_delta(Ls, size(Ls, 1), eta) for Ls in L)
end

const CASES = (
    (; name="intercept", brmi=intercept, default=intercept_default,
       block=:b_mu__g__by__b_g__by__b, priors=(LogNormal(-1, 0.75),), eta=1.0),
    (; name="slope", brmi=slope, default=slope_default,
       block=:b_mu__g__by__b_g__by__b, priors=(Exponential(2.0), LogNormal(0, 0.5)), eta=3.0),
    (; name="shared", brmi=shared, default=shared_default,
       block=:b_p_g__by__b, priors=(Exponential(1.5), Exponential(1.5)), eta=2.0),
)

STRATIFIED_RUNTIME && @stestset "BridgeStan: every stratum takes the configured prior" begin
    for case in CASES
        configured = instantiate(build(case.brmi)).model
        default = instantiate(build(case.default)).model
        @test BridgeStan.param_unc_names(configured) == BridgeStan.param_unc_names(default)
        K, S = length(case.priors), 2
        for u in (zeros(BridgeStan.param_unc_num(configured)),
                  collect(range(-0.43, 0.51; length=BridgeStan.param_unc_num(configured))))
            block = string(case.block)
            tau_values = stan_values(configured, u, block * "_tau_s_tau")
            tau = [[tau_values[(k, s)] for k in 1:K] for s in 1:S]
            L = K == 1 ? nothing : let L_values = stan_values(configured, u, block * "_L_s_L")
                [[L_values[(s, i, j)] for i in 1:K, j in 1:K] for s in 1:S]
            end
            lp(m) = BridgeStan.log_density(m, u; propto=false, jacobian=true)
            @test lp(configured) - lp(default) ≈ prior_delta(tau, L, case.priors, case.eta) atol=1e-10 rtol=1e-10
        end
    end
end

# The default twin's coordinate for a configured one: per-margin scale vectors
# `tau_k[s]` are columns of the default `tau[s, k]`.
rk_default_name(name) =
    Symbol(replace(string(name), r"\.tau_(\d+)\.(\d+)$" => s".tau.\2.\1"))

STRATIFIED_RUNTIME && @stestset "RKBRMI: every stratum takes the configured prior" begin
    for case in CASES
        rk, problem = consumer_problem(case.brmi)
        rk_default, problem_default = consumer_problem(case.default)
        names = coordinate_names(rk.model.layout)
        default_names = coordinate_names(rk_default.model.layout)
        @test sort(rk_default_name.(names)) == sort(default_names)
        default_position = Dict(n => i for (i, n) in enumerate(default_names))
        to_default = [default_position[rk_default_name(n)] for n in names]
        K = length(case.priors)
        for u in (zeros(length(names)), collect(range(-0.43, 0.51; length=length(names))))
            u_default = similar(u)
            u_default[to_default] = u
            nt = getproperty(constrain(rk.model.layout, u), case.block)
            tau = K == 1 || hasproperty(nt, :tau) ?
                [nt.tau[s, :] for s in axes(nt.tau, 1)] :
                [[getproperty(nt, Symbol(:tau_, k))[s] for k in 1:K]
                 for s in eachindex(nt.tau_1)]
            L = K == 1 ? nothing : [nt.L[:, :, s] for s in axes(nt.L, 3)]
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            @test value - LogDensityProblems.logdensity(problem_default, u_default) ≈
                  prior_delta(tau, L, case.priors, case.eta) atol=1e-10 rtol=1e-10
            h = 1e-6
            numeric = map(eachindex(u)) do i
                up, dn = copy(u), copy(u)
                up[i] += h
                dn[i] -= h
                (LogDensityProblems.logdensity(problem, up) -
                 LogDensityProblems.logdensity(problem, dn)) / 2h
            end
            @test gradient ≈ numeric rtol=1e-5 atol=1e-6
        end
    end
end

# RKBRMI lowers a shared stratified `|ID|` block as one component whose
# per-row draws every predictor naming the ID gathers its own column from.
STRATIFIED_RUNTIME && @stestset "RKBRMI: a shared stratified `|ID|` block, by hand" begin
    @test ranefcoefnames(shared, :p) == [(; predictor=:mu, coefficient=:Intercept),
                                         (; predictor=:sigma, coefficient=:Intercept)]
    rk, problem = consumer_problem(shared)
    layout = rk.model.layout
    stratum = [DF.b[findfirst(==(g), DF.g)] for g in 1:8]
    rows = eachindex(DF.y)
    for u in (zeros(layout.total), collect(range(-0.43, 0.51; length=layout.total)))
        nt = constrain(layout, u)
        block = nt.b_p_g__by__b
        beta, log_sigma = nt.pop_mu.beta_pop, nt.sigma_Intercept
        # diag(tau_s) * L_s * z_g for the stratum s of each group g
        b = [(block.tau[stratum[g], :] .* block.L[:, :, stratum[g]]) * block.z[g, :]
             for g in 1:8]
        mu = [beta[1] + beta[2] * DF.x[i] + b[DF.g[i]][1] for i in rows]
        sigma = [exp(log_sigma + b[DF.g[i]][2]) for i in rows]
        likelihood = sum(logpdf(Normal(mu[i], sigma[i]), DF.y[i]) for i in rows)
        prior = sum(logpdf(Normal(), v) for v in beta) + logpdf(Normal(), log_sigma) +
            sum(logpdf(Exponential(1.5), t) for t in block.tau) +
            sum(logpdf(LKJCholesky(2, 2.0), Cholesky(block.L[:, :, s], 'L', 0)) for s in 1:2) +
            sum(logpdf(Normal(), v) for v in block.z)
        @test LogDensityProblems.logdensity(problem, u) ≈
              likelihood + prior + ReactiveKernelsPPL.logjac(layout, u) atol=1e-10 rtol=1e-10
    end
end
