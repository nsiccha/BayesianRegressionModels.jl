# Globals that few units read in minibatch SVI (src/nlme_svi_sites.jl): window
# boundaries, field states, separators. A coordinate read by k of N units gets a
# likelihood gradient only when a reader is drawn, N/B times the reader's own,
# and Adam then left its mean and scale far from the family's optimum (snag
# svi-globals-read-f3eafb84: 270,000 nats below it on 2,048 window units). Each
# reader now keeps a Gaussian site for the coordinate, refreshed when it is
# visited, and the coordinate steps on the sum of its readers' sites.
#
# The end-to-end testsets fit window units of OU chains whose boundary states are
# read by two units each. The chain's parameters are data, so the target is
# exactly Gaussian: q* (exact means, precisions from the Hessian's blocks) is the
# family's optimum, and the boundaries' fitted means and sds are compared with it.
#
# RUN: julia --project=test test/nlme_svi_sparse_globals.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
using LinearAlgebra, Random, SparseArrays, Statistics
import LogDensityProblems

function svi_ou_path(x0, e, a, s)
    out = zeros(typeof(x0 * a * s), length(e))
    prev = x0
    for t in eachindex(e)
        prev = a * prev + s * e[t]
        out[t] = prev
    end
    out
end

# `S` stations, each an OU chain of `W` windows of `L` states, observed with noise
# sd 1. Window w of station s starts at boundary state xb[(s-1)(W+1)+w] and
# closes on the next one, so every interior boundary has two readers.
function svi_window_data(S, W, L; ℓ=50.0, τ=1.0, seed=1)
    rng = Xoshiro(seed)
    T = W * L + 1
    a, v = exp(-1 / ℓ), -expm1(-2 / ℓ)
    win, y, ids, lb, rb = String[], Vector{Vector{Float64}}(), Vector{Vector{Int}}(), Int[], Int[]
    next = 0
    for s in 1:S
        x = zeros(T)
        x[1] = τ * randn(rng)
        for t in 2:T
            x[t] = a * x[t-1] + τ * sqrt(v) * randn(rng)
        end
        x .+= randn(rng, T)
        for w in 1:W
            lo = (w - 1) * L + 1
            push!(win, "s$(s)w$w"); push!(y, x[lo:lo+L-1])
            push!(ids, collect(next+1:next+L-1)); next += L - 1
            push!(lb, (s - 1) * (W + 1) + w); push!(rb, (s - 1) * (W + 1) + w + 1)
        end
    end
    n = length(win)
    (; win, y, ids, lb, rb, state=collect(1:n), nint=fill(L - 1, n), closing=zeros(n),
       nb=S * (W + 1), ne=next, ar=a, sig=τ * sqrt(v))
end

svi_window_model(data) = @brm data begin
    xb[1:nb] .~ Normal(0.0, 30.0)
    e[1:ne] .~ Normal(0.0, 1.0)
    sd(mu, win) ~ LogNormal(-7.0, 0.01)
    mu ~ 0 + (1 | win)
    @plate for i in eachindex(state)
        h = svi_ou_path(xb[lb[i]], e[ids[i]], ar, sig)
        y[i] ~ Normal(mu[i] .+ vcat(xb[lb[i]], h), 1.0)
        closing[i] ~ Normal(xb[rb[i]] - ar * h[nint[i]], sig)
    end
end

# The exact Gaussian: H = -∇² log p from gradient differences (exact in every
# coordinate but the no-op subject sd, which sits at its prior's mode, where the
# target is quadratic in it to about e^-7), and its mean.
function svi_window_exact(rk)
    p = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    n = LogDensityProblems.dimension(p)
    names = string.(coordinate_names(rk.model.layout))
    scale = findall(c -> occursin(".tau", c), names)
    function hessian(x0)
        _, g0 = LogDensityProblems.logdensity_and_gradient(p, x0)
        H = zeros(n, n)
        for j in 1:n
            δ = j in scale ? 1e-4 : 1.0
            x = copy(x0); x[j] += δ
            H[:, j] .= -(LogDensityProblems.logdensity_and_gradient(p, x)[2] .- g0) ./ δ
        end
        Symmetric((H + H') / 2), g0
    end
    x = zeros(n); x[scale] .= -7.0
    for _ in 1:2
        H, g = hessian(x)
        x = x .+ (cholesky(H) \ g)
    end
    H, _ = hessian(x)
    (; H, μ=x, names)
end

# The boundary globals' means (in optimal sds) and sds (over the optimal) for
# family factor `q` with global positions `globals`.
function svi_boundaries(ex, q, globals)
    b = findall(k -> startswith(ex.names[globals[k]], "xb."), eachindex(globals))
    j = globals[b]
    h = [ex.H[k, k] for k in j]
    (; z=(q.μ[b] .- ex.μ[j]) .* sqrt.(h), r=exp.(q.ρ[b]) .* sqrt.(h), positions=b)
end

@stestset "a global is sparse when its readers are drawn less than once per batch" begin
    t = (; globals=[11, 12, 13, 14], units=[[1], [2], [3], [4]],
        touched=[[1, 2], [1, 3], [1, 3], [1]])
    uniform = fill(0.25, 4)
    sites = BRM._brm_svi_sites(t, 1, uniform)
    # Global 1 is read by every unit (one expected reader per batch); 2 by one
    # unit; 3 by two; 4 by none, so it is the global term's alone.
    @test sites.globals == [2, 3]
    @test sites.readers == [1, 2]
    @test sites.expected ≈ [0.25, 0.5]
    @test sites.entries == [[2], [2], [2], Int[]]
    @test BRM._brm_svi_sites(t, 2, uniform).globals == [2]
    @test BRM._brm_svi_sites(t, 4, uniform) === nothing
    # Work sampling: the expected reader draws follow the probabilities. At B = 2,
    # global 1 expects 2 reader draws, global 2 (unit 1) 1.2, global 3 (units 2
    # and 3) 0.7.
    work = BRM._brm_svi_sites(t, 2, [0.6, 0.3, 0.05, 0.05])
    @test work.globals == [3]
    @test work.expected ≈ [0.7]
    @test BRM._brm_svi_minibatch_global(nothing, 2)
    @test !BRM._brm_svi_minibatch_global(sites, 2) && BRM._brm_svi_minibatch_global(sites, 1)
    @test BRM._brm_svi_sites_diagnostics(sites, t) ==
        (; coordinates=[12, 13], readers=[1, 2], expected_readers=[0.25, 0.5])
    @test isempty(BRM._brm_svi_sites_diagnostics(nothing, t).coordinates)
end

# Visits of one reader with term -h (x - c)²/2 of global 1, its exact gradient at
# draws of q = N(μ, s²), μ and s held fixed: the site's precision tends to h and
# its gradient at the mean to the exact one, -h (μ - c).
function svi_site_visits(; h=40.0, c=0.3, μ=1.1, s=0.1, visits=600, seed=3, β=0.1)
    t = (; globals=[1, 2], units=[[3]], touched=[[1, 2]])
    sites = BRM._brm_svi_sites(t, 1, [0.5])
    q = (; μ=[μ, 0.0], ρ=log.([s, 1.0]), Β=zeros(2, 0))
    rng = Xoshiro(seed)
    history = NamedTuple[]
    for _ in 1:visits
        ε = randn(rng, 2)
        g = q.μ .+ exp.(q.ρ) .* ε
        G = -h * (g[1] - c)
        BRM._brm_svi_update_sites!(sites, t, q, g, ε, [1], [(; gg=[2G, 0.0])], β, 1.0)
        λ, ν = sites.precision[1][1], sites.pull[1][1]
        push!(history, (; λ, pull=ν - λ * μ))
    end
    sites, history
end

@stestset "a reader's site learns its term's curvature and pull" begin
    h, c, μ = 40.0, 0.3, 1.1
    sites, history = svi_site_visits(; h, c, μ)
    tail = history[end-299:end]
    @test 0.8h < mean(x.λ for x in tail) < 1.2h
    @test abs(mean(x.pull for x in tail) - (-h * (μ - c))) < 0.05h
    # The sums over readers are the site's own here.
    @test sites.precision_sum[1] ≈ last(history).λ
    @test sites.seen == [true]
    # One visit changes the coordinate's precision at most 2.5-fold: the first
    # visit, from no site, at most to 1.5 times the coordinate's precision 1/s².
    Λ = 1 / 0.1^2
    @test history[1].λ <= 1.5Λ + 1e-12
    t = (; globals=[1], units=[[2]], touched=[[1]])
    sites = BRM._brm_svi_sites(t, 1, [0.5])
    q = (; μ=[1.1], ρ=[log(0.1)], Β=zeros(1, 0))
    g = q.μ .+ 0.1                   # ε = 1; a reader curving a million times more
    BRM._brm_svi_update_sites!(sites, t, q, g, [1.0], [1], [(; gg=[2 * -1e6 * (g[1] - 0.3)])], 0.1, 1.0)
    @test sites.precision[1][1] ≈ 1.5Λ
end

@stestset "a site does not depend on where the coordinate's origin is" begin
    # The same reader in coordinates shifted by 1,000: identical precisions and
    # pulls. Damping the natural parameter ν instead adds (λ_old - λ_new) μ to the
    # pull, which drove window boundaries steadily off their optimum.
    _, a = svi_site_visits(; c=0.3, μ=1.1, visits=200)
    _, b = svi_site_visits(; c=1000.3, μ=1001.1, visits=200)
    @test all(isapprox(x.λ, y.λ; rtol=1e-6) for (x, y) in zip(a, b))
    @test all(isapprox(x.pull, y.pull; rtol=1e-6, atol=1e-6) for (x, y) in zip(a, b))
end

@stestset "a sparse global's gradient is its readers' sites" begin
    t = (; globals=[1, 2], units=[[3], [4]], touched=[[1], [1, 2]])
    sites = BRM._brm_svi_sites(t, 1, [0.5, 0.5])
    @test sites.globals == [2]
    sites.precision_sum[1], sites.pull_sum[1] = 30.0, 12.0
    q = (; μ=[0.5, 0.2], ρ=log.([1.0, 0.3]), Β=[0.1 0.2; 0.4 -0.5])
    # [μ; ρ; vec(Β)] for two globals and rank 2.
    grad = BRM._brm_svi_add_site_gradient!(collect(1.0:8.0), sites, q)
    @test grad[[1, 3, 5, 7]] == [1.0, 3.0, 5.0, 7.0]  # global 1: untouched
    @test grad[2] ≈ 2 + 12 - 30 * 0.2                 # μ: ν - λ μ
    @test grad[4] ≈ 4 - 30 * 0.3^2                    # ρ: -λ s²
    @test grad[6] ≈ 6 - 30 * 0.4                      # Β: -λ Β, column 1
    @test grad[8] ≈ 8 - 30 * -0.5                     # and column 2
    @test BRM._brm_svi_add_site_gradient!([1.0, 2.0], nothing, q) == [1.0, 2.0]
end

# Two stations of 32 windows of 8 states: 64 units, 62 boundaries read by two
# units each, at N/B = 32 (one reader draw per 16 iterations per boundary).
const SVI_WINDOW_DATA = svi_window_data(2, 32, 8)
const SVI_WINDOW_RK = RKBRMI(svi_window_model(SVI_WINDOW_DATA))
const SVI_WINDOW_EXACT = svi_window_exact(SVI_WINDOW_RK)
const SVI_WINDOW_OPT = (; batch_size=2, threads=1, unit_sampling=:uniform)

svi_window_nlme() = brm_nlme_model(SVI_WINDOW_RK;
    ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), subject_group=:win)

@stestset "boundaries read by two window units reach the family's optimum" begin
    m = svi_window_nlme()
    fit = brm_svi_fit(m; SVI_WINDOW_OPT..., rng=Xoshiro(101), ndraws=10)
    d = fit.diagnostics
    b = svi_boundaries(SVI_WINDOW_EXACT, d.q, d.globals)
    @test length(b.positions) == 62
    # Every boundary, and only they, step on sites; the no-op sd every unit reads
    # stays on the minibatch estimate.
    @test sort(d.sparse_globals.coordinates) == sort(d.globals[b.positions])
    @test all(==(2), d.sparse_globals.readers)
    @test all(≈(2 * 2 / 64), d.sparse_globals.expected_readers)
    @info "sites" mean_err_rms=sqrt(mean(abs2, b.z)) sd_median=median(b.r) sd_range=extrema(b.r)
    @test sqrt(mean(abs2, b.z)) < 1.0
    @test 0.9 < median(b.r) < 1.1
    @test 0.7 < minimum(b.r) && maximum(b.r) < 1.4

    # Control: the same run with the boundaries on the minibatch estimate, as
    # before the sites. It shows this check tells the two apart.
    t = BRM._brm_svi_target(m)
    t, q = BRM._brm_svi_initialize(t, nothing, MeanFieldGlobal(); start=nothing,
        inits=nothing, global_scale=0.1, unit_scale=1.0)
    c = BRM._brm_svi_controls(length(t.units); batch_size=2, max_iterations=100_000,
        lr_global=0.01, unit_step=0.1, polish_lr=0.05, window=nothing, patience=10,
        min_windows=40, polish_windows=20, unit_sampling=:uniform)
    run = BRM._brm_svi_run(Xoshiro(101), q, c, t)
    @test run.sites !== nothing
    run.sites = nothing
    BRM._brm_svi_fit!(run, t, c)
    old = svi_boundaries(SVI_WINDOW_EXACT, BRM._brm_svi_result_q(run), t.globals)
    @info "minibatch" mean_err_rms=sqrt(mean(abs2, old.z)) sd_median=median(old.r) sd_range=extrema(old.r)
    @test maximum(old.r) > 2 * maximum(b.r)
    @test sqrt(mean(abs2, old.z)) > sqrt(mean(abs2, b.z))
end

@stestset "a run with sites continues exactly from a checkpoint" begin
    m = svi_window_nlme()
    opt = (; SVI_WINDOW_OPT..., window=20, min_windows=6, patience=3, polish_windows=4)
    full = brm_svi_fit(m; opt..., rng=Xoshiro(7), ndraws=8)
    first_part = brm_svi_fit(m; opt..., rng=Xoshiro(7), stop_after=97, ndraws=0)
    state = brm_svi_optimizer_checkpoint(first_part)
    @test state.run.sites !== nothing && any(state.run.sites.seen)
    rest = brm_svi_resume(m, state; ndraws=8)
    @test svi_same_result(rest, full)
    @test rest.diagnostics.sparse_globals == full.diagnostics.sparse_globals
end
