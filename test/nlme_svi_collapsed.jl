# Sparse globals with their readers integrated out (`sparse_globals = :collapsed`,
# src/nlme_svi_collapsed.jl). Snag brm-svi-fit-gmrf-ab199f57: on a space-time toy of
# window units with a read-pair `gmrf` field read by many windows and boundary states
# read by two, brm_svi_fit's per-coordinate sites left the field and boundary means
# 1.7 / 1.1 posterior sds from the exact posterior after 78 passes over the data at
# fixed hyperparameters. Collapsed sites with a joint Newton step of the sparse
# globals reach the exact means once every reader has been visited.
#
# The toy here is that model, small: S stations, each an OU chain cut into windows,
# plus a field state per station-month that every window of the station-month reads.
# The field's sd and lengths and the chain's parameters are data, so the target is
# exactly Gaussian (but for the no-op subject sd, at its prior's mode), and its exact
# means come from the Hessian.
#
# RUN: julia --project=test test/nlme_svi_collapsed.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
using LinearAlgebra, Random, SparseArrays, Statistics
import LogDensityProblems

function svi_collapsed_path(x0, e, a, s)
    out = zeros(typeof(x0 * a * s), length(e))
    prev = x0
    for t in eachindex(e)
        prev = a * prev + s * e[t]
        out[t] = prev
    end
    out
end

# `S` stations over `M` months of `WM` windows of `L` states each. Window w of station
# s reads boundary states xb[(s-1)(W+1)+w] and xb[(s-1)(W+1)+w+1] and the field state
# of its station-month.
function svi_collapsed_data(S, M, WM, L; ℓ=20.0, τ=1.0, σf=1.0, seed=1)
    rng = Xoshiro(seed)
    east, north = 2 .* rand(rng, S), 2 .* rand(rng, S)
    W = M * WM
    T = W * L + 1
    a, v = exp(-1 / ℓ), -expm1(-2 / ℓ)
    month(t) = min((t - 1) ÷ (WM * L) + 1, M)
    f = σf .* randn(rng, S * M)
    win, y, ids, lb, rb = String[], Vector{Vector{Float64}}(), Vector{Vector{Int}}(), Int[], Int[]
    ce, cn, cm = Float64[], Float64[], Float64[]
    next = 0
    for s in 1:S
        x = zeros(T)
        x[1] = τ * randn(rng)
        for t in 2:T
            x[t] = a * x[t-1] + τ * sqrt(v) * randn(rng)
        end
        x .+= [f[(month(t) - 1) * S + s] for t in 1:T] .+ randn(rng, T)
        for w in 1:W
            lo = (w - 1) * L + 1
            push!(win, "s$(s)w$w"); push!(y, x[lo:lo+L-1])
            push!(ids, collect(next+1:next+L-1)); next += L - 1
            push!(lb, (s - 1) * (W + 1) + w); push!(rb, (s - 1) * (W + 1) + w + 1)
            push!(ce, east[s]); push!(cn, north[s]); push!(cm, Float64(month(lo)))
        end
    end
    n = length(win)
    (; win, y, ids, lb, rb, east=ce, north=cn, month=cm, state=collect(1:n),
       nint=fill(L - 1, n), closing=zeros(n), nb=S * (W + 1), ne=next, ar=a, sig=τ * sqrt(v),
       nbr=S - 1, sf0=σf, ls0=1.0, lm0=2.0)
end

svi_collapsed_model(data) = @brm data begin
    field ~ 0 + gmrf((east, north), month; neighbors=nbr)
    sd(field, gmrf((east, north), month)) = sf0
    length_scale(field, gmrf((east, north), month), space) = ls0
    length_scale(field, gmrf((east, north), month), time) = lm0
    xb[1:nb] .~ Normal(0.0, 30.0)
    e[1:ne] .~ Normal(0.0, 1.0)
    sd(mu, win) ~ LogNormal(-7.0, 0.01)
    mu ~ 0 + (1 | win)
    @plate for i in eachindex(state)
        h = svi_collapsed_path(xb[lb[i]], e[ids[i]], ar, sig)
        y[i] ~ Normal(mu[i] .+ field[state[i]] .+ vcat(xb[lb[i]], h), 1.0)
        closing[i] ~ Normal(xb[rb[i]] - ar * h[nint[i]], sig)
    end
end

# The exact Gaussian: H = -∇² log p from gradient differences (exact but in the no-op
# subject sd, which sits at its prior's mode), its mean, and the marginal sds.
function svi_collapsed_exact(rk)
    p = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    n = LogDensityProblems.dimension(p)
    names = string.(coordinate_names(rk.model.layout))
    scale = findall(c -> occursin(".tau", c), names)
    function hessian(x0)
        _, g0 = LogDensityProblems.logdensity_and_gradient(p, x0)
        H = zeros(n, n)
        for j in 1:n
            δ = j in scale ? 1e-4 : 1.0
            xp = copy(x0); xp[j] += δ
            H[:, j] .= -(LogDensityProblems.logdensity_and_gradient(p, xp)[2] .- g0) ./ δ
        end
        Symmetric((H + H') / 2), g0
    end
    x = zeros(n); x[scale] .= -7.0
    for _ in 1:2
        H, g = hessian(x)
        x = x .+ (cholesky(H) \ g)
    end
    H, g = hessian(x)
    @test maximum(abs, g) < 1e-2          # control: the mean is the mode
    (; H, μ=x, sd=sqrt.(diag(inv(cholesky(H)))), names, scale)
end

# rms error of a fit's means, in exact posterior sds: the sparse globals' and every
# unit coordinate's (a centering-free fit's factors are over the packed coordinates).
function svi_collapsed_errors(ex, d)
    sparse = d.sparse_globals.coordinates
    kg = findall(in(sparse), d.globals)
    zg = (d.q.μ[kg] .- ex.μ[d.globals[kg]]) ./ ex.sd[d.globals[kg]]
    zu = Float64[]
    for (i, block) in enumerate(d.units)
        keep = [k for (k, c) in enumerate(block) if !(c in ex.scale)]
        append!(zu, (d.q.M[i][keep] .- ex.μ[block[keep]]) ./ ex.sd[block[keep]])
    end
    (; sparse=sqrt(mean(abs2, zg)), units=sqrt(mean(abs2, zu)))
end

# Four stations, two months of four windows of 8 states: 32 units, 8 field states
# read by 4 windows each and 28 interior boundaries read by 2, every one sparse at
# batch 2 (N / B = 16 iterations per pass over the data).
const SVI_COLLAPSED_DATA = svi_collapsed_data(4, 2, 4, 8)
const SVI_COLLAPSED_RK = RKBRMI(svi_collapsed_model(SVI_COLLAPSED_DATA))
const SVI_COLLAPSED_EXACT = svi_collapsed_exact(SVI_COLLAPSED_RK)
const SVI_COLLAPSED_OPT = (; batch_size=2, threads=1, unit_sampling=:uniform)

svi_collapsed_nlme() = brm_nlme_model(SVI_COLLAPSED_RK;
    ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), subject_group=:win)

@stestset "collapsed sites bring field and boundary states to the exact means" begin
    m = svi_collapsed_nlme()
    ex = SVI_COLLAPSED_EXACT
    budget = (; SVI_COLLAPSED_OPT..., max_iterations=320, centering=nothing, ndraws=10)
    fit = brm_svi_fit(m; budget..., rng=Xoshiro(101), sparse_globals=:collapsed)
    d = fit.diagnostics
    sg = d.sparse_globals
    @test sg.mode === :collapsed
    names = ex.names[sg.coordinates]
    @test count(n -> occursin(".field.", n), names) == 8
    @test count(n -> startswith(n, "xb."), names) == 28
    @test length(names) == 36
    @test sg.units_seen == 32
    @test sg.solves == d.iterations
    @test sg.skipped_steps == 0
    # 2k probe evaluations per visit come on top of the units' own.
    reference = brm_svi_fit(m; budget..., rng=Xoshiro(101))
    @test !hasproperty(reference.diagnostics.sparse_globals, :mode)
    err = svi_collapsed_errors(ex, d)
    old = svi_collapsed_errors(ex, reference.diagnostics)
    @info "collapsed against sites, 20 passes" err old sg.prior_colors sg.prior_assemblies sg.fallbacks
    @test err.sparse < 0.05
    @test err.units < 0.05
    # Control: today's sites, same budget and seed, are far from the exact means.
    @test old.sparse > 10 * err.sparse
    @test sg.fallbacks == 0
    # The joint factor's sds are the sparse globals' exact marginal sds (the sites
    # integrate the units out exactly here), and they are diag(P⁻¹) of the stored P.
    kg = findall(in(sg.coordinates), d.globals)
    ratio = exp.(d.q.ρ[kg]) ./ ex.sd[d.globals[kg]]
    @info "sparse globals' sds over the exact marginal sds" extrema(ratio)
    @test all(r -> 0.97 < r < 1.03, ratio)
    sites = d.optimizer.run.sites
    S = sites.sum
    P = SparseMatrixCSC(size(S)..., S.colptr, S.rowval, sites.prior .+ nonzeros(S))
    @test exp.(2 .* d.q.ρ[sites.base.globals]) ≈ diag(inv(Matrix(P))) rtol = 1e-8
    # Control: the default's diagonal widths are the conditional ones, far narrower
    # for the field states.
    oldratio = exp.(reference.diagnostics.q.ρ[kg]) ./ ex.sd[d.globals[kg]]
    @info "default sites' sds over the exact marginal sds" extrema(oldratio)
end

@stestset "collapsed sites under the default centering and coupling" begin
    m = svi_collapsed_nlme()
    ex = SVI_COLLAPSED_EXACT
    fit = brm_svi_fit(m; SVI_COLLAPSED_OPT..., max_iterations=640, rng=Xoshiro(5),
        ndraws=2000, sparse_globals=:collapsed)
    d = fit.diagnostics
    @test d.sparse_globals.mode === :collapsed
    # The window units stay coupled to their sparse globals.
    sparse = Set(findall(in(d.sparse_globals.coordinates), d.globals))
    @test any(u -> any(in(sparse), u.columns), d.coupling.units)
    # Means and sds from the draws (packed coordinates), against the exact posterior.
    m̂ = vec(mean(fit.draws; dims=2))
    ŝ = vec(std(fit.draws; dims=2))
    keep = setdiff(1:length(ex.μ), ex.scale)
    z = (m̂[keep] .- ex.μ[keep]) ./ ex.sd[keep]
    sparse = d.globals[findall(in(d.sparse_globals.coordinates), d.globals)]
    r = ŝ[sparse] ./ ex.sd[sparse]
    @info "collapsed, default centering and coupling" rms=sqrt(mean(abs2, z)) worst=maximum(abs, z) sparse_sd_ratio=extrema(r)
    @test sqrt(mean(abs2, z)) < 0.1
    @test all(x -> 0.9 < x < 1.1, r)
end

@stestset "a collapsed run continues exactly from a checkpoint" begin
    m = svi_collapsed_nlme()
    opt = (; SVI_COLLAPSED_OPT..., window=20, min_windows=6, patience=3, polish_windows=4,
        sparse_globals=:collapsed)
    full = brm_svi_fit(m; opt..., rng=Xoshiro(7), ndraws=8)
    first_part = brm_svi_fit(m; opt..., rng=Xoshiro(7), stop_after=97, ndraws=0)
    state = brm_svi_optimizer_checkpoint(first_part)
    @test state.run.sites isa BRM._BRMSVICollapsedSites && any(state.run.sites.seen)
    rest = brm_svi_resume(m, state; ndraws=8)
    @test svi_same_result(rest, full)
    @test rest.diagnostics.sparse_globals == full.diagnostics.sparse_globals
end

@stestset "a collapsed run is the same at any number of tasks" begin
    # The probes run inside the threaded unit loop, each in its batch slot; the sites
    # and the solve follow serially in batch order. Needs real threads (`--threads=2`).
    if Threads.nthreads() < 2 || !BRM._brm_concurrent_gradients(nothing)
        @info "skipped: needs --threads=2 or more and Enzyme's concurrent-GC repair"
        @test true
    else
        m = svi_collapsed_nlme()
        opt = merge(SVI_COLLAPSED_OPT, (; max_iterations=200, sparse_globals=:collapsed,
            ndraws=8, window=20, min_windows=4, patience=2, polish_windows=3))
        one = brm_svi_fit(m; opt..., rng=Xoshiro(11))
        two = brm_svi_fit(m; merge(opt, (; threads=2))..., rng=Xoshiro(11))
        @test svi_same_result(one, two)
    end
end

@stestset "a reader's collapsed site is its Schur complement" begin
    # One unit z ∈ R² reading two sparse globals g ∈ R², term -½ [z; g]ᵀ H [z; g] + bᵀ[z; g].
    H = [4.0 1.0 1.5 0.0; 1.0 3.0 0.5 1.0; 1.5 0.5 2.0 0.3; 0.0 1.0 0.3 1.5]
    b = [0.4, -0.2, 0.7, 0.1]
    Λ = H[3:4, 3:4] - H[3:4, 1:2] * (H[1:2, 1:2] \ H[1:2, 3:4])
    # The profile gradient at a point (z0, g0): G_g - H_gz H_zz⁻¹ G_z.
    z0, g0 = [0.3, -0.1], [0.2, 0.5]
    G = b .- H * [z0; g0]
    Gp = G[3:4] - H[3:4, 1:2] * (H[1:2, 1:2] \ G[1:2])
    # The site's gradient at x is Gp - Λ (x - g0); it vanishes at the conditional
    # optimum of the globals given z integrated out.
    x̂ = g0 .+ Λ \ Gp
    full = H \ b
    @test x̂ ≈ full[3:4]
    sites = BRM._brm_svi_collapsed_sites(BRM._brm_svi_sites(
        (; globals=[1, 2], units=[[3, 4]], touched=[[1, 2]]), 1, [0.5]),
        (; globals=[1, 2], units=[[3, 4]], touched=[[1, 2]]))
    q = (; μ=[0.0, 0.0], ρ=[0.0, 0.0], Β=zeros(2, 0))
    BRM._brm_svi_collapsed_update!(sites, q, [1], [(; Λ, G=Gp, point=g0, C=nothing)], 1.0)
    @test Matrix(sites.sum) ≈ Λ
    @test sites.pull_sum ≈ Gp .+ Λ * g0
    # Moving the coordinates' origin moves ν, not the site's gradient at the mean.
    q2 = (; μ=[3.0, -2.0], ρ=[0.0, 0.0], Β=zeros(2, 0))
    s2 = deepcopy(sites)
    BRM._brm_svi_collapsed_update!(s2, q2, [1], [(; Λ, G=Gp, point=g0, C=nothing)], 0.5)
    @test s2.pull_sum .- Matrix(s2.sum) * q2.μ ≈ Gp .+ Λ * (g0 .- q2.μ)
end

@stestset "sparse_globals refusals" begin
    m = svi_collapsed_nlme()
    @test_throws ArgumentError brm_svi_fit(m; SVI_COLLAPSED_OPT..., max_iterations=2,
        sparse_globals=:joint)
    err = try
        brm_svi_fit(m; laplace=GaussianBlockLaplace(), sparse_globals=:collapsed)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("sparse_globals", err.msg)
end
