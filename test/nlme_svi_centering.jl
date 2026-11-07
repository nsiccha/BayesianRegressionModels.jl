# Learned partial centering in minibatch SVI: `brm_svi_fit(...; centering=
# LearnedCentering())` (src/nlme_svi_centering.jl) on RK-lowered models.
#
# The centered target is checked against the plain one where the controls are
# zero (draw cells) and against central finite differences of its value in the
# globals, the learned coordinates and the controls, through RK's own scale,
# correlation and law-argument values: a correlated block located at its
# population coefficients, a scalar block on a Poisson model, and an authored
# Gaussian vector the units own. A fit learns controls that rise with a unit's
# data, returns draws in the packed RK coordinates, and resumes bit for bit.
#
# RUN: julia --project=test test/nlme_svi_centering.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, Random, Statistics, WarmupHMC

include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_owned_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_streaming_fixture.jl"))
using Serialization

const BRM_ = BayesianRegressionModels

# Poisson counts with a random intercept and `nobs[i]` observations for unit i.
function svi_centering_pois_data(rng, nobs; a=0.5, sa=0.8)
    u = sa .* randn(rng, length(nobs))
    (; subject=["s$i" for i in eachindex(nobs)],
       y=[[rand(rng, Poisson(exp(a + u[i]))) for _ in 1:n] for (i, n) in enumerate(nobs)])
end

svi_centering_pois_model(data) = @brm data begin
    a ~ 1 + (1 | subject)
    @plate for i in eachindex(a)
        y[i] ~ Poisson(exp.(a[i]))
    end
end

# A centered target with its controls at `c` (one vector per unit).
function centered_target(m, c)
    t = BRM_._brm_svi_target(m)
    plan = BRM_._brm_svi_centering_plan(t, LearnedCentering())
    q = (; centering=(; start=0.5, step=0.05, x=[BRM_._brm_svi_control_logit.(ci) for ci in c],
        opt=[BRM_._BRMSVIAdam(length(ci), 0.01) for ci in c]))
    merge(t, (; centering=plan)), q
end

# Unit i's centered term with its complete global gradient: the unit term's,
# the locations' and the RK pullback of its scale and law adjoints.
function centered_term(t, q, i, g, u)
    plan = t.centering
    frame = BRM_._brm_svi_centering_frame(plan, t, g)
    gg, gu = zeros(length(g)), zeros(length(u))
    v, gc, a = BRM_._brm_svi_centered_unit_term!(gg, gu, plan, frame, t, q, i, g, u)
    gg .+= BRM_._brm_svi_centering_batch_gradient(plan, t, frame, [(; gc, adjoints=(a,))], [2.0])
    (; v, gg, gu, gc)
end

function fd_vector(h, x; ε=1e-6)
    map(eachindex(x)) do j
        xp, xm = copy(x), copy(x)
        xp[j] += ε; xm[j] -= ε
        (h(xp) - h(xm)) / (2ε)
    end
end

# The checks shared by every model: plain parity of draw cells at c = 0, and
# finite differences of the centered term at interior controls.
function check_centered_terms(m; rng=Xoshiro(3), units=1:2, kinds)
    t0 = BRM_._brm_svi_target(m)
    plan = BRM_._brm_svi_centering_plan(t0, LearnedCentering())
    @test Set(cell.kind for cells in plan.cells for cell in cells) == Set(kinds)
    u0 = first(BRM_._brm_nlme_prior_median_point(m)) .+ 0.2 .* randn(rng, length(m.unit))
    g = u0[t0.globals]
    for i in units
        n = plan.ncells[i]
        @test n > 0
        u = u0[t0.units[i]]
        # Zero controls: a draw cell is its packed coordinate; a vector element
        # is standardized against its law.
        t, q = centered_target(m, [zeros(k) for k in plan.ncells])
        r = centered_term(t, q, i, g, u)
        if all(c -> c.kind === :ranef, plan.cells[i])
            gg, gz = zeros(length(g)), zeros(length(u))
            v = BRM_._brm_svi_unit_term!(gg, gz, t0, i, g, u)
            @test r.v ≈ v rtol = 1e-12
            @test r.gu ≈ gz rtol = 1e-10
            @test r.gg ≈ gg rtol = 1e-10 atol = 1e-12
        end
        # Interior controls: every gradient against finite differences, the
        # globals' through RK's values of the scales, correlations and laws.
        c = [0.15 .+ 0.7 .* rand(rng, k) for k in plan.ncells]
        t, q = centered_target(m, c)
        r = centered_term(t, q, i, g, u)
        @test r.gu ≈ fd_vector(x -> centered_term(t, q, i, g, x).v, u) rtol = 1e-5 atol = 1e-7
        @test r.gg ≈ fd_vector(x -> centered_term(t, q, i, x, u).v, g) rtol = 1e-5 atol = 1e-7
        @test r.gc ≈ fd_vector(c[i]) do x
            cx = copy(c); cx[i] = x
            centered_term(centered_target(m, cx)..., i, g, u).v
        end rtol = 1e-5 atol = 1e-7
    end
end

@stestset "a correlated block located at its population coefficients" begin
    rng = Xoshiro(1)
    m = brm_nlme_model(RKBRMI(svi_gauss_model(svi_gauss_data(rng, 8))); ad_backend=NLME_AD)
    plan = BRM_._brm_svi_centering_plan(BRM_._brm_svi_target(m), LearnedCentering())
    @test plan.ncells == fill(2, 8)
    @test all(cells -> all(c -> c.located, cells), plan.cells)
    @test isempty(plan.uncentered)
    check_centered_terms(m; kinds=(:ranef,))
end

@stestset "a scalar block on a Poisson model" begin
    rng = Xoshiro(2)
    m = brm_nlme_model(RKBRMI(svi_centering_pois_model(svi_centering_pois_data(rng,
        [2, 3, 30, 4, 25, 1]))); ad_backend=NLME_AD)
    check_centered_terms(m; units=(1, 3), kinds=(:ranef,))
end

@stestset "an authored Gaussian vector the units own" begin
    m = brm_nlme_model(RKBRMI(svi_owned_model(svi_owned_data())); ad_backend=NLME_AD)
    plan = BRM_._brm_svi_centering_plan(BRM_._brm_svi_target(m), LearnedCentering())
    @test all(cells -> count(c -> c.kind === :vector, cells) == 2, plan.cells)
    check_centered_terms(m; kinds=(:ranef, :vector))
end

# Measured with this data (seeds 5 and 6, 2026-11-07): the plain mean-field fit
# put the population intercept and log-scale 0.16-0.50 NUTS sds off with sds of
# 0.20 of NUTS's; learned centering put them within 0.15 sds with sds of
# 0.78-0.91, at every control step (0.05, 0.2, 0.5) and start (0.1, 0.5, 0.9),
# with controls of 0.60-0.69 on the 40-row units and 0.39-0.59 on the 1-row ones.
@stestset "a fit learns controls that rise with a unit's data" begin
    rng = Xoshiro(4)
    nobs = repeat([1, 40], 10)
    data = svi_centering_pois_data(rng, nobs; sa=0.5)
    rk = RKBRMI(svi_centering_pois_model(data))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    u0 = first(BRM_._brm_nlme_prior_median_point(m))
    reference = reduce(hcat, map(1:4) do chain
        run = adaptive_warmup_mcmc(Xoshiro(100 + chain), problem;
            init=(; position=copy(u0), squared_scale=ones(length(u0))), n_draws=1000)
        convert(Matrix{Float64}, run.posterior_position)
    end)
    μr, σr = vec(mean(reference; dims=2)), vec(std(reference; dims=2))
    options = (; batch_size=5, ndraws=2000)
    plain = brm_svi_fit(m; options..., rng=Xoshiro(5))
    fit = brm_svi_fit(m; options..., rng=Xoshiro(5), centering=LearnedCentering())
    report = fit.diagnostics.centering
    @test length(report.cells) == length(nobs)
    @test isempty(report.uncentered)
    c = [cell.c for cell in report.cells]
    @test all(0 .< c .< 1)
    @test mean(c[nobs .== 40]) > mean(c[nobs .== 1]) + 0.05
    # Draws are packed RK points, the same coordinates as the plain fit's.
    @test size(fit.draws) == size(plain.draws)
    @test fit.coordinates == plain.coordinates
    globals = fit.diagnostics.globals
    mean_error(f) = abs.(vec(mean(f.draws; dims=2)) .- μr) ./ σr
    ratio(f) = vec(std(f.draws; dims=2)) ./ σr
    @test all(mean_error(fit)[globals] .< 0.3)
    @test all(ratio(fit)[globals] .> 0.6)
    @test all(ratio(plain)[globals] .< 0.4)
    @test maximum(mean_error(fit)[setdiff(eachindex(μr), globals)]) < 0.3
    @test plain.diagnostics.centering === nothing
end

@stestset "a centered fit resumes bit for bit" begin
    rng = Xoshiro(6)
    m = brm_nlme_model(RKBRMI(svi_gauss_model(svi_gauss_data(rng, 10))); ad_backend=NLME_AD)
    options = (; batch_size=4, max_iterations=300, ndraws=20, centering=LearnedCentering())
    whole = brm_svi_fit(m; rng=Xoshiro(7), options...)
    part = brm_svi_fit(m; rng=Xoshiro(7), options..., stop_after=120)
    @test !part.diagnostics.finished
    resumed = brm_svi_resume(m, brm_svi_optimizer_checkpoint(part); ndraws=20)
    @test resumed.draws == whole.draws
    @test resumed.diagnostics.q.centering.x == whole.diagnostics.q.centering.x
    @test svi_same_q(resumed.diagnostics.q, whole.diagnostics.q)
    @test_throws ArgumentError brm_svi_fit(m; centering=:learned)
    @test_throws ArgumentError brm_svi_fit(m; centering=LearnedCentering(),
        laplace=GaussianBlockLaplace())
end

@stestset "a centered window carries its controls and transports its prior" begin
    data = svi_stream_data(Xoshiro(31), ["a", "b", "c"], 6)
    initial = brm_svi_fit(svi_stream_model(data); rng=Xoshiro(32), max_iterations=300,
        ndraws=0, batch_size=2, centering=LearnedCentering())
    saved = brm_svi_checkpoint(initial)
    @test saved.layout.centering == (; start=0.5, step=0.05)
    @test all(u -> count(!isnan, u.centering) == 1, values(saved.units))
    io = IOBuffer()
    serialize(io, saved)
    state = deserialize(seekstart(io))
    next = svi_stream_data(Xoshiro(33), ["c", "a", "new"], 4)
    @test_throws "carries the previous fit's centering" brm_svi_fit(state,
        svi_stream_model(next); centering=LearnedCentering(), SVI_STREAM_QUICK...)
    update = brm_svi_fit(state, svi_stream_model(next); rng=Xoshiro(34), max_iterations=0,
        ndraws=5)
    report = update.diagnostics.centering
    @test report !== nothing
    m = update.diagnostics.model
    # A carried unit starts from its controls, a new one at the start value.
    for row in report.cells
        old = get(state.units, row.unit, nothing)
        expected = old === nothing ? 0.5 :
            BRM_._brm_svi_control(only(filter(!isnan, old.centering)))
        @test row.c ≈ expected
    end
    t0 = BRM_._brm_svi_target(m)
    t, q = BRM_._brm_svi_initialize(t0, state, state.q.family; start=nothing, inits=nothing,
        global_scale=0.1, unit_scale=1.0)
    t, q = BRM_._brm_svi_with_centering(t, q, LearnedCentering())
    rng = Xoshiro(35)
    g = q.μ .+ 0.1 .* randn(rng, length(q.μ))
    for (i, label) in enumerate(BRM_._brm_svi_unit_levels(t))
        u = q.M[i] .+ 0.3 .* randn(rng, length(q.M[i]))
        r = centered_term(t, q, i, g, u)
        if haskey(state.units, label)
            # Unchanged controls: the carried factor at u itself, plus the
            # likelihood at the packed coordinates.
            old = state.units[label]
            frame = BRM_._brm_svi_centering_frame(t.centering, t, g)
            y = BRM_._brm_svi_centered_coordinates(t.centering, frame, q, i, g, u)
            gg, gy = zeros(length(g)), zeros(length(y))
            lik = BRM_._brm_svi_unit_partition!(gg, gy, t, i, g, y, t.plan) +
                BRM_._brm_svi_field_term!(gg, gy, t, i, y, first(t.work))
            Σ = inv(old.Λ)
            @test r.v ≈ lik + logpdf(MvNormal(old.M, Symmetric(Σ)), u) rtol = 1e-10
            # Moved controls: the prior goes through the frame transport.
            q.centering.x[i] .+= 0.8
            r = centered_term(t, q, i, g, u)
            @test r.gu ≈ fd_vector(x -> centered_term(t, q, i, g, x).v, u) rtol = 1e-5 atol = 1e-7
            @test r.gg ≈ fd_vector(x -> centered_term(t, q, i, x, u).v, g) rtol = 1e-5 atol = 1e-7
            x0 = copy(q.centering.x[i])
            gx = fd_vector(x0) do x
                q.centering.x[i] .= x
                v = centered_term(t, q, i, g, u).v
                q.centering.x[i] .= x0
                v
            end
            c = BRM_._brm_svi_control.(x0)
            @test r.gc .* c .* (1 .- c) ≈ gx rtol = 1e-5 atol = 1e-7
        end
    end
    # The absent unit keeps its factor and controls in the next checkpoint.
    carried = brm_svi_checkpoint(update)
    @test carried.units["b"].centering == state.units["b"].centering
    @test carried.units["b"].M == state.units["b"].M
end
