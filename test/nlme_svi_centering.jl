# Learned partial centering in minibatch SVI: `brm_svi_fit(...; centering=
# LearnedCentering())` (src/nlme_svi_centering.jl) on RK-lowered models.
#
# The centered target is checked against the plain one where the controls are
# zero (draw cells) and against central finite differences of its value in the
# globals, the learned coordinates and the controls, through RK's own scale,
# correlation and law-argument values: a correlated block located at its
# population coefficients, a scalar block on a Poisson model, an authored
# Gaussian vector the units own, and a block over a `shared_groups` grouping
# whose levels units own (an absent carried unit's level mapped through its
# controls in the next window). A sparse global that is a centred cell's scale
# steps on sites that see its readers' whole centered terms. A grouped HSGP's
# weights are cells scaled by its spectral scales, and its centered fit beats
# the plain one against NUTS. A fit learns controls that rise with a unit's
# data, returns draws in the packed RK coordinates, and resumes bit for bit.
#
# RUN: julia --project=test test/nlme_svi_centering.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random, Statistics, WarmupHMC

include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_owned_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_streaming_fixture.jl"))
include(joinpath(@__DIR__, "nlme_grouped_hsgp_fixture.jl"))
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
        # Zero controls: a draw cell (a random effect's or an HSGP level's
        # standardized weight) is its packed coordinate; a vector element is
        # standardized against its law.
        t, q = centered_target(m, [zeros(k) for k in plan.ncells])
        r = centered_term(t, q, i, g, u)
        if all(c -> c.kind in (:ranef, :hsgp), plan.cells[i])
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

# A device grouping crossed with the subjects (`shared_groups`): each subject's
# rows read a device of its own, except the subjects in `shared`, whose rows all
# read "dshared".
function svi_centering_device_data(rng, subjects; n=8, shared=())
    subject, device, x, y = String[], String[], Float64[], Float64[]
    for s in subjects, _ in 1:n
        h = sum(Int, codeunits(s))
        push!(subject, s)
        push!(device, s in shared ? "dshared" : "d" * s)
        push!(x, randn(rng))
        push!(y, 0.8 + 0.4sin(h) + 0.3cos(h) * x[end] + 0.5randn(rng))
    end
    (; subject, device, x, y)
end

# The units' block has no intercept, so the population intercept is the device
# block's location: a device's total is `mu_Intercept + τ z`.
svi_centering_device_model(data) = @brm data begin
    mu ~ 1 + (1 | device) + (0 + x | subject)
    sd(mu, device) ~ LogNormal(log(0.5), 0.4)
    y ~ Normal(mu, 0.5)
end

# Here the units' block absorbs the intercept, so the device block's cells are
# located at 0.
svi_centering_device_subject_model(data) = @brm data begin
    mu ~ 1 + (1 | subject) + (1 | device)
    sd(mu, device) ~ LogNormal(log(0.5), 0.4)
    y ~ Normal(mu, 0.5)
end

const SVI_CENTERING_DEVICES = (; subject_group=:subject, shared_groups=(:device,))

# snag svi-learned-cent-4ab361d3: a shared grouping's block was in neither
# `cells` nor `uncentered`.
@stestset "a non-finite centered term carries packed points, not learned coordinates" begin
    rng = Xoshiro(2)
    rk = RKBRMI(svi_centering_pois_model(svi_centering_pois_data(rng, [2, 3, 30, 4, 25, 1])))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    plan = BRM_._brm_svi_centering_plan(BRM_._brm_svi_target(m), LearnedCentering())
    t, q = centered_target(m, [fill(0.6, k) for k in plan.ncells])
    full = merge(q, BRM_._brm_svi_params(t, m.unit[t.globals], MeanFieldGlobal();
        global_scale=0.1, unit_scale=1.0))
    # exp of the linear predictor overflows at this finite point of unit 3.
    full.M[3] .= 1e4
    e = try
        BRM_._brm_svi_iteration!(Xoshiro(4), t, full, BRM_._brm_svi_adam(full, 0.01), [5, 3],
            0.01, 0.1)
        nothing
    catch err
        err
    end
    @test e isa BRMSVINonfiniteError
    p = only(e.terms)
    @test p.unit == 3 && p.draw_finite
    # The packed points are the map of the factor's draws: back through the learned
    # coordinates they are a pair around the factor's mean at the point's own globals.
    g = p.points.plus[t.globals]
    frame = BRM_._brm_svi_centering_frame(plan, t, g)
    u₊ = BRM_._brm_svi_learned_coordinates(plan, frame, full, 3, g, p.points.plus[t.units[3]])
    u₋ = BRM_._brm_svi_learned_coordinates(plan, frame, full, 3, g, p.points.minus[t.units[3]])
    @test (u₊ .+ u₋) ./ 2 ≈ full.M[3]
    # Positive control: the map moves them, so a point holding the learned coordinates fails.
    @test p.points.plus[t.units[3]] != u₊
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    @test !isfinite(LogDensityProblems.logdensity(problem, p.points.plus))
end

@stestset "a shared grouping's levels a unit owns are centred in that unit" begin
    subjects = ["s$i" for i in 1:6]
    data = svi_centering_device_data(Xoshiro(41), subjects; shared=("s5", "s6"))
    for (model, located) in ((svi_centering_device_model, true),
            (svi_centering_device_subject_model, false))
        m = brm_nlme_model(RKBRMI(model(data)); ad_backend=NLME_AD, SVI_CENTERING_DEVICES...)
        t = BRM_._brm_svi_target(m)
        @test BRM_._brm_svi_unit_levels(t) == subjects
        plan = BRM_._brm_svi_centering_plan(t, LearnedCentering())
        # Every unit keeps its own block's cell; units 1-4 add their device's.
        for (i, cells) in enumerate(plan.cells)
            devices = [c for c in cells if c.block.group === :device]
            @test count(c -> c.block.group === :subject, cells) == 1
            @test [c.level for c in devices] == (i <= 4 ? ["d" * subjects[i]] : String[])
            @test all(c -> c.located == located && c.coefficient === :Intercept, devices)
        end
        # The device two units read is global, and says so.
        @test length(plan.uncentered) == 1
        row = only(plan.uncentered)
        @test (row.kind, row.block.group, row.level) == (:ranef, :device, "dshared")
        @test row.reason == "its draws are not one unit's"
        @test only(row.coordinates) in m.view.coordinates[t.globals]
        check_centered_terms(m; units=(1, 5), kinds=(:ranef,))
    end
    # The fit's report: four device cells, the shared device uncentred.
    m = brm_nlme_model(RKBRMI(svi_centering_device_model(data)); ad_backend=NLME_AD,
        SVI_CENTERING_DEVICES...)
    fit = brm_svi_fit(m; batch_size=2, max_iterations=200, ndraws=10, rng=Xoshiro(42),
        centering=LearnedCentering())
    report = fit.diagnostics.centering
    @test sort([c.level for c in report.cells if c.block.group === :device]) ==
        ["ds1", "ds2", "ds3", "ds4"]
    @test [r.level for r in report.uncentered] == ["dshared"]
    @test all(c -> 0 < c.c < 1, report.cells)
end

# A window keeps the shared grouping's level table, so a carried unit absent
# from it still holds its device's draw, by packed name, with no reader. Its
# draws reach the packed coordinates through its own controls, at the draw's
# location and scale: z = (u - c μ) / τ^c.
@stestset "an absent unit's centred shared level maps through its controls" begin
    first_window = svi_centering_device_data(Xoshiro(43), ["a", "b", "c"])
    initial = brm_svi_fit(svi_centering_device_model(first_window); rng=Xoshiro(44),
        max_iterations=300, ndraws=0, batch_size=2, centering=LearnedCentering(),
        SVI_CENTERING_DEVICES...)
    state = brm_svi_checkpoint(initial)
    unit = state.units["b"]
    k = only(findall(key -> hasproperty(key, :coordinate), unit.keys))
    @test !isnan(unit.centering[k])
    next = svi_centering_device_model(svi_centering_device_data(Xoshiro(45), ["c", "a"]))
    update = brm_svi_fit(state, next; rng=Xoshiro(46), max_iterations=0, ndraws=20)
    m = update.diagnostics.model
    j = findfirst(==(unit.keys[k].coordinate), m.view.coordinates)
    @test j !== nothing
    @test all(isfinite, update.draws[j, :])
    t = BRM_._brm_svi_target(m; previous=state)
    @test j in t.ownership.absent
    plan = BRM_._brm_svi_centering_plan(t, LearnedCentering())
    @test haskey(plan.absent_blocks, j)
    @test isempty(plan.uncentered)
    names = string.(m.view.coordinates[t.globals])
    g = first(BRM_._brm_nlme_prior_median_point(m))[t.globals] .+
        0.3 .* randn(Xoshiro(47), length(t.globals))
    frame = BRM_._brm_svi_centering_frame(plan, t, g)
    μ = g[findfirst(==("mu_Intercept"), names)]
    τ = exp(g[findfirst(n -> occursin("device.tau", n), names)])
    c = BRM_._brm_svi_control(unit.centering[k])
    z = randn(Xoshiro(48), length(unit.keys))
    expected = copy(z)
    expected[k] = (z[k] - c * μ) / τ^c
    @test BRM_._brm_svi_absent_draw!(copy(z), plan, frame, unit, [k], [j]) ≈ expected
end

# Sixteen units, two of them in a stratum of their own with 30 rows each. At
# batch 2 that stratum's sd has fewer than one reader per batch on average, so
# it steps on its readers' sites (src/nlme_svi_sites.jl), and it is the scale of
# both readers' centred cells.
function svi_centering_strat_data(; seed=1, common=14, rare=2, nc=3, nr=30)
    rng = Xoshiro(seed)
    stratum = vcat(fill("common", common), fill("rare", rare))
    nobs = vcat(fill(nc, common), fill(nr, rare))
    u = [s == "rare" ? 1.5randn(rng) : 0.3randn(rng) for s in stratum]
    (; subject=["s$i" for i in eachindex(stratum)], stratum,
       y=[1.0 + u[i] .+ 0.5 .* randn(rng, nobs[i]) for i in eachindex(stratum)])
end

svi_centering_strat_model(data) = @brm data begin
    sigma ~ Exponential(1)
    a ~ 1 + (1 | gr(subject, by=stratum))
    @plate for i in eachindex(a)
        y[i] ~ Normal(a[i], sigma)
    end
end

# Measured with this data (seeds 11 and 12, 2026-11-07): the rare stratum's sd
# came out 0.08 and 0.10 NUTS sds off with an sd 1.01 and 0.96 times NUTS's, as
# with every global on the minibatch estimate (0.04, 0.12; 1.02, 0.94), and the
# plain fit 1.02 and 0.87 off at 0.14 times. The rare units' controls were
# 0.95-0.97, the others' 0.31-0.55. At the probe's point below, the unit term's
# direct gradient in that sd was -10.6 and +5.3, the complete one -2.4 and -0.006.
@stestset "a sparse global's site sees its reader's whole centered term" begin
    data = svi_centering_strat_data()
    rk = RKBRMI(svi_centering_strat_model(data))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    names = string.(coordinate_names(rk.model.layout))
    t0 = BRM_._brm_svi_target(m)
    N = length(t0.units)
    sites = BRM_._brm_svi_sites(t0, 2, fill(1 / N, N))
    @test names[t0.globals[sites.globals]] == ["b_subject__by__stratum.tau.2.1"]
    rare = findall(!isempty, sites.entries)
    common = findfirst(isempty, sites.entries)
    @test sort(BRM_._brm_svi_unit_levels(t0)[rare]) == ["s15", "s16"]

    # Each reader's site gradient is its complete centered gradient: the batch's
    # sum for a lone reader, its own pullback when both are drawn.
    plan = BRM_._brm_svi_centering_plan(t0, LearnedCentering())
    t, q = centered_target(m, [fill(0.85, k) for k in plan.ncells])
    u0 = first(BRM_._brm_nlme_prior_median_point(m)) .+ 0.3 .* randn(Xoshiro(3), length(m.unit))
    g = u0[t.globals]
    frame = BRM_._brm_svi_centering_frame(plan, t, g)
    function pieces(i)
        u = u0[t.units[i]]
        gg, gu = zeros(length(g)), zeros(length(u))
        v, gc, a = BRM_._brm_svi_centered_unit_term!(gg, gu, plan, frame, t, q, i, g, u)
        touched = BRM_._brm_svi_unit_touched(t, i)
        (; term=(; value=v, gg=2 .* gg[touched], gz=gu, gd=zero(gu)),
           centered=(; gc, adjoints=(a, a)), direct=gg, complete=centered_term(t, q, i, g, u).gg,
           touched)
    end
    for batch in ([rare[1], common], rare, [common, rare[2], rare[1]])
        rs = pieces.(batch)
        terms = [r.term for r in rs]
        weights = fill(N / length(batch), length(batch))
        centered = [r.centered for r in rs]
        chain = BRM_._brm_svi_centering_batch_gradient(plan, t, frame, centered, weights)
        BRM_._brm_svi_centering_site_terms!(terms, sites, plan, t, frame, centered, batch,
            weights, chain)
        for (k, i) in enumerate(batch), entry in sites.entries[i]
            p = rs[k].touched[entry]
            @test terms[k].gg[entry] / 2 ≈ rs[k].complete[p] rtol = 1e-10 atol = 1e-12
            # The direct gradient alone is not it: the check tells them apart.
            @test abs(rs[k].direct[p] - rs[k].complete[p]) > 1
        end
    end
    check_centered_terms(m; units=rare, kinds=(:ranef,))

    # A centered fit steps the sd on its readers' sites and recovers it.
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    x0 = first(BRM_._brm_nlme_prior_median_point(m))
    reference = reduce(hcat, map(1:4) do chain
        run = adaptive_warmup_mcmc(Xoshiro(100 + chain), problem;
            init=(; position=copy(x0), squared_scale=ones(length(x0))), n_draws=1000)
        convert(Matrix{Float64}, run.posterior_position)
    end)
    μr, σr = vec(mean(reference; dims=2)), vec(std(reference; dims=2))
    fit = brm_svi_fit(m; batch_size=2, threads=1, unit_sampling=:uniform, ndraws=10,
        rng=Xoshiro(11), centering=LearnedCentering())
    d = fit.diagnostics
    sd = t0.globals[only(sites.globals)]
    @test d.sparse_globals.coordinates == [sd]
    k = findfirst(==(sd), d.globals)
    @test abs(d.q.μ[k] - μr[sd]) / σr[sd] < 0.3
    @test 0.8 < exp(d.q.ρ[k]) / σr[sd] < 1.25
    rare_levels = BRM_._brm_svi_unit_levels(t0)[rare]
    c_rare = [cell.c for cell in d.centering.cells if cell.unit in rare_levels]
    c_common = [cell.c for cell in d.centering.cells if !(cell.unit in rare_levels)]
    @test minimum(c_rare) > maximum(c_common) + 0.2
end

# Grouped HSGPs whose levels are the units (test/nlme_grouped_hsgp_fixture.jl):
# each level's standardized weights are cells scaled by the term's spectral
# scales, the program value `<id>.spectral_scale`, a row per level when a
# hyper-predictor scales `sd` by level.
@stestset "a grouped HSGP's weights scaled by its spectral scales" begin
    for (brmi, rows) in ((flat_hsgp(GH_FLAT), false), (plate_hyper(GH_PLATE), true))
        m = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
        t = BRM_._brm_svi_target(m)
        plan = BRM_._brm_svi_centering_plan(t, LearnedCentering())
        hsgp = [cell for cells in plan.cells for cell in cells if cell.kind === :hsgp]
        @test length(hsgp) == 3length(t.units)
        @test all(cell -> !cell.located, hsgp)
        spectral = only(f for f in plan.families if f.spectral && f.stratum === nothing ||
            f.spectral && rows && f.stratum == 1)
        @test endswith(string(spectral.record.scale), ".spectral_scale")
        @test count(f -> f.spectral, plan.families) == (rows ? length(t.units) : 1)
        @test !any(r -> r.kind === :hsgp && any(c -> occursin(".z.", string(c)),
            r.coordinates), plan.uncentered)
        check_centered_terms(m; kinds=(:ranef, :hsgp))
    end
    # The floor: exact above exp(-110), exp(-150) at zero, slope 1 at both ends.
    @test BRM_._brm_svi_spectral_floor(1e-20) == exp(log(1e-20))
    @test BRM_._brm_svi_spectral_floor(0.0) ≈ exp(-150.0)
    @test BRM_._brm_svi_spectral_slope(0.0) == 1.0
    @test BRM_._brm_svi_spectral_slope(0.3) == 1.0
    for s in (exp(-140.0), exp(-150.0), exp(-155.0))
        h = 1e-6 * s
        fd = (BRM_._brm_svi_spectral_floor(s + h) - BRM_._brm_svi_spectral_floor(s - h)) / 2h
        @test BRM_._brm_svi_spectral_slope(s) ≈ fd rtol = 1e-5
    end
    # A centered fit learns a control per weight.
    m = brm_nlme_model(RKBRMI(flat_hsgp(GH_FLAT)); ad_backend=NLME_AD)
    fit = brm_svi_fit(m; batch_size=2, max_iterations=400, ndraws=20, rng=Xoshiro(9),
        centering=LearnedCentering())
    c = [cell.c for cell in fit.diagnostics.centering.cells if cell.kind === :hsgp]
    @test length(c) == 12 && all(0 .< c .< 1)
    @test all(isfinite, fit.draws)
end

# Curves by subject, alternating 40 and 3 rows: a grouped HSGP's weights are
# informed on the rich units and near their prior on the poor ones.
function svi_centering_curve_data(; seed=1, N=12, rich=40, poor=3)
    rng = Xoshiro(seed)
    subject, t, y = String[], Float64[], Float64[]
    for i in 1:N
        n = isodd(i) ? rich : poor
        a, b, c = 0.8randn(rng), 0.8randn(rng), 0.5randn(rng)
        for _ in 1:n
            x = 2rand(rng) - 1
            push!(subject, "s$i"); push!(t, x)
            push!(y, a * sin(pi * x) + b * cos(pi * x / 2) + c * x + 0.3randn(rng))
        end
    end
    (; subject, t, y)
end
svi_centering_curve_model(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(t; by=subject, k=5, domain=(-1.0, 1.0)) + (1 | subject)
    y ~ Normal(mu, sigma)
end

# Measured with this data against 4x1000 NUTS draws (seeds 5, 6, 7, 2026-11-08).
# With every cell starting at 0.5, the length scale ended 1.7 NUTS sds off with
# 0.02 of NUTS's sd and the ELBO 5-6 nats below the plain fit's: HSGP cells start
# from min(start, 0.1). With that start the length scale was 0.14-0.25 NUTS sds
# off (plain 0.44-0.49) with 0.22-0.25 of NUTS's sd, the HSGP sd 0.54-0.69 off
# (plain 0.79-0.84), the unit coordinates' rms error 0.10-0.13 (plain 0.17-0.18),
# and the ELBO 1.7-2.1 nats above plain.
# The length scale is weakly identified, and its mean wanders with where
# exploration stops (seeds 5-7, 2026-11-10): the centered fit's error was
# 0.19-0.31 NUTS sds after the 40 windows of the two-window stop
# (`trend_windows=0`), 0.32-0.38 after the 46-47 of the trend stop (BRM 0.2.264),
# 0.47-0.50 after 200 and 0.21-0.28 after 400, while the plain fit's was 0.53-0.76
# throughout; so the bound below sits between the two at the default stop.
@stestset "a grouped HSGP's centered fit against NUTS" begin
    rk = RKBRMI(svi_centering_curve_model(svi_centering_curve_data()))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    u0 = first(BRM_._brm_nlme_prior_median_point(m))
    reference = reduce(hcat, map(1:4) do chain
        run = adaptive_warmup_mcmc(Xoshiro(100 + chain), problem;
            init=(; position=copy(u0), squared_scale=ones(length(u0))), n_draws=1000)
        convert(Matrix{Float64}, run.posterior_position)
    end)
    μr, σr = vec(mean(reference; dims=2)), vec(std(reference; dims=2))
    names = string.(coordinate_names(rk.model.layout))
    rho, sd = findfirst(==("hsgp_t.rho"), names), findfirst(==("hsgp_t.sigma"), names)
    options = (; batch_size=4, ndraws=2000)
    # Plain: independent factors. With the default linear coupling and no centering
    # (seed 5, BRM 0.2.283) the length scale was 0.20 NUTS sds off and the HSGP sd
    # 0.49, against the centered fit's 0.38 and 0.71: this check compares centering
    # with the independent factorization.
    plain = brm_svi_fit(m; options..., rng=Xoshiro(5), centering=nothing, coupling=nothing)
    fit = brm_svi_fit(m; options..., rng=Xoshiro(5), centering=LearnedCentering())
    c = [cell.c for cell in fit.diagnostics.centering.cells if cell.kind === :hsgp]
    @test length(c) == 5 * 12
    mean_error(f) = abs.(vec(mean(f.draws; dims=2)) .- μr) ./ σr
    ratio(f) = vec(std(f.draws; dims=2)) ./ σr
    elbo(f) = mean(f.diagnostics.elbo[end-99:end])
    locals = setdiff(eachindex(μr), fit.diagnostics.globals)
    @test mean_error(fit)[rho] < 0.45 < mean_error(plain)[rho]
    @test ratio(fit)[rho] > 0.15
    @test mean_error(fit)[sd] < mean_error(plain)[sd]
    @test sqrt(mean(abs2, mean_error(fit)[locals])) < 0.15
    @test elbo(fit) > elbo(plain) + 1
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
    # Plain: the independent factorization, the comparison this check is about.
    plain = brm_svi_fit(m; options..., rng=Xoshiro(5), centering=nothing, coupling=nothing)
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
    options = (; batch_size=4, max_iterations=300, ndraws=20)
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
    explicit = brm_svi_fit(m; rng=Xoshiro(7), options..., centering=LearnedCentering())
    @test whole.draws == explicit.draws
    @test whole.diagnostics.q.centering.x == explicit.diagnostics.q.centering.x
    # Both a fit and its serialized checkpoint project to the same conditional
    # Gaussian start. The source is unchanged and the default learns afresh.
    state = brm_svi_checkpoint(whole)
    legacy_fit = merge(whole, (; diagnostics=Base.structdiff(whole.diagnostics,
        NamedTuple{(:centering_maps,)})))
    @test brm_svi_checkpoint(legacy_fit).units == state.units
    io = IOBuffer(); serialize(io, state)
    start = brm_svi_combine(m, [whole])
    restored = brm_svi_combine(m, [deserialize(seekstart(io))])
    @test start.globals == restored.globals
    @test start.units == restored.units
    @test start.receipt.centering.projection == :conditional_at_saved_global_mean
    @test start.receipt.centering.slices == [true]
    started = brm_svi_fit(m; start=whole, max_iterations=0, ndraws=0)
    @test started.diagnostics.centering !== nothing
    @test state.units == brm_svi_checkpoint(whole).units
    plain = brm_svi_fit(m; rng=Xoshiro(7), batch_size=4, max_iterations=50, ndraws=0,
        centering=nothing)
    @test brm_svi_combine(m, [plain]) isa BRMSVIStart
    plain_part = brm_svi_fit(m; rng=Xoshiro(7), batch_size=4, max_iterations=50,
        stop_after=20, ndraws=0, centering=nothing)
    plain_resumed = brm_svi_resume(m, brm_svi_optimizer_checkpoint(plain_part); ndraws=0)
    @test plain_resumed.diagnostics.centering === nothing
    @test svi_same_q(plain_resumed.diagnostics.q, plain.diagnostics.q)
end

@stestset "a centered window carries its controls and transports its prior" begin
    data = svi_stream_data(Xoshiro(31), ["a", "b", "c"], 6)
    initial = brm_svi_fit(svi_stream_model(data); rng=Xoshiro(32), max_iterations=300,
        ndraws=0, batch_size=2)
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
    @test carried.units["b"].centering_maps == state.units["b"].centering_maps
    moved = brm_svi_fit(carried, svi_stream_model(next); rng=Xoshiro(36),
        max_iterations=20, ndraws=0)
    later = brm_svi_checkpoint(moved)
    @test later.units["b"].centering_maps == state.units["b"].centering_maps
    @test later.q.μ != state.q.μ
    returned = brm_svi_fit(later, svi_stream_model(data); max_iterations=0, ndraws=0)
    @test brm_svi_checkpoint(returned).units["b"].centering == state.units["b"].centering
    @test brm_svi_checkpoint(returned).units["b"].M == state.units["b"].M
    absent_rows = data.subject .== "b"
    union_data = map((x, y) -> vcat(x, y[absent_rows]), next, data)
    union_rk = RKBRMI(svi_stream_model(union_data), state.preprocessing;
        resample_groups=(:subject,))
    combined = brm_svi_combine(union_rk, [later])
    @test length(combined.units) == 4
    @test combined.receipt.centering.slices == [true]
    # The absent unit's site uses the frame in which it was last fitted,
    # even though the stream's global scale moved. Combination removes that
    # frame's prior and applies the combined population prior once: check
    # the resulting physical moments with independent scalar Gaussian algebra.
    packed = brm_svi_fit(union_rk; start=combined, centering=nothing,
        max_iterations=0, ndraws=0)
    target = BRM_._brm_svi_target(packed.diagnostics.model)
    centered, factors = BRM_._brm_svi_with_centering(target,
        deepcopy(packed.diagnostics.q), LearnedCentering())
    i = only(findall(==("b"), BRM_._brm_svi_unit_levels(target)))
    now = only(BRM_._brm_svi_centering_maps(centered, factors)[i])
    old = later.units["b"]
    was = only(old.centering_maps)
    @test !("b" in combined.receipt.units.uncarried)
    old_mean = only(was.natural.factor * (was.C \ (old.M - was.b)) + was.natural.location)
    old_variance = (only(was.natural.factor) / only(was.C))^2 / only(old.Λ)
    old_prior_variance = only(was.natural.factor)^2
    new_prior_variance = only(now.natural.factor)^2
    expected_variance = inv(inv(old_variance) - inv(old_prior_variance) + inv(new_prior_variance))
    expected_mean = expected_variance * (old_mean / old_variance -
        only(was.natural.location) / old_prior_variance +
        only(now.natural.location) / new_prior_variance)
    actual_mean = only(now.natural.factor * combined.units["b"].M + now.natural.location)
    actual_variance = only(now.natural.factor)^2 / only(combined.units["b"].Λ)
    @test actual_mean ≈ expected_mean
    @test actual_variance ≈ expected_variance
end
