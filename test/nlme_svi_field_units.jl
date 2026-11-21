# A read-pair field's states in units' own factors (`field_states=:units`): the pieces add
# up to the whole model, in value and gradient, and the public fit runs, resumes and stays
# the same at any number of tasks. A new observation window reuses the fitted support.
include(joinpath(@__DIR__, "fitting_packages.jl"))
using Test, BayesianRegressionModels, LinearAlgebra, Random, Statistics
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, DifferentiationInterface
const BRM = BayesianRegressionModels
const FIELD_AD = AutoEnzyme(; mode=Enzyme.Reverse)

include("svi_field_fixture.jl")

function check_whole_model(t, rk, m)
    sampler = prepare_sampler(rk.model, rk_bound_plan(rk), m.unit; backend=FIELD_AD)
    for offset in (-0.2, 0.0, 0.25)
        u = [0.1sin(i) + offset for i in eachindex(m.unit)]
        BRM._brm_svi_field_joint!(t, u)
        g = u[t.globals]
        value, gg = BRM._brm_svi_global_term(t, g)
        grad = zeros(length(u))
        grad[t.globals] .= gg
        for (i, block) in enumerate(t.units)
            gi, gz = zeros(length(g)), zeros(length(block))
            value += BRM._brm_svi_unit_term!(gi, gz, t, i, g, u[block])
            grad[t.globals] .+= gi
            grad[block] .+= gz
        end
        expected_grad = similar(u)
        expected, _ = sampler_value_and_gradient!(sampler, expected_grad, u)
        @test value ≈ expected atol=2e-9 rtol=2e-11
        @test grad ≈ expected_grad atol=2e-8 rtol=2e-9
    end
end

include("svi_field_centering_oracle.jl")

data = station_data()
rk = RKBRMI(field_model(data))
names = String.(BRM._rk_layout_coordinate_names(rk))
states = findall(n -> occursin(".field.", n), names)

@testset "connected units own the states they read" begin
    m = brm_nlme_model(rk; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    t = BRM._brm_svi_target(m; field_states=:units, separator=:none)
    @test isempty(intersect(states, t.globals))
    @test sort!(reduce(vcat, [filter(in(states), block) for block in t.units])) == states
    # A station's block is its reads and its effect, whatever the field's size.
    @test all(block -> length(block) == 4, t.units)
    @test any(i -> !isempty(t.fields.blanket[i + 1]), eachindex(t.units))
    check_whole_model(t, rk, m)
    check_centered_field(m, rk)
end

@testset "a batch takes only the derivatives its terms read" begin
    # Enough stations that a unit reads some rows only as a parent's scale and
    # its neighbors read rows it does not: the oracle compares the term on the
    # rows a batch prepares with the term on every row, with all derivatives.
    wide = station_data(; stations=12, times=collect(0.0:5.0), reads=3, seed=11)
    rkw = RKBRMI(field_model(wide))
    mw = brm_nlme_model(rkw; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    counts = check_centered_field(mw, rkw)
    @test counts.parent_rows > 0
    @test counts.value_rows > 0
end

@testset "the coupling baseline is independent of neighboring field draws" begin
    m = brm_nlme_model(rk; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    t = BRM._brm_svi_target(m; field_states=:units, separator=:none)
    q = BRM._brm_svi_params(t, m.unit[t.globals], MeanFieldGlobal();
        global_scale=0.1, unit_scale=1.0)
    i = findfirst(i -> !isempty(t.fields.blanket[i + 1]), eachindex(t.units))
    neighbor = first(t.fields.blanket[i + 1])
    baseline_target = BRM._brm_svi_coupling_baseline_target(t)
    values = map((-0.1, 0.3)) do offset
        u = copy(m.unit)
        u[t.units[neighbor]] .+= offset
        BRM._brm_svi_field_joint!(t, u)
        b = zeros(length(q.M[i]))
        BRM._brm_svi_coupling_baseline!(b, baseline_target, q, i, nothing, nothing,
            first(t.plans), first(t.work))
        gg, gz = zeros(length(q.μ)), zeros(length(q.M[i]))
        BRM._brm_svi_unit_term!(gg, gz, t, i, q.μ, q.M[i])
        (; baseline=copy(b), full_gradient=gz)
    end
    @test values[1].baseline == values[2].baseline
    # Positive control: the field contribution really does depend on those draws.
    @test norm(values[1].full_gradient - values[2].full_gradient) > 1e-6
end

@testset "a state read by two units stays global" begin
    two = station_data(; stations=4, seed=3)
    # Move s2's reads onto s1's position: a state both read at the same time is shared.
    east = copy(two.east); north = copy(two.north)
    i = findfirst(==("s1"), two.site)
    east[two.site .== "s2"] .= two.east[i]
    north[two.site .== "s2"] .= two.north[i]
    input = merge(two, (; east, north))
    rk2 = RKBRMI(field_model(input))
    m2 = brm_nlme_model(rk2; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    t2 = BRM._brm_svi_target(m2; field_states=:units, separator=:none)
    names2 = String.(BRM._rk_layout_coordinate_names(rk2))
    field2 = findall(n -> occursin(".field.", n), names2)
    site_times(site) = unique(two.time[two.site .== site])
    common = length(intersect(site_times("s1"), site_times("s2")))
    @test common >= 2
    @test length(t2.units) == 4
    @test length(intersect(field2, t2.globals)) == common
    @test sort!(vcat(intersect(field2, t2.globals), reduce(vcat, [filter(in(field2), block) for block in t2.units]))) == field2
    check_whole_model(t2, rk2, m2)
    check_centered_field(m2, rk2)
end

@testset "the public fit runs, reads back and continues" begin
    m = brm_nlme_model(rk; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    controls = (; batch_size=3, max_iterations=40, ndraws=5, separator=:none, field_states=:units,
        centering=LearnedCentering(; fields=true))
    fit = brm_svi_fit(m; rng=Xoshiro(2), threads=1, controls...)
    @test fit.diagnostics.iterations == 40
    @test count(cell -> cell.kind === :gmrf, fit.diagnostics.centering.cells) == length(states)
    # Without `fields=true` the states keep their packed coordinates, and the
    # report says how to centre them.
    packed = brm_svi_fit(m; rng=Xoshiro(2), threads=1, controls..., max_iterations=2,
        centering=LearnedCentering())
    @test !any(cell -> cell.kind === :gmrf, packed.diagnostics.centering.cells)
    @test any(row -> occursin("fields=true", row.reason) &&
        any(name -> occursin(".field.", string(name)), row.coordinates),
        packed.diagnostics.centering.uncentered)
    @test !hasproperty(packed.diagnostics.sampling, :field_neighbor_calls)
    @test all(isfinite, fit.diagnostics.elbo)
    check_field_start_and_projection(m, fit)
    @test size(fit.draws, 1) == length(names)
    @test all(isfinite, fit.draws)
    # A stopped run continues to the same place as an unbroken one.
    stopped = brm_svi_fit(m; rng=Xoshiro(2), threads=1, controls..., stop_after=15)
    checkpoint = brm_svi_optimizer_checkpoint(stopped)
    @test checkpoint.identity.field_states === :units
    resumed = brm_svi_resume(m, checkpoint; threads=1, ndraws=5)
    @test resumed.diagnostics.elbo == fit.diagnostics.elbo
    @test resumed.draws == fit.draws
    @test sum(fit.diagnostics.sampling.field_neighbor_calls) > 0
    @test resumed.diagnostics.sampling.field_neighbor_calls == fit.diagnostics.sampling.field_neighbor_calls
    @test resumed.diagnostics.unit_gradient_calls == fit.diagnostics.unit_gradient_calls
end

@testset "a new observation window carries the field Gaussian on its fitted support" begin
    data = station_data(; stations=4, seed=3)
    # Include shared field coordinates as well as coordinates held by the stations.
    first_site = findfirst(==("s1"), data.site)
    data.east[data.site .== "s2"] .= data.east[first_site]
    data.north[data.site .== "s2"] .= data.north[first_site]
    controls = (; threads=1, batch_size=3, max_iterations=6, ndraws=3,
        centering=nothing, coupling=nothing)
    fitted = brm_svi_fit(field_model(data); controls..., ad_backend=FIELD_AD, rng=Xoshiro(81),
        subject_group=:site, unit_partition=:connected, separator=:none, field_states=:units)
    state = brm_svi_checkpoint(fitted)
    @test state.layout.options.field_states === :units
    @test any(name -> occursin(".field.", string(name)), state.globals)
    keep = 1:2:length(data.y)
    window = (; east=data.east[keep], north=data.north[keep], time=data.time[keep],
        site=data.site[keep], x=0.2 .- data.x[keep], y=randn(Xoshiro(82), length(keep)),
        state=collect(eachindex(keep)), nb=data.nb)
    update = brm_svi_fit(state, field_model(window); controls..., ad_backend=FIELD_AD, rng=Xoshiro(83))
    carried = brm_svi_checkpoint(update)
    @test carried.updates == 1 && carried.layout.options.field_states === :units
    m = update.diagnostics.model
    stopped = brm_svi_fit(state, m; controls..., rng=Xoshiro(83), stop_after=2)
    resumed = brm_svi_resume(m, stopped; threads=1, ndraws=3)
    @test resumed.diagnostics.elbo == update.diagnostics.elbo
    @test resumed.draws == update.draws
    @test_throws "field_states" brm_svi_fit(state, m; controls..., field_states=:global)

    # Whole new-window likelihood from RK, plus the saved Gaussians. In particular
    # the original Vecchia law must contribute neither its value nor its gradients.
    target = BRM._brm_svi_target(m; previous=state, field_states=:units, separator=:none)
    t, _ = BRM._brm_svi_initialize(target, state, state.q.family; start=nothing,
        inits=nothing, global_scale=1.0, unit_scale=1.0)
    sampler = prepare_sampler(m.backend.model, rk_bound_plan(m.backend), m.unit; backend=FIELD_AD)
    levels = BRM._brm_svi_unit_levels(t)
    for offset in (-0.15, 0.2)
        u = offset .+ 0.1 .* sin.(eachindex(m.unit))
        whole_grad = similar(u)
        whole, _ = sampler_value_and_gradient!(sampler, whole_grad, u)
        prior, prior_grad = BRM._rk_nlme_population_prior_value_gradient(m.population_prior, u)
        g = u[t.globals]
        δ = g - state.q.μ
        variance = exp.(2 .* state.q.ρ)
        expected = whole - prior - sum(δ.^2 ./ variance .+ log.(2π .* variance)) / 2
        expected_grad = whole_grad - prior_grad
        expected_grad[t.globals] .-= δ ./ variance
        value, gg = BRM._brm_svi_global_term(t, g)
        grad = zeros(length(u))
        grad[t.globals] .= gg
        for (i, block) in enumerate(t.units)
            saved = state.units[levels[i]]
            keys = BRM._brm_svi_owned_unit_keys(m, t.units, i)
            order = [only(findall(==(key), saved.keys)) for key in keys]
            diff = u[block] - saved.M[order]
            precision = saved.Λ[order, order]
            expected += (logdet(Symmetric(precision)) - length(block) * log(2π) -
                dot(diff, precision, diff)) / 2
            expected_grad[block] .-= precision * diff
            gz = zeros(length(block))
            value += BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[block])
            grad[t.globals] .+= gg
            grad[block] .+= gz
        end
        @test value ≈ expected atol=2e-9 rtol=2e-11
        @test grad ≈ expected_grad atol=2e-8 rtol=2e-9
    end
    # A prepared model can have the same packed field length and different locations.
    moved = RKBRMI(field_model(merge(window, (; east=window.east .+ 10))))
    @test_throws "preprocessing changed" brm_svi_fit(state, moved; controls..., ad_backend=FIELD_AD)
    # Identical field geometry with a previously private state read by a different station.
    swapped = replace(window.site, "s3"=>"s4", "s4"=>"s3")
    @test_throws ArgumentError brm_svi_fit(state, field_model(merge(window, (; site=swapped)));
        controls..., ad_backend=FIELD_AD)
end

@testset "a non-finite unit's point holds this iteration's field states" begin
    m = brm_nlme_model(rk; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    t = BRM._brm_svi_target(m; separator=:none, field_states=:units)
    @test t.fields !== nothing
    t.fields.others .= 0.1 .* eachindex(m.unit)
    g = fill(0.25, length(t.globals)); z = fill(-3.0, length(t.units[2]))
    u = BRM._brm_svi_term_point(t, g, t.units[2], z)
    @test u[t.globals] == g && u[t.units[2]] == z
    other = setdiff(eachindex(u), t.globals, t.units[2])
    @test !isempty(other) && u[other] == t.fields.others[other]
    # A plain target's rest is the model's point, and the reading is left alone.
    @test t.fields.others == 0.1 .* eachindex(m.unit)
    plain = BRM._brm_svi_target(m; separator=:none)
    @test BRM._brm_svi_term_point(plain, fill(0.25, length(plain.globals)))[
        setdiff(eachindex(m.unit), plain.globals)] == m.unit[setdiff(eachindex(m.unit), plain.globals)]
end

@testset "a held target is reused only under the same field_states" begin
    # `_brm_svi_checkout!` keys the model's held target on `field_states`: a `:units`
    # target holds read states in units' blocks, with the field prior split by owner.
    m = brm_nlme_model(rk; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    controls = (; batch_size=3, max_iterations=6, ndraws=0, separator=:none, threads=1)
    units = brm_svi_fit(m; rng=Xoshiro(6), controls..., field_states=:units)
    @test !units.diagnostics.target.reused
    again = brm_svi_fit(m; rng=Xoshiro(6), controls..., field_states=:units)
    @test again.diagnostics.target.reused
    @test again.diagnostics.elbo == units.diagnostics.elbo
    global_fit = brm_svi_fit(m; rng=Xoshiro(6), controls..., field_states=:global)
    @test !global_fit.diagnostics.target.reused
    @test length(global_fit.diagnostics.globals) > length(units.diagnostics.globals)
    # A resume passes the saved field_states and finds its target again.
    part = brm_svi_fit(m; rng=Xoshiro(7), controls..., field_states=:units, stop_after=2)
    @test !part.diagnostics.target.reused
    rest = brm_svi_resume(m, part; ndraws=0, threads=1)
    @test rest.diagnostics.target.reused
    @test rest.diagnostics.units == units.diagnostics.units
    # A `:units` snapshot is plain data like any other: a process without BRM reads it.
    mktempdir() do dir
        path = joinpath(dir, "latest.jls")
        brm_svi_fit(m; rng=Xoshiro(8), controls..., field_states=:units, snapshot=path)
        script = "using Serialization; s = deserialize(ARGS[1]); " *
            "print(length(s.q.M), ' ', isdefined(Main, :BayesianRegressionModels))"
        cmd = addenv(`$(Base.julia_cmd()) --startup-file=no -e $script $path`,
            "JULIA_LOAD_PATH" => "@stdlib")
        @test read(cmd, String) == "$(length(units.diagnostics.units)) false"
    end
end

Threads.nthreads() > 1 && @testset "a run is the same at any number of tasks" begin
    m = brm_nlme_model(rk; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    controls = (; batch_size=3, max_iterations=25, ndraws=4, separator=:none, field_states=:units)
    for centering in (LearnedCentering(), LearnedCentering(; fields=true))
        one = brm_svi_fit(m; rng=Xoshiro(4), threads=1, controls..., centering)
        two = brm_svi_fit(m; rng=Xoshiro(4), threads=2, controls..., centering)
        @test one.diagnostics.elbo == two.diagnostics.elbo
        @test one.draws == two.draws
    end
end

@testset "a fixed field law needs only its bound coefficient rows" begin
    fixed_data = station_data(; stations=4, seed=3)
    # Include both shared and unit-owned states in the factor-level start.
    first_site = findfirst(==("s1"), fixed_data.site)
    fixed_data.east[fixed_data.site .== "s2"] .= fixed_data.east[first_site]
    fixed_data.north[fixed_data.site .== "s2"] .= fixed_data.north[first_site]
    fixed = @brm fixed_data begin
        sigma ~ Exponential(1)
        field ~ 0 + gmrf((east, north), time; neighbors=nb)
        length_scale(field, gmrf((east, north), time), space) = 0.3
        length_scale(field, gmrf((east, north), time), time) = 3.0
        mu ~ 1 + (0 + x | site)
        @plate for s in groups(site)
            y[s] ~ Normal(mu[s] .+ field[s], sigma)
        end
    end
    rk_fixed = RKBRMI(fixed)
    m = brm_nlme_model(rk_fixed; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:subjects)
    check_centered_field(m, rk_fixed)
    fit = brm_svi_fit(m; rng=Xoshiro(23), batch_size=2, max_iterations=3, ndraws=2,
        separator=:none, field_states=:units, unit_reoptimize=1,
        centering=LearnedCentering(; fields=true))
    @test all(isfinite, fit.draws)
    saved = brm_svi_checkpoint(fit)
    start = brm_svi_combine(m, [saved]; field_states=:units)
    initialized = brm_svi_fit(m; start, field_states=:units, max_iterations=0, ndraws=0,
        centering=LearnedCentering(; fields=true))
    state = brm_svi_checkpoint(initialized)
    source = BRM._brm_svi_source("field test", state, "initialized fit")
    @test initialized.diagnostics.start.applied.units == length(m.view.levels)
    @test any(name -> occursin(".field.", string(name)), state.globals)
    @test !isempty(state.layout.field_projection)
    for (label, unit) in state.units
        fields = findall(key -> hasproperty(key, :coordinate) &&
            haskey(state.layout.field_projection, key.coordinate), unit.keys)
        isempty(fields) && continue
        supplied = start.units[label]
        order = [only(findall(isequal(unit.semantic[k]), supplied.keys)) for k in fields]
        @test source.units[label].M[fields] ≈ supplied.M[order] atol=1e-10
        # At fixed neighboring units, the within-owner covariance of the packed
        # field coordinates is the block the slice combination supplied.
        names = [unit.keys[k].coordinate for k in fields]
        A = zeros(length(names), length(names))
        for (i, name) in enumerate(names)
            row = state.layout.field_projection[name]
            for (input, w) in zip(row.inputs, row.weights)
                j = findfirst(==(input), names)
                j === nothing || (A[i, j] += w)
            end
        end
        @test A * inv(Symmetric(unit.Λ))[fields, fields] * A' ≈
            inv(Symmetric(supplied.Λ))[order, order] atol=1e-10
    end
end

@testset "channels through a sampled factor" begin
    base = station_data(; stations=4, seed=5)
    channel = repeat(["a", "b"]; outer=length(base.y))
    input = (; east=repeat(base.east; inner=2), north=repeat(base.north; inner=2),
        time=repeat(base.time; inner=2), site=repeat(base.site; inner=2), x=repeat(base.x; inner=2),
        y=repeat(base.y; inner=2), channel, state=collect(1:2length(base.y)), nb=3)
    model = @brm input begin
        sigma ~ Exponential(1)
        Omega ~ LKJCholesky(2, 2.0)
        field ~ 0 + gmrf((east, north), time, channel; neighbors=nb, cor=Omega)
        sd(field, gmrf((east, north), time, channel)) ~ Exponential(1)
        length_scale(field, gmrf((east, north), time, channel), space) ~ LogNormal(log(0.3), 0.7)
        length_scale(field, gmrf((east, north), time, channel), time) = 3.0
        mu ~ 1 + (0 + x | site)
        @plate for s in eachindex(state)
            y[s] ~ Normal(mu[s] .+ field[state[s]], sigma)
        end
    end
    rkc = RKBRMI(model)
    mc = brm_nlme_model(rkc; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    tc = BRM._brm_svi_target(mc; field_states=:units, separator=:none)
    check_whole_model(tc, rkc, mc)
    check_centered_field(mc, rkc)
end

@testset "refusals name the reason" begin
    m = brm_nlme_model(rk; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    @test_throws ArgumentError BRM._brm_svi_target(m; field_states=:maybe)
    plain = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + (0 + x | site)
        y ~ Normal(mu, sigma)
    end
    m0 = brm_nlme_model(RKBRMI(plain); ad_backend=FIELD_AD, subject_group=:site)
    err = try BRM._brm_svi_target(m0; field_states=:units); nothing catch e; e end
    @test err isa ArgumentError && occursin("neighbors=m", sprint(showerror, err))
    xy = [0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5]
    triangles = [1 2 5; 2 3 5; 3 4 5; 4 1 5]
    input = merge(data, (; xy, triangles))
    mesh = @brm input begin
        sigma ~ Exponential(1)
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        mu ~ 1 + (0 + x | site)
        @plate for s in eachindex(state)
            y[s] ~ Normal(mu[s] .+ field[state[s]], sigma)
        end
    end
    mm = brm_nlme_model(RKBRMI(mesh); ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    err = try BRM._brm_svi_target(mm); nothing catch e; e end
    @test err isa BRM.BRMNLMEViewError && occursin("neighbors=m", sprint(showerror, err))
end

@testset "learned field maps carry across fixed-support windows" begin
    data = station_data(; stations=4,seed=7)
    opts = (; threads=1,batch_size=2,max_iterations=7,ndraws=3,unit_curvature=:pair)
    fresh = brm_svi_fit(field_model(data); opts...,ad_backend=FIELD_AD,subject_group=:site,
        unit_partition=:connected,field_states=:units,separator=:none,rng=Xoshiro(303),
        centering=LearnedCentering(;delay=false,fields=true))
    keep = 1:2:length(data.y)
    newdata = (;east=data.east[keep],north=data.north[keep],time=data.time[keep],
        site=data.site[keep],x=0.2 .- data.x[keep],y=randn(Xoshiro(301),length(keep)),
        state=collect(eachindex(keep)),nb=data.nb)
    state = brm_svi_checkpoint(fresh)
    result = brm_svi_fit(state,field_model(newdata);opts...,ad_backend=FIELD_AD,rng=Xoshiro(304))
    check_field_window(fresh,result)
    m = result.diagnostics.model
    paused = brm_svi_fit(state,m;opts...,rng=Xoshiro(304),stop_after=2)
    resumed = brm_svi_resume(m,paused;threads=1,ndraws=3)
    @test resumed.diagnostics.elbo == result.diagnostics.elbo
    @test resumed.draws == result.draws
    # The same fitted field can lose all observations of one of its owners.
    # Give its saved conditional a nonzero slope to exercise the path to g.
    label = ("s1",)
    saved = state.units[label]
    saved.coupling.W .= 0.08
    keep = [j for j in 1:2:length(data.y) if data.site[j] != only(label)]
    partial = (;east=data.east[keep],north=data.north[keep],time=data.time[keep],
        site=data.site[keep],x=data.x[keep],y=data.y[keep],state=collect(eachindex(keep)),nb=data.nb)
    absent = brm_svi_fit(state,field_model(partial);opts...,ad_backend=FIELD_AD,rng=Xoshiro(305))
    @test !(label in absent.diagnostics.unit_levels)
    @test brm_svi_checkpoint(absent).units[label].M == saved.M
    check_field_window(state,absent)

end


@testset "packed field states stream under learned centering of the other cells" begin
    data = station_data(; stations=4, seed=7)
    opts = (; threads=1, batch_size=2, max_iterations=7, ndraws=3, unit_curvature=:pair)
    fresh = brm_svi_fit(field_model(data); opts..., ad_backend=FIELD_AD, subject_group=:site,
        unit_partition=:connected, field_states=:units, separator=:none, rng=Xoshiro(403),
        centering=LearnedCentering(; delay=false))
    @test !isempty(fresh.diagnostics.centering.cells)
    @test !any(cell -> cell.kind === :gmrf, fresh.diagnostics.centering.cells)
    state = brm_svi_checkpoint(fresh)
    @test !haskey(state.layout.centering, :fields)
    # One station has no observation in the new window.
    label = ("s1",)
    keep = [j for j in 1:2:length(data.y) if data.site[j] != only(label)]
    window = (; east=data.east[keep], north=data.north[keep], time=data.time[keep],
        site=data.site[keep], x=0.2 .- data.x[keep], y=randn(Xoshiro(401), length(keep)),
        state=collect(eachindex(keep)), nb=data.nb)
    update = brm_svi_fit(state, field_model(window); opts..., ad_backend=FIELD_AD, rng=Xoshiro(404))
    @test all(isfinite, update.draws)
    @test !any(cell -> cell.kind === :gmrf, update.diagnostics.centering.cells)
    @test !hasproperty(update.diagnostics.sampling, :field_neighbor_calls)
    @test !(label in update.diagnostics.unit_levels)
    carried = brm_svi_checkpoint(update)
    @test carried.units[label].M == state.units[label].M
    @test !haskey(carried.layout.centering, :fields)
    m = update.diagnostics.model
    paused = brm_svi_fit(state, m; opts..., rng=Xoshiro(404), stop_after=2)
    resumed = brm_svi_resume(m, paused; threads=1, ndraws=3)
    @test resumed.diagnostics.elbo == update.diagnostics.elbo
    @test resumed.draws == update.draws
    # A state saved with learned field coordinates keeps them in the next window.
    learned = brm_svi_fit(field_model(data); opts..., ad_backend=FIELD_AD, subject_group=:site,
        unit_partition=:connected, field_states=:units, separator=:none, rng=Xoshiro(403),
        centering=LearnedCentering(; delay=false, fields=true))
    kept = brm_svi_fit(brm_svi_checkpoint(learned), field_model(window); opts...,
        ad_backend=FIELD_AD, rng=Xoshiro(404))
    @test brm_svi_checkpoint(kept).layout.centering.fields === true
    @test any(cell -> cell.kind === :gmrf, kept.diagnostics.centering.cells)
end
