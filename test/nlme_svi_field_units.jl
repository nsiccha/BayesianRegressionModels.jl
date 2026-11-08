# A read-pair field's states in units' own factors (`field_states=:units`): the pieces add
# up to the whole model, in value and gradient, and the public fit runs, resumes and stays
# the same at any number of tasks. Mesh fields and streaming are refused by name.
using Test, BayesianRegressionModels, LinearAlgebra, Random, Statistics
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, DifferentiationInterface
const BRM = BayesianRegressionModels
const FIELD_AD = AutoEnzyme(; mode=Enzyme.Reverse)

# Stations at fixed positions read the field at several of a shared set of times.
function station_data(; stations=6, times=[0.0, 1.0, 2.0, 4.0], reads=3, rows=2, seed=1, channels=nothing)
    rng = Xoshiro(seed)
    px, py = rand(rng, stations), rand(rng, stations)
    east, north, time, site = Float64[], Float64[], Float64[], String[]
    for s in 1:stations, t in sort!(randperm(rng, length(times))[1:reads]), _ in 1:rows
        push!(east, px[s]); push!(north, py[s]); push!(time, times[t]); push!(site, "s$s")
    end
    n = length(east)
    x = randn(rng, n)
    y = sin.(3 .* east) .+ cos.(time) .+ 0.2 .* x .+ 0.3 .* randn(rng, n)
    (; east, north, time, site, x, y, state=collect(1:n), nb=3)
end

function field_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        field ~ 0 + gmrf((east, north), time; neighbors=nb)
        sd(field, gmrf((east, north), time)) ~ Exponential(1)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(0.3), 0.7)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(3.0), 0.7)
        mu ~ 1 + (0 + x | site)
        @plate for s in eachindex(state)
            y[s] ~ Normal(mu[s] .+ field[state[s]], sigma)
        end
    end
end

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
end

@testset "the public fit runs, reads back and continues" begin
    m = brm_nlme_model(rk; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    controls = (; batch_size=3, max_iterations=40, ndraws=5, separator=:none, field_states=:units)
    fit = brm_svi_fit(m; rng=Xoshiro(2), threads=1, controls...)
    @test fit.diagnostics.iterations == 40
    @test all(isfinite, fit.diagnostics.elbo)
    @test size(fit.draws, 1) == length(names)
    @test all(isfinite, fit.draws)
    # A stopped run continues to the same place as an unbroken one.
    stopped = brm_svi_fit(m; rng=Xoshiro(2), threads=1, controls..., stop_after=15)
    checkpoint = brm_svi_optimizer_checkpoint(stopped)
    @test checkpoint.identity.field_states === :units
    resumed = brm_svi_resume(m, checkpoint; threads=1, ndraws=5)
    @test resumed.diagnostics.elbo == fit.diagnostics.elbo
    @test resumed.draws == fit.draws
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
    one = brm_svi_fit(m; rng=Xoshiro(4), threads=1, controls...)
    two = brm_svi_fit(m; rng=Xoshiro(4), threads=2, controls...)
    @test one.diagnostics.elbo == two.diagnostics.elbo
    @test one.draws == two.draws
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
