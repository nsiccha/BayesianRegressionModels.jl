# The footprint `gmrf` term on RKBRMI: density and native reverse derivative against
# the plain core, state plates under `locality = :error`, frozen replay and refusals.
using Test, BayesianRegressionModels, LinearAlgebra, Random, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
import DifferentiationInterface as DI
import LogDensityProblems as LDP
const BRM = BayesianRegressionModels

function vecchia_rk_data(; seed=11, channels=false)
    rng = MersenneTwister(seed)
    npositions, times = 9, [0.0, 1.0, 2.0, 80.0]
    x, y = 10 .* rand(rng, npositions), 10 .* rand(rng, npositions)
    east = repeat(x; outer=length(times))
    north = repeat(y; outer=length(times))
    time = repeat(times; inner=npositions)
    keep = [i for i in eachindex(time) if !(time[i] == 80.0 && isodd(i))]
    east, north, time = east[keep], north[keep], time[keep]
    if channels
        channel = repeat(["gauge", "radar"]; outer=length(keep))
        east, north, time = repeat(east; inner=2), repeat(north; inner=2), repeat(time; inner=2)
        return (; east, north, time, channel, y=0.3 .* randn(rng, length(time)))
    end
    (; east, north, time, y=0.3 .* randn(rng, length(time)))
end

function vecchia_rk_backend(model; case_id="gmrf-vecchia")
    artifact = BRM.emit_rk_artifact(model; case_id)
    bound = BRM.rk_translate_artifact(artifact)
    built = build_kernel(bound)
    (; rk=RKBRMI(model, artifact.plan, built), bound, built, plan=artifact.plan)
end

vecchia_rk_problem(rk) = rk_logdensity_problem(rk; ad_backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))

function vecchia_gradient_check(fx, u; h=1e-6, tolerance=2e-4)
    sampler = prepare_sampler(fx.built, fx.bound, u; backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    query = prepare_query(fx.built, fx.bound, :sampler)
    value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ Base.invokelatest(query, u) atol=1e-9
    for j in eachindex(u)
        plus, minus = copy(u), copy(u)
        plus[j] += h
        minus[j] -= h
        @test gradient[j] ≈ (Base.invokelatest(query, plus) - Base.invokelatest(query, minus)) / 2h atol=tolerance
    end
    value
end

@testset "A model-scope footprint field: density and reverse derivative" begin
    data = vecchia_rk_data()
    model = @brm data begin
        field ~ 0 + gmrf((east, north), time; neighbors=4)
        sd(field, gmrf((east, north), time)) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(3.0), 0.4)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(1.7), 0.4)
        y ~ Normal(field, 0.2)
    end
    fx = vecchia_rk_backend(model)
    f = BRM._brm_vecchia_footprint(data.east, data.north, data.time, 1; neighbors=4)
    names = BRM._rk_layout_coordinate_names(fx.rk)
    @test length(names) == f.ncoordinates + 3
    rng = MersenneTwister(1)
    u = 0.2 .* randn(rng, length(names))
    position(suffix) = only(i for (i, name) in enumerate(names) if endswith(string(name), suffix))
    sigma = exp(u[position(".sigma")])
    spatial = exp(u[position(".spatial_length")])
    temporal = exp(u[position(".temporal_length")])
    field = u[[i for (i, name) in enumerate(names) if occursin(".field.", string(name))]]
    index = BRM._brm_vecchia_field_index(f, [1], data.east, data.north, data.time, ones(Int, length(data.y)))
    expected = BRM._brm_vecchia_logdensity(field, spatial, temporal, ones(1, 1), f) +
        sum(logpdf.(Normal.(sigma .* field[index], 0.2), data.y)) +
        logpdf(Exponential(0.7), sigma) + log(sigma) +
        logpdf(LogNormal(log(3.0), 0.4), spatial) + log(spatial) +
        logpdf(LogNormal(log(1.7), 0.4), temporal) + log(temporal)
    @test Base.invokelatest(prepare_query(fx.built, fx.bound, :sampler), u) ≈ expected atol=1e-8
    @test vecchia_gradient_check(fx, u) ≈ expected atol=1e-8
    # The derivative comes from the rule, not from Enzyme over the states.
    problem = vecchia_rk_problem(fx.rk)
    @test LDP.logdensity(problem, u) ≈ expected atol=1e-8
    value, gradient = LDP.logdensity_and_gradient(problem, u)
    @test value ≈ expected atol=1e-8
    @test gradient[position(".sigma")] != 0
end

@testset "Preparation builds each footprint once and keeps fields apart" begin
    data = vecchia_rk_data()
    # Two fields on one time column have different footprints.
    input = (; data..., east2=data.east .+ 0.5, north2=data.north)
    model = @brm input begin
        a ~ 0 + gmrf((east, north), time; neighbors=3)
        b ~ 0 + gmrf((east2, north2), time; neighbors=2)
        y ~ Normal(a .+ b, 0.2)
    end
    plan = BRM._brm_rk_plan(model)
    supports = [only(v) for (k, v) in plan.columns if occursin("_footprint", string(k))]
    @test length(supports) == 2
    @test all(s -> s isa BRM._BRMGMRFFootprintSupport, supports)
    @test Set(s.geometry.neighbors for s in supports) == Set([2, 3])
    @test Set(first(s.geometry.east) in input.east2 for s in supports) == Set([true, false])
    # The emitted graph reads the bound support and never rebuilds it.
    emitted = BRM._rk_emit_ast(plan)
    @test !occursin("brm_gmrf_footprint_support", string(emitted.main, emitted.defs))
end

@testset "Term priors may read data, as on the mesh form" begin
    data = (; vecchia_rk_data()..., amplitude=0.7)
    model = @brm data begin
        field ~ 0 + gmrf((east, north), time; neighbors=3)
        sd(field, gmrf((east, north), time)) ~ Exponential(amplitude)
        y ~ Normal(field, 0.2)
    end
    fx = vecchia_rk_backend(model; case_id="gmrf-vecchia-data-prior")
    u = 0.1 .* randn(MersenneTwister(5), fx.built.layout.total)
    names = BRM._rk_layout_coordinate_names(fx.rk)
    sigma = exp(u[only(i for (i, name) in enumerate(names) if endswith(string(name), ".sigma"))])
    baseline = Base.invokelatest(prepare_query(fx.built, fx.bound, :sampler), u)
    other = (; data..., amplitude=1.4)
    moved = vecchia_rk_backend(@brm(other, begin
        field ~ 0 + gmrf((east, north), time; neighbors=3)
        sd(field, gmrf((east, north), time)) ~ Exponential(amplitude)
        y ~ Normal(field, 0.2)
    end); case_id="gmrf-vecchia-data-prior-moved")
    changed = Base.invokelatest(prepare_query(moved.built, moved.bound, :sampler), u)
    @test changed - baseline ≈ logpdf(Exponential(1.4), sigma) - logpdf(Exponential(0.7), sigma) atol=1e-9
end

@testset "Correlated channels through a sampled LKJ factor" begin
    data = vecchia_rk_data(; channels=true)
    model = @brm data begin
        Omega ~ LKJCholesky(2, 2.0)
        field ~ 0 + gmrf((east, north), time, channel; neighbors=3, cor=Omega)
        length_scale(field, gmrf((east, north), time, channel), space) = 3.0
        y ~ Normal(field, 0.2)
    end
    fx = vecchia_rk_backend(model; case_id="gmrf-vecchia-channels")
    rng = MersenneTwister(2)
    vecchia_gradient_check(fx, 0.15 .* randn(rng, fx.built.layout.total))
end

@testset "State plates read the field under locality = :error" begin
    data = vecchia_rk_data()
    positions = unique(zip(data.east, data.north))
    unit = [findfirst(==((a, b)), positions) for (a, b) in zip(data.east, data.north)]
    input = (; data..., unit, state=collect(eachindex(data.y)), rate=fill(1.0, length(positions)))
    model = @brm input begin
        field ~ 0 + gmrf((east, north), time; neighbors=3)
        length_scale(field, gmrf((east, north), time), space) = 3.0
        length_scale(field, gmrf((east, north), time), time) = 1.7
        sd(field, gmrf((east, north), time)) = 0.5
        @plate for s in eachindex(state)
            y[s] ~ Normal(exp(field[state[s]]) * rate[unit[s]], 0.2)
        end
    end
    fx = vecchia_rk_backend(model; case_id="gmrf-vecchia-plate")
    u = 0.1 .* randn(MersenneTwister(3), fx.built.layout.total)
    vecchia_gradient_check(fx, u)
    cell = prepare_cell_query(fx.built, fx.bound, :y; locality=:error)
    footprint = ReactiveKernels.partition_footprint_kernel(cell.kernel)
    # One observation reads one field coordinate, not the whole field.
    @test length(Base.invokelatest(footprint, u, 1).unconstrained) == 1
    # Units of positions keep their own coordinates under the strict partition.
    partition = prepare_partition(fx.built, fx.bound, :y; by=:unit, locality=:error)
    labels = partition_labels(partition)
    @test length(labels) == length(unique(unit))
    sampler = prepare_partition_sampler(fx.built, fx.bound, :y, u; by=:unit,
        backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    reads = partition_footprint(sampler, first(labels))
    @test length(reads) == count(==(first(labels)), unit)
    value, _ = partition_value_and_gradient!(sampler, zeros(length(reads)), u, first(labels))
    @test isfinite(value)
end

@testset "Frozen replay reads fitted pairs and refuses others" begin
    data = vecchia_rk_data()
    builder = @brm begin
        field ~ 0 + gmrf((east, north), time; neighbors=3)
        y ~ Normal(field, 0.2)
    end
    model = builder(data)
    fx = vecchia_rk_backend(model; case_id="gmrf-vecchia-replay")
    rows = [5, 1, 9]
    subset = (; east=data.east[rows], north=data.north[rows], time=data.time[rows], y=data.y[rows])
    replay = BRM.reprocess(fx.rk, subset)
    @test BRM._rk_layout_coordinate_names(replay) == BRM._rk_layout_coordinate_names(fx.rk)
    u = 0.1 .* randn(MersenneTwister(4), fx.built.layout.total)
    problem = vecchia_rk_problem(replay)
    names = BRM._rk_layout_coordinate_names(fx.rk)
    field = u[[i for (i, name) in enumerate(names) if occursin(".field.", string(name))]]
    @test isfinite(LDP.logdensity(problem, u))
    # A fitted state replays without the training rows.
    state = brm_fitted_state(fx.rk)
    from_state = RKBRMI(BRM._brm_rebind_brmi(model, subset), state)
    @test LDP.logdensity(vecchia_rk_problem(from_state), u) ≈ LDP.logdensity(problem, u) atol=1e-9
    moved = (; east=[data.east[1] + 0.01], north=[data.north[1]], time=[data.time[1]], y=[0.0])
    refusal = try
        LDP.logdensity(vecchia_rk_problem(BRM.reprocess(fx.rk, moved)), u)
        nothing
    catch error
        sprint(showerror, error)
    end
    @test refusal !== nothing && occursin("not in the fitted footprint", refusal)
end

@testset "The same reads in another row order give the same coordinates and density" begin
    # A tied lattice, where the maximin order has to break ties.
    grid = [(Float64(i), Float64(j)) for i in 1:4 for j in 1:4]
    times = [0.0, 1.0]
    rows = [(g..., t) for g in grid for t in times]
    rng = MersenneTwister(21)
    data = (; east=[r[1] for r in rows], north=[r[2] for r in rows], time=[r[3] for r in rows],
        y=0.3 .* randn(rng, length(rows)))
    p = randperm(rng, length(rows))
    builder = @brm begin
        field ~ 0 + gmrf((east, north), time; neighbors=3)
        y ~ Normal(field, 0.2)
    end
    a = vecchia_rk_backend(builder(data); case_id="gmrf-vecchia-rows")
    b = vecchia_rk_backend(builder(map(column -> column[p], data)); case_id="gmrf-vecchia-permuted")
    @test BRM._rk_layout_coordinate_names(a.rk) == BRM._rk_layout_coordinate_names(b.rk)
    u = 0.1 .* randn(rng, a.built.layout.total)
    @test LDP.logdensity(vecchia_rk_problem(a.rk), u) ≈ LDP.logdensity(vecchia_rk_problem(b.rk), u) atol=1e-9
end

@testset "Refused where the representation does not apply" begin
    data = vecchia_rk_data()
    # Exactly one representation.
    both = (; data..., xy=[0.0 0.0; 1 0; 0 1], triangles=[1 2 3])
    @test_throws ArgumentError RKBRMI(@brm both begin
        field ~ 0 + gmrf((east, north), time; neighbors=3, mesh=(vertices=xy, cells=triangles))
        y ~ Normal(field, 0.2)
    end)
    # The neighbour count is a number the model reads, from data too (user
    # principle `0ikeidi`), within the same bounds as a literal.
    for k in (0, 101, 2.5)
        counted = (; data..., k)
        @test_throws ArgumentError RKBRMI(@brm counted begin
            field ~ 0 + gmrf((east, north), time; neighbors=k)
            y ~ Normal(field, 0.2)
        end)
    end
    # One model-scope field: independent cell fields keep the mesh form.
    grouped = (; data..., window=repeat([1, 2]; outer=length(data.y) ÷ 2 + 1)[eachindex(data.y)])
    @test_throws Exception RKBRMI(@brm grouped begin
        @plate for p in groups(window)
            field ~ 0 + gmrf((east[p], north[p]), time[p]; neighbors=3)
            y[p] ~ Normal(field, 0.2)
        end
    end)
    # Embedded Laplace does not integrate it; the field stays out of its latent blocks.
    model = @brm data begin
        field ~ 0 + gmrf((east, north), time; neighbors=3)
        y ~ Normal(field, 0.2)
    end
    declaration = Symbol("gmrf_field_east_north_time")
    @test_throws Exception brm_laplace_problem(RKBRMI(model);
        ad_backend=DI.AutoEnzyme(; mode=Enzyme.Reverse), integrate=(declaration,))
end
