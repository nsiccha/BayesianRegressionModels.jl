using Test, BayesianRegressionModels, LinearAlgebra, Random, Distributions
using StanBlocks, ReactiveKernels, ReactiveKernelsPPL, Enzyme
import BridgeStan as BS
import DifferentiationInterface as DI
import LogDensityProblems as LDP
const BRM = BayesianRegressionModels
include("testset_filter.jl")

function multiresolution_data(; channels=false)
    rng = MersenneTwister(91)
    P = 7
    x, y = 10 .* rand(rng, P), 10 .* rand(rng, P)
    time = repeat([0.0, 1.0, 13.0, 72.0, 180.0]; inner=P)
    keep = [i for i in eachindex(time) if mod(i, 4) != 0]
    east, north = repeat(x; outer=5)[keep], repeat(y; outer=5)[keep]
    time = time[keep]
    if channels
        return (; east=repeat(east; inner=2), north=repeat(north; inner=2),
            time=repeat(time; inner=2), channel=repeat(["radar", "gauge"]; outer=length(time)),
            y=0.3 .* randn(rng, 2length(time)))
    end
    (; east, north, time, y=0.3 .* randn(rng, length(time)))
end
function multiresolution_model(data; channels=false)
    if channels
        return @brm data begin
            Omega ~ LKJCholesky(2, 2.0)
            field ~ 0 + gmrf((east, north), time, channel; multiresolution=3, cor=Omega)
            sd(field, gmrf((east, north), time, channel)) = 1.1
            length_scale(field, gmrf((east, north), time, channel), space) = 3.0
            length_scale(field, gmrf((east, north), time, channel), time) = 13.0
            y ~ Normal(field, 0.2)
        end
    end
    @brm data begin
        field ~ 0 + gmrf((east, north), time; multiresolution=3)
        sd(field, gmrf((east, north), time)) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(3.0), 0.4)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(13.0), 0.4)
        y ~ Normal(field, 0.2)
    end
end
function multiresolution_native(model; id="gmrf-multiresolution")
    artifact = BRM.emit_rk_artifact(model; case_id=id)
    bound = BRM.rk_translate_artifact(artifact)
    built = build_kernel(bound)
    (; bound, built, rk=RKBRMI(model, artifact.plan, built))
end
multiresolution_problem(rk) = rk_logdensity_problem(rk;
    ad_backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))

@stestset "Stan spatial basis, compressed reader, and normalized prior agree with the core" begin
    data = multiresolution_data(; channels=true)
    sb = SBBRMI(multiresolution_model(data; channels=true); mod=@__MODULE__)
    check = StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false)
    check.ok || println(check)
    @test check.ok
    prepared = only(e.const_.prepared for e in values(sb.preproc) if e.kind === :gmrf)
    g = prepared.state.support
    t = BRM._sb_gmrf_multiresolution_data(prepared)
    weights = BRM.brm_gmrf_multiresolution_basis(3.0, t.east, t.north, t.origin, t.item_ptr,
        t.item_position, t.item_lag, t.neighbor_ptr, t.neighbor, t.state_pattern,
        t.basis_ptr, t.basis_node, t.M, t.minimum_distance)
    @test weights ≈ BRM._brm_multiresolution_basis(g.spatial, 3.0).weights atol=1e-12
    field = 0.2 .* sin.(1:g.ncoordinates)
    readvalues = BRM.brm_gmrf_multiresolution_read(field, weights, t.C, t.n_read,
        t.basis_ptr, t.basis_node, t.read_ptr, t.path_ptr, t.mapid,
        t.run_ptr, t.run_start, t.run_stop, t.run_delta)
    @test readvalues[t.index] ≈ [BRM._brm_multiresolution_value(field, weights, g, r)
        for r in prepared.state.reads] atol=1e-14
    L = [1.0 0.0; 0.4 sqrt(0.84)]
    @test BRM.brm_gmrf_multiresolution_logdensity(field, 13.0, L, t.times, t.path_ptr, t.path_time) ≈
        BRM._brm_multiresolution_density(field, 13.0, L, g).value atol=1e-10
end

@stestset "Native density and reverse derivative against an independently assembled target" begin
    data = multiresolution_data()
    model = multiresolution_model(data)
    fx = multiresolution_native(model)
    names = BRM._rk_layout_coordinate_names(fx.rk)
    u = 0.1 .* sin.(1:length(names))
    pos(s) = only(i for (i, n) in enumerate(names) if endswith(string(n), s))
    sigma, ell, theta = exp.(u[[pos(".sigma"), pos(".spatial_length"), pos(".temporal_length")]])
    field = u[[i for (i,n) in enumerate(names) if occursin(".field.", string(n))]]
    g = BRM._brm_multiresolution_support(data.east, data.north, data.time; rank=3)
    weights = BRM._brm_multiresolution_basis(g.spatial, ell).weights
    reads, _ = BRM._brm_multiresolution_rows(g, [1], data.east, data.north, data.time, ones(Int,length(data.y)))
    expected = BRM._brm_multiresolution_density(field, theta, ones(1,1), g).value +
        sum(logpdf(Normal(sigma * BRM._brm_multiresolution_value(field, weights, g, r), 0.2), y)
            for (r,y) in zip(reads,data.y)) + logpdf(Exponential(0.7),sigma) + log(sigma) +
        logpdf(LogNormal(log(3.0),0.4),ell) + log(ell) +
        logpdf(LogNormal(log(13.0),0.4),theta) + log(theta)
    query = prepare_query(fx.built, fx.bound, :sampler)
    @test Base.invokelatest(query, u) ≈ expected atol=1e-8
    sampler = prepare_sampler(fx.built, fx.bound, u; backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    original = copy(u)
    value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ expected atol=1e-8
    @test u == original
    h = 1e-6
    for j in eachindex(u)
        plus, minus = copy(u), copy(u)
        plus[j] += h; minus[j] -= h
        @test gradient[j] ≈ (Base.invokelatest(query,plus) - Base.invokelatest(query,minus)) / 2h atol=2e-4 rtol=2e-5
    end
end

@stestset "Compiled Stan and native reverse agree through physical coordinate transport" begin
    for channels in (false, true)
        model = multiresolution_model(multiresolution_data(; channels); channels)
        fx = multiresolution_native(model; id="gmrf-multiresolution-transport-$channels")
        sb = SBBRMI(model; mod=@__MODULE__)
        stan = BRM.stan_instantiate(sb)
        transport = brm_coordinate_transport(fx.rk, sb, BS.param_unc_names(stan.model))
        u = 0.1 .* sin.(1:fx.built.layout.total)
        sampler = prepare_sampler(fx.built, fx.bound, u; backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
        points = [zeros(length(u)), u]
        if !channels
            tiny = copy(u)
            names = BRM._rk_layout_coordinate_names(fx.rk)
            for suffix in (".spatial_length", ".temporal_length")
                tiny[only(i for (i,n) in enumerate(names) if endswith(string(n),suffix))] = -365.0
            end
            push!(points,tiny)
        end
        for point in points
            @test brm_check_coordinate_transport(transport, fx.rk, stan.model, point).max_error < 1e-10
            value, gradient = sampler_value_and_gradient!(sampler, similar(u), point)
            sv, sg = BS.log_density_gradient(stan.model, brm_rk_to_stan(transport, point);
                propto=false, jacobian=true)
            @test value ≈ sv atol=2e-8 rtol=2e-10
            @test gradient ≈ brm_stan_to_rk(transport, sg) atol=3e-7 rtol=3e-7
        end
    end
end

@stestset "Primal coordinate transport and compiled Stan derivative control" begin
    for channels in (false,true)
        data=multiresolution_data(;channels)
        model=multiresolution_model(data;channels)
        fx=multiresolution_native(model;id="multiresolution-primal-$channels")
        query=prepare_query(fx.built,fx.bound,:sampler)
        sb=SBBRMI(model;mod=@__MODULE__)
        stan=BRM.stan_instantiate(sb)
        tr=brm_coordinate_transport(fx.rk,sb,BS.param_unc_names(stan.model))
        u=.1 .* sin.(1:fx.built.layout.total)
        for point in (zeros(length(u)),u)
            @test brm_check_coordinate_transport(tr,fx.rk,stan.model,point).max_error < 1e-10
            native=Base.invokelatest(query,point)
            sv,sg=BS.log_density_gradient(stan.model,brm_rk_to_stan(tr,point);propto=false,jacobian=true)
            @test native ≈ sv atol=2e-8 rtol=2e-10
            gradient=brm_stan_to_rk(tr,sg)
            h=1e-6
            for j in eachindex(point)
                plus,minus=copy(point),copy(point)
                plus[j]+=h;minus[j]-=h
                fd=(Base.invokelatest(query,plus)-Base.invokelatest(query,minus))/(2h)
                @test fd ≈ gradient[j] atol=2e-4 rtol=2e-5
            end
        end
    end
end

@stestset "Strict cell locality retains only a read's shared OU innovations" begin
    data = multiresolution_data()
    input = (; data..., state=collect(eachindex(data.y)))
    model = @brm input begin
        field ~ 0 + gmrf((east,north),time; multiresolution=3)
        sd(field,gmrf((east,north),time)) = 0.5
        length_scale(field,gmrf((east,north),time),space) = 3.0
        length_scale(field,gmrf((east,north),time),time) = 13.0
        @plate for s in eachindex(state)
            y[s] ~ Normal(exp(field[state[s]]),0.2)
        end
    end
    fx = multiresolution_native(model; id="gmrf-multiresolution-locality")
    u = 0.1 .* sin.(1:fx.built.layout.total)
    cell = prepare_cell_query(fx.built,fx.bound,:y; locality=:error)
    footprint = ReactiveKernels.partition_footprint_kernel(cell.kernel)
    g = BRM._brm_multiresolution_support(data.east,data.north,data.time; rank=3)
    for i in (1, length(data.y))
        read = BRM._brm_multiresolution_read(g,data.east[i],data.north[i],data.time[i])
        expected = [BRM._brm_multiresolution_read_coordinate(g,read,j)
            for j in BRM._brm_multiresolution_basis_range(g,read)]
        actual = Base.invokelatest(footprint,u,i).unconstrained
        @test sort(actual) == sort(expected)
        @test length(actual) < length(u)
    end
end

# Under a learned spatial length the basis is one whole value per length, not a
# value with rows: a row is built from its parents' rows, so a row form would need
# a large share of the support. A strict partition therefore holds it as a shared
# value, refreshed when the length changes. A unit gradient pays one whole
# pullback; a cell pass pays one for the whole batch.
@stestset "Strict units under learned lengths read one shared basis" begin
    data = multiresolution_data()
    positions = Dict{Tuple{Float64,Float64},Int}()
    unit = [get!(positions, p, length(positions) + 1) for p in zip(data.east, data.north)]
    input = (; data..., state=collect(eachindex(data.y)))
    model = @brm input begin
        field ~ 0 + gmrf((east,north),time; multiresolution=3)
        sd(field,gmrf((east,north),time)) ~ Exponential(0.7)
        length_scale(field,gmrf((east,north),time),space) ~ LogNormal(log(3.0),0.4)
        length_scale(field,gmrf((east,north),time),time) ~ LogNormal(log(13.0),0.4)
        @plate for s in eachindex(state)
            y[s] ~ Normal(exp(field[state[s]]),0.2)
        end
    end
    fx = multiresolution_native(model; id="gmrf-multiresolution-learned-locality")
    names = BRM._rk_layout_coordinate_names(fx.rk)
    pos(s) = only(i for (i, n) in enumerate(names) if endswith(string(n), s))
    states = [i for (i, n) in enumerate(names) if occursin(".field.", string(n))]
    g = BRM._brm_multiresolution_support(data.east, data.north, data.time; rank=3)
    reads, _ = BRM._brm_multiresolution_rows(g, [1], data.east, data.north, data.time, ones(Int, length(data.y)))
    # An independent evaluation of a unit's reads from the core.
    function oracle(u, rows)
        sigma, ell = exp(u[pos(".sigma")]), exp(u[pos(".spatial_length")])
        weights = BRM._brm_multiresolution_basis(g.spatial, ell).weights
        sum(logpdf(Normal(exp(sigma * BRM._brm_multiresolution_value(u[states], weights, g, reads[r])), 0.2),
            data.y[r]) for r in rows)
    end
    rows(label) = findall(==(label), unit)
    u = 0.1 .* sin.(1:fx.built.layout.total)
    labels = sort!(unique(unit))
    # The spelling the guide gives for a model whose units own no coordinates:
    # any parameter, with every level shared.
    entry = first(fx.built.layout.entries)
    shared = (entry.name => fill(missing, isempty(entry.dims) ? entry.size : first(entry.dims)),)
    by = (; y=unit)

    query = prepare_partition(fx.built, fx.bound, :y; by, locality=:error, partitioned=shared, u0=u)
    weights = only(s for s in ReactiveKernels.partition_locality(query) if endswith(string(s.value), ".weights"))
    @test weights.status === :input
    for label in labels
        @test query(u, label) ≈ oracle(u, rows(label)) atol=1e-10
    end
    # A new length refreshes the shared basis; so does coming back to the old one.
    moved = copy(u)
    moved[pos(".spatial_length")] += 0.7
    @test !(oracle(moved, rows(labels[1])) ≈ oracle(u, rows(labels[1])))
    for point in (moved, u)
        for label in labels
            @test query(point, label) ≈ oracle(point, rows(label)) atol=1e-10
        end
    end

    backend = DI.AutoEnzyme(; mode=Enzyme.Reverse)
    sampler = prepare_partition_sampler(fx.built, fx.bound, :y, u; by, backend, locality=:error, partitioned=shared)
    h = 1e-6
    function central(i, label)
        plus, minus = copy(u), copy(u)
        plus[i] += h; minus[i] -= h
        (oracle(plus, rows(label)) - oracle(minus, rows(label))) / 2h
    end
    unit_gradients = zeros(length(u), length(labels))
    for (k, label) in enumerate(labels)
        footprint = partition_footprint(sampler, label)
        expected = sort!(unique!([pos(".sigma"); pos(".spatial_length");
            [states[BRM._brm_multiresolution_read_coordinate(g, reads[r], j)]
                for r in rows(label) for j in BRM._brm_multiresolution_basis_range(g, reads[r])]]))
        @test footprint == expected
        gradient = zeros(length(footprint))
        value, _ = partition_value_and_gradient!(sampler, gradient, u, label)
        @test value ≈ oracle(u, rows(label)) atol=1e-10
        unit_gradients[footprint, k] .= gradient
        for i in eachindex(u)
            @test unit_gradients[i, k] ≈ central(i, label) atol=2e-4 rtol=2e-5
        end
    end

    # One pass over the batch's cells, one shared pullback, the same derivatives.
    compact = prepare_compact_cell_ad(ReactiveKernels.prepare_ad, fx.built, fx.bound, :y, u, Int[], backend;
        partitioned=shared, locality=:error)
    accumulator = prepare_cell_accumulator(compact, backend, u; order=0)
    pullback = prepare_cell_pullback(compact, backend, u)
    for batch_labels in (labels[1:1], labels[2:3], labels)
        gradient = zeros(length(u))
        total = Ref(0.0)
        cell_pullback!(gradient, pullback, u) do batch
            for label in batch_labels, cell in rows(label)
                total[] += first(accumulate_cell!(batch, accumulator, nothing, 1.0, Float64[], Int[], cell))
            end
        end
        @test total[] ≈ sum(oracle(u, rows(label)) for label in batch_labels) atol=1e-10
        @test gradient ≈ vec(sum(unit_gradients[:, findall(in(batch_labels), labels)]; dims=2)) atol=1e-10
    end
end

@stestset "Frozen replay keeps shared coordinate identities; fresh fits rebuild their support" begin
    data = multiresolution_data()
    model = multiresolution_model(data)
    fx = multiresolution_native(model; id="gmrf-multiresolution-replay")
    sb = SBBRMI(model; mod=@__MODULE__)
    rows = [5,1,9,5]
    subset = (; (k=>v[rows] for (k,v) in pairs(data))...)
    native = BRM.reprocess(fx.rk,subset)
    @test BRM._rk_layout_coordinate_names(native) == BRM._rk_layout_coordinate_names(fx.rk)
    u = 0.1 .* sin.(1:fx.built.layout.total)
    from_state = RKBRMI(BRM._brm_rebind_brmi(model,subset),brm_fitted_state(fx.rk))
    @test LDP.logdensity(multiresolution_problem(native),u) ≈
        LDP.logdensity(multiresolution_problem(from_state),u) atol=1e-9
    replay = BRM.reprocess(sb,subset)
    key(d,suffix) = only(k for k in keys(d) if endswith(string(k),suffix))
    @test replay.data[key(replay.data,"_index")] == sb.data[key(sb.data,"_index")][rows]
    @test replay.data[key(replay.data,"_n_field")] == sb.data[key(sb.data,"_n_field")]
    fresh = BRM.reprocess(sb,subset; freeze_constants=false)
    g = BRM._brm_multiresolution_support(subset.east,subset.north,subset.time; rank=3)
    @test fresh.data[key(fresh.data,"_n_field")] == g.ncoordinates
    moved = (; east=[data.east[1]+0.01], north=[data.north[1]],time=[data.time[1]],y=[0.0])
    @test_throws ArgumentError BRM.reprocess(sb,moved)
    @test_throws Exception LDP.logdensity(multiresolution_problem(BRM.reprocess(fx.rk,moved)),u)
end

@stestset "Two fields retain distinct innovations and known channel matrices" begin
    data = multiresolution_data(; channels=true)
    input = (; data..., east2=data.east .+ .37, north2=data.north,
        corr=[1.0 .4; .4 1.0])
    model = @brm input begin
        a ~ 0 + gmrf((east,north),time,channel; multiresolution=3, cor=corr)
        b ~ 0 + gmrf((east2,north2),time,channel; multiresolution=2, cor=corr)
        y ~ Normal(a .+ b, .2)
    end
    fx = multiresolution_native(model; id="gmrf-multiresolution-two-fields")
    terms = [t for p in BRM._rk_replay_predictors(fx.rk.plan) for t in p.terms if t.kind === :gmrf]
    supports = [BRM.brm_gmrf_multiresolution_support(
        (fx.rk.plan.columns[n] for n in values(t.options.fit))..., t.options.multiresolution)
        for t in terms]
    emitted = BRM._rk_emit_ast(fx.rk.plan)
    @test occursin("brm_gmrf_multiresolution_support", string(emitted.main, emitted.defs))
    @test all(v -> !(v isa AbstractVector{<:BRM._BRMGMRFMultiresolutionSupport}),
        values(fx.rk.plan.columns))
    @test length(supports) == 2
    @test Set(s.geometry.spatial.rank for s in supports) == Set([2,3])
    @test !isequal(supports[1].geometry.spatial.conditionals.east,
        supports[2].geometry.spatial.conditionals.east)
    sb = SBBRMI(model; mod=@__MODULE__)
    stan = BRM.stan_instantiate(sb)
    tr = brm_coordinate_transport(fx.rk,sb,BS.param_unc_names(stan.model))
    u = .1 .* cos.(1:fx.built.layout.total)
    @test brm_check_coordinate_transport(tr,fx.rk,stan.model,u).max_error < 1e-10
    problem = multiresolution_problem(fx.rk)
    v,g = LDP.logdensity_and_gradient(problem,u)
    sv,sg = BS.log_density_gradient(stan.model,brm_rk_to_stan(tr,u);propto=false,jacobian=true)
    @test v ≈ sv atol=2e-8
    @test g ≈ brm_stan_to_rk(tr,sg) atol=3e-7 rtol=3e-7
end

@stestset "Descriptions and invalid declarations" begin
    data = multiresolution_data()
    builder = @brm begin
        field ~ 0 + gmrf((east,north),time; multiresolution=3)
        y ~ Normal(field,.2)
    end
    d = brm_descriptor(builder,data;mod=@__MODULE__,name=:shared_ou)
    @test occursin("temporal exponential covariance is exact",string(brm_description(d)))
    @test occursin("zeitliche Kovarianz",string(brm_description(d;texts=description_texts(:de))))
    @test_throws ArgumentError SBBRMI(@brm data begin
        field ~ 0 + gmrf((east,north),time; multiresolution=3,neighbors=3)
        y ~ Normal(field,.2)
    end)
    @test_throws ArgumentError SBBRMI(@brm data begin
        field ~ 0 + gmrf((east,north),time; multiresolution=0)
        y ~ Normal(field,.2)
    end)
end

@stestset "Stan prior RNG uses stationary shared OU paths" begin
    data = (; east=[0.,1.,0.,1.], north=zeros(4),time=[0.,0.,1.,72.],
        channel=["a","b","b","a"],corr=[1.0 .4;.4 1.0])
    model = @brm data begin
        field ~ 0 + gmrf((east,north),time,channel;multiresolution=2,cor=corr)
        sd(field,gmrf((east,north),time,channel)) = 1.2
        length_scale(field,gmrf((east,north),time,channel),space) = .4
        length_scale(field,gmrf((east,north),time,channel),time) = 13.0
        y ~ Normal(field,.2)
    end
    sb = SBBRMI(model;mod=@__MODULE__)
    problem = BRM.stan_instantiate(sb)
    @test BS.param_unc_num(problem.model) == 0
    descriptor = brm_descriptor(sb;highlights=())
    names = BS.param_names(problem.model;include_tp=true,include_gq=true)
    indices = brm_term_coordinates(descriptor,:field,names;
        term=:gmrf_field_east_north_time_channel,parameter=:field).coordinates
    rng = BS.StanRNG(problem.model,617)
    draws = reduce(hcat,(BS.param_constrain(problem.model,Float64[];
        include_tp=true,include_gq=true,rng)[indices] for _ in 1:12000))
    s = only(e.const_.prepared.state for e in values(sb.preproc) if e.kind === :gmrf)
    g = s.support
    coordinates = [BRM._brm_multiresolution_coordinate(g,s.channels,j) for j in 1:g.ncoordinates]
    expected = [a.innovation==b.innovation ?
        exp(-abs(a.time-b.time)/13.0)*(a.channel==b.channel ? 1.0 : .4) : 0.0
        for a in coordinates,b in coordinates]
    empirical = draws*draws'/size(draws,2)
    @test maximum(abs,vec(sum(draws;dims=2)/size(draws,2))) < .04
    @test maximum(abs,diag(empirical).-1) < .04
    @test maximum(abs,empirical-expected) < .04
end
