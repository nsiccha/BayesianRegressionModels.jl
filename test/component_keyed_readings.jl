module KeyedReadingsTest
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using ReactiveKernelsDistributionKernels
using Statistics: mean, std

@brm function sensor(record, signal)
    offset ~ Normal(0, 1)
    loc = signal[record.picks] .+ offset
    total = sum(loc)
    return (; total, loc)
end
const model = @brm begin
    gain ~ Normal(0, 1)
    @plate for i in eachindex(x)
        signal[i] = gain * x[i]
    end
    for record in records
        sensors[record.id] ~ sensor(record, signal)
    end
    y ~ Normal(sensors.total, 1)
end

@brm function station(spec, slope)
    offset ~ Normal(slope, 0.3)
    drift ~ Normal(offset, 0.4)
    innovation = drift - offset
    baseline = spec.times .* slope
    adjusted = baseline .+ drift
    return (; baseline, adjusted, drift, innovation)
end

const monitoring = @brm begin
    slope ~ Normal(0, 1)
    y ~ Normal(slope, 1)
    for spec in instruments
        stations[spec.id] ~ station(spec, slope)
    end
end

@stestset "Unobserved keyed member returns at fitted points" begin
    training = (; y=[0.4, 0.6], instruments=[
        (; id=:west, times=[-1.0]), (; id=:east, times=[1.0, 2.0, 3.0])])
    changed = (; training..., instruments=[
        (; id=:east, times=Float64[]), (; id=:west, times=[2.0, 0.0])])
    fit, problem = consumer_problem(monitoring(training))
    U = reshape([-1.2, 0.4, 2.0], 1, :)
    @test BRM._rk_layout_coordinate_names(fit) == [:slope]
    oracle(u) = logpdf(Normal(), only(u)) + sum(logpdf.(Normal(only(u), 1), training.y))
    check_consumer_point(problem, [0.4], oracle)
    for (r, data) in ((fit, training), (reprocess(fit, changed), changed)), spec in data.instruments
        key = (:stations, spec.id)
        values = rk_draw_values(r, U, (:baseline,); submodel=key)
        for j in axes(U, 2)
            @test values.baseline[j] ≈ spec.times .* U[1, j]
        end
        # An unconstrained posterior read supplies only the fitted coordinates (1skhpag).
        position = spec.id === :east ? 1 : 2
        @test_throws "requires generated draws for stations_member_all.drift.$position." rk_draw_values(
            r, U, (:baseline, :drift); submodel=key)
        @test_throws "no returned graph value" rk_draw_values(r, U, (:unknown,); submodel=key)
    end
    @testset "Seeded conditional draws shared across member fields and designs" begin
        posterior = repeat(U, 1, 100)
        batch = rk_generated_draws(fit, posterior; seed=29)
        @test length(batch) == size(posterior, 2)
        replay = reprocess(fit, changed)
        other = RKBRMI(monitoring(changed))
        @test_throws "compatible frozen design" rk_draw_values(other, batch, (:drift,);
            submodel=(:stations, :east))
        repeated = rk_generated_draws(fit, posterior; seed=29)
        changed_seed = rk_generated_draws(fit, posterior; seed=30)
        for spec in changed.instruments
            key = (:stations, spec.id)
            # Separate reads of the same batch must retain the same member draw.
            drift = rk_draw_values(fit, batch, (:drift,); submodel=key).drift
            expected = rk_draw_values(replay, batch, (:baseline, :adjusted, :drift, :innovation); submodel=key)
            @test drift == expected.drift
            @test expected.baseline == [spec.times .* slope for slope in posterior[1, :]]
            @test expected.adjusted == [base .+ d for (base, d) in zip(expected.baseline, drift)]
            @test rk_draw_values(fit, repeated, (:drift,); submodel=key).drift == drift
            @test rk_draw_values(fit, changed_seed, (:drift,); submodel=key).drift != drift
            # The unreturned offset and the conditional innovation are distinct
            # parts of the same physical realization on both designs.
            @test expected.innovation == rk_draw_values(fit, batch, (:innovation,); submodel=key).innovation
            residuals = drift .- vec(posterior)
            @test abs(std(residuals .- expected.innovation) - 0.3) < 0.06
            @test abs(std(expected.innovation) - 0.4) < 0.06
            @test abs(mean(residuals)) < 0.10
            @test abs(std(residuals) - 0.5) < 0.07
        end
        east = rk_draw_values(fit, batch, (:drift,); submodel=(:stations, :east)).drift
        west = rk_draw_values(fit, batch, (:drift,); submodel=(:stations, :west)).drift
        @test east != west
    end
    check_consumer_point(problem, [0.4], oracle)
    @test BRM._rk_layout_coordinate_names(fit) == [:slope]
end

@stestset "Read one component's scalar and variable-length returns by key" begin
    data = (; x=[2., -3., .75], records=[(; id=:b, picks=[1, 3]), (; id=:a, picks=[2])], y=[.3, .1])
    changed = (; data..., records=[(; id=:a, picks=Int[]), (; id=:b, picks=[2, 1, 3])], y=reverse(data.y))
    trained = model(data)
    rk, _ = consumer_problem(trained)
    sb = SBBRMI(trained; mod=@__MODULE__, total_groups=())
    U = [.5 -.25; .1 -.4; .25 .7] # gain, then offsets in key order
    @test_throws ArgumentError rk_draw_values(rk, U, (:loc,); submodel=(:sensors, :missing))
    @test_throws ArgumentError rk_draw_values(rk, U, (:unreturned,); submodel=(:sensors, :a))
    batch = rk_generated_draws(rk, U; seed=17)
    key = (:sensors, :b)
    @test rk_draw_values(rk, batch, (:loc, :total); submodel=key) ==
        rk_draw_values(rk, U, (:loc, :total); submodel=key)
    path = joinpath(mkpath(joinpath(tempdir(), "brm-rk-consumer")), "keyed-reading-returns.stan")
    for (design, r, s) in ((data, rk, sb), (changed, reprocess(rk, changed), reprocess(sb, changed)))
        stan = BRM.stan_instantiate(s; path).model
        names = BridgeStan.param_names(stan; include_tp=true)
        descriptor = brm_descriptor(s)
        problem = rk_logdensity_problem(r; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        draws = Dict(rec.id => rk_draw_values(r, U, (:total, :loc); submodel=(:sensors, rec.id))
            for rec in design.records)
        reference(rec, u) = u[1] .* design.x[rec.picks] .+ u[rec.id === :a ? 2 : 3]
        oracle(u) = sum(logpdf.(Normal(), u)) + sum(logpdf(Normal(sum(reference(rec, u)), 1), y)
            for (rec, y) in zip(design.records, design.y))
        for (j, u) in enumerate(eachcol(U))
            point = collect(u)
            value, gradient = check_consumer_point(problem, point, oracle)
            stan_value, stan_gradient = BridgeStan.log_density_gradient(stan, point; propto=false)
            @test stan_value ≈ value
            @test stan_gradient ≈ gradient
            constrained = BridgeStan.param_constrain(stan, point; include_tp=true)
            for record in design.records
                submodel = (:sensors, record.id)
                expected = reference(record, u)
                @test draws[record.id].total[j] ≈ sum(expected)
                @test draws[record.id].loc[j] ≈ expected
                indices = brm_output_coordinates(descriptor, :loc, names; submodel)
                @test constrained[indices] ≈ expected
            end
        end
    end
end

@brm function shifted_reading(x)
    bias ~ Normal(0, 1)
    return (; fitted=x + bias)
end
const single_field = @brm begin
    sensor ~ shifted_reading(x)
    y ~ Normal(sensor.fitted, 1)
end

@stestset "A single named return feeds the parent likelihood" begin
    sb = SBBRMI(single_field((; x=.75, y=-.2)); mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(stan_code(sb); warn_pedantic=false).ok
end

@brm function observed_sensor(y, picks, signal, heavy)
    offset ~ Normal(0, 0.6)
    if heavy
        @plate for s in eachindex(y)
            picked[s] = signal[s][picks[s]]
            loc[s] = picked[s] .+ offset
            y[s] .~ student_t.(6, loc[s], 1)
        end
    else
        @plate for s in eachindex(y)
            picked[s] = signal[s][picks[s]]
            loc[s] = picked[s] .+ offset
            y[s] .~ Normal.(loc[s], 1)
        end
    end
    return (; picked, loc)
end
const single = @brm begin
    gain ~ Normal(0, 1)
    @plate for s in eachindex(x)
        signal[s] = gain .* x[s]
    end
    device ~ observed_sensor(y, picks, signal, heavy)
end

@stestset "A component observes readings at the supplied subject trajectories" begin
    x = [[2.5, 1.], [.2, 1.1, .8]]
    for (heavy, picks, y) in ((false, [[1, 2], Int[]], [[.9, .5], Float64[]]),
                              (true, [Int[], [3]], [Float64[], [.3]]))
        model = single((; x, picks, y, heavy))
        rk, problem = consumer_problem(model)
        sb = SBBRMI(model; mod=@__MODULE__, total_groups=())
        stan = consumer_stan(model, "sensor-returned-pair-$heavy"; mod=@__MODULE__).model
        descriptor = brm_descriptor(sb)
        names = BridgeStan.param_names(stan; include_tp=true, include_gq=true)
        selected = reduce(vcat, (row[idx] for (row, idx) in zip(x, picks)))
        @test LogDensityProblems.dimension(problem) == 2
        family = heavy ? TDist(6) : Normal()
        function oracle(u)
            gain, offset = u
            value = logpdf(Normal(), gain) + logpdf(Normal(0, .6), offset)
            for s in eachindex(y), (row, measurement) in zip(picks[s], y[s])
                value += logpdf(family, measurement - (gain * x[s][row] + offset))
            end
            value
        end
        for u in ([.3, .4], [-.2, .15])
            value, gradient = check_consumer_point(problem, u, oracle)
            stan_value, stan_gradient = BridgeStan.log_density_gradient(stan, u; propto=false)
            @test stan_value ≈ value
            @test stan_gradient ≈ gradient
            constrained = BridgeStan.param_constrain(stan, u; include_tp=true, include_gq=true,
                rng=BridgeStan.StanRNG(stan, 1))
            for (field, expected) in ((:picked, u[1] .* selected), (:loc, u[1] .* selected .+ u[2]))
                indices = brm_output_coordinates(descriptor, field, names; submodel=:device)
                @test constrained[indices] ≈ expected
            end
        end
    end
end

@brm function reading_sensor(record, signal)
    offset ~ Normal(0, 1)
    @plate for s in eachindex(record.y)
        picked[s] = signal[s][record.picks[s]]
        loc[s] = picked[s] .+ offset
        record.y[s] .~ Normal.(loc[s], record.scale)
    end
    return (; picked, loc)
end

const observed_collection = @brm begin
    gain ~ Normal(0, 1)
    @plate for s in eachindex(x)
        signal[s] = gain .* x[s]
    end
    for record in records
        sensors[record.id] ~ reading_sensor(record, signal)
    end
end

const optional_calibration = @brm begin
    gain ~ Normal(0, 1)
    @plate for s in eachindex(x)
        signal[s] = gain .* x[s]
    end
    for record in records
        sensors[record.id] ~ reading_sensor(record, signal)
    end
    calibration_bias ~ Normal(0, 1)
    @plate for j in eachindex(t)
        calibration[j] ~ Normal(calibration_bias * t[j], 1)
    end
    reference ~ Normal(0, 1)
end

@stestset "Omitting a separate response keeps the record members' likelihoods" begin
    x = [[2., 1.], [3.]]
    record = (; id=:a, picks=[[2, 1], Int[]], y=[[0.1, 0.8], Float64[]], scale=0.7)
    design = (; records=[record], x, t=[1., 2.], reference=[0.4])
    for calibration in (nothing, [0.2, -0.1])
        data = isnothing(calibration) ? design : merge(design, (; calibration))
        rk, problem = consumer_problem(optional_calibration(data))
        names = coordinate_names(rk.model.layout)
        bias = Symbol("sensors_member_all.offset.1")
        expected = isnothing(calibration) ? [:gain, bias] : [:gain, bias, :calibration_bias]
        @test Set(names) == Set(expected)
        function oracle(u)
            gain, offset = u[coordinate(names, :gain)], u[coordinate(names, bias)]
            score = sum(logpdf.(Normal(), u)) + logpdf(Normal(), only(design.reference))
            for s in eachindex(x), (row, y) in zip(record.picks[s], record.y[s])
                score += logpdf(Normal(gain * x[s][row] + offset, record.scale), y)
            end
            isnothing(calibration) && return score
            delta = u[coordinate(names, :calibration_bias)]
            score + sum(logpdf(Normal(delta * t, 1), y) for (t, y) in zip(design.t, calibration))
        end
        check_consumer_point(problem, collect(range(-0.3, 0.2; length=length(names))), oracle)
    end
end

const calibrated_sensors = @brm begin
    gain ~ Normal(0, 1)
    @plate for s in eachindex(x)
        signal[s] = gain .* x[s]
    end
    for record in records
        sensors[record.id] ~ reading_sensor(record, signal)
    end
    @plate for s in eachindex(calibration)
        calibration[s] .~ Normal.(signal[s], 1)
    end
end

@stestset "A fitted ragged calibration keeps each sensor's reading axis" begin
    x = [[2., 1.], [3.]]
    calibration = [[0.2, -0.1], [0.4]]
    record = (; id=:a, picks=[[2, 1], Int[]], y=[[0.1, 0.8], Float64[]], scale=0.7)
    rk, problem = consumer_problem(calibrated_sensors((; records=[record], x, calibration)))
    names = coordinate_names(rk.model.layout)
    bias = Symbol("sensors_member_all.offset.1")
    @test Set(names) == Set([:gain, bias])
    function oracle(u)
        gain, offset = u[coordinate(names, :gain)], u[coordinate(names, bias)]
        score = sum(logpdf.(Normal(), u))
        for s in eachindex(x)
            for (input, y) in zip(x[s], calibration[s])
                score += logpdf(Normal(gain * input, 1), y)
            end
            for (row, y) in zip(record.picks[s], record.y[s])
                score += logpdf(Normal(gain * x[s][row] + offset, record.scale), y)
            end
        end
        score
    end
    check_consumer_point(problem, [0.3, -0.2], oracle)
end

@stestset "Repeated sensors score and return their own readings on replay" begin
    a = (; id=:a, picks=[[2, 1], Int[]], y=[[.3, -.2], Float64[]], scale=.7)
    b = (; id=:b, picks=[[1], [2]], y=[[.4], [.8]], scale=1.2)
    data = (; x=[[.5, 1.], [-.3, .7]], records=[b, a])
    empty_b = (; b..., picks=[Int[], Int[]], y=[Float64[], Float64[]])
    changed = (; x=reverse.(data.x), records=[a, empty_b])
    no_subjects_a = (; a..., picks=Vector{Int}[], y=Vector{Float64}[])
    leading_empty = (; data..., records=[b, no_subjects_a])
    trained = observed_collection(data)
    fit = RKBRMI(trained)
    sb = SBBRMI(trained; mod=@__MODULE__, total_groups=())
    path = joinpath(mkpath(joinpath(tempdir(), "brm-rk-consumer")), "keyed-observed-readings.stan")
    U = [.5 -.25; .1 -.4; .25 .7] # gain, offset for a, offset for b
    for (design, rk, sfit) in ((data, fit, sb),
            (changed, reprocess(fit, changed), reprocess(sb, changed)),
            (leading_empty, reprocess(fit, leading_empty), reprocess(sb, leading_empty)))
        stan = BRM.stan_instantiate(sfit; path).model
        names = BridgeStan.param_names(stan; include_tp=true, include_gq=true)
        descriptor = brm_descriptor(sfit)
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        @test LogDensityProblems.dimension(problem) == size(U, 1)
        draws = Dict(rec.id => rk_draw_values(rk, U, (:picked, :loc); submodel=(:sensors, rec.id))
            for rec in design.records)
        picked(rec, s, u) = u[1] .* design.x[s][rec.picks[s]]
        location(rec, s, u) = picked(rec, s, u) .+ u[rec.id === :a ? 2 : 3]
        function oracle(u)
            value = sum(logpdf.(Normal(), u))
            for rec in design.records, s in eachindex(rec.y)
                for (y, loc) in zip(rec.y[s], location(rec, s, u))
                    value += logpdf(Normal(loc, rec.scale), y)
                end
            end
            value
        end
        for (j, u) in enumerate(eachcol(U))
            point = collect(u)
            value, gradient = check_consumer_point(problem, point, oracle)
            stan_value, stan_gradient = BridgeStan.log_density_gradient(stan, point; propto=false)
            @test stan_value ≈ value
            @test stan_gradient ≈ gradient
            constrained = BridgeStan.param_constrain(stan, point; include_tp=true, include_gq=true,
                rng=BridgeStan.StanRNG(stan, j))
            for rec in design.records
                for (field, value) in ((:picked, picked), (:loc, location))
                    expected = reduce(vcat, (value(rec, s, u) for s in eachindex(rec.y)); init=Float64[])
                    @test draws[rec.id][field][j] ≈ expected
                    indices = brm_output_coordinates(descriptor, field, names; submodel=(:sensors, rec.id))
                    @test constrained[indices] ≈ expected
                end
            end
        end
    end
end

# A member owns a site margin in the parent's correlated block and indexes it,
# with the parent's trajectories, in its own subject plate.
@brm function site_reading(entry, curve)
    offset ~ Normal(0.0, 1.0)
    shift ~ 0 + (1 | parentblock(:p) | entry.site)
    scale ~ LogNormal(-0.7, 1.0)
    @plate for s in eachindex(entry.y)
        loc[s] = offset .+ shift[entry.rows[s]] .+ curve[s][entry.stack[s]]
        entry.y[s] .~ Normal.(loc[s], scale)
    end
    return (; loc)
end
const site_model = @brm begin
    level ~ 1 + (1 | p | site)
    effect(level, Intercept) ~ Normal(0.0, 1.0)
    sd(:, p) ~ Exponential(0.3333333333333333)
    cor(:, p) ~ LKJCholesky(n_margins, 2.0)
    @plate for i in eachindex(level)
        curve[i] = exp(level[i]) .* exp.(-0.1 .* read_time[i])
    end
    for entry in records
        channels[entry.id] ~ site_reading(entry, curve)
    end
end

@stestset "Group redraws preserve a member's observed readings and return its new locations" begin
    function design(sites, picks)
        record(id, stack, value) = (; id, stack, site=sites,
            rows=[fill(i, length(s)) for (i, s) in pairs(stack)],
            y=[value .+ i .+ 0.1 .* s for (i, s) in pairs(stack)])
        records = [record(:b, reverse(picks), 2.0), record(:a, picks, 1.0)]
        (; site=sites, read_time=[[1., 2., 3.] for _ in sites], records, n_margins=3)
    end
    training = design([1, 2, 3], [[1], [1, 2], [3]])
    fit = SBBRMI(site_model(training); mod=@__MODULE__)
    dir = mkpath(joinpath(tempdir(), "brm-rk-consumer"))
    fitted = BRM.stan_instantiate(fit; path=joinpath(dir, "member-reading-fit.stan")).model
    from = BridgeStan.param_unc_names(fitted)
    point = collect(range(-0.3, 0.3; length=length(from)))
    for sites in (training.site, [3, 4, 5])
        dense = design(sites, [[2, 1], [1, 2, 3], [3, 1, 2]])
        replay = reprocess(fit, dense; resample_groups=[:site])
        descriptor = brm_descriptor(replay)
        kinds = Dict(o.name => o.kind for o in descriptor.outputs)
        @test kinds[:b_p_site_z_flat] === :generated_quantity
        @test kinds[:b_p_site_L] === kinds[:b_p_site_tau] === :parameter
        readings = stan_data(replay)[:channels_all_entry_y].mem
        @test readings == reduce(vcat, [reduce(vcat, r.y) for r in sort(dense.records; by=r -> r.id)])

        model = BRM.stan_instantiate(replay; path=joinpath(dir, "member-reading-redraw.stan")).model
        to = BridgeStan.param_unc_names(model)
        @test Set(to) == Set(filter(name -> !startswith(name, "b_p_site_z_flat."), from))
        transported = vec(transport_draws(fit, replay, reshape(point, 1, :), from, to))
        @test transported == [point[findfirst(==(name), from)] for name in to]
        names = BridgeStan.param_names(model; include_tp=true, include_gq=true)
        draws = [BridgeStan.param_constrain(model, transported; include_tp=true,
            include_gq=true, rng=BridgeStan.StanRNG(model, seed)) for seed in (7, 13)]
        level = brm_output_coordinates(descriptor, :level, names)
        for r in dense.records
            idx = brm_output_coordinates(descriptor, :loc, names; submodel=(:channels, r.id))
            @test length(idx) == sum(length, r.stack)
            @test draws[1][idx] != draws[2][idx]
            for draw in draws
                # Within a site both the member offset and the shared margin cancel.
                # The remaining differences identify the member's actual read times.
                first_read = 1
                for (i, rows) in pairs(r.stack)
                    last_read = first_read + length(rows) - 1
                    pred = draw[idx[first_read:last_read]]
                    expected = exp(draw[level[i]]) .* exp.(-0.1 .* dense.read_time[i][rows])
                    @test diff(pred) ≈ diff(expected)
                    first_read = last_read + 1
                end
            end
        end
    end
end

# The same model without components.
const site_twin = @brm begin
    level ~ 1 + (1 | p | site)
    effect(level, Intercept) ~ Normal(0.0, 1.0)
    sd(:, p) ~ Exponential(0.3333333333333333)
    cor(:, p) ~ LKJCholesky(n_margins, 2.0)
    a_offset ~ Normal(0.0, 1.0)
    a_shift ~ 0 + (1 | p | a_site)
    a_scale ~ LogNormal(-0.7, 1.0)
    b_offset ~ Normal(0.0, 1.0)
    b_shift ~ 0 + (1 | p | b_site)
    b_scale ~ LogNormal(-0.7, 1.0)
    ya ~ Normal(a_offset + a_shift[a_rows] + exp(level[a_rows]) * a_decay, a_scale)
    yb ~ Normal(b_offset + b_shift[b_rows] + exp(level[b_rows]) * b_decay, b_scale)
end

@stestset "A member with a shared site margin returns its fitted readings" begin
    reads = [[1, 2, 3], [1, 3], [2, 3], [1, 2, 3]]
    entry(id, level) = (; id, stack=reads, y=[level .+ 0.1 .* r for r in reads],
        site=collect(1:4), rows=[fill(s, length(reads[s])) for s in eachindex(reads)])
    records = [entry(:b, -0.3), entry(:a, 0.2)]   # supplied out of key order
    read_time = [[2.0, 12.0, 48.0], [1.0, 9.0, 30.0], [3.0, 14.0, 50.0], [2.5, 11.0, 40.0]]
    m = site_model((; site=collect(1:4), read_time, records, n_margins=3))
    rk = RKBRMI(m)
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    flat(record) = (; rows=reduce(vcat, record.rows), y=reduce(vcat, record.y),
        decay=reduce(vcat, [exp.(-0.1 .* read_time[s][record.stack[s]]) for s in eachindex(reads)]))
    a, b = flat(records[2]), flat(records[1])
    twin = site_twin((; site=collect(1:4), n_margins=3, a_site=collect(1:4), b_site=collect(1:4),
        a_rows=a.rows, a_decay=a.decay, ya=a.y, b_rows=b.rows, b_decay=b.decay, yb=b.y))
    rkt, reference = consumer_problem(twin)
    names = coordinate_names(rk.model.layout)
    println("RK_NAMES=", names); flush(stdout)
    tnames = coordinate_names(rkt.model.layout)
    rename = Dict(Symbol("channels_member_all.offset.1") => :a_offset,
        Symbol("channels_member_all.offset.2") => :b_offset,
        Symbol("channels_member_all.scale.1") => :a_scale,
        Symbol("channels_member_all.scale.2") => :b_scale)
    to_twin = [coordinate(tnames, get(rename, name, name)) for name in names]
    @test sort(to_twin) == 1:length(tnames) == 1:23
    n = length(names)
    points = (zeros(n), collect(range(-0.37, 0.31; length=n)), [0.3sin(1.7i) for i in 1:n])
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(m, "keyed-shared-margin-readings"; mod=@__MODULE__)
    descriptor = brm_descriptor(sb)
    stan_names = BridgeStan.param_names(stan.model; include_tp=true, include_gq=true)
    transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    for (j, u) in enumerate(points)
        v = similar(u); v[to_twin] = u
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        expected, reference_gradient = LogDensityProblems.logdensity_and_gradient(reference, v)
        @test value ≈ expected atol=2e-11 rtol=2e-11
        @test gradient ≈ reference_gradient[to_twin] atol=2e-9 rtol=2e-9
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, stan_u; propto=false)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value)
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient)
        # The twin's own formula values give each reading's fitted location.
        parts = rk_draw_values(rkt, reshape(v, :, 1), (:level, :a_shift, :b_shift))
        level = only(parts.level)
        constrained = BridgeStan.param_constrain(stan.model, stan_u; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(stan.model, j))
        for (key, record, shift, offset) in ((:a, a, only(parts.a_shift), v[coordinate(tnames, :a_offset)]),
                                             (:b, b, only(parts.b_shift), v[coordinate(tnames, :b_offset)]))
            fitted = offset .+ shift[record.rows] .+ exp.(level[record.rows]) .* record.decay
            @test only(rk_draw_values(rk, reshape(u, :, 1), (:loc,); submodel=(:channels, key)).loc) ≈ fitted
            indices = brm_output_coordinates(descriptor, :loc, stan_names; submodel=(:channels, key))
            @test constrained[indices] ≈ fitted
        end
    end
end
end
