module CollectionNamedReturnsTest
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels

# A consumer project may refuse a prepared program that dispatches at run time
# (the ReactiveKernels preference `type_stability = "error"`).
strict(f) = ReactiveKernels._with_type_stability(f, :error, :all)

# A curve on the member's own grid beside a fitted scale.
@brm function scaled_reading(config)
    gain ~ Normal(0, 1)
    scale ~ LogNormal(-0.7, 1.0)
    curve = gain .* config.x
    @observe s walk member begin
        config.y[s] ~ Normal(walk .+ member.curve[config.pick[s]], member.scale)
    end
    return (; curve, scale)
end
const scaled_model = @brm begin
    for c in sensors
        channels[c.id] ~ scaled_reading(c)
    end
    @plate for s in eachindex(input)
        shared[s] = 0.5 * input[s]
        observe(channels, s, shared[s])
    end
end

# A per-site vector in the parent's correlated block beside two scalars.
@brm function site_reading(entry)
    offset ~ Normal(0.0, 1.0)
    shift ~ 0 + (1 | parentblock(:p) | entry.site)
    scale ~ LogNormal(-0.7, 1.0)
    @observe s trajectory member begin
        entry.y[s] ~ Normal(member.offset .+ member.shift[entry.rows[s]] .+
            trajectory[entry.stack[s]], member.scale)
    end
    return (; offset, shift, scale)
end
const site_model = @brm begin
    level ~ 1 + (1 | p | site)
    effect(level, Intercept) ~ Normal(0.0, 1.0)
    sd(:, p) ~ Exponential(0.3333333333333333)
    cor(:, p) ~ LKJCholesky(n_margins, 2.0)
    for entry in records
        channels[entry.id] ~ site_reading(entry)
    end
    @plate for i in eachindex(level)
        path = exp(level[i]) .* exp.(-0.1 .* read_time[i])
        curve[i] = path
        observe(channels, i, curve[i])
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

@stestset "A named return reads a member's curve beside its fitted scale" begin
    sensors = [(; id=:b, x=[2., 3.], pick=[[1], [2]], y=[[.3], [.2]]),
               (; id=:a, x=[4., 7., 8.], pick=[[2, 3], Int[]], y=[[.6, .8], Float64[]])]
    input = [3., 1.]
    m = scaled_model((; sensors, input))
    rk, problem = strict(() -> consumer_problem(m))
    names = coordinate_names(rk.model.layout)
    # Members are fitted in key order: a, then b.
    members = reverse(sensors)
    gain = [coordinate(names, Symbol("channels_member_all.gain.", i)) for i in 1:2]
    scale = [coordinate(names, Symbol("channels_member_all.scale.", i)) for i in 1:2]
    function oracle(u)
        value = zero(eltype(u))
        for (i, sensor) in enumerate(members)
            g, log_s = u[gain[i]], u[scale[i]]
            s = exp(log_s)
            value += logpdf(Normal(), g) + logpdf(LogNormal(-0.7, 1.0), s) + log_s
            for subject in eachindex(input), (row, y) in zip(sensor.pick[subject], sensor.y[subject])
                value += logpdf(Normal(0.5 * input[subject] + g * sensor.x[row], s), y)
            end
        end
        value
    end
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(m, "named-curve-and-scale"; mod=@__MODULE__)
    desc = brm_descriptor(sb)
    snames = BridgeStan.param_unc_names(stan.model)
    # Stan position of each RK coordinate, by the member's keyed address.
    stan_at = zeros(Int, 4)
    for (i, sensor) in enumerate(members), (field, at) in ((:gain, gain), (:scale, scale))
        stan_at[at[i]] = only(brm_output_coordinates(desc, field, snames;
            submodel=(:channels, sensor.id)))
    end
    @test sort(stan_at) == 1:4
    for u in (zeros(4), [0.14, -0.27, 0.31, -0.22], [-0.31, 0.21, -0.4, 0.17])
        value, gradient = check_consumer_point(problem, u, oracle)
        stan_u = similar(u); stan_u[stan_at] = u
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, stan_u; propto=false)
        @test stan_value ≈ value
        @test stan_gradient[stan_at] ≈ gradient
    end
end

@stestset "A named return reads a parent-block vector beside two scalars" begin
    reads = [[1, 2, 3], [1, 3], [2, 3], [1, 2, 3]]
    entry(id, level) = (; id, stack=reads, y=[level .+ 0.1 .* r for r in reads],
        site=collect(1:4), rows=[fill(s, length(reads[s])) for s in eachindex(reads)])
    records = [entry(:ast, -0.3), entry(:alt, 0.2)]   # supplied out of key order
    read_time = [[2.0, 12.0, 48.0], [1.0, 9.0, 30.0], [3.0, 14.0, 50.0], [2.5, 11.0, 40.0]]
    m = site_model((; site=collect(1:4), read_time, records, n_margins=3))
    rk, problem = strict(() -> consumer_problem(m))
    flat(record) = (; rows=reduce(vcat, record.rows), y=reduce(vcat, record.y),
        decay=reduce(vcat, [exp.(-0.1 .* read_time[s][record.stack[s]]) for s in eachindex(reads)]))
    a, b = flat(records[2]), flat(records[1])        # a = :alt, b = :ast
    twin = site_twin((; site=collect(1:4), n_margins=3, a_site=collect(1:4), b_site=collect(1:4),
        a_rows=a.rows, a_decay=a.decay, ya=a.y, b_rows=b.rows, b_decay=b.decay, yb=b.y))
    rkt, reference = consumer_problem(twin)
    names = coordinate_names(rk.model.layout)
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
    stan = consumer_stan(m, "named-parent-block-vector"; mod=@__MODULE__)
    transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    for u in points
        v = similar(u); v[to_twin] = u
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        expected, reference_gradient = LogDensityProblems.logdensity_and_gradient(reference, v)
        @test value ≈ expected atol=2e-11 rtol=2e-11
        @test gradient ≈ reference_gradient[to_twin] atol=2e-9 rtol=2e-9
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, stan_u; propto=false)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value)
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient)
    end
end

# A member's returned curve is either fitted or its record's own data vector
# (snag `component-data-o-a8b79baf`). A vector that reads record fields alone
# is data of each member, which only the member's return reads.
@brm function site_curve(entry)
    offset ~ Normal(0.0, 1.0)
    shift ~ 0 + (1 | parentblock(:p) | entry.site)
    scale ~ LogNormal(-0.7, 1.0)
    if entry.fitted
        gain ~ Normal(0.0, 1.0)
        curve = gain .* entry.x
    else
        curve = entry.baseline
    end
    @observe s trajectory member begin
        entry.y[s] ~ Normal(member.offset .+ member.shift[entry.subject_rows[s]] .+
            member.curve[entry.pick[s]] .+ trajectory[entry.stack[s]], member.scale)
    end
    return (; offset, shift, curve, scale)
end
# The same return without a guard: every member's curve is data.
@brm function site_baseline(entry)
    offset ~ Normal(0.0, 1.0)
    shift ~ 0 + (1 | parentblock(:p) | entry.site)
    scale ~ LogNormal(-0.7, 1.0)
    curve = entry.baseline
    @observe s trajectory member begin
        entry.y[s] ~ Normal(member.offset .+ member.shift[entry.subject_rows[s]] .+
            member.curve[entry.pick[s]] .+ trajectory[entry.stack[s]], member.scale)
    end
    return (; offset, shift, curve, scale)
end
const curve_parent = quote
    level ~ 1 + (1 | p | site)
    effect(level, Intercept) ~ Normal(0.0, 1.0)
    sd(:, p) ~ Exponential(0.3333333333333333)
    cor(:, p) ~ LKJCholesky(n_margins, 2.0)
    for entry in records
        channels[entry.id] ~ COMPONENT(entry)
    end
    @plate for i in eachindex(level)
        path = exp(level[i]) .* exp.(-0.1 .* read_time[i])
        decay[i] = path
        observe(channels, i, decay[i])
    end
end
with_component(x, component) = x === :COMPONENT ? component :
    x isa Expr ? Expr(x.head, (with_component(a, component) for a in x.args)...) : x
parent_of(component) = brm(with_component(curve_parent, component); mod=@__MODULE__)
const curve_model = parent_of(:site_curve)
const baseline_model = parent_of(:site_baseline)
# The same models without components: each member's curve read is a data
# column, or a fitted gain times one.
const curve_twin = @brm begin
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
    c_offset ~ Normal(0.0, 1.0)
    c_shift ~ 0 + (1 | p | c_site)
    c_scale ~ LogNormal(-0.7, 1.0)
    c_gain ~ Normal(0.0, 1.0)
    ya ~ Normal(a_offset + a_shift[a_rows] + a_curve + exp(level[a_rows]) * a_decay, a_scale)
    yb ~ Normal(b_offset + b_shift[b_rows] + b_curve + exp(level[b_rows]) * b_decay, b_scale)
    yc ~ Normal(c_offset + c_shift[c_rows] + c_gain * c_x + exp(level[c_rows]) * c_decay, c_scale)
end
const baseline_twin = @brm begin
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
    ya ~ Normal(a_offset + a_shift[a_rows] + a_curve + exp(level[a_rows]) * a_decay, a_scale)
    yb ~ Normal(b_offset + b_shift[b_rows] + b_curve + exp(level[b_rows]) * b_decay, b_scale)
end

const curve_reads = [[1, 2, 3], [1, 3], [2, 3], [1, 2, 3]]
const curve_times = [[2.0, 12.0, 48.0], [1.0, 9.0, 30.0], [3.0, 14.0, 50.0], [2.5, 11.0, 40.0]]
# Members carry curves of different lengths; `pick` reads one entry per read.
# The field `subject_rows` shares its name with the segment column BRM emits for
# each indexed read of a member vector, so the clauses above (two such reads)
# need three distinct Stan-legal names for it.
function curve_entry(id, level; fitted=false, baseline=Float64[], x=Float64[], pick)
    (; id, fitted, baseline, x, pick, stack=curve_reads,
        y=[level .+ 0.1 .* r for r in curve_reads], site=collect(1:4),
        subject_rows=[fill(s, length(curve_reads[s])) for s in eachindex(curve_reads)])
end
function curve_flat(record, curve)
    (; rows=reduce(vcat, record.subject_rows), y=reduce(vcat, record.y),
        curve=reduce(vcat, [curve[record.pick[s]] for s in eachindex(curve_reads)]),
        decay=reduce(vcat, [exp.(-0.1 .* curve_times[s][record.stack[s]]) for s in eachindex(curve_reads)]))
end
# The twin's name of each member's own coordinate, by its plate and position.
member_names(plate, letters...) = Dict(
    Symbol("channels_member_", plate, ".", field, ".", i) => Symbol(letter, "_", field)
    for (i, letter) in enumerate(letters) for field in (:offset, :scale, :gain))
function curve_parity(model, twin, design, twin_design, rename, name)
    m = model(design)
    rk, problem = strict(() -> consumer_problem(m))
    rkt, reference = consumer_problem(twin(twin_design))
    names = coordinate_names(rk.model.layout)
    tnames = coordinate_names(rkt.model.layout)
    to_twin = [coordinate(tnames, get(rename, name, name)) for name in names]
    @test sort(to_twin) == 1:length(tnames)
    n = length(names)
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(m, name; mod=@__MODULE__)
    transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    for u in (zeros(n), collect(range(-0.37, 0.31; length=n)), [0.3sin(1.7i) for i in 1:n])
        v = similar(u); v[to_twin] = u
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        expected, reference_gradient = LogDensityProblems.logdensity_and_gradient(reference, v)
        @test value ≈ expected atol=2e-11 rtol=2e-11
        @test gradient ≈ reference_gradient[to_twin] atol=2e-9 rtol=2e-9
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, stan_u; propto=false)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value)
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient)
    end
    sb
end

@stestset "Generated names never end in a double underscore" begin
    # SBBRMI emits generated names as Stan identifiers; Stan reserves `x__`.
    taken = Set{Symbol}()
    names = [BRM._rk_ast_fresh_name("x", taken) for _ in 1:4]
    @test names == [:x, :x_, :x_2, :x_3]
    @test BRM._rk_ast_fresh_name("y_", Set([:y_])) === :y_1
    @test !any(name -> endswith(string(name), "__"), names)
end

@stestset "A named return holds a member's own data vector" begin
    # Two members return their record's curve (of lengths 3 and 2), one its
    # fitted curve; supplied out of key order.
    alt = curve_entry(:alt, 0.2; baseline=[0.3, -0.1, 0.45], pick=curve_reads)
    ast = curve_entry(:ast, -0.3; baseline=[-0.2, 0.15],
        pick=[[1, 2, 2], [1, 2], [2, 2], [1, 1, 2]])
    bil = curve_entry(:bil, 0.1; fitted=true, x=[0.5, 1.5, 2.5, 3.5],
        pick=[r .+ 1 for r in curve_reads])
    design(records) = (; site=collect(1:4), read_time=curve_times, records, n_margins=4)
    a, b, c = curve_flat(alt, alt.baseline), curve_flat(ast, ast.baseline), curve_flat(bil, bil.x)
    twin_design = (; site=collect(1:4), n_margins=4,
        a_site=collect(1:4), a_rows=a.rows, a_decay=a.decay, a_curve=a.curve, ya=a.y,
        b_site=collect(1:4), b_rows=b.rows, b_decay=b.decay, b_curve=b.curve, yb=b.y,
        c_site=collect(1:4), c_rows=c.rows, c_decay=c.decay, c_x=c.curve, yc=c.y)
    records = [ast, bil, alt]
    # Members with equal declarations share one plate (`channels_member_0`:
    # alt, then ast; `channels_member_1`: bil).
    sb = curve_parity(curve_model, curve_twin, design(records), twin_design,
        merge(member_names(0, :a, :b), member_names(1, :c)), "named-data-curve")
    # Replay on new curves binds them, as a fresh build on that data does.
    moved = [(; r..., baseline=r.baseline .+ 0.25) for r in records]
    replay = reprocess(sb, design(moved))
    fresh = SBBRMI(curve_model(design(moved)); mod=@__MODULE__, total_groups=())
    @test stan_code(replay) == stan_code(sb)
    @test stan_data(replay) == stan_data(fresh)
    @test stan_data(replay) != stan_data(sb)
end

@stestset "An unguarded named return holds each member's data vector" begin
    alt = curve_entry(:alt, 0.2; baseline=[0.3, -0.1, 0.45], pick=curve_reads)
    ast = curve_entry(:ast, -0.3; baseline=[-0.2, 0.15],
        pick=[[1, 2, 2], [1, 2], [2, 2], [1, 1, 2]])
    a, b = curve_flat(alt, alt.baseline), curve_flat(ast, ast.baseline)
    twin_design = (; site=collect(1:4), n_margins=3,
        a_site=collect(1:4), a_rows=a.rows, a_decay=a.decay, a_curve=a.curve, ya=a.y,
        b_site=collect(1:4), b_rows=b.rows, b_decay=b.decay, b_curve=b.curve, yb=b.y)
    curve_parity(baseline_model, baseline_twin,
        (; site=collect(1:4), read_time=curve_times, records=[ast, alt], n_margins=3),
        twin_design, member_names(:all, :a, :b), "unguarded-data-curve")
end

@brm function measured_grid(record)
    scale ~ LogNormal(0, 1)
    if record.scaled
        pattern = scale .* record.grid
    else
        pattern = record.grid
    end
    @plate for j in eachindex(record.y)
        loc[j] = pattern[record.rows[j]] .+ scale
        record.y[j] .~ Normal.(loc[j], 1)
    end
    return (; scale, pattern)
end
const grid_model = @brm begin
    for record in records
        channels[record.id] ~ measured_grid(record)
    end
end

@stestset "An observed member exposes both data and fitted return fields" begin
    plain = (; id=:a, scaled=false, grid=[2., -1.], rows=[[1, 2]], y=[[.7, 1.8]])
    scaled = (; id=:b, scaled=true, grid=[.5, 1.2, .7], rows=[[2, 1], [3]], y=[[.1, -.2], [.3]])
    design = (; records=[scaled, plain])
    changed = (; records=[(; plain..., grid=[-.4, .6]), (; scaled..., grid=[.8, 2., -.3])])
    fitted = RKBRMI(grid_model(design))
    point = [.2, -.3]
    for (rk, data) in ((fitted, design), (reprocess(fitted, changed), changed))
        names = coordinate_names(rk.model.layout)
        member(record) = Symbol("channels_member_", Int(record.scaled))
        scale_at(record, u) = exp(u[coordinate(names, Symbol(member(record), ".scale.1"))])
        pattern_at(record, u) = record.scaled ? scale_at(record, u) .* record.grid : record.grid
        for scope in rk_bound_plan(rk).submodel_scopes
            @test haskey(scope.locals, :component_result_pattern)
        end
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        function oracle(u)
            value = sum(logpdf.(Normal(), u))
            for record in data.records
                means = pattern_at(record, u) .+ scale_at(record, u)
                for (rows, y) in zip(record.rows, record.y)
                    value += sum(logpdf.(Normal.(means[rows], 1), y))
                end
            end
            value
        end
        check_consumer_point(problem, point, oracle)
        constrained = ReactiveKernelsPPL.constrain(rk.model.layout, point)
        for record in data.records
            draws = rk_draw_values(rk, reshape(point, :, 1), (:scale, :pattern);
                submodel=(:channels, record.id))
            @test only(constrained[member(record)].scale) ≈ scale_at(record, point)
            @test only(draws.scale) ≈ scale_at(record, point)
            @test only(draws.pattern) ≈ pattern_at(record, point)
        end
    end
end
end
