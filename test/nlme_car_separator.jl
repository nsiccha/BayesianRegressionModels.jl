# test/nlme_car_separator.jl — a `car(...)` field over areas nested in the NLME
# units, lowered by its unit border separator (`RKBRMI(brmi; car_units)`, todo
# 2026-10-28T13-18-22-697-0v882rf). Only the separator areas and each unit's
# interior sum are shared (θ); each unit's interior areas are that unit's
# standard-normal draws, and the field is the default lowering's.
#
# Covered: the separation's algebra (the shared and unit draws whiten the
# field's precision, interiors of different units share no edge, a unit's
# draws move only its own areas); each unit's sparse interior map, its inverse
# and its transpose, and their size, linear in the areas; the separated density
# against the default lowering's at the same field (a constant difference);
# the view's roles and owners; lockstep and unit-problem densities against the
# full RK density, for ICAR and BYM2; the coordinate transport to SBBRMI's field
# (points both ways, densities, gradients) and a full Stan start from NLME
# values; a unit label that is no subject, refused by name; frozen and rebuilt
# replays. Snag `car-units-border-aa5e8712`.
#
# Run: julia --project=test test/nlme_car_separator.jl [testset-filter...]

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, LinearAlgebra, Random
include(joinpath(@__DIR__, "nlme_unit_checks.jl"))
using StanBlocks, BridgeStan
using SparseArrays: spzeros

const SEP_CACHE = mkpath(joinpath(tempdir(), "brm-nlme-car-separator"))

# Six subjects, three areas each (subject a owns r1–r3, b r4–r6, …), on a
# chain r1 – … – r18 with two more edges across subjects and one inside a, as
# Gemeinden lie in Länder and border the Gemeinden of neighbouring Länder.
# Subject f also owns a component of its own (r19 – r20), subject c an island
# (r21), and r22 lies on an edge only: it has no rows, so no unit.
const SEP_DATA = let rng = Xoshiro(20261029)
    owner(j) = j <= 18 ? "abcdef"[cld(j, 3)] : j <= 20 ? 'f' : 'c'
    area = ["r$j" for j in vcat(1:21, 1:2:21)]
    subject = [string(owner(parse(Int, a[2:end]))) for a in area]
    n = length(area)
    x = round.(randn(rng, n); digits=2)
    y = round.(0.3 .+ 0.8 .* x .+ 0.4 .* randn(rng, n); digits=2)
    from = vcat(["r$j" for j in 1:17], ["r2", "r8", "r1", "r19", "r18"])
    to = vcat(["r$j" for j in 2:18], ["r5", "r14", "r3", "r20", "r22"])
    (; area, subject, x, y, from, to)
end

sep_icar(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject) + car(area; edges=(from, to), type=:icar)
    y ~ Normal(mu, sigma)
end
sep_bym2(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject) + car(area; edges=(from, to), type=:bym2)
    y ~ Normal(mu, sigma)
end

car_term(rk) = only(t for p in BRM._rk_replay_predictors(rk.plan) for t in p.terms
    if t.kind === :car)
coordinate_names(rk) = Vector{Symbol}(BRM._rk_layout_coordinate_names(rk))
message(f) = try f(); "" catch e; sprint(showerror, e) end

# A separated field's shared draws, and its unit draws in draw order (unit by
# unit, as `separation.draw_units` labels them).
shared_names(rk) = (o = car_term(rk).options;
    [Symbol(o.value, ".w.", k) for k in 1:o.separation.m])
draw_names(rk) = (o = car_term(rk).options;
    [Symbol(o.value, ".z_", b.unit, ".", j) for b in o.separation.blocks for j in 1:b.ndraws])

# The field `phi` (graph order) of a separated backend at its packed point `u`,
# from the emitted formula's data.
function separated_phi(rk, u)
    options = car_term(rk).options
    s, columns = options.separation, rk.plan.columns
    names = coordinate_names(rk)
    at(name) = u[findfirst(==(name), names)]
    w = at.(shared_names(rk))
    z = at.(draw_names(rk))
    x = BRM._brm_car_shared_values(w, only(columns[s.F]))
    parts = Any[x[columns[s.x_phi]]]
    for b in s.blocks
        xs = b.cols === nothing ? Float64[] : x[columns[b.cols]]
        push!(parts, BRM._brm_car_unit_field(z[b.draw_range], xs, only(columns[b.factor])))
    end
    vcat(parts...)[columns[s.restore]]
end

@stestset "the separation whitens the field's precision" begin
    rk = RKBRMI(sep_icar(SEP_DATA); car_units=:subject)
    options = car_term(rk).options
    columns = rk.plan.columns
    graph = (; n_struct=options.n_struct, node1=columns[options.node1],
        node2=columns[options.node2], comp_start=first.(options.components),
        comp_size=last.(options.components))
    n = graph.n_struct
    P = zeros(n, n)
    for (a, b) in zip(graph.node1, graph.node2)
        P[a, a] += 1; P[b, b] += 1; P[a, b] -= 1; P[b, a] -= 1
    end
    for (start, size) in options.components
        P[start:(start + size - 1), start:(start + size - 1)] .+= 1 / (0.001 * size)^2
    end
    # The field as a linear map of the shared and unit draws.
    s = options.separation
    names = coordinate_names(rk)
    field = vcat(shared_names(rk), draw_names(rk))
    @test s.m + s.ndraws == n
    M = reduce(hcat, (separated_phi(rk, setindex!(zeros(length(names)), 1.0,
        findfirst(==(f), names))) for f in field))
    @test M' * P * M ≈ I atol=1e-8
    # Interiors of different units share no edge, and a unit's draws move only
    # its own areas.
    for (a, b) in zip(graph.node1, graph.node2)
        s.shared[a] || s.shared[b] || @test s.node_units[findfirst(==(a),
            columns[options.struct_pos])] == s.node_units[findfirst(==(b), columns[options.struct_pos])]
    end
    for (k, label) in enumerate(s.draw_units)
        moved = findall(!iszero, M[:, s.m + k])
        nodes = [findfirst(==(p), columns[options.struct_pos]) for p in moved]
        @test all(node -> s.node_units[node] == label, nodes)
    end
    # The separator is one side of each border (a | b: r2, r3, a tie; b | c:
    # r6; c | d: r9; c | e: r8; d | e: r12; e | f: r15) and r22, of no unit;
    # the sums add one coordinate per subject on the chain.
    separator = sort!([options.nodes[k] for k in eachindex(options.nodes)
        if columns[options.struct_pos][k] <= n && s.shared[columns[options.struct_pos][k]]])
    @test separator == sort(["r2", "r3", "r6", "r8", "r9", "r12", "r15", "r22"])
    @test s.m == length(separator) + 6
end

@stestset "each unit's sparse interior map inverts and transposes exactly" begin
    for build in (sep_icar, sep_bym2)
        rk = RKBRMI(build(SEP_DATA); car_units=:subject)
        s = car_term(rk).options.separation
        rng = Xoshiro(11)
        for b in s.blocks
            f = only(rk.plan.columns[b.factor])
            @test (f.nareas, f.ndraws) == (b.areas, b.ndraws)
            z, xs = randn(rng, f.ndraws), randn(rng, f.ncols)
            phi = BRM._brm_car_unit_field(z, xs, f)
            @test BRM._brm_car_unit_draws(phi, xs, f) ≈ z atol=1e-12
            g = randn(rng, f.nareas)
            dz, dxs = BRM._brm_car_unit_pullback(g, f)
            z0, x0 = randn(rng, f.ndraws), randn(rng, f.ncols)
            @test dot(g, BRM._brm_car_unit_field(z0, x0, f)) ≈ dot(dz, z0) + dot(dxs, x0) rtol=1e-12
        end
    end
end

@stestset "a unit's interior map is linear in its areas, never n_u × n_u" begin
    # A 60 × 60 grid in four quadrant units, with one unit's corner cut off
    # as a component of its own: 3,600 areas, units of about 900.
    side = 60
    id(i, j) = (i - 1) * side + j
    edges = [(id(i, j), id(i, j + 1)) for i in 1:side for j in 1:(side - 1)]
    append!(edges, [(id(i, j), id(i + 1, j)) for i in 1:(side - 1) for j in 1:side])
    corner = [id(i, j) for i in 1:3 for j in 1:3]
    filter!(e -> (e[1] in corner) == (e[2] in corner), edges)
    n = side^2
    main = setdiff(1:n, corner)
    order = vcat(main, corner)
    position = invperm(order)
    graph = (; n_struct=n, node1=[position[a] for (a, _) in edges],
        node2=[position[b] for (_, b) in edges], comp_start=[1, length(main) + 1],
        comp_size=[length(main), length(corner)])
    quadrant(p) = 1 + (cld(p, side) > side ÷ 2) + 2 * (mod1(p, side) > side ÷ 2)
    unit = [quadrant(p) for p in order]
    s = BRM._brm_car_separation(graph, unit, 4)
    entries = sum(length(piece.factor.nzval) for u in s.units for piece in u.factor.pieces)
    @test entries <= 20 * n
    @test s.m <= 4 * side
    # The shared values' map is sparse too: one block per unit over the
    # separator areas it borders, no dense `m × m` factor.
    @test s.F isa BRM._BRMCarSharedFactor
    @test sum(length(b.factor.nzval) for b in s.F.blocks) <= 10 * n
    @test any(piece.kind === :component for u in s.units for piece in u.factor.pieces)
    # The whitening holds at this size too, on random directions.
    P = spzeros(n, n)
    for (a, b) in zip(graph.node1, graph.node2)
        P[a, a] += 1; P[b, b] += 1; P[a, b] -= 1; P[b, a] -= 1
    end
    for (start, size) in zip(graph.comp_start, graph.comp_size)
        r = start:(start + size - 1)
        P[r, r] .+= 1 / (0.001 * size)^2
    end
    ndraws = sum(u.factor.ndraws for u in s.units)
    @test s.m + ndraws == n
    function field(v)
        x = BRM._brm_car_shared_values(v[1:s.m], s.F)
        phi = zeros(n)
        for (k, area) in enumerate(s.x_areas)
            area > 0 && (phi[area] = x[k])
        end
        offset = s.m
        for u in s.units
            phi[u.areas] = BRM._brm_car_unit_field(v[offset .+ (1:u.factor.ndraws)],
                x[u.cols], u.factor)
            offset += u.factor.ndraws
        end
        phi
    end
    rng = Xoshiro(12)
    a, b = randn(rng, n), randn(rng, n)
    @test dot(field(a), P * field(b)) ≈ dot(a, b) rtol=1e-8
    @test dot(field(a), P * field(a)) ≈ dot(a, a) rtol=1e-8
end

@stestset "the separated density is the default one at the same field" begin
    for build in (sep_icar, sep_bym2)
        default = RKBRMI(build(SEP_DATA))
        rk = RKBRMI(build(SEP_DATA); car_units=:subject)
        names, default_names = coordinate_names(rk), coordinate_names(default)
        phi = Symbol(car_term(default).options.value, ".phi.")
        full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
        default_full = rk_logdensity_problem(default; ad_backend=NLME_AD)
        rng = Xoshiro(3)
        differences = map(1:3) do _
            u = 0.5 .* randn(rng, length(names))
            v = zeros(length(default_names))
            field = separated_phi(rk, u)
            for (j, name) in enumerate(default_names)
                text = String(name)
                v[j] = startswith(text, String(phi)) ? field[parse(Int, text[length(String(phi))+1:end])] :
                    u[findfirst(==(name), names)]
            end
            LogDensityProblems.logdensity(full, u) - LogDensityProblems.logdensity(default_full, v)
        end
        @test differences[2] ≈ differences[1] atol=1e-8
        @test differences[3] ≈ differences[1] atol=1e-8
    end
end

@stestset "only the separator and the interior sums are shared" begin
    for build in (sep_icar, sep_bym2)
        rk = RKBRMI(build(SEP_DATA); car_units=:subject)
        v = brm_nlme_view(rk)
        @test !any(==(:unclaimed), v.role)
        options = car_term(rk).options
        s = options.separation
        role(name) = v.role[findfirst(==(name), v.coordinates)]
        @test all(name -> role(name) === :shared_latent, shared_names(rk))
        # The scale (and mixing share), and the iid part of r22, of no unit.
        shared = build === sep_bym2 ? 2 + count(isnothing, s.node_units) : 1
        @test count(==(:shared_latent), v.role) == s.m + shared
        # Each interior draw is a unit draw of its subject, and so is each
        # area's BYM2 iid part; r22, of no unit, keeps a shared iid part.
        for (name, label) in zip(draw_names(rk), s.draw_units)
            c = findfirst(==(name), v.coordinates)
            @test v.role[c] === :subject_effect && v.levels[v.subject[c]] == label
        end
        if build === sep_bym2
            r22 = findfirst(==("r22"), options.nodes)
            @test s.node_units[r22] === nothing
            @test role(Symbol(options.value, ".theta.", r22)) === :shared_latent
        end
        @test BRM._brm_nlme_ragged(v)
        m = brm_nlme_model(rk; ad_backend=NLME_AD)
        check_lockstep(m, Xoshiro(5))
        check_unit_problems(m, Xoshiro(6))
        # Each unit's declared η Hessian pattern covers its dense η-HVP block:
        # an observation reading an interior area reads its unit's draws
        # through that unit's interior map.
        rng = Xoshiro(7)
        θ = 0.1 .* randn(rng, length(m.theta))
        σ = 0.1 .* randn(rng, length(m.sigma))
        H = [0.3 .* randn(rng, length(b)) for b in m.view.subject_coordinates]
        blocks = BRM._brm_nlme_hessian_blocks(m, θ, σ, H)
        for (P, block) in zip(BRM._brm_nlme_eta_sparsity(m), blocks)
            @test all(I -> P[I] || I[1] == I[2], findall(x -> abs(x) > 1e-10, block))
        end
    end
end

@stestset "a separated field moves through the coordinate transport to Stan's field" begin
    for build in (sep_icar, sep_bym2)
        brmi = build(SEP_DATA)
        rk = RKBRMI(brmi; car_units=:subject)
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        stan = StanBlocks.stan_instantiate(sb.model;
            path=joinpath(SEP_CACHE, string(nameof(build), ".stan")))
        t = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
        @test length(t.fields) == 1
        @test count(p -> p.relation === :car_separated, t.pairs) ==
            car_term(rk).options.n_struct
        problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
        rng = Xoshiro(13)
        for u in (zeros(length(t)), 0.4 .* randn(rng, length(t)))
            stan_u = brm_rk_point_to_stan(t, u)
            @test brm_stan_point_to_rk(t, stan_u) ≈ u atol=1e-10
            checked = brm_check_coordinate_transport(t, rk, stan.model, u)
            @test checked.fields == 1 && checked.max_error <= 1e-9
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                stan_gradient; propto=false, jacobian=true)
            @test value ≈ brm_stan_logdensity_to_rk(t, u, stan_value) atol=1e-8 rtol=1e-12
            @test gradient ≈ brm_stan_gradient_to_rk(t, u, stan_gradient) atol=1e-8 rtol=1e-10
        end
        # An NLME point of the separated model becomes a checked full Stan
        # start of the same physical point.
        map = brm_nlme_joint_map(rk, sb;
            path=joinpath(SEP_CACHE, string(nameof(build), "_joint.stan")))
        u = 0.3 .* randn(rng, length(t))
        values = brm_nlme_joint_values(map, brm_rk_point_to_stan(map.transport, u))
        role = map.view.role
        estimated = sort!(vcat(findall(==(:population), role), findall(==(:shared_latent), role)))
        start = brm_nlme_joint_start(map; theta=u[estimated], sigma=values.sigma,
            omega=values.omega, eta=values.eta, subjects=values.subjects)
        @test start.stan ≈ brm_rk_point_to_stan(map.transport, u) atol=1e-9
        @test start.checked.fields == 1
        # ... and back: that Stan point's NLME values are the RK point's (the
        # map `brm_nlme_start(m, map, stan_u)` reads its point through).
        back = brm_nlme_joint_values(map, start.stan)
        @test back.theta ≈ u[findall(==(:population), role)] atol=1e-9
        @test back.shared.values ≈ u[findall(==(:shared_latent), role)] atol=1e-9
        conditional = brm_nlme_conditional_problem(map, start; backend=rk, ad_backend=NLME_AD)
        @test isfinite(LogDensityProblems.logdensity(conditional, conditional.position))
    end
end

@stestset "car_units that are not the subjects are refused by name" begin
    rk = RKBRMI(sep_icar(SEP_DATA); car_units=[s == "a" ? "x" : s for s in SEP_DATA.subject])
    # refused: a unit's draws would belong to no subject, so the field has no
    # reading over these subjects (todo 2026-10-28T13-18-22-697-0v882rf).
    @test occursin("is not a level of the subject grouping", message(() -> brm_nlme_view(rk)))
end

@stestset "a replay keeps the fitted separation" begin
    rk = RKBRMI(sep_icar(SEP_DATA); car_units=:subject)
    fresh = merge(SEP_DATA, (; x=reverse(SEP_DATA.x)))
    replay = reprocess(rk, fresh)
    @test coordinate_names(replay) == coordinate_names(rk)
    @test car_term(replay).options.separation.fit.separation ===
        car_term(rk).options.separation.fit.separation
    rebuilt = reprocess(rk, fresh; freeze_constants=false)
    @test coordinate_names(rebuilt) == coordinate_names(rk)
    labelled = RKBRMI(sep_icar(SEP_DATA); car_units=SEP_DATA.subject)
    # refused: the labels name the fit's rows, not the new ones (todo
    # 2026-10-28T13-18-22-697-0v882rf).
    @test occursin("do not label the new data's rows",
        message(() -> reprocess(labelled, fresh; freeze_constants=false)))
end

println("nlme_car_separator.jl: all testsets passed")
