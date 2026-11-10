# Public synthetic acceptance for statistical implementations adopted by BRM.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Distributions, LinearAlgebra, Statistics, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels
const Prep = BRM.StatisticalPreparation

function library_build(name, body, columns)
    mod = Module(gensym(:BRMLibraryTest))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    Core.eval(mod, :(const owned = $(BRM.rkppl_model(name))))
    names = Tuple(keys(columns))
    bound = bind_data(lower_rkppl(body, names; mod, conditioned=(:y,)), columns)
    bound, build_kernel(bound)
end

function library_check(name, body, columns)
    bound, built = library_build(name, body, columns)
    u = built.layout.total <= 1 ? fill(0.13, built.layout.total) :
        collect(range(-0.17, 0.21; length=built.layout.total))
    q = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    g = similar(u)
    saved = copy(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test isfinite(value)
    @test isequal(u, saved)
    h = 1e-5
    fd = map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += h; minus[j] -= h
        (q(plus) - q(minus)) / (2h)
    end
    @test g ≈ fd atol=2e-5 rtol=2e-5
    # Reparse the adopted definition itself, then use the same public lowering.
    mod = Module(gensym(:BRMRetypedLibrary))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    for helper in (:hsgp_rho_floors, :hsgp_periodic_rho_floor, :hsgp_sqrt_spd,
            :hsgp_periodic_sqrt_spd, :hsgp_grouped_sqrt_spd)
        Core.eval(mod, :(const $helper = $(getproperty(Prep, helper))))
    end
    definition = Meta.parse(sprint(Base.show_unquoted, getproperty(BRM._BRM_STATISTICAL_MODELS, name)))
    Core.eval(mod, Expr(:macrocall, Symbol("@rkppl"), LineNumberNode(0), definition))
    Core.eval(mod, :(const owned = $(getproperty(mod, name))))
    rebound = bind_data(lower_rkppl(Meta.parse(sprint(Base.show_unquoted, body)),
        Tuple(keys(columns)); mod, conditioned=(:y,)), columns)
    rebuilt = build_kernel(rebound)
    @test coordinate_names(rebuilt.layout) == coordinate_names(built.layout)
    for preset in (:sampler, :prior, :likelihood)
        @test isequal(Base.invokelatest(prepare_query(built, bound, preset), u),
            Base.invokelatest(prepare_query(rebuilt, rebound, preset), u))
    end
    bound, built, u
end

@stestset "adopted BRM statistical library" begin
    n = 12
    x = collect(range(-1, 1; length=n))
    g = repeat(["b", "a", "c"], 4)
    s = repeat(["two", "one"], 6)
    base = (; x, g, s, y=fill(0.2, n))
    cases = (
        (:varying_coefs, :(begin r ~ owned(g); y .~ Normal.(r[g], 1) end)),
        (:varying_coefs_correlated, :(begin r ~ owned(g, 2); y .~ Normal.(r[g, 1] .+ x .* r[g, 2], 1) end)),
        (:varying_coefs_centered, :(begin r ~ owned(g); y .~ Normal.(r[g], 1) end)),
        (:varying_coefs_centered_correlated, :(begin r ~ owned(g, 2); y .~ Normal.(r[g, 1] .+ x .* r[g, 2], 1) end)),
        (:varying_stratified, :(begin r ~ owned(g, s); y .~ Normal.(r, 1) end)),
        (:varying_stratified_correlated, :(begin r ~ owned(g, s, 2); y .~ Normal.(r[:, 1] .+ x .* r[:, 2], 1) end)),
        (:monotonic, :(begin phi ~ Dirichlet([1.0, 1.0]); r ~ owned(c, phi); y .~ Normal.(r, 1) end)),
        (:differenced_ar1, :(begin beta ~ Beta(2, 2); sigma ~ Exponential(1); r ~ owned(beta, sigma); y .~ Normal.(r, 1) end)),
        (:r2d2_coefs, :(begin r ~ owned(X, [1.0, 1.0]); mu = X * r; y .~ Normal.(mu, 1) end)),
        (:horseshoe_coefs, :(begin r ~ owned(X); mu = X * r; y .~ Normal.(mu, 1) end)),
        (:penalized_smooth, :(begin r ~ owned(X, Z); y .~ Normal.(r, 1) end)),
        (:t2_smooth, :(begin r ~ owned(X, Zrr, Zrn, Znr); y .~ Normal.(r, 1) end)),
        (:hsgp_effect, :(begin r ~ owned(PHI, lambda); y .~ Normal.(r, 1) end)),
        (:hsgp_periodic_effect, :(begin r ~ owned(PHI, harmonics); y .~ Normal.(r, 1) end)),
        (:hsgp_grouped_effect, :(begin r ~ owned(PHI, lambda, g); y .~ Normal.(r, 1) end)),
        (:ordered_logistic, :(begin y ~ owned(x) end)),
    )
    for (name, body) in cases
        filter = get(ENV, "BRM_LIBRARY_FILTER", "")
        isempty(filter) || occursin(filter, string(name)) || continue
        @testset "$name" begin
            columns = Dict{Symbol,Any}(pairs(base))
            if name in (:r2d2_coefs, :horseshoe_coefs)
                columns[:X] = hcat(x, sin.(2x))
            elseif name === :penalized_smooth
                columns[:X], columns[:Z] = Prep.tps_basis(x; k=5)
            elseif name === :t2_smooth
                columns[:X], columns[:Zrr], columns[:Zrn], columns[:Znr] = Prep.t2_basis(x, sin.(2x); k=(3, 3))
            elseif name === :hsgp_periodic_effect
                columns[:PHI], columns[:harmonics] = Prep.hsgp_periodic_basis(x; k=3, period=2.0)
            elseif name in (:hsgp_effect, :hsgp_grouped_effect)
                columns[:PHI], columns[:lambda] = Prep.hsgp_basis(x; k=3,
                    by=name === :hsgp_grouped_effect ? g : nothing)
            elseif name === :ordered_logistic
                columns[:y] = repeat([1, 2, 3], 4)
            elseif name === :monotonic
                columns[:c] = repeat([1, 2, 3], 4)
            end
            bound, built, u = library_check(name, body, columns)
            if name === :varying_coefs_correlated
                nt = constrain(built.layout, u)
                B = nt.r.z * (Diagonal(nt.r.sd) * nt.r.L)'
                gi = [findfirst(==(v), sort(unique(g))) for v in g]
                mu = B[gi, 1] .+ x .* B[gi, 2]
                @test Base.invokelatest(prepare_query(built, bound, :likelihood), u) ≈ sum(logpdf.(Normal.(mu, 1), columns[:y]))
                prior = sum(logpdf.(truncated(Normal(), 0, Inf), nt.r.sd)) +
                    logpdf(LKJCholesky(2, 1), Cholesky(LowerTriangular(nt.r.L))) + sum(logpdf.(Normal(), nt.r.z))
                @test Base.invokelatest(prepare_query(built, bound, :prior), u) ≈ prior
            end
        end
    end
end

# `StatisticalPreparation.car_graph` over `test/car_term.jl`'s graph: component
# {a, b, c, d, h} (h has no rows), component {e, f}, island g (rows, no neighbours).
# Then the hand-written BYM2 recipe brm-use documents (references/rkppl-bodies.md):
# the graph bound as data, one soft sum-to-zero on a connected graph, against an
# independent oracle of the car term's documented law (terms.md, "Areal spatial
# effects").
@stestset "StatisticalPreparation.car_graph and a hand-written BYM2" begin
    area = ["a", "b", "c", "d", "e", "f", "g", "a", "e"]
    from, to = ["a", "b", "c", "a", "d", "e"], ["b", "c", "d", "c", "h", "f"]
    graph = Prep.car_graph(area, from, to)
    @test graph.nodes == ["a", "b", "c", "d", "e", "f", "g", "h"]
    @test graph.n_nodes == 8 && graph.area == [1, 2, 3, 4, 5, 6, 7, 1, 5]
    @test (graph.n_struct, graph.n_comp, graph.comp_start, graph.comp_size) == (7, 2, [1, 6], [5, 2])
    @test graph.struct_pos[7] == graph.n_struct + 1 && graph.is_struct[7] == 0.0 &&
        graph.inv_scale[7] == 0.0
    @test sort(collect(zip(min.(graph.node1, graph.node2), max.(graph.node1, graph.node2)))) ==
        [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (6, 7)]
    @test graph.scaling ≈ [BRM._brm_car_scaling_factor(5, [(1, 2), (2, 3), (3, 4), (1, 3), (4, 5)]),
        BRM._brm_car_scaling_factor(2, [(1, 2)])]
    @test graph.inv_scale[1:6] ≈ [fill(1 / graph.scaling[1], 4); fill(1 / graph.scaling[2], 2)]
    @test Prep.car_graph([1, 2], [1], [3]; nodes=[1, 2, 3, 4]).nodes == [1, 2, 3, 4]
    @test_throws ErrorException Prep.car_graph(["a"], ["a"], ["a"])

    station = ["s1", "s2", "s3", "s4", "s5", "s6", "s1", "s4"]
    sfrom, sto = ["s1", "s2", "s3", "s4", "s5", "s1"], ["s2", "s3", "s4", "s5", "s6", "s3"]
    y = collect(range(-0.3, 0.4; length=length(station)))
    G = Prep.car_graph(station, sfrom, sto)
    @test G.n_comp == 1 && all(==(1.0), G.is_struct)
    columns = (; y, station_pos=G.area, node1=G.node1, node2=G.node2, n_nodes=G.n_nodes,
        n_struct=G.n_struct, inv_scale=G.inv_scale, sum_scale=0.001 * G.n_struct)
    body = :(begin
        b0 ~ Normal(0, 1)
        sigma ~ truncated(Normal(0, 1), 0, Inf)
        rho ~ Beta(0.5, 0.5)
        theta[1:n_nodes] .~ Normal.(0, 1)
        phi[1:n_struct] ~ ICAR(node1, node2)
        phi_sum = sum(phi)
        phi_sum ~ Normal(0, sum_scale)
        effects = sigma .* (sqrt(rho) .* sqrt.(inv_scale) .* phi .+ sqrt(1 - rho) .* theta)
        y .~ Normal.(b0 .+ effects[station_pos], 1)
    end)
    mod = Module(gensym(:BRMHandBYM2))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    bound = bind_data(lower_rkppl(body, columns; mod, conditioned=(:y,)), columns)
    built = build_kernel(bound)
    u = collect(range(-0.17, 0.21; length=built.layout.total))
    nt = constrain(built.layout, u)
    effects = nt.sigma .* (sqrt(nt.rho) .* sqrt.(G.inv_scale) .* nt.phi .+ sqrt(1 - nt.rho) .* nt.theta)
    prior = logpdf(Normal(0, 1), nt.b0) + logpdf(truncated(Normal(), 0, Inf), nt.sigma) +
        logpdf(Beta(0.5, 0.5), nt.rho) + sum(logpdf.(Normal(), nt.theta)) -
        0.5 * sum(abs2, nt.phi[G.node1] .- nt.phi[G.node2]) +
        logpdf(Normal(0, 0.001 * G.n_struct), sum(nt.phi))
    @test Base.invokelatest(prepare_query(built, bound, :prior), u) ≈ prior rtol=1e-12
    @test Base.invokelatest(prepare_query(built, bound, :likelihood), u) ≈
        sum(logpdf.(Normal.(nt.b0 .+ effects[G.area], 1), y)) rtol=1e-12
    q = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    g = similar(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test isfinite(value)
    h = 1e-5
    fd = map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += h; minus[j] -= h
        (q(plus) - q(minus)) / (2h)
    end
    @test g ≈ fd atol=2e-5 rtol=2e-5
end
