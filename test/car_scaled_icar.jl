# test/car_scaled_icar.jl — the variance-scaled ICAR (Besag) field,
# `car(area; edges=(from, to), type=:icar, scaled=true)`: R-INLA's `besag` with
# `scale.model = TRUE` and Latte's default `BesagModel`. An area with neighbours
# adds `sigma * phi[j] / sqrt(s_c)`, with `s_c` the scaling factor of its
# connected component; an island adds `sigma * theta[i]`, one standard-normal
# `theta` per island (Freni-Sterrantino et al. 2018). It is the BYM2 field with
# its mixing share held at 1, so `sigma` is the field's marginal SD.
#
# The oracles below are written from that law alone: the scaling factors come
# from the dense pseudo-inverse of each hand-listed component's Laplacian.
# Snag `brm-car-icar-has-d25d72ec`, design decision `0u7hmag`.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using LinearAlgebra: pinv, diag
using Distributions: Normal, logpdf
import StanBlocks.stan: transpiles
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const SC_CACHE = joinpath(tempdir(), "brm-car-scaled-icar")
const SC_BS = StanBlocks.BridgeStan

# test/car_term.jl's graph: areas "a".."h". Component {a, b, c, d, h}: a–b, b–c,
# c–d, a–c, d–h; component {e, f}: e–f; "g" is an island with rows; "h" has no
# rows. Sorted areas a..h are 1..8; `struct_idx == [1, 2, 3, 4, 6, 7, 0, 5]`.
const SC_EDGES = (["a", "b", "c", "a", "d", "e"], ["b", "c", "d", "c", "h", "f"])
const SC_Y = [0.3, -0.2, 0.5, 1.1, -0.7, 0.0, 0.4, -0.1, 0.9]
sc_df(; area=["a", "b", "c", "d", "e", "f", "g", "a", "e"], y=SC_Y,
      from=SC_EDGES[1], to=SC_EDGES[2]) = (; area, y, from, to)
# The declared support adds "i", an island with no rows and no neighbours.
sc_df_nodes(; kwargs...) = merge(sc_df(; kwargs...),
    (; all_areas=["a", "b", "c", "d", "e", "f", "g", "h", "i"]))
# A graph without islands: a 4-cycle and an edge, every area with rows.
sc_df_connected() = (; area=["p", "q", "r", "s", "t", "u", "p", "t"],
    y=[0.2, -0.4, 0.1, 0.6, -0.3, 0.5, 0.0, 0.3],
    from=["p", "q", "r", "s", "t"], to=["q", "r", "s", "p", "u"])

sc_model(df=sc_df()) = @brm df begin
    mu ~ 1 + car(area; edges=(from, to), type=:icar, scaled=true)
    effect(mu, Intercept) ~ Normal(0, 1)
    y ~ Normal(mu, 1.0)
end
sc_model_nodes(df=sc_df_nodes()) = @brm df begin
    mu ~ 1 + car(area; edges=(from, to), nodes=all_areas, type=:icar, scaled=true)
    effect(mu, Intercept) ~ Normal(0, 1)
    y ~ Normal(mu, 1.0)
end

function sc_laplacian(members, from, to)
    Q = zeros(length(members), length(members))
    for (i, j) in zip(from, to)
        i in members && j in members || continue
        a, b = findfirst(==(i), members), findfirst(==(j), members)
        Q[a, a] += 1; Q[b, b] += 1; Q[a, b] -= 1; Q[b, a] -= 1
    end
    Q
end
sc_geomean(x) = exp(sum(log, x) / length(x))
sc_scaling(members, from, to) = sc_geomean(diag(pinv(sc_laplacian(members, from, to))))

const SC_COMPONENTS = (["a", "b", "c", "d", "h"], ["e", "f"])
const SC_PHI_ORDER = ["a", "b", "c", "d", "h", "e", "f"]

diagm_sqrt(v) = [i == j ? sqrt(v[i]) : 0.0 for i in eachindex(v), j in eachindex(v)]

sc_instantiate(sb) = begin
    code = StanBlocks.stan_code(sb.model)
    mkpath(SC_CACHE)
    StanBlocks.stan_instantiate(sb.model; path=joinpath(SC_CACHE, string(hash(code)) * ".stan"))
end
sc_index(names, name) = only(findall(==(name), names))
sc_indices(names, prefix) = findall(startswith(prefix * "."), names)

@testset "scaled ICAR: the graph's scaling factors and islands are data" begin
    sb = SBBRMI(sc_model(); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    for name in ("brm_icar_lpdf", "car_icar_scaled_effects", "car_mu_area_sigma",
                 "car_mu_area_phi", "car_mu_area_theta")
        @test occursin(name, code)
    end
    @test !occursin("car_mu_area_rho", code)
    @test sb.data[:car_mu_area_struct_idx] == [1, 2, 3, 4, 6, 7, 0, 5]
    @test sb.data[:car_mu_area_n_island] == 1
    @test sb.data[:car_mu_area_island_idx] == [0, 0, 0, 0, 0, 0, 1, 0]
    s1 = sc_scaling(SC_COMPONENTS[1], SC_EDGES...)
    s2 = sc_scaling(SC_COMPONENTS[2], SC_EDGES...)
    @test s2 ≈ 0.25
    @test sb.data[:car_mu_area_inv_scale] ≈ [1/s1, 1/s1, 1/s1, 1/s1, 1/s2, 1/s2, 0.0, 1/s1]
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # The declared support's rowless island "i" is the second island.
    nodes = SBBRMI(sc_model_nodes(); mod=@__MODULE__)
    @test nodes.data[:car_mu_area_n_island] == 2
    @test nodes.data[:car_mu_area_island_idx] == [0, 0, 0, 0, 0, 0, 1, 0, 2]
    # Without islands `theta` is empty; the program is still valid.
    connected = SBBRMI(sc_model(sc_df_connected()); mod=@__MODULE__)
    @test connected.data[:car_mu_area_n_island] == 0
    @test StanBlocks.stanc_check(StanBlocks.stan_code(connected.model); warn_pedantic=false).ok
end

@testset "scaled ICAR: sigma is the marginal SD of every component" begin
    # The field's prior covariance over a component's areas is
    # `sigma^2 * D pinv(Q_c) D` with `D = diag(sqrt.(inv_scale))`, the ICAR's
    # covariance under its sum-to-zero; the geometric mean of its diagonal is
    # `sigma^2` in every component, and an island's variance is `sigma^2`.
    sb = SBBRMI(sc_model(); mod=@__MODULE__)
    inv_scale = sb.data[:car_mu_area_inv_scale]
    nodes = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for members in SC_COMPONENTS
        k = [findfirst(==(m), nodes) for m in members]
        D = diagm_sqrt(inv_scale[k])
        cov = D * pinv(sc_laplacian(members, SC_EDGES...)) * D
        @test sc_geomean(diag(cov)) ≈ 1.0 rtol=1e-12
    end
end

@testset "scaled ICAR: refusals and defaults name the rule" begin
    df = sc_df()
    # scaled = false on BYM2 changes what its mixing share means
    @test_throws "type=:bym2, scaled=false" SBBRMI(@brm(df, begin
        mu ~ 1 + car(area; edges=(from, to), type=:bym2, scaled=false)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
    @test_throws "must be `true` or `false`" SBBRMI(@brm(df, begin
        mu ~ 1 + car(area; edges=(from, to), type=:icar, scaled=1)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
    @test_throws "has none" SBBRMI(@brm(df, begin
        mu ~ 1 + car(area; edges=(from, to), type=:icar, scaled=true)
        mixing(:, car(area)) ~ Uniform(0, 1)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
    # BYM2 is scaled by definition: saying so changes nothing
    plain = StanBlocks.stan_code(SBBRMI(@brm(df, begin
        mu ~ 1 + car(area; edges=(from, to), type=:bym2)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__).model)
    said = StanBlocks.stan_code(SBBRMI(@brm(df, begin
        mu ~ 1 + car(area; edges=(from, to), type=:bym2, scaled=true)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__).model)
    @test said == plain
    # and plain `:icar` keeps its unscaled law
    icar = SBBRMI(@brm(df, begin
        mu ~ 1 + car(area; edges=(from, to), type=:icar, scaled=false)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
    @test !occursin("car_mu_area_theta", StanBlocks.stan_code(icar.model))
    @test !haskey(icar.data, :car_mu_area_inv_scale)
end

# Julia oracle of the whole model density (propto=false, Jacobian on).
function sc_oracle(df; intercept, sigma, theta, phi)
    nodes = sort(unique(vcat(df.area, df.from, df.to, get(df, :all_areas, String[]))))
    islands = [n for n in nodes if !(n in SC_PHI_ORDER)]
    scale = Dict(l => sc_scaling(c, df.from, df.to) for c in SC_COMPONENTS for l in c)
    effect = Dict{String,Float64}()
    for (p, label) in enumerate(SC_PHI_ORDER)
        effect[label] = sigma * phi[p] / sqrt(scale[label])
    end
    for (i, label) in enumerate(islands)
        effect[label] = sigma * theta[i]
    end
    pos = Dict(l => p for (p, l) in enumerate(SC_PHI_ORDER))
    icar = -0.5 * sum((phi[pos[a]] - phi[pos[b]])^2 for (a, b) in zip(df.from, df.to))
    s2z = logpdf(Normal(0, 0.001 * 5), sum(phi[1:5])) + logpdf(Normal(0, 0.001 * 2), sum(phi[6:7]))
    prior = logpdf(Normal(0, 1), intercept) + logpdf(Normal(0, 1), sigma) + log(sigma) +
            sum(logpdf.(Normal(0, 1), theta)) + icar + s2z
    lik = sum(logpdf(Normal(intercept + effect[a], 1.0), y) for (a, y) in zip(df.area, df.y))
    prior + lik
end

@testset "scaled ICAR: BridgeStan density equals the Julia oracle; coordinates resolve" begin
    for (model, df, n_island) in ((sc_model, sc_df(), 1), (sc_model_nodes, sc_df_nodes(), 2))
        sb = SBBRMI(model(df); mod=@__MODULE__)
        problem = sc_instantiate(sb)
        sm = problem.model
        names = String.(SC_BS.param_names(sm))
        @test LogDensityProblems.dimension(problem) == 1 + 1 + n_island + 7
        q = zeros(LogDensityProblems.dimension(problem))
        theta = [-0.8, 0.6][1:n_island]
        phi = [0.4, -0.3, 0.2, 0.1, -0.4, 0.25, -0.25]
        q[sc_index(names, "car_mu_area_sigma")] = log(0.7)
        q[sc_indices(names, "car_mu_area_theta")] = theta
        q[sc_indices(names, "car_mu_area_phi")] = phi
        intercept_i = only(i for (i, n) in enumerate(names) if !startswith(n, "car_mu_area"))
        q[intercept_i] = 0.15
        lp = SC_BS.log_density(sm, q; propto=false, jacobian=true)
        @test lp ≈ sc_oracle(df; intercept=0.15, sigma=0.7, theta, phi) atol=1e-8
        cnames = SC_BS.param_names(sm; include_tp=true, include_gq=false)
        constrained = SC_BS.param_constrain(sm, q; include_tp=true, include_gq=false)
        d = brm_descriptor(sb)
        coords(p) = brm_term_coordinates(d, :mu, cnames; term=:car_mu_area, parameter=p).coordinates
        @test length(coords(:sd)) == 1
        @test length(coords(:structured)) == 7
        @test length(coords(:unstructured)) == n_island
        @test_throws Exception coords(:mixing)
        effects = constrained[coords(:effects)]
        @test effects[7] ≈ 0.7 * theta[1]                        # island g
        n_island == 2 && @test effects[9] ≈ 0.7 * theta[2]       # declared island i
        s1 = sc_scaling(SC_COMPONENTS[1], df.from, df.to)
        @test effects[1] ≈ 0.7 * phi[1] / sqrt(s1)                # a
        @test effects[5] ≈ 0.7 * phi[6] / sqrt(0.25)              # e
        @test constrained[coords(:value)] ≈ effects[sb.data[:car_mu_area_row_idx]]
        # The description names the scaled law, in English and in German.
        description = brm_description(d)
        @test description.complete
        @test occursin("so σ is a marginal SD", brm_description_markdown(description))
        german = brm_description(d; texts=description_texts(:de))
        @test occursin("sodass σ eine marginale SD ist", brm_description_markdown(german))
    end
end

@testset "scaled ICAR on RKBRMI: the same law as SBBRMI at transported coordinates" begin
    for (model, df) in ((sc_model, sc_df()), (sc_model_nodes, sc_df_nodes()),
                        (sc_model, sc_df_connected()))
        brmi = model(df)
        rk, problem = consumer_problem(brmi)
        sb = SBBRMI(brmi; mod=@__MODULE__)
        stan = sc_instantiate(sb)
        stan_names = SC_BS.param_unc_names(stan.model)
        transport = brm_coordinate_transport(rk, sb, stan_names)
        @test length(transport) == length(stan_names)
        for u in (zeros(length(transport)),
                  collect(range(-0.43, 0.51; length=length(transport))))
            stan_u = brm_rk_point_to_stan(transport, u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = SC_BS.log_density_gradient!(stan.model, stan_u, stan_gradient;
                propto=false, jacobian=true)
            @test value ≈ stan_value + transport.logdensity_offset atol=1e-9 rtol=1e-12
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=1e-8 rtol=1e-10
        end
    end
end

@testset "scaled ICAR on RKBRMI and SBBRMI: replay reads the fitted graph" begin
    rk = RKBRMI(sc_model_nodes())
    mu_at(backend, u) = begin
        bound = rk_bound_data(backend)
        query = ReactiveKernels.prepare(backend.model.spec;
            have=(:unconstrained, keys(bound)...), want=(:mu,), bound)
        value = Base.invokelatest(query, u)
        value isa Tuple ? only(value) : value
    end
    u = collect(range(-0.4, 0.5; length=length(coordinate_names(rk.model.layout))))
    train = mu_at(rk, u)
    # rows of the rowless declared island "i", the island "g" and area "a": the
    # two islands keep their own effects
    replay = reprocess(rk, (; area=["i", "g", "a"], y=zeros(3)))
    replayed = mu_at(replay, u)
    @test replayed[[2, 3]] ≈ train[[7, 1]] atol=1e-12 rtol=1e-12
    @test !(replayed[1] ≈ replayed[2])
    @test_throws "not an area of the fitted graph" reprocess(rk, (; area=["z"], y=[0.0]))
    sb = SBBRMI(sc_model_nodes(); mod=@__MODULE__)
    frozen = reprocess(sb, (; area=["i", "g", "a"], y=zeros(3)))
    @test frozen.data[:car_mu_area_row_idx] == [9, 7, 1]
    @test frozen.data[:car_mu_area_island_idx] == sb.data[:car_mu_area_island_idx]
end
