# A spatial field, `gmrf((east, north); …)`: no time axis, no temporal length,
# one channel. Checked against an independent dense FEM Matérn ν = 1 oracle,
# against the space-time field on one constant time with its temporal length
# held (the same law, since one time forms no OU transition), and across
# RKBRMI and SBBRMI: coordinate transport, replay, the sparse Laplace, the
# read-support forms, `@plate` cells, prediction, starts and descriptions.
# RUN: julia --project=test test/gmrf_spatial.jl
using Test, BayesianRegressionModels, LinearAlgebra, Distributions, Random
using StanBlocks, ReactiveKernels, ReactiveKernelsPPL, Enzyme
import BridgeStan as BS
import DifferentiationInterface as DI
import LogDensityProblems as LDP
const BRM = BayesianRegressionModels
const AD = DI.AutoEnzyme(; mode=Enzyme.Reverse)

const spatial_data = (; east=[0.1, 0.9, 0.7, 0.3, 0.55], north=[0.2, 0.4, 0.8, 0.5, 0.35],
    y=[0.4, -0.3, 0.2, 0.1, -0.2], unit=[10, 20, 10, 10, 20], t0=zeros(5),
    xy=[0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5], triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5])
const FIELD = "gmrf_field_east_north"

function spatial_model(input=spatial_data; support=:mesh, component=false)
    if support === :neighbors
        return @brm input begin
            field ~ 0 + gmrf((east, north); neighbors=2)
            sd(field, gmrf((east, north))) ~ Exponential(0.7)
            length_scale(field, gmrf((east, north))) ~ LogNormal(0.5, 0.8)
            y ~ Normal(field, 0.2)
        end
    elseif support === :multiresolution
        return @brm input begin
            field ~ 0 + gmrf((east, north); multiresolution=2)
            sd(field, gmrf((east, north))) ~ Exponential(0.7)
            length_scale(field, gmrf((east, north))) ~ LogNormal(0.5, 0.8)
            y ~ Normal(field, 0.2)
        end
    elseif component
        return @brm input begin
            field ~ 0 + gmrf((east, north); mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east, north))) ~ Exponential(0.7)
            length_scale(field, gmrf((east, north)), space) ~ LogNormal(0.5, 0.8)
            y ~ Normal(field, 0.2)
        end
    end
    @brm input begin
        field ~ 0 + gmrf((east, north); mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north))) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north))) ~ LogNormal(0.5, 0.8)
        y ~ Normal(field, 0.2)
    end
end

# The space-time field on one time (`t0`) with its temporal length held.
function one_time_model(input=spatial_data; support=:mesh)
    if support === :neighbors
        return @brm input begin
            field ~ 0 + gmrf((east, north), t0; neighbors=2)
            sd(field, gmrf((east, north), t0)) ~ Exponential(0.7)
            length_scale(field, gmrf((east, north), t0), space) ~ LogNormal(0.5, 0.8)
            length_scale(field, gmrf((east, north), t0), time) = 1.0
            y ~ Normal(field, 0.2)
        end
    elseif support === :multiresolution
        return @brm input begin
            field ~ 0 + gmrf((east, north), t0; multiresolution=2)
            sd(field, gmrf((east, north), t0)) ~ Exponential(0.7)
            length_scale(field, gmrf((east, north), t0), space) ~ LogNormal(0.5, 0.8)
            length_scale(field, gmrf((east, north), t0), time) = 1.0
            y ~ Normal(field, 0.2)
        end
    end
    @brm input begin
        field ~ 0 + gmrf((east, north), t0; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), t0)) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north), t0), space) ~ LogNormal(0.5, 0.8)
        length_scale(field, gmrf((east, north), t0), time) = 1.0
        y ~ Normal(field, 0.2)
    end
end

function rk_build(model; case_id="gmrf-spatial")
    artifact = BRM.emit_rk_artifact(model; case_id)
    bound = BRM.rk_translate_artifact(artifact)
    built = build_kernel(bound)
    RKBRMI(model, artifact.plan, built), bound, built
end

coordinate(names, name) = findfirst(==(Symbol(name)), names)
field_coordinates(names, declaration=FIELD) =
    [i for (i, n) in enumerate(names) if startswith(string(n), "$declaration.field.")]
error_text(f) = try f(); "" catch e; sprint(showerror, e) end
# The program after its functions block: the shared law functions name their
# temporal-length argument, but a spatial field's data, parameters and model
# hold none.
model_blocks(code) = last(split(code, "\ndata {"; limit=2))

# Independent dense oracle: the mass-lumped FEM Matérn ν = 1 precision
# Q = K C⁻¹ K / (4π κ²), K = κ² C + G, κ = √2/ℓ, on the fixture mesh.
function matern_covariance(ell)
    mesh = BRM._brm_gmrf_mesh(spatial_data.xy, spatial_data.triangles)
    kappa2 = 2 / ell^2
    K = Matrix(mesh.stiffness) + kappa2 * Diagonal(mesh.mass)
    inv(Symmetric(K * Diagonal(1 ./ mesh.mass) * K / (4pi * kappa2)))
end
function read_matrix(rows=eachindex(spatial_data.east))
    support = BRM.brm_gmrf_support(spatial_data.xy, spatial_data.triangles, [0.0], [1])
    reads = BRM.brm_gmrf_rows(support, spatial_data.east[rows], spatial_data.north[rows],
        zeros(length(rows)))
    A = zeros(length(rows), 5)
    for r in axes(A, 1), j in 1:3
        A[r, getproperty(reads, Symbol(:index, j))[r]] += getproperty(reads, Symbol(:weight, j))[r]
    end
    A
end
hyper_logprior(a, b) = logpdf(Exponential(0.7), exp(a)) + a + logpdf(LogNormal(0.5, 0.8), exp(b)) + b

@testset "Spatial gmrf: addresses, prepared state and refusals" begin
    model = spatial_model()
    terms = BRM._brm_term_address_map(model, :field)
    @test only(keys(terms)) == Symbol("gmrf((east,north))")
    term = only(only(values(terms)))
    @test map(s -> s.name, BRM._brm_term_prior_slots(term)) == (:sigma, :spatial_length)
    prepared = BRM._brm_prepare_term(term, :field, BRM._brm_backend_context(model))
    @test prepared.source == (:east, :north)
    @test prepared.state.times == [0.0]
    @test prepared.state.layout.ncoordinates == 5
    priors = BRM._brm_term_prior_expressions(prepared)
    @test BRM.getf(priors.sigma) === Exponential
    @test BRM.getargs(priors.spatial_length) == (0.5, 0.8)
    @test priors.temporal_length isa BRM._BRMKnownTermValue
    # `, space)` names the same, only length.
    spaced = BRM._brm_term_address_map(spatial_model(; component=true), :field)
    spaced_prepared = BRM._brm_prepare_term(only(only(values(spaced))), :field,
        BRM._brm_backend_context(spatial_model(; component=true)))
    @test BRM.getargs(BRM._brm_term_prior_expressions(spaced_prepared).spatial_length) == (0.5, 0.8)
    timed = @brm spatial_data begin
        field ~ 0 + gmrf((east, north); mesh=(vertices=xy, cells=triangles))
        length_scale(field, gmrf((east, north)), time) ~ LogNormal(0.5, 0.8)
        y ~ Normal(field, 0.2)
    end
    message = error_text(() -> BRM._brm_resolve_term_priors(timed))
    @test occursin("valid components are `space`", message)
    with_cor = merge(spatial_data, (; Omega=ones(1, 1)))
    cor_term() = only(only(values(BRM._brm_term_address_map(@brm(with_cor, begin
        field ~ 0 + gmrf((east, north); mesh=(vertices=xy, cells=triangles), cor=Omega)
        y ~ Normal(field, 0.2)
    end), :field))))
    temporal_term() = only(only(values(BRM._brm_term_address_map(@brm(spatial_data, begin
        field ~ 0 + gmrf((east, north); mesh=(vertices=xy, cells=triangles), temporal=:ou)
        y ~ Normal(field, 0.2)
    end), :field))))
    @test occursin("`cor=` correlates", error_text(() -> BRM._brm_gmrf_arguments(cor_term())))
    @test occursin("takes no `temporal=`", error_text(() -> BRM._brm_gmrf_arguments(temporal_term())))
end

@testset "Spatial gmrf on RKBRMI: dense oracle and the held one-time field" begin
    rk, bound, built = rk_build(spatial_model())
    names = BRM._rk_layout_coordinate_names(rk)
    @test built.layout.total == 7
    @test !any(n -> occursin("temporal", string(n)), names)
    A = read_matrix()
    z = field_coordinates(names)
    @test length(z) == 5
    function oracle(u)
        a, b = u[coordinate(names, "$FIELD.sigma")], u[coordinate(names, "$FIELD.spatial_length")]
        sigma, ell = exp(a), exp(b)
        hyper_logprior(a, b) + logpdf(MvNormal(zeros(5), Symmetric(matern_covariance(ell))), u[z]) +
            sum(logpdf.(Normal.(sigma .* (A * u[z]), 0.2), spatial_data.y))
    end
    u = 0.1 .* sin.(1:built.layout.total)
    query = prepare_query(built, bound, :sampler)
    for point in (zeros(length(u)), u)
        @test Base.invokelatest(query, point) ≈ oracle(point) atol=2e-8 rtol=2e-10
    end
    sampler = prepare_sampler(built, bound, u; backend=AD)
    value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ oracle(u) atol=2e-8 rtol=2e-10
    for j in eachindex(u)
        plus, minus = copy(u), copy(u)
        plus[j] += 2e-6
        minus[j] -= 2e-6
        @test gradient[j] ≈ (oracle(plus) - oracle(minus)) / 4e-6 atol=3e-5 rtol=3e-6
    end
    # The space-time field on one time, its temporal length held: the same law
    # on the same coordinates, which only carry the time column's name.
    twin, twin_bound, twin_built = rk_build(one_time_model(); case_id="gmrf-spatial-twin")
    twin_names = BRM._rk_layout_coordinate_names(twin)
    @test [replace(string(n), "_t0" => "") for n in twin_names] == string.(names)
    twin_query = prepare_query(twin_built, twin_bound, :sampler)
    for point in (zeros(length(u)), u)
        @test Base.invokelatest(twin_query, point) ≈ Base.invokelatest(query, point) atol=1e-12 rtol=1e-12
    end
    @testset "prior-median start" begin
        start = brm_start(rk, (;); base=:prior_median)
        @test start[coordinate(names, "$FIELD.sigma")] ≈ log(median(Exponential(0.7)))
        @test start[coordinate(names, "$FIELD.spatial_length")] ≈ 0.5
    end
    @testset "sparse Laplace integrates the field exactly" begin
        target = brm_laplace_problem(rk; ad_backend=AD, integrate=(Symbol(FIELD),))
        @test Set(target.names) == Set([Symbol("$FIELD.sigma"), Symbol("$FIELD.spatial_length")])
        marginal(φ) = begin
            a = φ[coordinate(target.names, "$FIELD.sigma")]
            b = φ[coordinate(target.names, "$FIELD.spatial_length")]
            Σ = exp(2a) * A * matern_covariance(exp(b)) * A' + 0.2^2 * I
            hyper_logprior(a, b) + logpdf(MvNormal(zeros(5), Symmetric(Σ)), spatial_data.y)
        end
        φ = [0.2, -0.3]
        for point in (zeros(2), φ)
            @test LDP.logdensity(target, point) ≈ marginal(point) atol=1e-7 rtol=1e-9
        end
        value, gradient = LDP.logdensity_and_gradient(target, φ)
        @test value ≈ marginal(φ) atol=1e-7 rtol=1e-9
        for j in 1:2
            plus, minus = copy(φ), copy(φ)
            plus[j] += 1e-5
            minus[j] -= 1e-5
            @test gradient[j] ≈ (marginal(plus) - marginal(minus)) / 2e-5 atol=1e-5 rtol=1e-5
        end
    end
end

@testset "Spatial gmrf on SBBRMI: Stan, transport, replay and descriptions" begin
    model = spatial_model()
    sb = SBBRMI(model; mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    check = StanBlocks.stanc_check(code; warn_pedantic=false)
    check.ok || println(check)
    @test check.ok
    @test occursin("brm_gmrf_lpdf", code)
    @test !occursin("temporal_length", model_blocks(code))
    @test sb.data[Symbol(FIELD, "_n_field")] == 5
    @test sb.data[Symbol(FIELD, "_times")] == [0.0]
    rk, bound, built = rk_build(model; case_id="gmrf-spatial-sb")
    stan = BRM.stan_instantiate(sb)
    transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
    @test length(transport.pairs) == 7
    u = 0.1 .* sin.(1:built.layout.total)
    sampler = prepare_sampler(built, bound, u; backend=AD)
    for point in (zeros(length(u)), u)
        @test brm_check_coordinate_transport(transport, rk, stan.model, point).max_error < 1e-10
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), point)
        sv, sg = BS.log_density_gradient(stan.model, brm_rk_to_stan(transport, point);
            propto=false, jacobian=true)
        @test value ≈ sv atol=2e-8 rtol=2e-10
        @test gradient ≈ brm_stan_to_rk(transport, sg) atol=3e-7 rtol=3e-7
    end
    @testset "replay: reprocess and exported fitted state" begin
        rows = [4, 2, 5]
        fresh = merge(spatial_data, (; (k => getproperty(spatial_data, k)[rows]
            for k in (:east, :north, :y, :unit, :t0))...))
        state = brm_fitted_state(rk)
        replay = reprocess(rk, fresh)
        from_state = RKBRMI(BRM._brm_rebind_brmi(model, fresh), state)
        @test BRM._rk_layout_coordinate_names(replay) == BRM._rk_layout_coordinate_names(rk)
        @test BRM._rk_layout_coordinate_names(from_state) == BRM._rk_layout_coordinate_names(rk)
        replay_stan = BRM.stan_instantiate(reprocess(sb, fresh))
        A = read_matrix(rows)
        names = BRM._rk_layout_coordinate_names(rk)
        z = field_coordinates(names)
        for problem in (rk_logdensity_problem(replay; ad_backend=AD),
                rk_logdensity_problem(from_state; ad_backend=AD)), point in (zeros(length(u)), u)
            sv = BS.log_density(replay_stan.model, brm_rk_to_stan(transport, point);
                propto=false, jacobian=true)
            @test LDP.logdensity(problem, point) ≈ sv atol=2e-8 rtol=2e-10
            a, b = point[coordinate(names, "$FIELD.sigma")], point[coordinate(names, "$FIELD.spatial_length")]
            @test sv ≈ hyper_logprior(a, b) +
                logpdf(MvNormal(zeros(5), Symmetric(matern_covariance(exp(b)))), point[z]) +
                sum(logpdf.(Normal.(exp(a) .* (A * point[z]), 0.2), fresh.y)) atol=2e-8 rtol=2e-10
        end
    end
    @testset "descriptor roles and descriptions" begin
        descriptor = brm_descriptor(sb; highlights=())
        entry = only(BRM._brm_term_coordinate_entries(model, :field))
        @test entry.term == Symbol(FIELD)
        for role in (:sd, :spatial_length)
            @test BRM._brm_term_parameter_output(descriptor, :field, entry, role).declaration !== nothing
        end
        @test prod(BRM._brm_term_parameter_output(descriptor, :field, entry, :field).dims) == 5
        @test !isempty(error_text(() -> BRM._brm_term_parameter_output(descriptor, :field, entry, :temporal_length)))
        for (language, axis) in ((:en, "no time axis"), (:de, "keine Zeitachse"))
            description = brm_description(descriptor; texts=description_texts(language))
            @test description.complete
            @test isempty(description.diagnostics)
            @test any(p -> occursin(axis, p) && occursin("Matérn", p), description.prose)
            @test !any(p -> occursin("OU", p), description.prose)
        end
    end
end

@testset "Spatial gmrf read-support forms: neighbors= and multiresolution=" begin
    for support in (:neighbors, :multiresolution)
        @testset "$support" begin
            model = spatial_model(; support)
            rk, bound, built = rk_build(model; case_id="gmrf-spatial-$support")
            names = BRM._rk_layout_coordinate_names(rk)
            @test !any(n -> occursin("temporal", string(n)), names)
            twin, twin_bound, twin_built = rk_build(one_time_model(; support);
                case_id="gmrf-spatial-$support-twin")
            @test [replace(string(n), "_t0" => "") for n in BRM._rk_layout_coordinate_names(twin)] ==
                string.(names)
            u = 0.1 .* sin.(1:built.layout.total)
            query = prepare_query(built, bound, :sampler)
            twin_query = prepare_query(twin_built, twin_bound, :sampler)
            for point in (zeros(length(u)), u)
                @test Base.invokelatest(twin_query, point) ≈ Base.invokelatest(query, point) atol=1e-12 rtol=1e-12
            end
            sb = SBBRMI(model; mod=@__MODULE__)
            code = StanBlocks.stan_code(sb.model)
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
            @test !occursin("temporal_length", model_blocks(code))
            stan = BRM.stan_instantiate(sb)
            transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
            sampler = prepare_sampler(built, bound, u; backend=AD)
            for point in (zeros(length(u)), u)
                value, gradient = sampler_value_and_gradient!(sampler, similar(u), point)
                sv, sg = BS.log_density_gradient(stan.model, brm_rk_to_stan(transport, point);
                    propto=false, jacobian=true)
                @test value ≈ sv atol=2e-8 rtol=2e-10
                @test gradient ≈ brm_stan_to_rk(transport, sg) atol=3e-7 rtol=3e-7
            end
            if support === :neighbors
                draws = hcat(zeros(length(u)), u)
                predicted = brm_gmrf_predict(rk, draws, :field; east=[0.5, 0.2], north=[0.5, 0.7])
                @test size(predicted) == (2, 2)
                @test all(isfinite, predicted)
                @test occursin("omit `time`", error_text(() ->
                    brm_gmrf_predict(rk, draws, :field; east=[0.5], north=[0.5], time=[0.0])))
            end
        end
    end
end

@testset "Spatial gmrf fields in @plate cells" begin
    input = spatial_data
    model = @brm input begin
        @plate for p in groups(unit)
            field ~ 0 + gmrf((east[p], north[p]); mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east[p], north[p]))) ~ Exponential(0.7)
            length_scale(field, gmrf((east[p], north[p]))) ~ LogNormal(0.5, 0.8)
            y[p] ~ Normal(field, 0.2)
        end
    end
    rk, bound, built = rk_build(model; case_id="gmrf-spatial-cells")
    names = BRM._rk_layout_coordinate_names(rk)
    @test built.layout.total == 2 * 7
    @test !any(n -> occursin("temporal", string(n)), names)
    cells = Dict(10 => [1, 3, 4], 20 => [2, 5])
    function oracle(u)
        lp = 0.0
        for (cell, label) in enumerate((10, 20))
            z = [i for (i, n) in enumerate(names) if startswith(string(n), "$FIELD.field.$cell.")]
            @assert length(z) == 5
            a = u[coordinate(names, "$FIELD.sigma.$cell")]
            b = u[coordinate(names, "$FIELD.spatial_length.$cell")]
            rows = cells[label]
            lp += hyper_logprior(a, b) +
                logpdf(MvNormal(zeros(5), Symmetric(matern_covariance(exp(b)))), u[z]) +
                sum(logpdf.(Normal.(exp(a) .* (read_matrix(rows) * u[z]), 0.2), spatial_data.y[rows]))
        end
        lp
    end
    u = 0.05 .* sin.(1:built.layout.total)
    query = prepare_query(built, bound, :sampler)
    for point in (zeros(length(u)), u)
        @test Base.invokelatest(query, point) ≈ oracle(point) atol=2e-8 rtol=2e-10
    end
    sb = SBBRMI(model; mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test !occursin("temporal_length", model_blocks(code))
    stan = BRM.stan_instantiate(sb)
    transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
    for point in (zeros(length(u)), u)
        sv = BS.log_density(stan.model, brm_rk_to_stan(transport, point); propto=false, jacobian=true)
        @test sv ≈ oracle(point) atol=2e-8 rtol=2e-10
    end
end

@testset "Spatial gmrf beside an intercept (the docs example)" begin
    model = @brm spatial_data begin
        mu ~ 1 + gmrf((east, north); mesh=(vertices=xy, cells=triangles))
        y ~ Normal(mu, 0.2)
        sd(mu, gmrf((east, north))) ~ Exponential(0.7)
        length_scale(mu, gmrf((east, north))) ~ LogNormal(log(0.4), 0.4)
    end
    sb = SBBRMI(model; mod=@__MODULE__)
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
    rk, bound, built = rk_build(model; case_id="gmrf-spatial-intercept")
    stan = BRM.stan_instantiate(sb)
    transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
    u = 0.1 .* cos.(1:built.layout.total)
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    for point in (zeros(length(u)), u)
        @test LDP.logdensity(problem, point) ≈ BS.log_density(stan.model,
            brm_rk_to_stan(transport, point); propto=false, jacobian=true) atol=2e-8 rtol=2e-10
    end
end
