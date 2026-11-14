include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicKernelObservationFamilies
using BayesianRegressionModels, StanBlocks, Distributions
import ReactiveKernels
import BayesianRegressionModels: _rk_observation_source!

# A public vector law with two aligned row arguments and one live shared
# scale. Its native scalar graph uses the exact same normalized law.
StanBlocks.@deffun begin
    @lhs @lpxf relative_normal_lpdf(y::vector[n], location::vector[n],
            reference::vector[n], scale::real)::real = begin
        result::real = 0.0
        for i in 1:n
            result += normal_lpdf(y[i], location[i] - reference[i], scale)
        end
        return result
    end
    relative_normal_lpdfs(y::vector[n], location::vector[n],
            reference::vector[n], scale::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = normal_lpdf(y[i], location[i] - reference[i], scale)
        end
        return result
    end
    relative_normal_rng(vector[n], location::vector[n],
            reference::vector[n], scale::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = normal_rng(location[i] - reference[i], scale)
        end
        return result
    end
end

StanBlocks.@deffun begin
    @lhs @lpxf relative_bound_normal_lpdf(y::vector[n], location::vector[n],
            reference::vector[n], scale::real)::real =
        relative_normal_lpdf(y, location, reference, scale)
    relative_bound_normal_lpdfs(y::vector[n], location::vector[n],
            reference::vector[n], scale::real)::vector[n] =
        relative_normal_lpdfs(y, location, reference, scale)
    relative_bound_normal_rng(vector[n], location::vector[n],
            reference::vector[n], scale::real)::vector[n] =
        relative_normal_rng(vector[n], location, reference, scale)
end

ReactiveKernels.@kernel relative_bound_graph(value, location, reference, scale) = begin
    relative_residual = (value - location + reference) / scale
    relative_log_scale = log(scale)
    relative_logdensity = -0.5 * log(2 * pi) - relative_log_scale -
        0.5 * relative_residual * relative_residual
    return relative_logdensity
end

function _rk_observation_source!(definitions, bindings, entry,
        ::typeof(relative_bound_normal))
    child = Symbol(entry, :_child)
    push!(bindings, child => relative_bound_graph)
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, location, reference, scale) = begin
        density = $child(value, location, reference, scale)
        return density
    end))
    :done
end

function _rk_observation_source!(definitions, bindings, entry,
        ::typeof(relative_normal))
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, location, reference, scale) = begin
        relative_residual = (value - location + reference) / scale
        relative_log_scale = log(scale)
        relative_logdensity = -0.5 * log(2 * pi) - relative_log_scale -
            0.5 * relative_residual * relative_residual
        return relative_logdensity
    end))
    :done
end

function build(data)
    @brm data begin
        a ~ Normal(0, 0.7)
        sigma ~ Exponential(0.9)
        nu ~ Exponential(0.2)
        locations ~ kernel(x, y) do xs, ys
            loc = xs * a
            ys ~ student_t(nu, loc, sigma)
            loc
        end
    end
end

function build_relative(data)
    @brm data begin
        a ~ Normal(0, 0.7)
        sigma ~ Exponential(0.9)
        locations ~ kernel(x, reference, y) do xs, refs, ys
            loc = xs * a
            retained_reference = refs * (1 + a)
            ys ~ relative_normal(loc, retained_reference, sigma)
            loc
        end
    end
end

function build_relative_direct(data)
    @brm data begin
        a ~ Normal(0, 0.7)
        sigma ~ Exponential(0.9)
        loc = x * a
        retained_reference = reference * (1 + a)
        y ~ relative_normal(loc, retained_reference, sigma)
    end
end
function build_relative_bound(data)
    @brm data begin
        a ~ Normal(0, 0.7)
        sigma ~ Exponential(0.9)
        locations ~ kernel(x, reference, y) do xs, refs, ys
            loc = xs * a
            retained_reference = refs * (1 + a)
            ys ~ relative_bound_normal(loc, retained_reference, sigma)
            loc
        end
    end
end
function _rk_observation_source!(definitions, bindings, entry,
        ::typeof(StanBlocks.bernoulli_logit))
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, logit) = begin
        binary_logdensity = if value == 0
            -log1p(exp(logit))
        else
            -log1p(exp(-logit))
        end
        return binary_logdensity
    end))
    :done
end
function build_binary(data)
    @brm data begin
        a ~ Normal(0, 0.7)
        locations ~ kernel(x, y) do xs, ys
            loc = xs * a
            ys ~ bernoulli_logit(loc)
            loc
        end
    end
end
end

@stestset "kernel cell-law retains exact Student-t constructor and observation axes" begin
    data = (;x=[[0.2,0.5],Float64[],[0.7]],
        y=[[0.1,0.4],Float64[],[-0.2]])
    saved = deepcopy(data)
    brmi = PublicKernelObservationFamilies.build(data)
    backend, problem = consumer_problem(brmi)
    @test coordinate_names(backend.model.layout) == [:a,:sigma,:nu]
    # One array per subject, observed per subject (RKPPL nested plates).
    @test isequal(backend.plan.columns[:y], data.y)
    oracle(u) = begin
        a,sigma,nu = u[1],exp(u[2]),exp(u[3])
        logpdf(Normal(0,0.7),a) +
            logpdf(Exponential(0.9),sigma) + u[2] +
            logpdf(Exponential(0.2),nu) + u[3] +
            sum(logpdf(LocationScale(x*a,sigma,TDist(nu)),y)
                for (xs,ys) in zip(data.x,data.y) for (x,y) in zip(xs,ys))
    end
    stan = consumer_stan(brmi,"kernel-student-t-law";mod=PublicKernelObservationFamilies)
    for u in ([0.0,0.0,0.0],[0.17,-0.21,0.3],[-0.1,0.2,-0.2])
        check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,[:a=>"a",:sigma=>"sigma",:nu=>"nu"],backend,u)
    end
    emitted = BRM._rk_emit_ast(backend.plan)
    @test occursin("y[i] .~ StudentT.(",sprint(Base.show_unquoted,emitted.main))
    @test isequal(data,saved)
end

# Julia's Student-t in a plate cell, `LocationScale(location, scale, TDist(nu))`:
# the spelling a top-level response takes, so a cell observes it too (brm-use
# kernel.md, "Cell bodies are Julia"; snag student-t-locati-6466e87a).
module PublicCellStudentT
using BayesianRegressionModels
shared(data) = @brm data begin
    nu ~ LogNormal(2, 0.5); sigma ~ Exponential(1); mu ~ Normal(0, 5)
    @plate for s in eachindex(y)
        y[s] ~ LocationScale(mu, sigma, TDist(nu))
    end
end
# Location, scale and degrees of freedom one per entry.
per_entry(data) = @brm data begin
    nu ~ LogNormal(2, 0.5); sigma ~ Exponential(1); mu ~ Normal(0, 5)
    @plate for s in eachindex(y)
        loc = mu .+ x[s]
        scale = sigma .* (1 .+ abs.(x[s]))
        y[s] ~ LocationScale(loc, scale, TDist(nu .* w[s]))
    end
end
# One response value per cell.
scalar_cells(data) = @brm data begin
    nu ~ LogNormal(2, 0.5); sigma ~ Exponential(1); mu ~ Normal(0, 5)
    @plate for s in eachindex(y)
        y[s] ~ LocationScale(mu + x[s], sigma, TDist(nu))
    end
end
other_base(data) = @brm data begin
    sigma ~ Exponential(1); mu ~ Normal(0, 5)
    @plate for s in eachindex(y)
        y[s] ~ LocationScale(mu, sigma, Normal(0, 1))
    end
end
end

const CELL_T = (; x=[[0.2, -0.5, 1.1], Float64[], [0.7, 0.0]],
    w=[[1.0, 0.5, 2.0], Float64[], [1.5, 0.8]],
    y=[[0.1, 0.4, -1.3], Float64[], [-0.2, 2.5]])

function cell_t_oracle(names, data, law)
    index = Dict(n => coordinate(names, n) for n in (:nu, :sigma, :mu))
    u -> begin
        nu, sigma, mu = exp(u[index[:nu]]), exp(u[index[:sigma]]), u[index[:mu]]
        logpdf(LogNormal(2, 0.5), nu) + u[index[:nu]] +
            logpdf(Exponential(1), sigma) + u[index[:sigma]] + logpdf(Normal(0, 5), mu) +
            sum((logpdf(law(nu, sigma, mu, entry...), last(entry))
                for entry in data); init=0.0)
    end
end

@stestset "plate cell observes LocationScale(location, scale, TDist(nu))" begin
    cases = (
        (PublicCellStudentT.shared, CELL_T,
            [(y,) for ys in CELL_T.y for y in ys],
            (nu, sigma, mu, y) -> LocationScale(mu, sigma, TDist(nu))),
        (PublicCellStudentT.per_entry, CELL_T,
            [(x, w, y) for (xs, ws, ys) in zip(CELL_T.x, CELL_T.w, CELL_T.y)
                for (x, w, y) in zip(xs, ws, ys)],
            (nu, sigma, mu, x, w, y) ->
                LocationScale(mu + x, sigma * (1 + abs(x)), TDist(nu * w))),
        (PublicCellStudentT.scalar_cells, (; x=[0.2, -0.5, 1.1], y=[0.1, 0.4, -1.3]),
            [(x, y) for (x, y) in zip([0.2, -0.5, 1.1], [0.1, 0.4, -1.3])],
            (nu, sigma, mu, x, y) -> LocationScale(mu + x, sigma, TDist(nu))))
    for (k, (build, data, entries, law)) in enumerate(cases)
        saved = deepcopy(data)
        brmi = build(data)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test sort(names) == [:mu, :nu, :sigma]
        oracle = cell_t_oracle(names, entries, law)
        # SBBRMI spells the same cell `student_t(nu, location, scale)`.
        stan = consumer_stan(brmi, "cell-student-t-$k"; mod=PublicCellStudentT)
        for u in ([0.0, 0.0, 0.0], [0.17, -0.21, 0.3], [-0.4, 0.2, -0.2])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, [:nu=>"nu", :sigma=>"sigma", :mu=>"mu"],
                backend, u)
        end
        # The law is RKPPL's StudentT; the cell never stores a `TDist` value.
        main = sprint(Base.show_unquoted, BRM._rk_emit_ast(backend.plan).main)
        @test occursin("StudentT.(", main)
        @test !occursin("TDist", main)
        @test isequal(data, saved)
    end
end

@stestset "plate cell LocationScale with a base other than TDist(nu) names the spelling" begin
    # As a top-level response (`_rk_ast_value_distribution`), RK observes
    # `LocationScale` as Student-t only; the refusal says what to write.
    message = try
        RKBRMI(PublicCellStudentT.other_base(CELL_T)); nothing
    catch err
        sprint(showerror, err)
    end
    @test message !== nothing
    @test occursin("must be `TDist(nu)`", message)
    @test occursin("LocationScale(location, scale, TDist(nu))", message)
end

@stestset "integer observed cell law retains original discrete response rows" begin
    data = (;x=[[0.2,0.5],Float64[],[0.7]],y=[[0,1],Int[],[1]])
    saved = deepcopy(data)
    brmi = PublicKernelObservationFamilies.build_binary(data)
    backend,problem = consumer_problem(brmi)
    # One array per subject, observed per subject through this module's
    # caller observation graph (RKPPL nested plates).
    @test eltype(eltype(backend.plan.columns[:y])) === Int
    @test isequal(backend.plan.columns[:y], data.y)
    @test coordinate_names(backend.model.layout) == [:a]
    oracle(u) = logpdf(Normal(0,0.7),u[1]) +
        sum(logpdf(Bernoulli(inv(1+exp(-x*u[1]))),y)
            for (xs,ys) in zip(data.x,data.y) for (x,y) in zip(xs,ys))
    stan = consumer_stan(brmi,"integer-kernel-law";mod=PublicKernelObservationFamilies)
    for u in ([0.0],[0.17],[-0.1])
        check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,[:a=>"a"],backend,u)
    end
    @test isequal(data,saved)
end

# kernel_expr is a pre-build replay and can retain KernelSpec reader calls;
# the built numerical graph is the place to verify composed scalar recipes.
function observation_graph_recipes(graph;depth=0)
    [(; depth=depth + entry.depth, outputs=sprint(show,entry.recipe.outputs),
        source=entry.recipe.source) for entry in recipe_inventory(graph)]
end

@stestset "caller observation graph retains vector arguments and visible scalar law" begin
  for route in (:kernel, :direct, :bound_child)
    data = (;x=[[0.2,0.5],Float64[],[0.7]],
        reference=[[0.1,0.4],Float64[],[-0.2]],
        y=[[0.1,0.4],Float64[],[-0.2]])
    if route === :direct
        data = map(values -> reduce(vcat,values),data)
    end
    saved = deepcopy(data)
    brmi = route === :kernel ? PublicKernelObservationFamilies.build_relative(data) :
        route === :direct ? PublicKernelObservationFamilies.build_relative_direct(data) :
        PublicKernelObservationFamilies.build_relative_bound(data)
    backend, problem = consumer_problem(brmi)
    @test coordinate_names(backend.model.layout) == [:a,:sigma]
    # A cell's per-subject response is observed per subject; flat data stays flat.
    @test isequal(backend.plan.columns[:y], data.y)
    oracle(u) = begin
        a,sigma = u[1],exp(u[2])
        rows = route !== :direct ? zip(data.x,data.reference,data.y) :
            [(data.x,data.reference,data.y)]
        logpdf(Normal(0,0.7),a) + logpdf(Exponential(0.9),sigma) + u[2] +
            sum(logpdf(Normal(x*a-r*(1+a),sigma),y)
                for (xs,rs,ys) in rows
                for (x,r,y) in zip(xs,rs,ys))
    end
    stan = consumer_stan(brmi,"relative-normal-graph-$route";
        mod=PublicKernelObservationFamilies)
    for u in ([0.0,0.0],[0.17,-0.21],[-0.1,0.2])
        check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,[:a=>"a",:sigma=>"sigma"],backend,u)
    end
    bound = BRM.rk_translate_artifact(BRM.emit_rk_artifact(brmi;
        case_id="relative-normal-graph-$route"))
    main = sprint(Base.show_unquoted, BRM._rk_emit_ast(backend.plan).main)
    records = observation_graph_recipes(kernel_graph(build_kernel(bound).spec))
    # The scalar law runs inside the observation plate, which a per-subject
    # response nests inside its subject plate.
    scalar = filter(record->record.depth==(route === :direct ? 1 : 2),records)
    println("OBSERVATION_BUILT_SCALAR_RECIPES=",scalar)
    @test any(record->occursin("relative_residual",record.outputs) &&
        isequal(record.source,:(((value-location)+reference)/scale)),scalar)
    @test any(record->occursin("relative_log_scale",record.outputs) &&
        isequal(record.source,:(log(scale))),scalar)
    @test any(record->occursin("relative_logdensity",record.outputs) &&
        isequal(record.source,:((-0.5*log(2*pi)-relative_log_scale)-
            0.5*relative_residual*relative_residual)),scalar)
    @test occursin((route === :direct ? "y" : "y[i]") * " .~ LogDensity.(y_scalar_logdensity,", main)
    for retired in ("y_logdensity_reader", "brm_logdensity_value", "y_law_argument")
        @test !occursin(retired, main)
    end
    @test isequal(data,saved)
  end
end
