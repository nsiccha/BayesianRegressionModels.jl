include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LinearAlgebra

# Exact covariance from the declared axes and length scales, independently
# of BRM's covariance source and native helper implementations.
function multiaxis_gp_covariance(axes, sigma, rho, jitter)
    widths = rho isa Real ? fill(rho, length(axes)) : rho
    [sigma^2 * exp(-sum(((axis[i] - axis[j]) / widths[d])^2
        for (d, axis) in enumerate(axes)) / 2) + (i == j ? jitter : 0.0)
        for i in eachindex(first(axes)), j in eachindex(first(axes))]
end

@stestset "multi-axis exact GP covariance density and gradients" begin
    # The first two rows coincide on every axis; jitter is added once, after
    # the covariance, and keeps the non-centered draw well defined.
    data = (; x=[-.4,-.4,.7], w=[.2,.2,-.6], t=[-.1,-.1,.5], y=[.2,-.1,.3])
    before = deepcopy(data)
    cases = (
        ("isotropic", (:x,:w), true, @brm(data, begin
            location ~ 0 + gp(x,w;jitter=.03)
            length_scale(:,gp(x,w)) ~ LogNormal(.1,.7)
            sd(:,gp(x,w)) ~ Exponential(1.3)
            y ~ Normal(location,1.)
        end)),
        ("anisotropic", (:x,:w,:t), false, @brm(data, begin
            location ~ 0 + gp(x,w,t;iso=false,jitter=.03)
            length_scale(:,gp(x,w,t)) ~ LogNormal(.1,.7)
            sd(:,gp(x,w,t)) ~ Exponential(1.3)
            y ~ Normal(location,1.)
        end)),
        ("one-axis-anisotropic", (:x,), false, @brm(data, begin
            location ~ 0 + gp(x;iso=false,jitter=.03)
            length_scale(:,gp(x)) ~ LogNormal(.1,.7)
            sd(:,gp(x)) ~ Exponential(1.3)
            y ~ Normal(location,1.)
        end)))
    for (label,columns,iso,model) in cases
        backend,problem = consumer_problem(model)
        check_rk_source_roundtrip(backend;ad_backend=AutoEnzyme(;mode=Enzyme.Reverse))
        axes = map(c->getproperty(data,c),columns)
        term = only(t for p in backend.plan.predictors for t in p.terms if t.kind === :gp)
        id = string(term.options.f)
        names = coordinate_names(backend.model.layout)
        rpos = iso ? [coordinate(names,Symbol(id,".rho"))] :
            [coordinate(names,Symbol(id,".rho.",j)) for j in eachindex(columns)]
        spos = coordinate(names,Symbol(id,".sigma"))
        zpos = [coordinate(names,Symbol(id,".z.",j)) for j in eachindex(data.y)]
        @test length(names) == length(rpos)+1+length(zpos)
        function parts(u)
            rho = iso ? exp(only(u[rpos])) : exp.(u[rpos])
            sigma,z = exp(u[spos]),u[zpos]
            K = multiaxis_gp_covariance(axes,sigma,rho,.03)
            f = cholesky(Symmetric(K)).L*z
            prior = sum(logpdf.(LogNormal(.1,.7),exp.(u[rpos]))) +
                logpdf(Exponential(1.3),sigma) + sum(logpdf.(Normal(),z))
            likelihood = sum(logpdf.(Normal.(f,1.),data.y))
            (;f,prior,likelihood,jacobian=sum(u[rpos])+u[spos])
        end
        oracle(u) = let p=parts(u); p.prior+p.likelihood+p.jacobian end
        emitted = BRM._rk_emit_ast(backend.plan)
        ext = Base.get_extension(BRM,:BayesianRegressionModelsReactiveKernelsExt)
        mod = ext._rk_emit_module(emitted)
        covariance = getfield(mod,:brm_gp_exp_quad_cov)
        rho = iso ? .8 : [.6+.5j/length(columns) for j in eachindex(columns)]
        expected = multiaxis_gp_covariance(axes,1.2,rho,.03)
        @test Base.invokelatest(covariance,axes...,1.2,rho,.03) ≈ expected atol=2e-15 rtol=2e-15
        # The emitted graph reads raw axes and works at a new row count.
        new_axes = ([.1,-.6,.8,.3],[.2,.4,-.1,.7],[-.3,.8,.2,-.4])[1:length(columns)]
        @test Base.invokelatest(covariance,new_axes...,1.2,rho,.03) ≈
            multiaxis_gp_covariance(new_axes,1.2,rho,.03) atol=2e-15 rtol=2e-15
        inventory = recipe_inventory(kernel_graph(backend.model.spec))
        @test any(e->e.kind===:plate,inventory)
        @test occursin("covariance",join(repr(e.recipe.outputs) for e in inventory))
        stan = consumer_stan(model,"exact-gp-"*label)
        stanid = "gp_"*join(string.(columns),"_")
        mapping = Pair{Symbol,String}[names[spos]=>stanid*"_sigma"]
        append!(mapping,[names[zpos[j]]=>stanid*"_z.$j" for j in eachindex(zpos)])
        append!(mapping,[names[rpos[j]]=>(iso ? stanid*"_rho" : stanid*"_rho.$j") for j in eachindex(rpos)])
        queries = Dict(key=>prepare_query(backend.model,rk_bound_plan(backend),key)
            for key in (:prior,:likelihood))
        for u in (zeros(length(names)),fill(.13,length(names)),collect(range(-.2,.3;length=length(names))))
            check_consumer_point(problem,u,oracle)
            check_consumer_stan(problem,stan,mapping,backend,u)
            p = parts(u)
            @test Base.invokelatest(queries[:prior],u) ≈ p.prior
            @test Base.invokelatest(queries[:likelihood],u) ≈ p.likelihood
        end
        @test isequal(data,before)
    end
end

# Two ordinary readers receive one named GP draw, without a second copy of
# its hyperparameters or standardized innovations.
multiaxis_gp_reader(f,x;scale=1.) = scale .* f .+ x
@stestset "multi-axis exact GP shared readers and known scales" begin
    data = (;x=[-.4,.1,.7],w=[.2,-.3,-.6],t=[-.1,.3,.5],y=[.2,-.1,.3],y2=[-.2,.4,.1])
    for known in ([.6,.9,1.2], .8)
        builder = @brm begin
            field ~ 0 + gp(x,w,t;iso=false,jitter=.02)
            length_scale(:,gp(x,w,t)) = known_scales
            sd(:,gp(x,w,t)) = 1.1
            first_reader = multiaxis_gp_reader(field,x)
            second_reader = multiaxis_gp_reader(field,w;scale=-.5)
            y ~ Normal(first_reader,1.)
            y2 ~ Normal(second_reader,1.)
        end
        backend,problem = consumer_problem(builder(merge(data,(;known_scales=known))))
        @test LogDensityProblems.dimension(problem) == 3
        K = multiaxis_gp_covariance((data.x,data.w,data.t),1.1,known,.02)
        L = cholesky(Symmetric(K)).L
        function oracle(u)
            f = L*u
            sum(logpdf.(Normal(),u)) + sum(logpdf.(Normal.(f.+data.x,1.),data.y)) +
                sum(logpdf.(Normal.(-.5f.+data.w,1.),data.y2))
        end
        pointwise = prepare_query(backend.model,rk_bound_plan(backend),:pointwise)
        for u in (zeros(3),[.2,-.3,.1])
            check_consumer_point(problem,u,oracle)
            result = Base.invokelatest(pointwise,u)
            f = L*u
            @test result.y ≈ logpdf.(Normal.(f.+data.x,1.),data.y)
            @test result.y2 ≈ logpdf.(Normal.(-.5f.+data.w,1.),data.y2)
        end
    end
end

@stestset "multi-axis exact GP construction" begin
    # The consumer's reported three-axis declaration, on synthetic data.
    data = (; east=[-0.3, 0.4], north=[0.1, -0.2], time=[-0.2, 0.3], y=[0.1, -0.1])
    model = @brm data begin
        mu ~ 1 + gp(east, north, time; iso=false, jitter=1e-12)
        effect(mu, Intercept) ~ Normal(0.0, 1.0)
        sd(mu, gp(east, north, time)) ~ Exponential(1.0)
        length_scale(mu, gp(east, north, time)) ~ LogNormal(0.0, 0.7)
        y ~ Normal(mu, 1.0)
    end
    backend, problem = consumer_problem(model)
    @test LogDensityProblems.dimension(problem) == 7
    @test isfinite(LogDensityProblems.logdensity(problem, zeros(7)))
end
