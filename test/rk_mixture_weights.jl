# Native mixture probabilities are ordinary model values. Public synthetic
# cases retain the authored laws and check every unconstrained coordinate.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const hierarchical_mixture = @brm begin
    logit(w) ~ 1 + (1 | analyte)
    mu ~ 1 + (1 | sample)
    y ~ MixtureModel([Normal(mu, sigma), Normal(mu, tau)], [1-w, w])
    effect(w, Intercept) ~ Normal(0, 1)
    effect(mu, Intercept) ~ Normal(0, 1)
    sd(w, analyte) ~ Exponential(1)
    sd(mu, sample) ~ Exponential(1)
    sigma ~ Exponential(1)
    tau ~ Exponential(1)
end

function mixture_population(nt, backend, predictor)
    record = only(r for r in BRM._rk_coordinate_records(backend.plan)
        if r.kind === :population && r.predictor === predictor &&
            r.coefficient === :Intercept)
    value = nt
    for name in split(String(record.declaration), '.')
        value = getproperty(value, Symbol(name))
    end
    index = get(record, :index, ())
    isempty(index) ? value : value[index...]
end

function mixture_group(nt, backend, group)
    bucket = only(b for b in backend.plan.regression.ranef_buckets if b.group === group)
    suffix = bucket.id === nothing ? string(bucket.group) : string(bucket.id, "_", bucket.group)
    getproperty(nt, Symbol("b_", suffix))
end

mixture_query(backend, preset, u) = Base.invokelatest(
    prepare_query(backend.model, rk_bound_plan(backend), preset), u)

function hierarchy_parts(backend, data, u)
    nt = constrain(backend.model.layout, u)
    aw, am = mixture_population(nt, backend, :w), mixture_population(nt, backend, :mu)
    rw, rm = mixture_group(nt, backend, :analyte), mixture_group(nt, backend, :sample)
    # Fitted levels are sorted; these data use 1:3 and 1:2, respectively.
    w = LogExpFunctions.logistic.(aw .+ only(rw.tau) .* vec(rw.z)[data.analyte])
    mu = am .+ only(rm.tau) .* vec(rm.z)[data.sample]
    ll = [logpdf(MixtureModel([Normal(mu[j], nt.sigma), Normal(mu[j], nt.tau)],
        [1-w[j], w[j]]), data.y[j]) for j in eachindex(data.y)]
    prior = logpdf(Normal(), aw) + logpdf(Normal(), am) +
        sum(logpdf.(Normal(), rw.z)) + sum(logpdf.(Normal(), rm.z)) +
        logpdf(Exponential(), only(rw.tau)) + logpdf(Exponential(), only(rm.tau)) +
        logpdf(Exponential(), nt.sigma) + logpdf(Exponential(), nt.tau)
    jac = log(only(rw.tau)) + log(only(rm.tau)) + log(nt.sigma) + log(nt.tau)
    (; ll, prior, jac)
end

@stestset "hierarchical mixture probabilities: density, Reverse, pointwise and replay" begin
    data = (; y=[-0.6, 0.2, 1.1, -0.3, 0.7, 1.8],
        analyte=[1,1,2,2,3,3], sample=[1,2,1,2,1,2])
    brmi = hierarchical_mixture(data)
    backend, problem = consumer_problem(brmi)
    @test backend.plan isa BRM._RKValuePlan
    @test Set(p.name for p in backend.plan.regression.predictors) == Set((:mu, :w))
    @test length(backend.plan.regression.ranef_buckets) == 2
    @test LogDensityProblems.dimension(problem) == 11
    @test isdefined(Enzyme.Compiler, :RestoreSafepointEffectsPass)
    println("LIBENZYME=", Enzyme.Compiler.Enzyme_jll.libEnzyme_path)
    artifact = BRM.emit_rk_artifact(brmi; case_id="hierarchical-mixture-probabilities")
    translated = rk_translate_artifact(artifact)
    rebuilt = Base.invokelatest(build_kernel, translated)
    replay_pointwise = prepare_query(rebuilt, translated, :pointwise)
    replay = prepare_sampler(rebuilt, translated, zeros(11);
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    stan = consumer_stan(brmi, "hierarchical-mixture-probabilities")
    transport = brm_coordinate_transport(backend, SBBRMI(brmi; total_groups=()),
        BridgeStan.param_unc_names(stan.model))
    mapping = [pair.rk => pair.stan for pair in transport.pairs]
    for u in (zeros(11), fill(0.13,11), collect(range(-0.4,0.5;length=11)))
        oracle(v) = (p = hierarchy_parts(backend,data,v); sum(p.ll)+p.prior+p.jac)
        value, gradient = check_consumer_point(problem,u,oracle)
        check_consumer_stan(problem,stan,mapping,backend,u)
        parts = hierarchy_parts(backend,data,u)
        @test mixture_query(backend,:prior,u) ≈ parts.prior
        @test mixture_query(backend,:likelihood,u) ≈ sum(parts.ll)
        @test mixture_query(backend,:pointwise,u).y ≈ parts.ll
        @test isequal(mixture_query(backend,:pointwise,u),
            Base.invokelatest(replay_pointwise,u))
        replay_gradient = similar(u)
        replay_value, _ = sampler_value_and_gradient!(replay,replay_gradient,u)
        @test isequal(value,replay_value)
        @test gradient ≈ replay_gradient atol=2e-13 rtol=2e-13
        @test all(isfinite, gradient)
    end
    # Reprocess on a different row count/order while retaining the fitted
    # grouping levels. The probability predictor owns the new observation axis.
    changed = (; y=[0.4,-0.1,1.2,0.6], analyte=[3,1,2,3], sample=[2,1,2,1])
    rebound = reprocess(backend, changed)
    @test coordinate_names(rebound.model.layout) == coordinate_names(backend.model.layout)
    u = collect(range(-0.2,0.3;length=11))
    @test mixture_query(rebound,:pointwise,u).y ≈ hierarchy_parts(rebound,changed,u).ll
    check_rk_source_roundtrip(rebound; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
end

@stestset "mixture probabilities with only a weight predictor and general K" begin
    data = (; x=[-1.0,0.2,0.7,1.3], y=[-0.5,0.1,0.8,1.4])
    brmi = @brm data begin
        logit(w) ~ 1 + x
        effect(w, :) ~ Normal(0, 1)
        y ~ MixtureModel([Normal(-1,0.5),Normal(0,1),Normal(2,0.8)],
            [(1-w)/2, (1-w)/2, w])
    end
    backend, problem = consumer_problem(brmi)
    @test LogDensityProblems.dimension(problem) == 2
    for u in (zeros(2), [0.3,-0.4])
        # Formula design order is intercept, x.
        w = LogExpFunctions.logistic.(u[1] .+ u[2].*data.x)
        ll = [logpdf(MixtureModel([Normal(-1,0.5),Normal(0,1),Normal(2,0.8)],
            [(1-w[j])/2,(1-w[j])/2,w[j]]),data.y[j]) for j in eachindex(data.y)]
        @test mixture_query(backend,:pointwise,u).y ≈ ll
        check_consumer_point(problem,u,v -> begin
            p = LogExpFunctions.logistic.(v[1] .+ v[2].*data.x)
            sum(logpdf.(Normal(),v)) + sum(logpdf(MixtureModel(
                [Normal(-1,0.5),Normal(0,1),Normal(2,0.8)],
                [(1-p[j])/2,(1-p[j])/2,p[j]]),data.y[j]) for j in eachindex(data.y))
        end)
    end
    for u in ([-1000.0,0.0], [1000.0,0.0])
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem,u)
        @test isfinite(value)
        @test all(isfinite,gradient)
    end
end

@stestset "named mixture probability expressions" begin
    data = (; y=[-0.3,0.2,1.1], x=[-0.5,0.0,0.7])
    brmi = @brm data begin
        logit(w) ~ 1 + x
        effect(w, :) ~ Normal(0,1)
        probabilities = [(1-w)/2, (1-w)/2, w]
        y ~ MixtureModel([Normal(-1,1),Normal(0,1),Normal(2,1)], probabilities)
    end
    backend, problem = consumer_problem(brmi)
    oracle(v) = sum(logpdf.(Normal(),v)) + sum(eachindex(data.y)) do j
        w = LogExpFunctions.logistic(v[1] + v[2]*data.x[j])
        logpdf(MixtureModel([Normal(-1,1),Normal(0,1),Normal(2,1)],
            [(1-w)/2,(1-w)/2,w]), data.y[j])
    end
    for u in ([0.2,-0.1], [-0.3,0.4])
        check_consumer_point(problem,u,oracle)
    end
end

@stestset "computed shared mixture probabilities retain scalar priors" begin
    brmi = @brm (; y=[-0.3,0.2,1.1]) begin
        a ~ Normal(0,1)
        w = 1 / (1 + exp(-a))
        y ~ MixtureModel([Normal(0,1),Normal(2,1)], [1-w,w])
    end
    backend, problem = consumer_problem(brmi)
    oracle(v) = logpdf(Normal(),v[1]) + sum(logpdf(MixtureModel(
        [Normal(0,1),Normal(2,1)], [1-LogExpFunctions.logistic(v[1]),
        LogExpFunctions.logistic(v[1])]),y) for y in [-0.3,0.2,1.1])
    check_consumer_point(problem,[0.2],oracle)
end

@stestset "computed shared general-K probabilities retain every scalar prior" begin
    data = (; y=[-0.3,0.2,1.1,0.6])
    brmi = @brm data begin
        a ~ Normal(0,1)
        b ~ Normal(0,1)
        total = 1 + exp(a) + exp(b)
        probabilities = [1/total, exp(a)/total, exp(b)/total]
        y ~ MixtureModel([Normal(-1,1),Normal(0,1),Normal(2,1)], probabilities)
    end
    backend, problem = consumer_problem(brmi)
    @test LogDensityProblems.dimension(problem) == 2
    names = coordinate_names(backend.model.layout)
    ia, ib = coordinate(names,:a), coordinate(names,:b)
    oracle(v) = begin
        weights = [1.0,exp(v[ia]),exp(v[ib])]
        weights ./= sum(weights)
        law = MixtureModel([Normal(-1,1),Normal(0,1),Normal(2,1)],weights)
        sum(logpdf.(Normal(),v)) + sum(logpdf.(law,data.y))
    end
    for u in (zeros(2),[0.2,-0.3])
        check_consumer_point(problem,u,oracle)
    end
end

@stestset "tuple mixture probability expressions" begin
    data = (; y=[-0.3,0.2,1.1,0.6], x=[-0.5,0.0,0.7,0.2])
    brmi = @brm data begin
        logit(w) ~ 1 + x
        effect(w, :) ~ Normal(0,1)
        y ~ MixtureModel((Normal(-1,1),Normal(0,1),Normal(2,1)),
            ((1-w)/2,(1-w)/2,w))
    end
    backend, problem = consumer_problem(brmi)
    oracle(v) = sum(logpdf.(Normal(),v)) + sum(eachindex(data.y)) do j
        w = LogExpFunctions.logistic(v[1] + v[2]*data.x[j])
        logpdf(MixtureModel([Normal(-1,1),Normal(0,1),Normal(2,1)],
            [(1-w)/2,(1-w)/2,w]), data.y[j])
    end
    check_consumer_point(problem,[0.2,-0.1],oracle)
end

@stestset "mixture probability data and invalid live rows" begin
    builder = @brm begin
        mu ~ Normal(0,1)
        y ~ MixtureModel([Normal(mu,1),Normal(2,1)], probabilities)
    end
    y = [-0.3,0.2,1.1]
    for probabilities in ([0.25,0.75], [[1.0,0.0],[0.2,0.8],[0.0,1.0]])
        backend, problem = consumer_problem(builder((; y, probabilities)))
        u = [0.1]
        rows = eltype(probabilities) <: Real ? fill(probabilities,3) : probabilities
        oracle(v) = logpdf(Normal(),v[1]) + sum(logpdf(MixtureModel(
            [Normal(v[1],1),Normal(2,1)], rows[j]), y[j]) for j in eachindex(y))
        check_consumer_point(problem,u,oracle)
    end
    for (probabilities, diagnostic) in (
            ([[0.2,0.8],[0.4,0.4],[0.5,0.5]], "mixture weights must sum to 1"),
            ([[0.2,0.8],[-0.1,1.1],[0.5,0.5]], "mixture weights must be nonnegative"))
        err = try
            RKBRMI(builder((; y, probabilities)))
        catch e
            e
        end
        @test err isa ReactiveKernelsPPL.ContractValidationError
        @test occursin(diagnostic, sprint(showerror, err))
    end
    invalid = @brm (; y, x=[0.0,0.5,1.0]) begin
        w ~ 1 + x
        effect(w, :) ~ Normal(0,1)
        y ~ MixtureModel([Normal(0,1),Normal(2,1)], [1-w,w])
    end
    backend = RKBRMI(invalid)
    @test mixture_query(backend,:likelihood,[0.2,0.1]) > -Inf
    @test mixture_query(backend,:likelihood,[0.2,1.0]) == -Inf
end
