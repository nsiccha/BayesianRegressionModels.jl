# Executable authoring convention, including complete printed BRM replay.
# Compare each backend with the same laws written before their readers.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
include(joinpath(@__DIR__, "..", "docs", "examples", "structure_first.jl"))

function check_ordered_density(reference, candidate, refnames, names)
    @test sort(string.(names)) == sort(string.(refnames))
    permutation = [only(findall(==(name), refnames)) for name in names]
    n = length(refnames)
    for point in (zeros(n), [0.2sin(i) for i in 1:n])
        reference_value, reference_gradient =
            LogDensityProblems.logdensity_and_gradient(reference, point)
        value, gradient = LogDensityProblems.logdensity_and_gradient(candidate, point[permutation])
        @test isfinite(reference_value)
        @test value ≈ reference_value atol=2e-10 rtol=2e-10
        @test gradient ≈ reference_gradient[permutation] atol=2e-9 rtol=2e-9
    end
end

function check_structure_first(builder, original, data, label; check_rk=true)
    source = string(builder)
    replay = brm(Meta.parse("begin\n" * source * "\nend"); mod=StructureFirstExamples)
    @test string(replay) == source
    models = (original(data), builder(data), replay(data))
    backends = check_rk ? map(RKBRMI, models) : nothing
    if check_rk
        problems = map(rk -> rk_logdensity_problem(rk;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), backends)
        names = map(rk -> coordinate_names(rk.model.layout), backends)
        for k in 2:3
            check_ordered_density(problems[1], problems[k], names[1], names[k])
        end
    end
    sb = map(m -> SBBRMI(m; mod=StructureFirstExamples), models)
    @test BRM.stan_code(sb[2]) == BRM.stan_code(sb[3])
    stan = map(enumerate(models[1:2])) do (k, model)
        consumer_stan(model, "structure-first-$(label)-$(k)"; mod=StructureFirstExamples)
    end
    stan_names = map(p -> BridgeStan.param_unc_names(p.model), stan)
    check_ordered_density(stan[1], stan[2], stan_names[1], stan_names[2])
    (; models, backends, sb, stan_names)
end

@stestset "composed structures precede scalar and addressed priors" begin
    data = (; subject=["a", "a", "b", "b"], y=[0.1, 0.3, -0.2, 0.4])
    original = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | subject)
        effect(mu, Intercept) ~ Normal(0, 1)
        sd(mu, subject) ~ Exponential(1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(mu[i], sigma)
        end
    end
    source = string(StructureFirstExamples.joint)
    @test startswith(source, "mu ~ 1 + (1 | subject)\n@plate")
    @test first(findfirst("end", source)) < first(findfirst("effect(mu", source))
    @test endswith(source, "sigma ~ Exponential(1)")
    check_structure_first(StructureFirstExamples.joint, original, data, "scalar")
end

@stestset "correlation, scale and coefficient priors share the final block" begin
    original = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        effect(mu, Intercept) ~ Normal(0, 1)
        effect(mu, x) ~ Normal(0, 0.5)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(2, 2)
        y ~ Normal(mu, sigma)
    end
    data = (; subject=["a", "a", "b", "b"],
        x=[-1.0, 1.0, -1.0, 1.0], y=[0.1, 0.3, -0.2, 0.4])
    check_structure_first(StructureFirstExamples.correlated_model, original, data, "correlation")
end

@stestset "declared vectors and scalar hyperpriors follow the plate" begin
    original = @brm begin
        sigma ~ Exponential(1)
        tau ~ Exponential(1)
        x[1:n] .~ Normal.(0, tau)
        @plate for i in eachindex(y)
            y[i] ~ Normal(x[i], sigma)
        end
    end
    data = (; n=3, y=[0.1, -0.2, 0.4])
    result = check_structure_first(StructureFirstExamples.vector_model, original, data, "vector")
    @test any(d -> d.target === :x && d.role === :prior && d.dimension == (:x_n,),
        generative_plan(result.sb[2]).declarations)
    @test count(n -> startswith(string(n), "x."),
        coordinate_names(result.backends[2].model.layout)) == data.n
end

@stestset "trailing cell-local priors keep their replication and identity" begin
    original = @brm begin
        @plate for i in eachindex(y)
            theta ~ Normal(0, 1)
            y[i] ~ Normal(theta, 0.5)
        end
    end
    data = (; y=[0.1, -0.2, 0.4])
    result = check_structure_first(StructureFirstExamples.local_model, original, data, "local")
    @test length(coordinate_names(result.backends[2].model.layout)) == length(data.y)
end

@stestset "Stan cell-local hyperpriors follow their deterministic readers" begin
    original = @brm begin
        @plate for i in eachindex(y)
            location ~ Normal(0, 1)
            log_scale ~ Normal(0, 0.3)
            center = 2 * location
            theta ~ Normal(center, exp(log_scale))
            y[i] ~ Normal(theta, 0.5)
        end
    end
    builder = @brm begin
        @plate for i in eachindex(y)
            center = 2 * location
            y[i] ~ Normal(theta, 0.5)
            theta ~ Normal(center, exp(log_scale))
            location ~ Normal(0, 1)
            log_scale ~ Normal(0, 0.3)
        end
    end
    data = (; y=[0.1, -0.2, 0.4])
    result = check_structure_first(builder, original, data, "local-hyperpriors"; check_rk=false)
    @test length(result.stan_names[2]) == 3length(data.y)
end

@stestset "Stan trailing cell prior feeds a scan without reordering its steps" begin
    original = @brm begin
        @plate for i in eachindex(y)
            theta ~ Normal(0, 1)
            @scan begin
                h[1] ~ Normal(theta, 1)
                for t in 2:T
                    h[t] ~ Normal(h[t - 1], 0.2)
                end
            end
            y[i] ~ Normal(h, 0.5)
        end
    end
    builder = @brm begin
        @plate for i in eachindex(y)
            @scan begin
                h[1] ~ Normal(theta, 1)
                for t in 2:T
                    h[t] ~ Normal(h[t - 1], 0.2)
                end
            end
            y[i] ~ Normal(h, 0.5)
            theta ~ Normal(0, 1)
        end
    end
    data = (; y=[[0.1, -0.2, 0.4], [0.2, 0.3, -0.1]], T=3)
    result = check_structure_first(builder, original, data, "local-scan"; check_rk=false)
    @test length(result.stan_names[2]) == (data.T + 1)length(data.y)
end
