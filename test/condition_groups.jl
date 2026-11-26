# julia --project=test test/condition_groups.jl
module TestConditionGroups
using BayesianRegressionModels, StanBlocks, BridgeStan, Test, Statistics
const BRM = BayesianRegressionModels

const model = @brm begin
    level ~ 0 + factor(kind) + (1 | area) + (1 | site)
    sd(level, site) ~ Exponential(1.)
    extra_sd ~ Exponential(1.)
    noise = sqrt.(se .* se .+ coarse .* extra_sd .* extra_sd)
    reading .~ Normal.(level, noise)
    log(mu) ~ 0 + offset(level) + coarse + (1 | batch) + offset(log_n)
    count ~ NegativeBinomial2(mu, 4.)
end
const train = (; kind=["a","b","a","b"], area=[1,2,1,2], site=1:4, batch=[1,1,2,2],
               se=fill(.5,4), coarse=[0.,1.,0.,0.], log_n=fill(log(3.),4),
               reading=[1.,0.,2.,1.], count=[3,2,4,1])
const future = (; kind=["a","b","b","a"], area=[1,2,2,1], site=[17,4,17,39], batch=[3,3,3,4],
                se=[.5,.5,1.,2.], coarse=[0.,1.,0.,0.], log_n=[0.,1.,0.,1.],
                reading=[-2.,2.,1.,3.])

@testset "Gaussian measurement conditions each replayed intercept" begin
    joint = model(train)
    context = BRM._brm_backend_context(joint)
    projected = BRM._adaptive_numeric_columns(joint, context, :mu, length(train.site),
                                               name -> name === :coarse ? 1 : 0)
    @test only(projected).label === :coarse
    @test only(projected).values == train.coarse
    sb = SBBRMI(joint; mod=@__MODULE__, total_groups=())
    selection = (; site=:reading)
    replay = reprocess(sb, future; condition_groups=selection, resample_groups=:batch)
    plan = reprocess(generative_plan(sb), future; condition_groups=selection, resample_groups=:batch)
    @test Set(plan.cv_groups) == Set([:site,:batch])
    @test BRM.stan_code(plan) == BRM.stan_code(replay)
    @test restan_data(sb, future; condition_groups=selection, resample_groups=:batch) == BRM.stan_data(plan)
    @test only(filter(b -> b.group === :site, ranef_blocks(replay))).generated
    @test length(brm_outputs(brm_descriptor(replay); component=:conditional_moments)) == 1
    @test_throws ArgumentError reprocess(sb, future; condition_groups=selection, resample_groups=:site)
    @test_throws ArgumentError reprocess(sb, future; condition_groups=(site=:count,), resample_groups=:batch)
    @test_throws ArgumentError reprocess(sb, merge(future,(;count=ones(Int,4)));
        condition_groups=(site=:count,), resample_groups=:batch)
    @test_throws ArgumentError reprocess(sb, future; condition_groups=(batch=:reading,), resample_groups=:site)
    @test_throws "updates must be independent" reprocess(sb, future;
        condition_groups=(site=:reading, area=:reading), resample_groups=:batch)

    rebound = reprocess(replay, merge(future,(; reading=-future.reading)))
    @test BRM.stan_code(rebound) == BRM.stan_code(replay)
    @test_throws ArgumentError reprocess(replay, Base.structdiff(future, (; reading=nothing)))
    @test_throws ArgumentError reprocess(sb, merge(future,(;count=ones(Int,4)));
        condition_groups=selection, resample_groups=:batch, held_out=:reading)

    mktempdir() do dir
        from = BRM.stan_instantiate(sb; path=joinpath(dir,"training.stan")).model
        to = BRM.stan_instantiate(replay; path=joinpath(dir,"forecast.stan")).model
        training_names = string.(BridgeStan.param_unc_names(from))
        forecast_names = string.(BridgeStan.param_unc_names(to))
        u = zeros(1, length(training_names))
        u[1, only(findall(==("cat_level_kind_beta.1"),training_names))] = .7
        u[1, only(findall(==("cat_level_kind_beta.2"),training_names))] = -.4
        v = vec(BRM.transport_draws(sb,replay,u,training_names,forecast_names))
        names = string.(BridgeStan.param_names(to; include_tp=true,include_gq=true))
        col(key) = only(findall(==(key),names))
        cols = [col("level.$i") for i in 1:4]
        mus = [col("mu.$i") for i in 1:4]
        rng = BridgeStan.StanRNG(to, 925)
        draws = reduce(hcat, [BridgeStan.param_constrain(to,v; include_tp=true,include_gq=true,rng)
                              for _ in 1:4000])
        profile = [.7,-.4,-.4,.7]
        effects = draws[cols,:] .- profile
        @test effects[1,:] ≈ effects[3,:]
        @test log.(draws[mus[1],:] ./ draws[mus[3],:]) ≈ fill(profile[1]-profile[3],4000)
        for site in unique(future.site)
            rows = findall(==(site),future.site)
            weights = inv.(future.se[rows].^2 .+ future.coarse[rows])
            variance = inv(1 + sum(weights))
            location = variance * sum(weights .* (future.reading[rows] .- profile[rows]))
            b = effects[first(rows),:]
            @test mean(b) ≈ location atol=5sqrt(variance/length(b))
            @test var(b) ≈ variance rtol=.09
        end
        @test abs(cor(effects[1,:],effects[2,:])) < .06
    end
end

@testset "single observation with a scalar sampling error and default group scale" begin
    direct = @brm begin
        level ~ 0 + (1 | site)
        reading .~ Normal.(level, 0.5)
        log(mu) ~ 0 + offset(level)
        count ~ NegativeBinomial2(mu, 3.)
    end
    sb = SBBRMI(direct(train); mod=@__MODULE__, total_groups=())
    replay(x) = reprocess(sb, (; site=[42], reading=[x]); condition_groups=(site=:reading,))
    mktempdir() do dir
        function samples(x)
            target = BRM.stan_instantiate(replay(x); path=joinpath(dir,"one.stan")).model
            v = zeros(length(BridgeStan.param_unc_names(target)))
            names = string.(BridgeStan.param_names(target; include_tp=true,include_gq=true))
            k = only(findall(==("level.1"),names))
            # Discrete outcome draws consume a variable number of random bits.
            # Pair seeds per replication so subsequent draws stay aligned.
            [BridgeStan.param_constrain(target,v; include_tp=true,include_gq=true,
                rng=BridgeStan.StanRNG(target, 109+i))[k] for i in 1:3000]
        end
        a, b = samples(-2.), samples(2.)
        @test b .- a ≈ fill(3.2,length(a))
        @test mean(a) ≈ -1.6 atol=.04
        @test var(b) ≈ .2 atol=.02

        # One observed zero count at the first of two visits sharing this site.
        # The second visit's expected count integrates exp(level) against the
        # joint Normal prior, measurement density and one NB2 probability.
        pzero(t) = (3 / (3 + exp(t)))^3
        grid = -3:.005:5
        mass = @. exp(-grid^2/2 - (2-grid)^2/(2*.5^2)) * pzero(grid)
        expected = sum(exp.(grid) .* mass) / sum(mass)
        w = pzero.(b)
        w ./= sum(w)
        rate = exp.(b)
        estimate = sum(w .* rate)
        @test estimate ≈ expected atol=5sqrt(sum(abs2, w .* (rate .- estimate)))
    end
end
end
