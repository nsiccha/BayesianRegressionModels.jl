include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@stestset "unconditioned BRM retains generated roles and zero fitted coordinates" begin
    regression = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    direct = @brm begin
        a ~ Normal(0, 1)
        y ~ Normal(a, 1)
    end
    ragged_prior = @brm begin
        sigma_add ~ Exponential(1)
        sigma_prop ~ Exponential(1)
        log_CL ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(log_CL)
            loc[i] = exp(log_CL[i]) .+ 0.0 .* obs_idx[obs[i]]
            pk_conc[obs[i]] ~ censored(
                Normal(loc[i], addprop(loc[i], sigma_add, sigma_prop)); lower=pk_lloq[obs[i]])
        end
    end
    ragged_data = (;
        subject=["s2", "s1"],
        obs_subject=["s1", "s2", "s1", "s2", "s2", "s1", "s2"],
        obs_idx=collect(1.0:7.0),
        pk_lloq=[0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16])
    for (label, factory, data) in (
        ("prior-regression", regression, (; x=[-0.2, 0.4, 0.8])),
        ("prior-direct-no-data", direct, (;)),
        ("prior-ragged-bounds", ragged_prior, ragged_data))
        before = deepcopy(data)
        brmi = factory(data)
        backend = check_rk_source_roundtrip(RKBRMI(brmi))
        @test backend.plan isa BRM._RKUnconditionedPlan
        @test isempty(coordinate_names(backend.model.layout))
        @test isempty(BRM._rk_emit_ast(backend.plan).main.args)
        @test isempty(BRM._rk_observed_names(backend.plan))
        @test any(d -> d.role === :observation && d.data_source === nothing,
            backend.plan.generative.declarations)
        @test occursin("native generated draws = unavailable", BRM.show_rk_plan(backend.plan))
        @test BRM.stan_code(backend.plan.generative) ==
            BRM.stan_code(SBBRMI(brmi))
        if label == "prior-ragged-bounds"
            # The bound read through the binding, gathered per subject.
            bound = only(k for k in keys(backend.plan.generative.data)
                if occursin("pk_lloq", string(k)) && endswith(string(k), "_ragged"))
            @test backend.plan.generative.data[bound] ==
                [[0.11, 0.13, 0.14, 0.16], [0.10, 0.12, 0.15]]
            @test !haskey(backend.plan.columns, :pk_conc)
        elseif label == "prior-regression"
            @test backend.plan.generative.data[:x] == data.x
        end
        problem = rk_logdensity_problem(backend;
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, Float64[])
        @test value == 0.0
        @test isempty(gradient)
        stan = consumer_stan(brmi, label)
        @test BridgeStan.param_unc_num(stan.model) == 0
        sv, sg = BridgeStan.log_density_gradient(stan.model, Float64[];
            propto=false, jacobian=true)
        @test sv == value
        @test isempty(sg)
        # Both backends start at the empty point, and the coordinate
        # transport pairs the two empty layouts.
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        for b in (:zero, :prior_median)
            base = b === :zero ? Float64[] : b
            @test brm_start(backend, (;); base) == Float64[]
            @test brm_start(sb, stan.model, (;); base) == Float64[]
        end
        # BridgeStan names this model's parameters `[""]`: none.
        t = brm_coordinate_transport(backend, sb, BridgeStan.param_unc_names(stan.model))
        @test isempty(t.pairs) && isempty(t.rk_names) && isempty(t.stan_names)
        @test brm_rk_to_stan(t, Float64[]) == Float64[]
        @test brm_start(sb, stan.model; rk=backend, chains=2) == [Float64[], Float64[]]
        given = Dict("prior-regression" => (; sigma=1.0),
            "prior-direct-no-data" => (; a=0.1),
            "prior-ragged-bounds" => (; sigma_add=0.2))[label]
        for start in (() -> brm_start(backend, given), () -> brm_start(sb, stan.model, given))
            err = try
                start()
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("zero sampled coordinates", err.msg)
            @test occursin("`$(only(keys(given)))`", err.msg)
        end
        artifact = BRM.emit_rk_artifact(brmi; case_id=label)
        rebuilt = build_kernel(BRM.rk_translate_artifact(artifact))
        @test isempty(coordinate_names(rebuilt.layout))
        @test isequal(data, before)
    end
end

# An unconditioned program samples nothing, so every start is the empty point
# and an initial value has no coordinate to set; snag unconditioned-br-7e674a7f
# (before, `brm_start` died in a MethodError on the plan's coordinate records).
@stestset "unconditioned BRM starts at the empty point and refuses initial values" begin
    declared = @brm begin
        z[1:n] .~ LogNormal.(mu, sd)
        y ~ Normal(z, obs_sd)
    end
    grouped = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | p | g)
        y ~ Normal(mu, sigma)
    end
    for (label, brmi, inits, given) in (
        ("declared-vector", declared((; n=2, mu=-0.2, sd=0.3, obs_sd=1.0)),
            (; z=exp.([-0.3, 0.0])), ("`z`",)),
        ("declared-vector-zero-extent", declared((; n=0, mu=-0.2, sd=0.3, obs_sd=1.0)),
            (; z=Float64[]), ("`z`",)),
        ("grouped", grouped((; x=[-0.2, 0.4, 0.8, 0.1], g=["a", "b", "a", "c"])),
            (; mu=(; Intercept=0.1), sigma=1.2, p=(; sd=0.5)), ("`mu`", "`sigma`", "`p`")))
        rk = RKBRMI(brmi)
        @test rk.plan isa BRM._RKUnconditionedPlan
        @test isempty(BRM._rk_coordinate_records(rk.plan; term_starts=true))
        for base in (Float64[], :prior_median)
            @test brm_start(rk, (;); base) == Float64[]
        end
        @test brm_start(rk) == Float64[]
        @test brm_start(rk; chains=2) == [Float64[], Float64[]]
        err = try
            brm_start(rk, inits)
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("unconditioned program", err.msg)
        @test all(name -> occursin(name, err.msg), given)
    end
end

@stestset "bound baseline retains inference while absent endpoint forward-simulates" begin
    data = (; baseline=[0.1, -0.3, 0.5], x=[-0.2, 0.4, 0.8])
    original = deepcopy(data)
    for value_route in (false, true)
        brmi = if value_route
            @brm data begin
                a ~ Normal(0, 1)
                unused ~ Exponential(2)
                loc = a + x
                baseline ~ Normal(a, 1)
                y ~ Normal(loc, 1)
            end
        else
            @brm data begin
                a ~ Normal(0, 1)
                unused ~ Exponential(2)
                mu ~ 1
                baseline ~ Normal(mu, 1)
                y ~ Normal(mu, 1)
            end
        end
        backend, problem = consumer_problem(brmi)
        @test !(backend.plan isa BRM._RKUnconditionedPlan)
        @test length(coordinate_names(backend.model.layout)) == 1
        @test BRM._rk_observed_names(backend.plan) == (:baseline,)
        stan = consumer_stan(brmi, "prior-baseline-$value_route")
        @test BridgeStan.param_unc_num(stan.model) == 1
        oracle(u) = logpdf(Normal(), u[1]) +
            sum(logpdf.(Normal(u[1], 1), data.baseline))
        for u in ([0.0], [0.23], [-0.17])
            value, gradient = check_consumer_point(problem, u, oracle)
            sv, sg = BridgeStan.log_density_gradient(stan.model, u;
                propto=false, jacobian=true)
            @test value ≈ sv atol=2e-12 rtol=2e-12
            @test gradient ≈ sg atol=2e-12 rtol=2e-12
        end
        @test isequal(data, original)
    end
end
