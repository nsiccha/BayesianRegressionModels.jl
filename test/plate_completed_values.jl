# A cell reads a completed vector (or an assignment of it); it does not
# observe that vector again. Run with julia --project=test this_file.jl.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

const COMPLETION_DATA = let
    kreis = repeat(["a", "b", "c"]; inner=4)
    v = Union{Missing,Float64}[11, missing, missing, 31, 5, 7, 9, 12,
        missing, missing, missing, missing]
    hid = [ismissing(x) && k == "a" ? k : "" for (x, k) in zip(v, kreis)]
    # The supplied total constrains the two rows labelled "a" by hid.
    (; v, x=randn(Xoshiro(1), 12), kreis, hid, listed=["a"], total=[62.0],
        cells=["c"], obs_c=[55.0], pos_c=findall(==("c"), kreis), kreis_c=fill("c", 4))
end

function completion_model(data; gathered=false, shares=true, scaled=false)
    completion = shares ? :(mi(v)::shares(total, groups(hid => listed)) ~ LogNormal(mu, sigma)) :
        :(mi(v) ~ LogNormal(mu, sigma))
    assignment = scaled ? :(v_c = 2 * v[pos_c]) : :(v_c = v[pos_c])
    grouping = gathered ? :(rows_c = groups(kreis_c => cells)) :
        :(rows_c = groups(kreis => cells))
    location = gathered ? :(sum(v_c[rows_c[kk]]) + 10.0) :
        scaled ? :(2 * sum(v[rows_c[kk]]) + 10.0) : :(sum(v[rows_c[kk]]) + 10.0)
    body = quote
        sigma ~ Exponential(1)
        mu ~ 1 + x
        $completion
        $grouping
        @plate for kk in eachindex(obs_c)
            obs_c[kk] ~ Normal($location, 2.0)
        end
    end
    gathered && push!(body.args, assignment) # A forward read must work too.
    BRM.brm(body; mod=@__MODULE__)(data)
end

completion_points(n) = (zeros(n), fill(0.1, n), collect(range(-0.3, 0.4; length=n)))

function completion_stan(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    stan = BRM.stan_instantiate(sb)
    (; brmi, sb, stan, names=BridgeStan.param_unc_names(stan.model))
end

# Independent density on the physical values, including every Jacobian.
function completion_oracle(data, constrained; shares=true, scaled=false)
    sigma = constrained["sigma"]
    beta = [constrained["pop_mu_beta_pop.$i"] for i in 1:2]
    mu = beta[1] .+ beta[2] .* data.x
    v = [constrained["v.$i"] for i in eachindex(data.v)]
    value = logpdf(Exponential(1), sigma) + log(sigma) + sum(logpdf.(Normal(), beta))
    value += sum(logpdf.(LogNormal.(mu, sigma), v))
    missing_rows = findall(ismissing, data.v)
    value += sum(log, v[missing_rows])
    if shares
        for (label, total) in zip(data.listed, data.total)
            observed = findall(i -> data.hid[i] == label && !ismissing(data.v[i]), eachindex(v))
            value -= log(total - sum(v[observed]; init=0.0))
        end
    end
    value + logpdf(Normal((scaled ? 2 : 1) * sum(v[data.pos_c]) + 10, 2), only(data.obs_c))
end

@stestset "completed plate values: both spellings equal the full density" begin
    for shares in (true, false), scaled in (false, true)
        println("completion parity: shares=$shares scaled=$scaled"); flush(stdout)
        data = COMPLETION_DATA
        direct = completion_stan(completion_model(data; shares, scaled))
        gathered = completion_stan(completion_model(data; shares, scaled, gathered=true))
        @test direct.names == gathered.names
        @test length(direct.names) == (shares ? 8 : 9)
        physical_names = BridgeStan.param_names(direct.stan.model; include_tp=true)
        for u in completion_points(length(direct.names))
            values = Dict(zip(physical_names,
                BridgeStan.param_constrain(direct.stan.model, u; include_tp=true)))
            expected = completion_oracle(data, values; shares, scaled)
            gd, gg = similar(u), similar(u)
            vd, _ = BridgeStan.log_density_gradient!(direct.stan.model, u, gd; propto=false)
            vg, _ = BridgeStan.log_density_gradient!(gathered.stan.model, u, gg; propto=false)
            @test vd ≈ expected atol=1e-10 rtol=1e-12
            @test vg ≈ vd atol=1e-10 rtol=1e-12
            @test gg ≈ gd atol=1e-10 rtol=1e-12
        end
        for completions in (:centered, :noncentered), gather in (false, true)
            println("RK completion parity: $completions gathered=$gather"); flush(stdout)
            compiled = gather ? gathered : direct
            rk = RKBRMI(compiled.brmi; completions)
            problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            transport = brm_coordinate_transport(rk, compiled.sb, compiled.names)
            @test LogDensityProblems.dimension(problem) == length(direct.names)
            for u in completion_points(length(transport))
                su = brm_rk_point_to_stan(transport, u)
                sg = similar(su)
                sv, _ = BridgeStan.log_density_gradient!(compiled.stan.model, su, sg; propto=false)
                rv, rg = LogDensityProblems.logdensity_and_gradient(problem, u)
                @test isfinite(rv) && all(isfinite, rg)
                @test rv ≈ brm_stan_logdensity_to_rk(transport, u, sv) atol=1e-9 rtol=1e-11
                @test rg ≈ brm_stan_gradient_to_rk(transport, u, sg) atol=1e-8 rtol=1e-10
            end
        end
    end
end

@stestset "a response observed elsewhere is only read by a subset cell" begin
    data = (; y=[1.0, 2.0, 3.0], g=["a", "b", "b"], cells=["b"], obs=[4.0])
    brmi = @brm data begin
        mu ~ Normal(0, 1)
        y ~ Normal(mu, 1)
        rows = groups(g => cells)
        @plate for k in eachindex(obs)
            obs[k] ~ Normal(sum(y[rows[k]]), 2.0)
        end
    end
    compiled = completion_stan(brmi)
    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in ([0.0], [0.3], [-0.2])
        expected = logpdf(Normal(), only(u)) + sum(logpdf.(Normal(only(u), 1), data.y)) +
            logpdf(Normal(5, 2), only(data.obs))
        @test BridgeStan.log_density(compiled.stan.model, u; propto=false) ≈ expected
        check_consumer_point(problem, u, x -> logpdf(Normal(), only(x)) +
            sum(logpdf.(Normal(only(x), 1), data.y)) + logpdf(Normal(5, 2), only(data.obs)))
    end
end

@stestset "completed plate values: replay and emitted source" begin
    data = COMPLETION_DATA
    changed = merge(data, (; obs_c=[49.0], total=[51.0], pos_c=[12, 9, 11, 10],
        kreis_c=fill("c", 4), x=reverse(data.x)))
    for gathered in (false, true)
        brmi = completion_model(data; gathered)
        direct = completion_model(changed; gathered)
        sb = SBBRMI(brmi; mod=@__MODULE__)
        @test stan_data(reprocess(sb, changed)) == stan_data(SBBRMI(direct; mod=@__MODULE__))
        rk = check_rk_source_roundtrip(RKBRMI(brmi))
        replay = rk_logdensity_problem(reprocess(rk, changed);
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        rebuilt = rk_logdensity_problem(RKBRMI(direct);
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        for u in completion_points(LogDensityProblems.dimension(replay))
            @test LogDensityProblems.logdensity(replay, u) ≈
                LogDensityProblems.logdensity(rebuilt, u) atol=1e-10 rtol=1e-12
        end
    end
end

@stestset "grouped assignments: predictor gather and row validation" begin
    data = COMPLETION_DATA
    direct = @brm data begin
        mu ~ 1 + x
        rows = groups(kreis => cells)
        @plate for k in eachindex(obs_c)
            obs_c[k] ~ Normal(sum(mu[rows[k]]), 2.0)
        end
    end
    gathered = @brm data begin
        mu ~ 1 + x
        mu_c = mu[pos_c]
        rows = groups(kreis_c => cells)
        @plate for k in eachindex(obs_c)
            obs_c[k] ~ Normal(sum(mu_c[rows[k]]), 2.0)
        end
    end
    sd, sg = completion_stan(direct), completion_stan(gathered)
    pd = rk_logdensity_problem(RKBRMI(direct); ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    pg = rk_logdensity_problem(RKBRMI(gathered); ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in completion_points(2)
        @test BridgeStan.log_density(sd.stan.model, u; propto=false) ≈
            BridgeStan.log_density(sg.stan.model, u; propto=false)
        @test LogDensityProblems.logdensity(pd, u) ≈ LogDensityProblems.logdensity(pg, u)
    end
    bad = merge(data, (; kreis_c=["c", "c", "c"]))
    for backend in (SBBRMI, RKBRMI)
        @test_throws "has 4 entries" backend(completion_model(bad; gathered=true))
        observed = @brm (; y=[1.0, 2.0], g=["a", "b"], cells=["a"]) begin
            mu ~ Normal(0, 1)
            rows = groups(g => cells)
            @plate for k in eachindex(cells)
                y[rows[k]] ~ Normal(mu, 1.0)
            end
        end
        @test_throws "belong to no cell" backend(observed)
    end
end
