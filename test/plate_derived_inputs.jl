# Computed arrays keep their own axes when read directly by a plate cell.
# Run: julia --project=test test/plate_derived_inputs.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const derived_plate_data = (; subject=["b", "a"],
    x=[[1.0, 2.0], [3.0]], y=[[0.7, 1.1], [0.5]])

function derived_plate_mapping(names)
    map(names) do name
        s = string(name)
        stan = if (m = match(r"^(.*)\.z\.(\d+)\.1$", s)) !== nothing
            "$(m[1])_z_flat.$(m[2])"
        elseif (m = match(r"^(.*)\.tau\.(\d+)$", s)) !== nothing
            "$(m[1])_tau.$(m[2])"
        elseif (m = match(r"^(.*)_Intercept$", s)) !== nothing
            "pop_$(m[1])_beta_pop.1"
        else
            s
        end
        name => stan
    end
end

@stestset "derived formula arrays are direct plate inputs" begin
    builder = @brm begin
        # Reader first: source order is not dependency order.
        @plate for g in eachindex(x)
            mu[g] = rate[g] .* x[g]
            y[g] ~ Normal(mu[g], 1.0)
        end
        rate = exp(log_rate)
        log_rate ~ 1 + (1 | subject)
        effect(log_rate, Intercept) ~ Normal(0.0, 1.0)
        sd(log_rate, subject) ~ Exponential(1.0)
    end
    data = derived_plate_data
    saved = deepcopy(data)
    rk, problem = consumer_problem(builder(data))
    stan = consumer_stan(builder(data), "plate-derived-formula")
    names = coordinate_names(rk.model.layout)
    intercept = coordinate(names, :log_rate_Intercept)
    scale = only(findall(n -> occursin(".tau.", string(n)), names))
    z = findall(n -> occursin(".z.", string(n)), names)
    @test length(names) == 4
    @test length(z) == 2
    function oracle(u, data)
        sigma = exp(u[scale])
        levels = sort(unique(data.subject))
        rate = [exp(u[intercept] + sigma * u[z[findfirst(==(s), levels)]])
            for s in data.subject]
        logpdf(Normal(), u[intercept]) + logpdf(Exponential(), sigma) + u[scale] +
            sum(logpdf.(Normal(), u[z])) +
            sum(sum(logpdf.(Normal.(rate[g] .* data.x[g], 1.0), data.y[g]))
                for g in eachindex(data.x))
    end
    points = (zeros(4), [-0.3, 0.2, 0.4, -0.1], [0.1, -0.4, -0.2, 0.3])
    for u in points
        check_consumer_point(problem, u, u -> oracle(u, data))
        check_consumer_stan(problem, stan, derived_plate_mapping(names), rk, u)
    end
    # Replay changes both subject order and the lengths inside each cell.
    fresh = (; subject=["a", "b"], x=[[0.5], [1.0, 2.0, 4.0]],
        y=[[0.1], [0.2, 0.4, 0.8]])
    replay = reprocess(rk, fresh)
    replay_problem = rk_logdensity_problem(replay;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in points
        check_consumer_point(replay_problem, u, u -> oracle(u, fresh))
    end
    @test isequal(data, saved)
end

@stestset "derived arrays own their axes and can supply the plate range" begin
    # A sampled scalar is shared; a declared vector supplies its own size.
    # A gather and concatenation deliberately change that size, so following
    # the assignment's dependencies to infer its cell axis would be wrong.
    cases = (
        ("chain", :(rate = exp(level)), :(result = 2 * rate), [1, 2, 3], 2.0),
        ("gather", :(rate = exp(level)), :(result = rate[pos]), [3, 1], 1.0),
        ("concat", :(rate = exp(level)), :(result = vcat(rate, rate)),
            [1, 2, 3, 1, 2, 3], 1.0),
    )
    for (label, first_assignment, second_assignment, positions, multiplier) in cases
        builder = brm(quote
            @plate for g in eachindex(result)
                mu[g] = result[g] + shift
            end
            y ~ Normal(mu, 0.7)
            $second_assignment
            $first_assignment
            level[1:3] .~ Normal.(0.0, 0.8)
            shift ~ Normal(0.0, 0.5)
        end; mod=@__MODULE__)
        data = (; pos=[3, 1], y=collect(range(-0.2, 0.4; length=length(positions))))
        brmi = builder(data)
        rk, problem = consumer_problem(brmi)
        stan = consumer_stan(brmi, "plate-derived-" * label)
        names = coordinate_names(rk.model.layout)
        li = [coordinate(names, Symbol("level.", j)) for j in 1:3]
        si = coordinate(names, :shift)
        oracle(u) = sum(logpdf.(Normal(0.0, 0.8), u[li])) +
            logpdf(Normal(0.0, 0.5), u[si]) +
            sum(logpdf.(Normal.(multiplier .* exp.(u[li[positions]]) .+ u[si], 0.7), data.y))
        for u in (zeros(4), [0.1, -0.2, 0.3, 0.2])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, derived_plate_mapping(names), rk, u)
        end
    end
end
