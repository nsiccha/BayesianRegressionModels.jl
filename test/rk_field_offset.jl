# Declared scalar plate outputs and explicitly indexed helper outputs in
# coefficient-one offsets. Run: julia --project=test test/rk_field_offset.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const FIELD_OFFSET_DATA = (;
    x=[-0.6, 0.2, 0.8, -0.3, 0.5],
    fixed=[0.1, -0.2, 0.3, 0.0, -0.1],
    read_row=[5, 2, 2, 1], y=[0.9, 1.3, 1.1, 0.7])

const FIELD_OFFSET_READER = @brm begin
    log(quantity) ~ 1 + offset(latent) + offset(fixed)
    readout = quantity[read_row]
    y ~ Normal(readout, 0.6)
    effect(quantity, Intercept) ~ Normal(0.1, 0.9)
    theta ~ Normal(0, 1)
end

const FIELD_OFFSET_PRODUCERS = (
    last=@brm(begin
        @plate for i in eachindex(x)
            latent[i] = theta * x[i]
        end
    end),
    first=@brm(begin
        @plate for i in eachindex(x)
            latent[i] = theta * x[i]
            auxiliary[i] = latent[i] + 1
        end
    end),
    scaled=@brm(begin
        @plate for i in eachindex(x)
            half_latent[i] = 0.5 * theta * x[i]
        end
        latent = 2 * half_latent
    end),
)

function field_offset_point(rk, theta, level)
    brm_start(rk, (; theta, quantity=(; Intercept=level)); base=:prior_median)
end

@stestset "a declared scalar field is a coefficient-one offset" begin
    for (label, field) in pairs(FIELD_OFFSET_PRODUCERS)
        # Reader-first composition must retain the producer and its parameters.
        builder = merge(FIELD_OFFSET_READER, field)
        data = FIELD_OFFSET_DATA
        saved = deepcopy(data)
        brmi = builder(data)
        rk, problem = consumer_problem(brmi)
        names = coordinate_names(rk.model.layout)
        @test length(names) == 2
        itheta = coordinate(names, :theta)
        ilevel = coordinate(names, :quantity_Intercept)
        oracle(u) = begin
            quantity = exp.(u[ilevel] .+ u[itheta] .* data.x .+ data.fixed)
            logpdf(Normal(), u[itheta]) + logpdf(Normal(0.1, 0.9), u[ilevel]) +
                sum(logpdf.(Normal.(quantity[data.read_row], 0.6), data.y))
        end
        stan = consumer_stan(brmi, "rk-field-offset-$label")
        mapping = [:theta => "theta", :quantity_Intercept => "pop_log_quantity_beta_pop.1"]
        for (theta, level) in ((-0.2, 0.1), (0.4, -0.3), (0.0, 0.5))
            u = field_offset_point(rk, theta, level)
            expected = exp.(level .+ theta .* data.x .+ data.fixed)
            values = rk_draw_values(rk, reshape(u, :, 1), (:latent, :quantity, :readout))
            @test only(values[:latent]) ≈ theta .* data.x
            @test only(values[:quantity]) ≈ expected
            @test only(values[:readout]) ≈ expected[data.read_row]
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, rk, u)
        end
        @test isequal(data, saved)
    end
end

field_offset_values(theta, x) = reverse(theta .* x)

@stestset "an explicit gather supplies an opaque helper's row axis" begin
    reader = @brm begin
        latent = whole[position]
        log(quantity) ~ 1 + offset(latent) + offset(fixed)
        y ~ Normal(quantity, 0.6)
        effect(quantity, Intercept) ~ Normal(0.1, 0.9)
        theta ~ Normal(0, 1)
    end
    builder = merge(@brm(begin
        whole = field_offset_values(theta, x)
    end), reader)
    # Native RK reads the ordinary Julia helper. The independent Stan twin
    # expresses that helper's body with Stan builtins, without a native provider.
    stan_builder = merge(@brm(begin
        whole = reverse(theta * x)
    end), reader)
    # Unequal lengths and repeated/reordered positions make the row declaration
    # observable: the helper's input axis cannot stand in for the gather's.
    data = (; x=[-0.6, 0.2, 0.8, -0.3, 0.5, 0.9],
        position=[6, 2, 2, 1], fixed=[0.1, -0.2, 0.3, 0.0],
        y=[0.9, 1.3, 1.1, 0.7])
    saved = deepcopy(data)
    brmi = builder(data)
    rk, problem = consumer_problem(brmi)
    names = coordinate_names(rk.model.layout)
    @test length(names) == 2
    itheta = coordinate(names, :theta)
    ilevel = coordinate(names, :quantity_Intercept)
    oracle(u) = begin
        values = reverse(u[itheta] .* data.x)[data.position]
        quantity = exp.(u[ilevel] .+ values .+ data.fixed)
        logpdf(Normal(), u[itheta]) + logpdf(Normal(0.1, 0.9), u[ilevel]) +
            sum(logpdf.(Normal.(quantity, 0.6), data.y))
    end
    stan = consumer_stan(stan_builder(data), "rk-field-offset-gather")
    mapping = [:theta => "theta", :quantity_Intercept => "pop_log_quantity_beta_pop.1"]
    for (theta, level) in ((-0.2, 0.1), (0.4, -0.3), (0.0, 0.5))
        u = field_offset_point(rk, theta, level)
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, rk, u)
        values = rk_draw_values(rk, reshape(u, :, 1), (:whole, :latent, :quantity))
        expected = reverse(theta .* data.x)[data.position]
        @test only(values[:whole]) ≈ reverse(theta .* data.x)
        @test only(values[:latent]) ≈ expected
        @test only(values[:quantity]) ≈ exp.(level .+ expected .+ data.fixed)
    end
    @test isequal(data, saved)
end

@stestset "offset field reads stay local to the selected row" begin
    builder = merge(FIELD_OFFSET_READER, FIELD_OFFSET_PRODUCERS.last)
    costs = []
    for n in (16, 1024)
        data = (; x=collect(range(-0.6, 0.7; length=n)), fixed=fill(0.1, n),
            read_row=reverse(collect(1:n)), y=fill(0.9, n))
        rk = RKBRMI(builder(data))
        u = field_offset_point(rk, 0.2, -0.1)
        names = coordinate_names(rk.model.layout)
        itheta = coordinate(names, :theta)
        ilevel = coordinate(names, :quantity_Intercept)
        query = prepare_cell_query(rk.model, rk_bound_plan(rk), :y; locality=:error)
        sampler = prepare_cell_sampler(rk.model, rk_bound_plan(rk), :y, u;
            backend=AutoEnzyme(; mode=Enzyme.Reverse), locality=:error)
        gradient = similar(u)
        for i in (n, 2, 2, 1)
            row = data.read_row[i]
            quantity = exp(u[ilevel] + u[itheta] * data.x[row] + data.fixed[row])
            expected = logpdf(Normal(quantity, 0.6), data.y[i])
            @test query(u, i) ≈ expected
            value, _ = cell_value_and_gradient!(sampler, gradient, u, i)
            @test value ≈ expected
            derivative = (data.y[i] - quantity) * quantity / 0.6^2
            @test gradient[ilevel] ≈ derivative
            @test gradient[itheta] ≈ derivative * data.x[row]
        end
        for _ in 1:3
            query(u, 2)
            cell_value_and_gradient!(sampler, gradient, u, 2)
        end
        bytes = (
            value=minimum(@allocated(query(u, 2)) for _ in 1:3),
            gradient=minimum(@allocated(cell_value_and_gradient!(sampler, gradient, u, 2))
                for _ in 1:3))
        push!(costs, bytes)
        println("FIELD_OFFSET_LOCALITY n=$n bytes=$bytes")
    end
    @test costs[2].value <= costs[1].value + 1024
    @test costs[2].gradient <= costs[1].gradient + 1024
end

@stestset "a plate offset does not flatten an unrelated cell axis" begin
    # There are three cells and three flattened values, but they are different
    # axes. A length-only test would silently accept this accidental equality.
    # Use an identity link: a log link can reject the composed predictor earlier,
    # before the native kernel checks its cells during density evaluation.
    data = (; x=[[0.2, 0.4], Float64[], [0.8]], fixed=zeros(3), y=ones(3))
    for expression in (:raw_latent, :(2 * raw_latent))
        builder = brm(quote
            theta ~ Normal(0, 1)
            @plate for i in eachindex(x)
                raw_latent[i] = theta .* x[i]
            end
            latent = $expression
            quantity ~ 1 + offset(latent) + offset(fixed)
            effect(quantity, Intercept) ~ Normal(0.1, 0.9)
            y ~ Normal(quantity, 0.6)
        end; mod=@__MODULE__)
        @test_throws r"offset of a plate output requires one real scalar per cell" begin
            rk = RKBRMI(builder(data))
            problem = rk_logdensity_problem(rk;
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            u = field_offset_point(rk, 0.2, 0.1)
            LogDensityProblems.logdensity(problem, u)
        end
    end
end
