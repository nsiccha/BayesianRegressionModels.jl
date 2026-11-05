include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module ComponentCellReturns
using BayesianRegressionModels

@brm function paired(x, scale)
    @plate for i in eachindex(x)
        z ~ Normal(0, scale)
        value[i] = [x[i] + z, x[i] - z]
    end
    return value
end

@brm function nested(x, scale)
    inner ~ paired(x, scale)
    return inner
end

@brm function aliased(x, scale)
    @plate for i in eachindex(x)
        z ~ Normal(0, scale)
        value[i] = [x[i] + z, x[i] - z]
    end
    alias = value
    return alias
end

const model = @brm begin
    field ~ nested(x, scale)
    scale ~ Exponential(1)
    @plate for i in eachindex(y)
        y[i] ~ Normal(field[row[i]][channel[i]], 0.3)
    end
end
end

@stestset "component cell returns follow direct and aliased outputs" begin
    @testset "$component n=$n" for component in (:paired, :aliased), n in (2, 4)
        data = (; x=collect(range(-0.4, 0.7; length=n)), row=[n, 1, n],
            channel=[2, 1, 1], y=[0.2, -0.1, 0.3])
        builder = brm(quote
            field ~ $component(x, scale)
            scale ~ Exponential(1)
            @plate for i in eachindex(y)
                y[i] ~ Normal(field[row[i]][channel[i]], 0.3)
            end
        end; mod=ComponentCellReturns)
        rk = RKBRMI(builder(data))
        point = zeros(rk.model.layout.total)
        @test only(rk_draw_values(rk, reshape(point, :, 1), (:field,))[:field]) ==
            [[x, x] for x in data.x]
        sb = SBBRMI(builder(data); mod=ComponentCellReturns)
        @test StanBlocks.stanc_check(stan_code(sb); warn_pedantic=false).ok
    end
end

@stestset "nested component returns retain vector cells" begin
    data = (; x=[-0.4, 0.2, 0.7, 1.1], row=[4, 2, 2, 1, 3],
        channel=[2, 1, 2, 1, 2], y=[0.2, -0.1, 0.3, 0.1, -0.2])
    model = ComponentCellReturns.model(data)
    rk, problem = consumer_problem(model)
    u = zeros(LogDensityProblems.dimension(problem))
    values = only(rk_draw_values(rk, reshape(u, :, 1), (:field,))[:field])
    @test values == [[x, x] for x in data.x]
    @test LogDensityProblems.logdensity(problem, u) ≈
        logpdf(Exponential(1), 1) + 4logpdf(Normal(), 0) +
        sum(logpdf.(Normal.(data.x[data.row], 0.3), data.y))
    query = prepare_cell_query(rk.model, rk_bound_plan(rk), :y; locality=:error)
    supports = [ReactiveKernels.partition_footprint(query.kernel, u, i).unconstrained
        for i in eachindex(data.y)]
    @test all(s -> s !== :whole && length(s) == 1, supports)
    @test supports[2] == supports[3]
    @test isempty(intersect(supports[1], supports[2]))
    sb = SBBRMI(model; mod=ComponentCellReturns)
    @test brm_output(brm_descriptor(sb), :value; submodel=(:field, :inner)) !== nothing
    stan = consumer_stan(model, "nested-vector-cell-return"; mod=ComponentCellReturns)
    mapping = [pair.rk => pair.stan for pair in
        brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model)).pairs]
    names = coordinate_names(rk.model.layout)
    scale = coordinate(names, :scale)
    draws = [coordinate(names, Symbol("field.inner.z.", i)) for i in eachindex(data.x)]
    function oracle(point)
        means = data.x[data.row] .+ ifelse.(data.channel .== 1, 1, -1) .* point[draws][data.row]
        logpdf(Exponential(1), exp(point[scale])) + point[scale] +
            sum(logpdf.(Normal(0, exp(point[scale])), point[draws])) +
            sum(logpdf.(Normal.(means, 0.3), data.y))
    end
    for point in (u, collect(range(-0.2, 0.3; length=length(u))))
        check_consumer_point(problem, point, oracle)
        check_consumer_stan(problem, stan, mapping, rk, point)
    end
end
