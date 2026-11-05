include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module ComponentDirectRows
using BayesianRegressionModels

@brm function rows(x, shared)
    @plate for j in eachindex(x)
        z ~ Normal(0, 1)
        value[j] = [x[j] + z, 2x[j] + 3z] .* sum(shared)
    end
    return value
end

@brm function nested(x, shared)
    inner ~ rows(x, shared)
    return inner
end

function builder(component; gather=false, anchor=:y)
    read = gather ? :(field[row[i]]) : :(field[i])
    brm(quote
        field ~ $component(x, shared)
        @plate for i in eachindex($anchor)
            total[i] = sum($read)
            y[i] ~ Normal(total[i], 0.3)
        end
    end; mod=@__MODULE__)
end
end

@stestset "direct component reads use returned cells, not shared inputs" begin
    data = (; x=[-0.4, 0.2, 0.7, 1.1], shared=reshape(collect(0.1:0.1:0.6), 2, 3),
        row=[1, 2, 3, 4], y=[0.2, -0.1, 0.3, -0.2])
    for component in (:rows, :nested), gather in (false, true)
        @testset "$component gather=$gather" begin
            model = ComponentDirectRows.builder(component; gather)(data)
            rk, problem = consumer_problem(model)
            u = zeros(rk.model.layout.total)
            @test only(rk_draw_values(rk, reshape(u, :, 1), (:total,))[:total]) ≈
                3 .* data.x .* sum(data.shared)
            sb = SBBRMI(model; mod=ComponentDirectRows)
            stan = consumer_stan(model, "direct-component-$component-$gather";
                mod=ComponentDirectRows)
            mapping = [pair.rk => pair.stan for pair in
                brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model)).pairs]
            names = coordinate_names(rk.model.layout)
            path = component === :nested ? "field.inner.z." : "field.z."
            draws = [coordinate(names, Symbol(path, i)) for i in eachindex(data.x)]
            function oracle(point)
                means = (3 .* data.x .+ 4 .* point[draws]) .* sum(data.shared)
                sum(logpdf.(Normal(), point[draws])) +
                    sum(logpdf.(Normal.(means, 0.3), data.y))
            end
            for point in (u, fill(0.13, length(u)), collect(range(-0.2, 0.3; length=length(u))))
                check_consumer_point(problem, point, oracle)
                check_consumer_stan(problem, stan, mapping, rk, point)
            end
            query = prepare_cell_query(rk.model, rk_bound_plan(rk), :y; locality=:error)
            supports = [ReactiveKernels.partition_footprint(query.kernel, u, i).unconstrained
                for i in eachindex(data.y)]
            @test all(i -> supports[i] !== :whole && Set(supports[i]) == Set([draws[i]]),
                eachindex(data.y))
        end
    end
end

@stestset "a returned component value can supply the only plate extent" begin
    builder = brm(quote
        field ~ nested(x, shared)
        @plate for i in eachindex(field)
            total[i] = sum(field[i])
        end
        observed ~ Normal(sum(total), 0.3)
    end; mod=ComponentDirectRows)
    for n in (3, 5)
        data = (; x=collect(range(-0.2, 0.5; length=n)),
            shared=reshape(collect(0.1:0.1:0.6), 2, 3), observed=0.1)
        rk, problem = consumer_problem(builder(data))
        u = zeros(LogDensityProblems.dimension(problem))
        @test only(rk_draw_values(rk, reshape(u, :, 1), (:total,))[:total]) ≈
            3 .* data.x .* sum(data.shared)
        sb = SBBRMI(builder(data); mod=ComponentDirectRows)
        @test StanBlocks.stanc_check(stan_code(sb); warn_pedantic=false).ok
    end
end
