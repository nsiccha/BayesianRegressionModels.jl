# Selecting several vector-valued cells preserves a collection of cells at
# ordinary helper boundaries, including repeated and reordered selections.
# Run: julia --startup-file=no --project=test test/plate_selected_collection.jl
using Test, BayesianRegressionModels, StanBlocks
import BridgeStan
using Distributions: Normal, logpdf
const BRM = BayesianRegressionModels

StanBlocks.@deffun selected_cell_weighted(rows, weights::vector[n], channel::int)::real = begin
    value = 0.0
    for t in 1:n
        value += weights[t] * rows[t][channel]
    end
    value
end

function selected_collection_builder(; side=false)
    outputs = Any[:(cells[i] = [a + axis[i], 2.0 * a - 0.5 * axis[i]]),
                  :(spare[i] = axis[i])]
    side || reverse!(outputs)
    brm(quote
        a ~ Normal(0, 1)
        @plate for j in eachindex(groups)
            location[j] = selected_cell_weighted(cells[groups[j]], weights[j], channel[j])
            y[j] ~ Normal(location[j], 0.7)
        end
        @plate for i in eachindex(axis)
            $(outputs...)
        end
    end; mod=@__MODULE__)
end

function selected_collection_problem(sb, tag)
    BRM.stan_instantiate(sb; path=joinpath(mktempdir(), tag * ".stan"))
end

@testset "selected vector cells remain indexable in ordinary helpers" begin
    inputs = (; axis=[0.1, 0.7, -0.2, 1.4, -0.6],
        groups=[[3, 1], [2], [1, 3, 3]], weights=[[0.2, 0.8], [1.0], [0.1, 0.4, 0.5]],
        channel=[1, 2, 2])
    data = (; inputs..., y=[0.2, -0.3, 0.5])
    before = deepcopy(data)
    slope = [sum(inputs.weights[j]) * (inputs.channel[j] == 1 ? 1 : 2) for j in 1:3]
    intercept = [sum(inputs.weights[j] .* inputs.axis[inputs.groups[j]]) *
        (inputs.channel[j] == 1 ? 1 : -0.5) for j in 1:3]
    for side in (false, true)
        builder = selected_collection_builder(; side)
        sb = SBBRMI(builder(data); mod=@__MODULE__)
        @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
        problem = selected_collection_problem(sb, "selected-collection-$side")
        @test BridgeStan.param_unc_num(problem.model) == 1
        for a in (-0.3, 0.0, 0.6)
            means = slope .* a .+ intercept
            expected = logpdf(Normal(0, 1), a) + sum(logpdf.(Normal.(means, 0.7), data.y))
            expected_gradient = -a + sum(slope .* (data.y .- means)) / 0.7^2
            value, gradient = BridgeStan.log_density_gradient(problem.model, [a]; propto=false)
            @test value ≈ expected atol=2e-11 rtol=2e-11
            @test only(gradient) ≈ expected_gradient atol=2e-11 rtol=2e-11
        end
        prior = SBBRMI(builder(inputs); mod=@__MODULE__)
        @test StanBlocks.stanc_check(BRM.stan_code(prior); warn_pedantic=false).ok
        prior_problem = selected_collection_problem(prior, "selected-collection-prior-$side")
        @test BridgeStan.param_unc_num(prior_problem.model) == 0
        names = BridgeStan.param_names(prior_problem.model; include_tp=true, include_gq=true)
        name = String(brm_output(brm_descriptor(prior), :location).name)
        indices = [only(findall(==(name * "." * string(i)), names)) for i in 1:3]
        a_index = only(findall(==("a"), names))
        rng = BridgeStan.StanRNG(prior_problem.model, 49)
        for _ in 1:3
            draw = BridgeStan.param_constrain(prior_problem.model, Float64[];
                include_tp=true, include_gq=true, rng)
            @test all(isfinite, draw)
            @test draw[indices] ≈ slope .* draw[a_index] .+ intercept atol=2e-12 rtol=2e-12
        end
        @test isequal(data, before)
    end
end
