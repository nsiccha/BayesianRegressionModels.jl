using Test
using BayesianRegressionModels
using Distributions
using LogDensityProblems
using StanBlocks

const BRM = BayesianRegressionModels
const BridgeStan = StanBlocks.BridgeStan

@testset "Dirichlet concentration calls retain sampled hyperparameters" begin
    data = (; category=[1, 3, 2], y=[0.1, 0.7, 0.4])
    symmetric = @brm begin
        concentration ~ Exponential(2.0)
        share ~ Dirichlet(3, concentration)
        y ~ Normal(share[category], 0.2)
    end
    heterogeneous = @brm begin
        concentration ~ Exponential(2.0)
        share ~ Dirichlet([concentration, 1.0, concentration + 1.0])
        y ~ Normal(share[category], 0.2)
    end
    alphas = (c -> fill(c, 3), c -> [c, 1.0, c + 1.0])
    cache = mkpath(joinpath(tempdir(), "brm-hierarchical-simplex"))
    for (builder, alpha) in zip((symmetric, heterogeneous), alphas)
        code = BRM.stan_code(SBBRMI(builder(data)))
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test occursin("simplex[", code)
        descriptor = brm_descriptor(builder, data; mod=@__MODULE__, highlights=())
        problem = brm_execute(descriptor, :instantiate;
            path=joinpath(cache, string(descriptor.id) * ".stan"))
        # One concentration and a 3-simplex (two free coordinates).
        @test LogDensityProblems.dimension(problem) == 3
        for raw in (zeros(3), [0.4, -0.3, 0.2])
            names = BridgeStan.param_names(problem.model)
            values = BridgeStan.param_constrain(problem.model, raw)
            physical = Dict(zip(names, values))
            c = physical["concentration"]
            share = [physical["share.$k"] for k in 1:3]
            @test sum(share) ≈ 1
            # The sampled concentration reaches the Dirichlet at every draw.
            expected = logpdf(Exponential(2.0), c) + logpdf(Dirichlet(alpha(c)), share) +
                sum(logpdf.(Normal.(share[data.category], 0.2), data.y))
            @test BridgeStan.log_density(problem.model, raw; propto=false, jacobian=false) ≈
                expected atol=1e-10
        end
    end
end
