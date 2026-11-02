using Test, BayesianRegressionModels, StanBlocks, BridgeStan
using WarmupHMC, Enzyme, LinearAlgebra, Random, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
using Distributions: Normal, Uniform

const BRM = BayesianRegressionModels
const AC = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)

data = (; g=repeat(["a", "b", "c"]; inner=3),
        x=repeat([-0.8, 0.1, 0.7]; outer=3),
        y=[-0.4, 0.2, 0.7, -0.3, 0.5, 1.1, -1.0, -0.1, 0.2])

const BOUNDED = @brm begin
    mu ~ 1 + x + (1 + x | g)
    effect(mu, :) ~ Normal(0, 2)
    sd(mu, g) ~ Uniform(0.3, 4.7)
    y ~ Normal(mu, 0.8)
end

fd(f, q; h=1e-6) = [begin
    a, b = copy(q), copy(q)
    a[i] += h; b[i] -= h
    (f(a) - f(b)) / (2h)
end for i in eachindex(q)]

@testset "bounded random-effect scales use the compiled physical frame" begin
    for centered in (false, true)
        sb = SBBRMI(BOUNDED(data); mod=@__MODULE__,
                     centered_groups=centered ? [:g] : Symbol[])
        problem = Base.invokelatest(StanBlocks.stan_instantiate, sb.model)
        names = BridgeStan.param_unc_names(problem.model)
        cnames = BridgeStan.param_names(problem.model)
        block = only(adaptive_centering_blocks(sb, names))
        @test block isa BRM.R2D2AdaptiveCenteringBlock
        d = brm_descriptor(sb)
        raw = 0.3 .* randn(Xoshiro(82), length(names))
        physical = BridgeStan.param_constrain(problem.model, raw)
        outputs = brm_outputs(d; component=:scale)
        scale_output = only(outputs)
        tau = physical[vec(brm_output_coordinates(scale_output, cnames))]
        @test BRM._adaptive_block_taus(raw, block) ≈ tau
        Lout = only(brm_outputs(d; component=:correlation_cholesky))
        L = reshape(physical[vec(brm_output_coordinates(Lout, cnames))], 2, 2)
        C = Diagonal(tau) * L
        @test BRM._adaptive_block_cholesky(raw, block) ≈ C
        # Keep a centered array-of-vectors' effect segment intact; only the
        # scalar scale carrier is reordered here.
        permutation = collect(eachindex(names))
        scale_indices = BRM._brm_element_coordinates(scale_output, names)
        permutation[scale_indices] = reverse(scale_indices)
        shuffled = only(adaptive_centering_blocks(sb, names[permutation]))
        @test BRM._adaptive_block_cholesky(raw[permutation], shuffled) ≈ C
        beta = [only(brm_population_effect_coordinates(d, :mu, names;
                      coefficient=term).coordinates) for term in (:Intercept, :x)]

        for control in (0.0, 0.37, 1.0)
            wrapped = adaptive_centering_problem(sb, problem, AutoEnzyme();
                                                 centeredness=control)
            source = copy(raw)
            expected = copy(raw)
            for g in axes(block.effects, 2)
                z = raw[block.effects[:, g]]
                expected[block.effects[:, g]] = centered ? C * z : z
                for k in 1:2
                    m = sum(C[k, j] * z[j] for j in 1:k-1; init=0.0)
                    location = m + (centered ? 0.0 : raw[beta[k]])
                    source[block.effects[k, g]] = control * location + C[k,k]^control * z[k]
                end
            end
            lj, mapped = WarmupHMC.reparametrizer(wrapped)(source)
            @test mapped ≈ expected
            @test lj ≈ size(block.effects, 2) * (Int(centered)-control) * sum(log, diag(C)) atol=1e-12
            lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, source)
            @test lp ≈ lj + LogDensityProblems.logdensity(problem, expected)
            @test gradient ≈ fd(q -> LogDensityProblems.logdensity(wrapped, q), source) rtol=3e-5 atol=3e-5
        end
        # Centered replay must read the same scale transform as adaptation.
        if centered
            replay = BRM._ranef_centered_frame(block.ranef, names, generative_plan(sb))
            @test BRM._adaptive_block_cov_chol(raw, replay) ≈ C
        end
    end
end

@testset "bounded scale atoms: stable tails and unchanged raw logs" begin
    for lower in (0.0, 0.3), upper in (4.7, Inf), q in (-30.0, -0.4, 0.6, 20.0)
        expected = isfinite(upper) ? lower + (upper-lower) / (1+exp(-q)) : lower+exp(q)
        s = BRM._AdaptiveLogScale(0.0, Tuple{Float64,Int}[],
            Tuple{Float64,Float64,Int}[], BRM._AdaptiveSimplexLog[],
            [(1.0, 1, lower, upper)])
        table = BRM._AdaptiveLogScaleTable([s])
        @test exp(BRM._adaptive_log_scale([q], table, 1)) ≈ expected
        @test BRM._adaptive_scale_coordinates(s) == [1]
    end
    @test BRM._adaptive_log_constrain(-1000., 0., 4.7) ≈ log(4.7)-1000.
    @test BRM._adaptive_log_constrain(-1000., 0., Inf) === -1000.
end
