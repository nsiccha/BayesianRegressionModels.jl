# Run: julia --startup-file=no --project=test test/description_comparisons.jl
#
# Comparisons are described (snag description-cove-e12e9953). `==`, `!=`, `<`,
# `<=`, `>` and `>=` are built-in descriptions, so a model deriving numeric
# indicators the documented way, `bottle = 1.0 * (vessel == 2) + 1.0 *
# (vessel == 3)`, is complete. A comparison used as a value reads as the
# indicator 1{P}, as `factor` dummies do; `|` and `&` joining comparisons (in a
# `@plate` cell, on scalars) read as one indicator of their logical or/and, a
# nested connective parenthesized. A formula's `(1 | g)` keeps its group term.
using Test, BayesianRegressionModels, Distributions

const DATA = let n = 24
    (; vessel=repeat([1, 2, 3, 4], 6), dose=collect(range(0.5, 6; length=n)),
       g=repeat(["a", "b", "c"], 8), y=sin.(1:n))
end
const CELLS = (; subject=[1, 2, 3], vessel=[2, 3, 1], obs_subject=[1, 2, 3, 1, 2, 3],
    y=[0.4, 1.2, 0.6, 0.5, 1.1, 0.7])
describe(builder; kwargs...) =
    brm_description(SBBRMI(builder; mod=@__MODULE__, total_groups=()); kwargs...)
equation(r, lhs) = only(filter(e -> startswith(e, lhs), r.equations))

@testset "model-scope indicators of comparisons are described" begin
    model = @brm DATA begin
        bottle = 1.0 * (vessel == 2) + 1.0 * (vessel == 3)
        large = 1.0 * (vessel == 3)
        other = 1.0 * (vessel != 1)
        window = 1.0 * (dose > 1) * (dose <= 5) + 1.0 * (dose < 0.75) + 1.0 * (dose >= 5.5)
        sigma ~ Exponential(1)
        mu ~ 1 + bottle + large + other + window
        y ~ Normal(mu, sigma)
    end
    r = describe(model)
    @test r.complete
    @test isempty(r.diagnostics)
    @test equation(r, "\\mathrm{bottle}=") == "\\mathrm{bottle}=\\left(\\left(1.0 \\cdot " *
        "\\mathbf 1\\{\\mathrm{vessel} = 2\\}\\right) + \\left(1.0 \\cdot \\mathbf 1\\{\\mathrm{vessel} = 3\\}\\right)\\right)"
    @test equation(r, "\\mathrm{large}=") == "\\mathrm{large}=\\left(1.0 \\cdot \\mathbf 1\\{\\mathrm{vessel} = 3\\}\\right)"
    @test equation(r, "\\mathrm{other}=") == "\\mathrm{other}=\\left(1.0 \\cdot \\mathbf 1\\{\\mathrm{vessel} \\ne 1\\}\\right)"
    window = equation(r, "\\begin{aligned}\\mathrm{window}")
    for indicator in ("\\mathbf 1\\{\\mathrm{dose} > 1\\} \\cdot \\mathbf 1\\{\\mathrm{dose} \\le 5\\}",
            "\\mathbf 1\\{\\mathrm{dose} < 0.75\\}", "\\mathbf 1\\{\\mathrm{dose} \\ge 5.5\\}")
        @test occursin(indicator, window)
    end
    @test !occursin("\\mathrm{==}", brm_description_markdown(r))
    # Operators are standard notation: the German page writes the same equations.
    german = describe(model; texts=description_texts(:de))
    @test german.complete
    @test equation(german, "\\mathrm{large}=") == equation(r, "\\mathrm{large}=")
end

@testset "or and and of comparisons in a @plate cell read as one indicator" begin
    cell(statement) = Core.eval(@__MODULE__, :(@brm CELLS begin
        sigma ~ Exponential(1)
        eta ~ 1 + (1 | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(subject)
            $statement
            loc[i] = eta[i] + ind
            y[obs[i]] ~ Normal(loc[i], sigma)
        end
    end))
    for (statement, expected) in (
            :(ind = 1.0 * ((vessel[i] == 2) | (vessel[i] == 3))) =>
                "{\\mathrm{vessel}}_{i} = 2 \\lor {\\mathrm{vessel}}_{i} = 3",
            :(ind = 1.0 * ((vessel[i] > 1) & (vessel[i] < 3))) =>
                "{\\mathrm{vessel}}_{i} > 1 \\land {\\mathrm{vessel}}_{i} < 3",
            :(ind = 1.0 * (((vessel[i] == 1) | (vessel[i] == 2)) & (vessel[i] != 2))) =>
                "\\left({\\mathrm{vessel}}_{i} = 1 \\lor {\\mathrm{vessel}}_{i} = 2\\right) \\land {\\mathrm{vessel}}_{i} \\ne 2")
        r = describe(cell(statement))
        @test r.complete
        @test equation(r, "\\mathrm{loc.ind}=") == "\\mathrm{loc.ind}=\\left(1.0 \\cdot \\mathbf 1\\{" * expected * "\\}\\right)"
    end
end

@testset "a formula's group bar keeps its group term" begin
    r = describe(@brm DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + dose + (1 | g)
        y ~ Normal(mu, sigma)
    end)
    @test r.complete
    @test equation(r, "\\mathrm{mu}=") == "\\mathrm{mu}=\\beta_{mu,Intercept} + \\beta_{mu,dose}\\,\\mathrm{dose} + " *
        "\\mathbf z_{mu,1,j}^{\\mathsf T}\\mathbf b_{1,g_j}"
end
