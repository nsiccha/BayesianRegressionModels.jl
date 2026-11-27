using Test, BayesianRegressionModels, Distributions

# Independent toy counts: no downstream case-study model or data.
const BB_DESCRIPTION_DATA = (; trials=[3, 8, 5, 11], hits=[1, 4, 2, 7],
    x=[-0.8, 0.2, 0.7, 1.4])
const BB_DESCRIPTION_MODELS = (
    @brm(BB_DESCRIPTION_DATA, begin
        p ~ Beta(2, 3)
        phi ~ Exponential(0.5)
        hits ~ BetaBinomial2(trials, p, phi)
    end),
    @brm(BB_DESCRIPTION_DATA, begin
        eta ~ 1 + x
        phi ~ Exponential(0.5)
        @plate for i in eachindex(hits)
            hits[i] ~ BetaBinomial2(trials[i], inv_logit(eta[i]), phi)
        end
    end),
    @brm(BB_DESCRIPTION_DATA, begin
        p ~ Beta(2, 3)
        phi ~ Exponential(0.5)
        @plate for i in eachindex(hits)
            hits[i] ~ BetaBinomial2(trials[i], p, phi)
        end
    end),
)

@testset "BetaBinomial2 descriptions preserve the prepared model" begin
    for model in BB_DESCRIPTION_MODELS
        sb = SBBRMI(model)
        code, data = stan_code(sb), deepcopy(sb.data)
        d = brm_descriptor(sb)
        en = brm_description(d)
        de = brm_description(d; texts=description_texts(:de))
        for (r, convention, prose) in ((en, "trials,mean,precision", "mean success probability p"),
                                      (de, "Versuche,Mittelwert,Präzision", "Erfolgswahrscheinlichkeit p"))
            @test r.complete
            @test isempty(r.diagnostics)
            @test any(p -> occursin(prose, p) && occursin("pφ", p) &&
                occursin("np(1−p)(n+φ)/(1+φ)", p), r.prose)
            @test any(e -> occursin("\\operatorname{BetaBinomial}_{\\mathrm{" * convention * "}}(", e), r.equations)
            @test !occursin("No scientific description", brm_description_markdown(r))
            laws = filter(c -> c.callable === BetaBinomial2, brm_description_components(r))
            @test length(laws) == 1
            @test length(only(laws).arguments) == 3
        end
        @test en.model_id == de.model_id == d.id
        @test map(p -> p.id, en.priors) == map(p -> p.id, de.priors)
        @test any(p -> p.id == (:parameter, :phi), en.priors)
        @test stan_code(sb) == code
        @test isequal(sb.data, data)
    end
end

@testset "BetaBinomial2 argument order and count moments" begin
    sb = SBBRMI(@brm((; hits=[1, 4, 6]), hits ~ BetaBinomial2(8, 0.35, 6.5)))
    r = brm_description(sb)
    @test r.complete
    @test any(e -> occursin("\\operatorname{BetaBinomial}_{\\mathrm{trials,mean,precision}}(8,0.35,6.5)", e), r.equations)
    # Check the reported moments against the normalized public count law.
    n, p, phi = 8, 0.35, 6.5
    probabilities = pdf.(BetaBinomial2(n, p, phi), 0:n)
    @test sum(probabilities) ≈ 1
    @test sum((0:n) .* probabilities) ≈ n*p
    @test sum(((0:n) .- n*p).^2 .* probabilities) ≈ n*p*(1-p)*(n+phi)/(1+phi)
end
