# test/family_surfaces.jl — executable contracts for custom likelihood surfaces.
#
# Run on a capable host:
#   julia --startup-file=no --project=. test/family_surfaces.jl

using Test
using BayesianRegressionModels
using Distributions: BetaBinomial, LocationScale, TDist
using LogExpFunctions: logit
using StanBlocks

df = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    prog=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    math=[41.0, 48.0, 52.0, 57.0, 63.0, 69.0],
    y_zip=[0, 1, 0, 2, 3, 0],
    y_nb=[0, 1, 2, 4, 3, 6],
    y_t=[-0.8, -0.2, 0.1, 0.7, 1.0, 1.4],
    trials=[8, 8, 10, 10, 12, 12],
    y_bb_shapes=[1, 2, 3, 4, 5, 7],
    y_bb_mean_precision=[0, 1, 4, 6, 8, 11],
    y_cat=[20, 10, 30, 20, 30, 10],
)

family_builder = @brm begin
    log(lambda) ~ 1 + x
    y_zip ~ ZeroInflatedPoisson(lambda, 0.25)

    # Exact catalogue shape for bambi:negative_binomial_interaction and
    # bambi:plot_pred_nb: the interaction includes a transformed operand.
    log(mu) ~ 0 + prog + zscale(math) + prog & zscale(math)
    log(phi) ~ 1
    y_nb ~ NegativeBinomial2(mu, phi)

    loc ~ 1 + x
    log(scale) ~ 1
    y_t ~ LocationScale(loc, scale, TDist(4.0))

    y_bb_shapes ~ BetaBinomial(trials, 2.0, 4.0)

    logit(bb_mean) ~ 1 + x
    log(bb_precision) ~ 1
    y_bb_mean_precision ~ BetaBinomial2(trials, bb_mean, bb_precision)

    cat_eta2 ~ 1 + x
    cat_eta3 ~ 1 + prog
    y_cat ~ CategoricalLogit(cat_eta2, cat_eta3)
end

@testset "SBBRMI lowers density, pointwise log-lik and RNG paths" begin
    plan = generative_plan(family_builder, df; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(plan)

    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("zero_inflated_poisson(", code)
    @test occursin("neg_binomial_2(", code)
    @test occursin("student_t(", code)
    @test occursin("beta_binomial(", code)
    @test occursin("bb_mean .* bb_precision", code)
    @test occursin("(1 - bb_mean) .* bb_precision", code)
    @test occursin("brm_categorical_logit(", code)
    @test occursin("categorical_logit_lpmf(", code)
    @test occursin("y_cat_categorical_logits", code)
    @test occursin("int_prog_x_zscale_math", code)

    sb = SBBRMI(family_builder(df); mod=@__MODULE__)
    math_mu = sum(df.math) / length(df.math)
    math_sd = sqrt(sum((x - math_mu)^2 for x in df.math) / length(df.math))
    expected_interaction = df.prog .* ((df.math .- math_mu) ./ math_sd)
    @test sb.data[:int_prog_x_zscale_math] ≈ expected_interaction
    @test sb.data[:y_cat] == [2, 1, 3, 2, 3, 1]
    @test sb.preproc[:y_cat].const_.levels == [10, 20, 30]

    new_df = merge(df, (;
        prog=[2.0, 1.0, 0.0, 0.0, 1.0, 2.0],
        math=[45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
    ))
    replayed = reprocess(sb, new_df)
    expected_replayed = new_df.prog .* ((new_df.math .- math_mu) ./ math_sd)
    @test replayed.data[:int_prog_x_zscale_math] ≈ expected_replayed
    @test replayed.data[:y_cat] == sb.data[:y_cat]
    unseen_cat = merge(new_df, (; y_cat=[10, 20, 30, 40, 10, 20]))
    @test_throws "not a training level" reprocess(sb, unseen_cat)

    for target in (:y_zip, :y_nb, :y_t, :y_bb_shapes,
                   :y_bb_mean_precision, :y_cat)
        declaration = only(d for d in plan.declarations if d.target === target)
        @test declaration.role === :observation
        @test !isnothing(declaration.draw)
        @test occursin(string(declaration.draw), code)
        @test occursin(string(target, "_likelihood"), code)
    end

    families = Dict(d.target => d.family for d in plan.declarations
                    if d.role === :observation)
    @test families[:y_zip] === :zero_inflated_poisson
    @test families[:y_nb] === :neg_binomial_2
    @test families[:y_t] === :student_t
    @test families[:y_bb_shapes] === :beta_binomial
    @test families[:y_bb_mean_precision] === :beta_binomial
    @test families[:y_cat] === :brm_categorical_logit
end
