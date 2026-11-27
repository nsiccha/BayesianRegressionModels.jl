# Replaying an ordinal fit needs its fitted levels, not future response values.
# Run: julia --startup-file=no --project=test test/ordinal_forecast.jl
using Test
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using CategoricalArrays

const BRM = BayesianRegressionModels

const ordinal_forecast_builder = @brm begin
    eta ~ 0 + x
    log(disc) ~ 0 + x
    nu ~ 1 + z
    grade ~ Ordinal(Cumulative(), LogitLink(), eta)
    stage ~ Ordinal(StoppingRatio(), ProbitLink(), eta; per_threshold=(w,))
    rating ~ Ordinal(Cumulative(), CloglogLink(), 0.0; discrimination=disc)
    detected ~ BernoulliLogit(nu)
end
const ordinal_forecast_data = (;
    x=[-1.0, -0.4, 0.2, 0.8, 1.4, 2.0], w=[0., 1., 0., 1., 0., 1.],
    grade=categorical(["low", "mid", "high", "low", "mid", "high"];
        levels=["low", "mid", "high"], ordered=true),
    stage=[1, 2, 3, 2, 1, 3], rating=[1, 2, 3, 1, 2, 3],
    z=[-2., 2.], detected=[0, 1])
const ordinal_forecast_rows = (; x=[-0.5, 0.5, 1.5], w=[1., 0., 1.], z=[0., 1.])
const ordinal_forecast_labels = (; grade=["mid", "mid", "mid"],
    stage=[2, 2, 2], rating=[2, 2, 2], detected=[1, 0])

@testset "ordinal forecasts keep fitted levels and accept no response values" begin
    fit = SBBRMI(ordinal_forecast_builder(ordinal_forecast_data); mod=@__MODULE__)
    original_data = deepcopy(fit.data)
    forecast = reprocess(fit, ordinal_forecast_rows)
    @test forecast.held_out == Set((:grade, :stage, :rating, :detected))
    for response in (:grade, :stage, :rating)
        @test forecast.preproc[response] === fit.preproc[response]
        @test forecast.preproc[response].const_.n_levels == 3
        @test forecast.data[Symbol(response, :_thresholds_length)] == 2
    end
    code = BRM.stan_code(forecast)
    checked = StanBlocks.stanc_check(code; warn_pedantic=false)
    checked.ok || @error "ordinal forecast rejected by stanc" output=checked.output
    @test checked.ok
    descriptor = brm_descriptor(forecast)
    @test :predict in [operation.name for operation in descriptor.operations]
    @test :fit ∉ [operation.name for operation in descriptor.operations]
    for response in (:grade, :stage, :rating, :detected)
        @test !haskey(BRM.stan_data(forecast), response)
        @test !occursin(string(response, "_likelihood"), code)
    end
    # A response on another, shorter axis stays conditioned if supplied.
    partial = reprocess(fit, merge(ordinal_forecast_rows, (; detected=[0, 1])))
    @test partial.held_out == Set((:grade, :stage, :rating))
    @test :fit in [operation.name for operation in brm_descriptor(partial).operations]
    @test occursin("detected_likelihood", BRM.stan_code(partial))
    @test BRM.stan_data(partial)[:detected] == [0, 1]

    # A later replay can restore labels and can drop them again. The fitted
    # vocabulary stays ordered even when the new rows contain only one level.
    labelled_rows = merge(ordinal_forecast_rows, ordinal_forecast_labels)
    control = reprocess(fit, labelled_rows; held_out=(:grade, :stage, :rating, :detected))
    restored = reprocess(forecast, labelled_rows)
    @test BRM.stan_code(restored) == BRM.stan_code(control)
    @test BRM.stan_data(restored) == BRM.stan_data(control)
    @test BRM.stan_code(reprocess(forecast, ordinal_forecast_rows)) == code
    @test BRM.stan_code(reprocess(restored, ordinal_forecast_rows)) == code
    @test_throws r"level" reprocess(forecast,
        merge(labelled_rows, (; grade=["unseen", "mid", "low"])))
    @test isequal(fit.data, original_data)

    # Fitted unconstrained draws feed the forecast directly. Supplying arbitrary
    # valid placeholder classes gives the same predictive draws; the true
    # future classes are not needed even by that older, values-bearing route.
    fit_problem = BRM.stan_instantiate(fit)
    forecast_problem = BRM.stan_instantiate(forecast)
    control_problem = BRM.stan_instantiate(control)
    fit_names = BridgeStan.param_unc_names(fit_problem.model)
    @test BridgeStan.param_unc_names(forecast_problem.model) == fit_names
    @test BridgeStan.param_unc_names(control_problem.model) == fit_names
    q = fill(0.1, LogDensityProblems.dimension(fit_problem))
    for problem in (fit_problem, forecast_problem)
        density, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(density)
        @test all(isfinite, gradient)
    end
    predictions = brm_execute(descriptor, :predict;
        problem=forecast_problem, draws=q, seed=7319)
    placeholders = brm_execute(brm_descriptor(control), :predict;
        problem=control_problem, draws=q, seed=7319)
    for response in (:grade, :stage, :rating)
        key = Symbol(response, :_gen)
        @test length(getproperty(predictions, key)) == length(ordinal_forecast_rows.x)
        @test all(1 .<= getproperty(predictions, key) .<= 3)
        @test getproperty(predictions, key) == getproperty(placeholders, key)
    end
    @test length(predictions.detected_gen) == length(ordinal_forecast_rows.z)
end

@testset "ordinal forecast links, one-level fits, and group redraws" begin
    for structure in (Cumulative, StoppingRatio), link in (LogitLink, ProbitLink, CloglogLink)
        builder = brm(quote
            eta ~ 0 + x
            y ~ Ordinal($structure(), $link(), eta)
        end; mod=@__MODULE__)
        for categories in ([1, 2, 3, 1], [1, 1, 1, 1])
            fit = SBBRMI(builder((; x=[-1., 0., 1., 2.], y=categories)); mod=@__MODULE__)
            forecast = reprocess(fit, (; x=[0.2, 0.7]))
            @test forecast.data[:y_thresholds_length] == maximum(categories) - 1
            @test StanBlocks.stanc_check(BRM.stan_code(forecast); warn_pedantic=false).ok
        end
    end
    # The threshold design supplies a row axis even when eta or discrimination
    # is scalar. Exercise all four family signatures used to infer that axis.
    for eta_arg in (:eta, 0.0), disc_arg in (:disc, 1.0)
        builder = brm(quote
            eta ~ 0 + x
            log(disc) ~ 0 + x
            y ~ Ordinal(StoppingRatio(), LogitLink(), $eta_arg;
                discrimination=$disc_arg, per_threshold=(w,))
        end; mod=@__MODULE__)
        fit = SBBRMI(builder((; x=[-1., 0., 1., 2.], w=[0., 1., 0., 1.],
            y=[1, 2, 3, 1])); mod=@__MODULE__)
        forecast = reprocess(fit, (; x=[0.2, 0.7], w=[0., 1.]))
        @test StanBlocks.stanc_check(BRM.stan_code(forecast); warn_pedantic=false).ok
    end
    builder = @brm begin
        eta ~ 0 + x + (1 | group)
        y ~ Ordinal(Cumulative(), LogitLink(), eta)
    end
    fit = SBBRMI(builder((; x=[-1., 0., 1., 2.], y=[1, 2, 3, 1],
        group=[:a, :a, :b, :b])); mod=@__MODULE__, total_groups=())
    forecast = reprocess(fit, (; x=[0.2, 0.7], group=[:c, :c]); resample_groups=[:group])
    @test forecast.held_out == Set((:y,))
    @test forecast.data[:y_thresholds_length] == 2
    @test StanBlocks.stanc_check(BRM.stan_code(forecast); warn_pedantic=false).ok
end
