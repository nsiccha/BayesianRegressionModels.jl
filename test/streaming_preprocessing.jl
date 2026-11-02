using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Statistics, Serialization

module StreamingPreprocessingModel
using BayesianRegressionModels
build(data) = @brm data begin
    mu ~ 0 + zscale(x)
    effect(mu, :) ~ Normal(0, 1)
    y ~ Normal(mu, 0.7)
end
end

@testset "Streaming windows use frozen preprocessing without reading training rows" begin
    point = [0.4]
    window = (; x=[-1.2, 0.8, 1.1, 3.7], y=[-0.1, 0.3, 0.7, 1.0])
    summary_bytes = Int[]
    for n in (32, 512, 4096)
        training = (; x=collect(range(-3.0, 5.0; length=n)), y=zeros(n))
        center, scale = mean(training.x), std(training.x)
        rk = RKBRMI(StreamingPreprocessingModel.build(training))
        saved = brm_fitted_state(rk)
        # The frozen artifact must survive serialization independently of the
        # original backend and the arrays from which its anchors were fitted.
        io = IOBuffer()
        serialize(io, saved)
        push!(summary_bytes, position(io))
        seekstart(io)
        saved = deserialize(io)
        previous = reprocess(rk, window)
        expected = ReactiveKernelsPPL.prepare_query(previous.model,
            rk_bound_plan(previous), :likelihood)(point)
        analytic = sum(zip(window.x, window.y)) do (x, y)
            residual = y-point[1]*(x-center)/scale
            -(log(2pi*0.7^2)+residual^2/0.7^2)/2
        end
        @test expected ≈ analytic atol=1e-11

        fill!(training.x, NaN)
        fill!(training.y, NaN)
        for _ in 1:3
            replay = RKBRMI(StreamingPreprocessingModel.build(window), saved)
            density = ReactiveKernelsPPL.prepare_query(replay.model,
                rk_bound_plan(replay), :likelihood)
            @test density(point) ≈ analytic atol=1e-11
            # Fit-role columns retain only an absent-input shape descriptor,
            # never the training values. Match the existing fitted-state test.
            @test occursin("Absent", string(nameof(typeof(rk_bound_data(replay).x_fit))))
            @test rk_bound_data(replay).x_fit_mean ≈ center
            @test rk_bound_data(replay).x_fit_sd ≈ scale
        end
    end
    @test maximum(summary_bytes)-minimum(summary_bytes) <= 16
end
