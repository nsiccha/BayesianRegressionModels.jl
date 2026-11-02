using Test, LinearAlgebra
include("streaming_oracles.jl")
const SO = StreamingOracles

@testset "Gaussian window oracle counts the prior once" begin
    # Fixed dimension two: the dense algebra here is an independent oracle,
    # not a data-sized covariance or a metric passed to any sampler.
    phase = 2pi .* (0:31) ./ 32
    x, z = cos.(phase), sin.(phase)
    data = (; x1=(x .+ 0.02z)./sqrt(2), x2=(x .- 0.02z)./sqrt(2),
        y=0.6x .- 0.006z .+ 0.03cos.(2phase))
    design = hcat(data.x1, data.x2)
    precision = Matrix{Float64}(I, 2, 2) + 100design'design
    h = 100design'data.y
    mean = precision\h
    covariance = inv(precision)
    log_evidence = -length(data.y)*log(2pi*0.01)/2 - logdet(precision)/2 -
        (100sum(abs2, data.y)-dot(h, mean))/2
    for width in (1, 2, 4, 8, 16, 32)
        summary = SO.gaussian_prior()
        for first in 1:width:32
            rows = first:first+width-1
            window = (; (name => values[rows] for (name, values) in pairs(data))...)
            summary = SO.gaussian_window(summary, window)
        end
        @test collect(summary.mean) ≈ mean atol=1e-11
        @test collect(summary.covariance) ≈ [covariance[1,1], covariance[1,2], covariance[2,2]] atol=1e-11
        @test summary.log_evidence ≈ log_evidence atol=1e-10
        @test summary.observations == 32
    end
    previous = SO.gaussian_window(SO.gaussian_prior(), data)
    empty_window = (; x1=Float64[], x2=Float64[], y=Float64[])
    @test SO.gaussian_window(previous, empty_window) == previous
end

@testset "Static-parameter Kalman oracle agrees with conditional integration" begin
    ys = Union{Float64,Missing}[0.4, -0.3, missing, 0.7, 1.0, 0.2, 0.8]
    summary = SO.kalman_prior()
    grid = range(-9.0, 9.0; length=12001)
    for i in eachindex(ys)
        previous = summary
        summary = SO.kalman_step(summary, ys[i])
        logweights = [SO.kalman_loglikelihood(@view(ys[1:i]), level) -
            (log(2pi)+level^2)/2 for level in grid]
        peak = maximum(logweights)
        weights = exp.(logweights .- peak)
        integral = sum(weights)*step(grid)
        weights ./= sum(weights)
        mean = sum(grid .* weights)
        variance = sum((grid .- mean).^2 .* weights)
        @test summary.mean[1] ≈ mean atol=1e-10
        @test summary.covariance[1] ≈ variance atol=1e-10
        @test summary.log_evidence ≈ log(integral)+peak atol=1e-10
        @test summary.log_evidence-previous.log_evidence ≈ summary.predictive_logscore atol=1e-12
        if ismissing(ys[i])
            @test summary.predictive_logscore == 0
            @test summary.mean[1] == previous.mean[1]
            @test summary.covariance[1] == previous.covariance[1]
        end
    end
    @test summary.covariance[1] < 1
end
