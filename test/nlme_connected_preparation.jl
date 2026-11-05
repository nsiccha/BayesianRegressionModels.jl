# Preparation-only scaling receipt on actual BRM/RK structural supports.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Statistics
include(joinpath(@__DIR__, "nlme_connected_fixture.jl"))

for n in (8, 32, 128, 512)
    left = collect(1:n)
    right = [isodd(i) ? i + 1 : i for i in left]
    rk = RKBRMI(connected_model(connected_data(n, left, right)))
    view = brm_nlme_view(rk; unit_partition=:connected)
    @test length(view.unit_members) == n ÷ 2
    @test all(members -> length(members) == 2, view.unit_members)
    timings = [@timed brm_nlme_view(rk; unit_partition=:connected) for _ in 1:3]
    println((; subjects=n, units=length(view.levels),
        component_sizes=unique(length.(view.unit_members)),
        median_seconds=median(t.time for t in timings),
        median_bytes=median(t.bytes for t in timings)))
end
