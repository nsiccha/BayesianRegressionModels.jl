# A unit's evaluation costs its footprint and prior support, whatever the number of
# globals, and its compact gradients agree with the dense interface.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, Random
include("nlme_svi_owned_fixture.jl")
const BRM = BayesianRegressionModels

prepared(n; shared) = BRM._brm_svi_target(brm_nlme_model(
    RKBRMI(svi_owned_model(svi_owned_data(collect(1:n); shared)));
    ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)))

@testset "compact unit evaluation agrees with the dense interface" begin
    t = prepared(8; shared=true)
    D = length(t.globals)
    rng = Xoshiro(11)
    g = t.m.unit[t.globals] .+ 0.2 .* randn(rng, D)
    for i in eachindex(t.units)
        z = t.m.unit[t.units[i]] .+ 0.2 .* randn(rng, length(t.units[i]))
        before = copy(t.plan.point)
        gg, gz = fill(NaN, D), fill(NaN, length(z))
        dense = BRM._brm_svi_unit_term!(gg, gz, t, i, g, z)
        lg, lz = zeros(D), zeros(length(z))
        compact = BRM._brm_svi_unit_local!(lg, lz, t, i, g, z, t.plan, first(t.work))
        @test compact == dense && lg == gg && lz == gz
        # Nothing outside what the unit declares it touched was written.
        untouched = setdiff(eachindex(t.globals), t.touched[i])
        @test all(iszero, lg[untouched])
        # The base point is restored exactly, for the next unit and the next worker.
        @test t.plan.point == before
    end
end

@testset "a unit's work does not grow with the globals it does not read" begin
    # Each odd `z` beyond the shared first has no reader, so it is a global, and the
    # unit count sets how many: the same unit shape against several hundred globals.
    measured = map((6, 192)) do n
        t = prepared(n; shared=true)
        D = length(t.globals)
        g, z = t.m.unit[t.globals], t.m.unit[t.units[1]]
        gg, gz = zeros(D), zeros(length(z))
        local_call() = BRM._brm_svi_unit_local!(gg, gz, t, 1, g, z, t.plan, first(t.work))
        local_call(); local_call()
        (; D, bytes=@allocated(local_call()), footprint=length(t.plan.footprints[1]),
           touched=length(t.touched[1]))
    end
    small, large = measured
    @test large.D > 10small.D
    @test large.footprint == small.footprint
    @test large.touched == small.touched
    @test large.bytes == small.bytes
end
