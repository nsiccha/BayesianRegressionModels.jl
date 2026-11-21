# One coefficient row of a read-pair field's Vecchia law, as SVI's learned field
# centering reads it from the emitted RK graph: its values and its derivatives
# in the two field lengths, on a small and a large field with the same number
# of parents. The printed cost is a measurement, not an assertion: inside a
# prepared query one reverse call still copies the fixed footprint geometry, so
# a row's derivative cost grows with the field. ReactiveKernels 0.1.351 removed
# that copy for a rule called directly; the prepared-query case is open on the
# Enzyme side (ReactiveKernels snag native-reverse-r-b8de62b9).
include(joinpath(@__DIR__, "fitting_packages.jl"))
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, LinearAlgebra, Random, Statistics
const BRM = BayesianRegressionModels
include("svi_field_fixture.jl")

function check_field_row(stations; parents=12)
    data = station_data(; stations, times=collect(0.0:7.0), reads=4, rows=1)
    rk = RKBRMI(field_model(merge(data, (; nb=10))))
    record = BRM._brm_gmrf_footprint_record(rk, :field)
    names = BRM._rk_layout_coordinate_names(rk)
    lengths = [only(findall(==(Symbol(record.declaration, ".", local_name)), names))
        for local_name in (record.locals.spatial_length, record.locals.temporal_length)]
    base = zeros(length(names))
    hyper = [-1.2, 0.8]
    reader = BRM._rk_svi_conditional_reader(rk, record, base, lengths,
        AutoEnzyme(; mode=Enzyme.Reverse))
    width = length.(reader.layout.inputs)
    # The same number of parents on both fields, so the rows are comparable.
    k = findfirst(==(parents), width)
    @test k !== nothing
    J = reader.coefficients(hyper, k, true)
    @test size(J) == (parents + 1, 1 + length(hyper))
    @test reader.coefficients(hyper, k, false) == J[:, 1:1]
    for j in eachindex(hyper)
        h = zero(hyper)
        h[j] = 1e-5
        central = vec(reader.coefficients(hyper + h, k, false) -
            reader.coefficients(hyper - h, k, false)) / 2e-5
        @test central ≈ J[:, j+1] rtol=2e-6 atol=2e-7
    end
    # A row read only for its scale takes that entry's derivatives alone and
    # marks the others as not taken.
    S = reader.coefficients(hyper, k, true; scale_only=true)
    @test S[:, 1] == J[:, 1]
    @test S[end, :] == J[end, :]
    @test all(isnan, S[1:end-1, 2:end])
    forward = [@timed(reader.coefficients(hyper, k, false)) for _ in 1:10]
    jacobian = [@timed(reader.coefficients(hyper, k, true)) for _ in 1:10]
    scale = [@timed(reader.coefficients(hyper, k, true; scale_only=true)) for _ in 1:10]
    cost = (; stations, states=length(width), packed=length(base), parents,
        value_bytes=Int(median(r.bytes for r in forward)),
        scale_bytes=Int(median(r.bytes for r in scale)),
        jacobian_bytes=Int(median(r.bytes for r in jacobian)),
        jacobian_microseconds=round(median(r.time for r in jacobian) * 1e6; digits=1))
    @info "One coefficient row of the field law" cost...
    cost
end

@testset "one coefficient row and its derivatives at two field sizes" begin
    small = check_field_row(64)
    large = check_field_row(1024)
    @test large.states == 16 * small.states
    # Reading a row's values does not depend on the field's size.
    @test large.value_bytes <= small.value_bytes + 1024
end
