# Bounded selected-unit work control; timings are observations, not a
# hardware-independent performance threshold.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface
include("nlme_svi_owned_fixture.jl")
const BRM = BayesianRegressionModels

@testset "authored unit queries keep a fixed structural size" begin
    for n in (6, 24, 96)
        m = brm_nlme_model(RKBRMI(svi_owned_model(svi_owned_data(collect(1:n))));
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        preparation = @timed BRM._brm_svi_target(m)
        t = preparation.value
        @test length(t.globals) == 4
        @test all(==(3), length.(t.units))
        @test length(BRM._rk_svi_factor_coordinates(t.prior.queries[1])) == 4
        @test length(BRM._rk_svi_factor_coordinates(t.prior.queries[2])) == 4
        @test length(t.plan.footprints[1]) == 6
        g, z = fill(0.1, 4), fill(0.2, 3)
        gg, gz = zeros(4), zeros(3)
        evaluate() = BRM._brm_svi_unit_term!(gg, gz, t, 1, g, z)
        evaluate()
        evaluate()
        allocated = @allocated for _ in 1:100
            evaluate()
        end
        elapsed = @elapsed for _ in 1:100
            evaluate()
        end
        println("SVI_OWNED_WORK units=", n, " packed=", length(m.unit),
            " globals=", length(t.globals), " local=", length(t.units[1]),
            " prior_support=", length(BRM._rk_svi_factor_coordinates(t.prior.queries[2])),
            " likelihood_support=", length(t.plan.footprints[1]),
            " prior_retained_bytes=", Base.summarysize(t.prior),
            " preparation_bytes=", preparation.bytes,
            " bytes_per_unit=", allocated / 100, " seconds_per_unit=", elapsed / 100)
    end
end
