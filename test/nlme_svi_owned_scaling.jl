# Bounded selected-unit work control; timings are observations, not a
# hardware-independent performance threshold.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface
include("nlme_svi_owned_fixture.jl")
const BRM = BayesianRegressionModels

function owned_work_measure(f)
    f()
    f()
    bytes = @allocated for _ in 1:100
        f()
    end
    seconds = @elapsed for _ in 1:100
        f()
    end
    (; bytes=bytes / 100, seconds=seconds / 100)
end

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
        unit_work = owned_work_measure(evaluate)
        prior_work = owned_work_measure() do
            fill!(gg, 0)
            fill!(gz, 0)
            BRM._brm_svi_unit_prior!(gg, gz, t.prior, t, 1, g, z)
        end
        packed = zeros(length(g) + length(z))
        BRM._brm_svi_joint!(packed, t.layouts[1], g, z)
        packed_gradient = similar(packed)
        likelihood_work = owned_work_measure() do
            BRM._brm_nlme_partition_sum!(packed_gradient, t.plan, m.unit, 1, packed)
        end
        global_prior_work = owned_work_measure(() -> BRM._brm_svi_global_term(t, g))
        q = BRM._brm_svi_params(t, g, MeanFieldGlobal(); global_scale=0.1, unit_scale=1.0)
        x = BRM._brm_svi_global_vector(q)
        opt = BRM._BRMSVIAdam(length(x), 0.0)
        epsilon, eta = fill(0.1, 4), Float64[]
        # Zero step sizes hold the numerical state fixed while timing all
        # gradient, Adam, eigensolver and QR arithmetic. These are operation
        # costs for fixed D=4 and d=3, not fit-convergence measurements.
        global_update_work = owned_work_measure() do
            gradient = BRM._brm_svi_global_gradient(q, g, epsilon, eta)
            BRM._brm_svi_step!(opt, x, gradient)
            BRM._brm_svi_set_global!(q, x)
        end
        local_epsilon = fill(0.1, 3)
        local_gradient = -local_epsilon
        local_mean_gradient = zeros(3)
        factor_work = owned_work_measure() do
            BRM._brm_svi_unit_step!(q, 1, local_mean_gradient, local_gradient,
                local_epsilon, 0.0)
        end
        println("SVI_OWNED_WORK units=", n, " packed=", length(m.unit),
            " globals=", length(t.globals), " local=", length(t.units[1]),
            " prior_support=", length(BRM._rk_svi_factor_coordinates(t.prior.queries[2])),
            " likelihood_support=", length(t.plan.footprints[1]),
            " prior_retained_bytes=", Base.summarysize(t.prior),
            " preparation_bytes=", preparation.bytes,
            " preparation_seconds=", preparation.time,
            " unit=", unit_work, " unit_prior=", prior_work,
            " likelihood=", likelihood_work, " global_prior=", global_prior_work,
            " global_update=", global_update_work, " dense_unit_update=", factor_work,
            " dense_unit_state_bytes=", Base.summarysize((q.M[1], q.Λ[1], q.U[1])))
    end
end
