# Captured source must preserve the backend's actual submodels and callable
# module, so an upstream sequential interpreter can read the authored sites.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels
const PPL = ReactiveKernelsPPL

module CapturedSourceModels
using BayesianRegressionModels, ReactiveKernels
const BRM = BayesianRegressionModels

ar_model(d) = @brm d begin
    mu ~ 1 + ar(t; p=1)
    y ~ Normal(mu, 1)
end
dar_model(d) = @brm d begin
    mu ~ 1 + dar(t)
    y ~ Normal(mu, 1)
end
scaled_model(d) = @brm d begin
    mu ~ 1 + zscale(x)
    y ~ Normal(mu, 1)
end
function accumulated_input(x, gain)
    carry = 0.0
    map(x) do value
        carry = 0.7carry + gain * value
    end
end
function BRM._rk_callable_source!(definitions, bindings, entry, ::typeof(accumulated_input))
    push!(definitions, :(ReactiveKernels.@kernel $entry(x, gain) = begin
        values = ReactiveKernels.scan(x; init=0.0) do carry, value
            next = 0.7carry + gain * value
            (next, next)
        end
        return values
    end))
    :done
end
provider_model(d) = @brm d begin
    gain ~ Normal(0, 1)
    loc = accumulated_input(x, gain)
    y ~ Normal(loc, 1)
end
end

source_inputs(rk) = BRM._rk_source_data_columns(rk.plan, BRM._rk_retained_program(rk))

function check_captured_density(rk; columns=source_inputs(rk))
    captured = rkppl_model(rk)
    @test captured isa PPL.RKPPLModel
    @test rkppl_model(rk) === captured
    # Bound graph inputs can include derived definitions. Re-lower the captured
    # source with its source inputs, leaving those definitions to the model.
    lowered = PPL.lower_rkppl(captured, columns; conditioned=(:y,))
    plan = PPL.bind_data(lowered, columns)
    rebuilt = PPL.build_kernel(plan)
    @test PPL.coordinate_names(rebuilt.layout) == PPL.coordinate_names(rk.model.layout)
    n = rk.model.layout.total
    u = n == 1 ? [0.17] : collect(range(-0.2, 0.3; length=n))
    for preset in (:prior, :likelihood, :sampler)
        original = PPL.prepare_query(rk.model, rk_bound_plan(rk), preset)
        replay = PPL.prepare_query(rebuilt, plan, preset)
        @test replay(u) ≈ original(u) atol=1e-12 rtol=1e-12
    end
    captured
end

@testset "RK captured source preserves submodels and provider graphs" begin
    data = (; t=collect(1.0:4.0), x=[-1.0, 0.0, 1.0, 2.0], y=[0.3, -0.1, 0.7, 0.4])
    for model in (CapturedSourceModels.ar_model, CapturedSourceModels.dar_model,
                  CapturedSourceModels.provider_model)
        rk = RKBRMI(model(data))
        check_captured_density(rk)
        @test rkppl_model(deepcopy(rk)) === rkppl_model(rk)
    end
end

@testset "RK captured source remains separate from frozen binding" begin
    training = (; x=[-2.0, 0.0, 2.0], y=[-0.3, 0.4, 0.9])
    design = (; x=[8.0, 9.0], y=[1.2, 1.3])
    rk = RKBRMI(CapturedSourceModels.scaled_model(training))
    replay = reprocess(rk, design)
    @test rkppl_model(replay) === rkppl_model(rk)
    check_captured_density(replay)
    fitted = brm_fitted_state(rk)
    datafree = RKBRMI(CapturedSourceModels.scaled_model(design), fitted)
    # The captured model is unbound. Give this independent source rebuild the
    # original fit-role inputs; its density must match the data-free backend,
    # whose actual binding instead uses the frozen summaries.
    check_captured_density(datafree; columns=source_inputs(replay))
end
