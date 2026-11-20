module ComponentSettingsTest
using Test, BayesianRegressionModels
using ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels

@brm function calibration(config, x)
    if config.curve && config.family == :shifted
        shift ~ Normal(0, 1)
        out = x + shift
    elseif config.curve
        scale ~ Exponential(1)
        out = scale * x
    else
        out = x
    end
    return out
end

const model = @brm begin
    value ~ calibration(settings, x)
    y ~ Normal(value, 1)
end

@testset "Data-known component declarations" begin
    for (config, locals) in (((curve=true, family=:shifted), [:shift]),
                             (Dict(:curve=>true, :family=>:scaled), [:scale]),
                             ((curve=false,), Symbol[]))
        design = (; settings=config, x=[1., 2.], y=[1.5, 2.5])
        bound = model(design)
        @test only(brm_components(bound)).path == (:value,)
        rk = RKBRMI(bound)
        @test ReactiveKernelsPPL.coordinate_names(rk.model.layout) ==
            [Symbol("value.", key) for key in locals]
        sb = SBBRMI(bound; mod=@__MODULE__, total_groups=())
        descriptor = brm_descriptor(sb)
        for key in (:shift, :scale)
            output = brm_outputs(descriptor; submodel=:value, logical=key)
            @test isempty(output) == !(key in locals)
        end
        @test BayesianRegressionModels.stan_code(reprocess(sb, design)) ==
            BayesianRegressionModels.stan_code(sb)
    end
    design = (; x=[1.], y=[1.], settings=(curve=true, family=:shifted))
    sb = SBBRMI(model(design); mod=@__MODULE__, total_groups=())
    @test_throws "different declarations" reprocess(sb, merge(design, (; settings=(curve=false,))))
end
end
