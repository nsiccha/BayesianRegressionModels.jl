module ComponentCollectionsTest
using Test, BayesianRegressionModels, StanBlocks
using ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels

@brm function sensor(settings)
    if settings.calibrated
        shift ~ Normal(0, settings.scale)
        reading = settings.offset + shift
    else
        reading = settings.offset
    end
    return reading
end

const signals = @brm begin
    for item in sensors
        channels[item.id] ~ sensor(item)
    end
    y ~ Normal(channels, 1)
end

@testset "Indexed component declarations and fitted label identity" begin
    sensors = [(id=:c, calibrated=true, offset=-0.2, scale=0.5),
               (id=:b, calibrated=false, offset=0.4, scale=1.2),
               (id=:a, calibrated=true, offset=1.0, scale=0.7)]
    design(records) = (; sensors=records, y=zeros(length(records)))
    @test brm_columns(signals) == (:sensors, :y)
    @test_throws "duplicate keys" signals(design([sensors[1], sensors[1]]))
    for selected in (sensors, sensors[[3, 2, 1]], sensors[1:1], sensors[2:2], sensors[1:0])
        source = signals(design(selected))
        @test Set(c.path for c in brm_components(source)) ==
              Set((:channels, row.id) for row in selected)
        rk = RKBRMI(source)
        sb = SBBRMI(source; mod=@__MODULE__, total_groups=())
        @test rk.model.layout.total == count(x -> x.calibrated, selected)
        active = sort([row.id for row in selected if row.calibrated])
        u = collect(1.0:length(active))
        for (records, r, s) in ((selected, rk, sb),
                (reverse(selected), reprocess(rk, design(reverse(selected))),
                 reprocess(sb, design(reverse(selected)))))
            expected = [rec.offset + (rec.calibrated ? u[findfirst(==(rec.id), active)] : 0)
                        for rec in records]
            @test only(rk_draw_values(r, reshape(u, :, 1), (:channels,)).channels) ≈ expected
            descriptor = brm_descriptor(s)
            @test stan_code(s) == stan_code(sb)
            for row in records
                address = (:channels, row.id)
                results = brm_outputs(descriptor; logical=:shift, submodel=address)
                @test isempty(results) == !row.calibrated
                row.calibrated || continue
                carrier = only(results)
                names = reverse([string(carrier.name, ".", i) for i in eachindex(active)])
                @test brm_output_coordinates(descriptor, :shift, names; submodel=address) ==
                      [findfirst(==(row.id), reverse(active))]
            end
        end
    end
end
end
