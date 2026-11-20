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
    rk = RKBRMI(signals(design(sensors)))
    @test_throws "membership or declarations changed" reprocess(rk, design(sensors[2:2]))
    fresh = reprocess(rk, design(sensors[2:2]); freeze_constants=false)
    @test fresh.model.layout.total == 0
end
end

module CollectionSmoothsTest
using Test, BayesianRegressionModels, StanBlocks
import BridgeStan

@brm function trend(config)
    if config.smooth
        population ~ 0 + hsgp(config.time; k=config.k, c=1.5)
        if config.periodic
            seasonal ~ 0 + hsgp(config.time; k=config.period_modes,
                cov=:periodic, period=config.period)
            mean_curve = population + seasonal
        else
            mean_curve = population
        end
        if config.individual
            deviations ~ 0 + hsgp(config.time; k=config.k, by=config.unit, anchor=0.0)
            value = sum(mean_curve + deviations)
        else
            value = sum(mean_curve)
        end
    else
        value = config.offset
    end
    return value
end

const model = @brm begin
    for e in records
        curves[e.id] ~ trend(e)
    end
    y ~ Normal(curves, 1)
end

const population_reference = @brm begin
    curve ~ 0 + hsgp(time; k=k, c=1.5)
    y ~ Normal(sum(curve), 1)
end
const periodic_reference = @brm begin
    curve ~ 0 + hsgp(time; k=k, c=1.5)
    seasonal ~ 0 + hsgp(time; k=kp, cov=:periodic, period=period)
    y ~ Normal(sum(curve + seasonal), 1)
end

@testset "Each configured time smooth matches a model on its own clock" begin
    a = (; id=:a, smooth=true, periodic=false, individual=false, time=[.3,.7,1.1], k=3)
    b = (; id=:b, smooth=false, offset=.24)
    c = (; a..., id=:c, k=4, time=[-.2,.15,.9,1.7])
    d = (; a..., id=:d, periodic=true, time=[2.,2.7,3.1], period_modes=2, period=2.1)
    ys = [.8,-.1,.18,.42]
    function compiled(source, label)
        fit = SBBRMI(source; mod=@__MODULE__, total_groups=())
        path = joinpath(mkpath(joinpath(tempdir(), "brm-rk-consumer")), label * ".stan")
        stan = BayesianRegressionModels.stan_instantiate(fit; path).model
        (; desc=brm_descriptor(fit), stan, names=BridgeStan.param_unc_names(stan))
    end
    combined = compiled(model((; records=[d,a,b,c], y=ys)), "independent-clocks")
    refs = [compiled(population_reference((; a.time, a.k, y=ys[2:2])), "clock-population"),
            compiled(population_reference((; c.time, c.k, y=ys[4:4])), "clock-population"),
            compiled(periodic_reference((; d.time, d.k, kp=d.period_modes, d.period, y=ys[1:1])), "clock-periodic")]
    mappings = map((a,c,d), refs) do record, ref
        mapping = zeros(Int, length(ref.names))
        terms = record.periodic ? ((:population,:curve), (:seasonal,:seasonal)) : ((:population,:curve),)
        for (joint, single) in terms, parameter in (:length_scale,:sd,:basis_weights)
            src = brm_term_coordinates(ref.desc, single, ref.names; parameter)
            dst = brm_term_coordinates(combined.desc, joint, combined.names;
                parameter, submodel=(:curves, record.id))
            mapping[src.coordinates] = dst.coordinates
        end
        mapping
    end
    n = length(combined.names)
    @test sort(reduce(vcat, mappings)) == 1:n
    for u in (zeros(n), collect(range(-.35,.4; length=n)), [0.23sin(i) for i in 1:n])
        value, grad = BridgeStan.log_density_gradient(combined.stan, u; propto=false)
        expected = -0.5*(ys[3] - b.offset)^2 - 0.5*log(2pi)
        for (ref, coords) in zip(refs, mappings)
            part, part_grad = BridgeStan.log_density_gradient(ref.stan, u[coords]; propto=false)
            expected += part
            @test grad[coords] ≈ part_grad
        end
        @test value ≈ expected
    end
end

@testset "Repeat and replay individually configured time smooths" begin
    a = (; id=:a, smooth=true, periodic=false, individual=false,
        time=[0.2, 0.5, 0.7], k=3)
    b = (; id=:b, smooth=false, offset=0.2)
    c = (; a..., id=:c, time=[-1., 0., 1., 2.], k=4)
    d = (; c..., id=:d, periodic=true, period_modes=2, period=3.5,
        individual=true, unit=["one", "one", "two", "two"])
    design(records) = (; records, y=zeros(length(records)))
    periodic = (; d..., individual=false)
    sb = SBBRMI(model(design([a,b,c,periodic])); mod=@__MODULE__, total_groups=())
    compiled = StanBlocks.stanc_check(stan_code(sb); warn_pedantic=false)
    @test isempty(compiled.output) && compiled.ok
    @test Set(c.path for c in brm_components(sb)) == Set((:curves,id) for id in (:a,:b,:c,:d))
    desc = brm_descriptor(sb)
    @test isempty(filter(o -> endswith(string(o.name), "_beta_raw"), brm_outputs(desc; submodel=(:curves,:b))))
    for (id, expected) in ((:a, 1:3), (:c, 4:7))
        weights = only(filter(o -> endswith(string(o.name), "_beta_raw"), brm_outputs(desc; submodel=(:curves,id))))
        names = [string(weights.name, ".", i) for i in 1:7]
        @test brm_output_coordinates(weights, names) == expected
        result = brm_term_coordinates(desc, :population, reverse(names);
            parameter=:basis_weights, submodel=(:curves,id))
        @test result.coordinates == 8 .- expected
    end
    replay = reprocess(sb, design([periodic,c,b,a]))
    @test stan_code(replay) == stan_code(sb)
    # The same model with its subject-specific smooth also exercises native
    # StanBlocks' promotion of per-cell, data-only integer index arrays.
    grouped = SBBRMI(model(design([a,b,c,d])); mod=@__MODULE__, total_groups=())
    grouped_check = StanBlocks.stanc_check(stan_code(grouped); warn_pedantic=false)
    @test grouped_check.ok
end
end
