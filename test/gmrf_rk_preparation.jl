include("gmrf_preparation.jl")

@testset "GMRF emitted data preparation retains fitted axes and local rows" begin
    support = BRM.brm_gmrf_support(data.xy, data.triangles, data.time, data.channel)
    @test support.layout.times == [0.0, 0.3, 1.7]
    @test support.layout.channels == ["a", "b"]
    @test support.geometry.ncoordinates == 30
    @test support.geometry.mass === support.mesh.factor_mass
    @test support.geometry.times === support.layout.times
    rows = BRM.brm_gmrf_rows(support, data.east, data.north, data.time, data.channel)
    @test rows.weight1 + rows.weight2 + rows.weight3 ≈ ones(4)
    @test rows.index1[1] > 5 # channel b, first time
    @test rows.index1[2] in 11:15 # channel a, second time
    selected = [3, 1]
    replay = BRM.brm_gmrf_rows(support, data.east[selected], data.north[selected],
        data.time[selected], data.channel[selected])
    @test all(getproperty(replay, k) == getproperty(rows, k)[selected] for k in keys(rows))
    @test_throws ArgumentError BRM.brm_gmrf_rows(support, [0.1], [0.2], [2.0], ["a"])
    @test_throws ArgumentError BRM.brm_gmrf_rows(support, [0.1], [0.2], [0.0], ["new"])
    empty_rows = BRM.brm_gmrf_rows(support, Float64[], Float64[], Float64[], String[])
    @test all(isempty, values(empty_rows))
end

@testset "GMRF formula planning keeps numerical preparation in emitted source" begin
    for has_channels in (false, true)
        model = if has_channels
            @brm data begin
                field ~ 0 + gmrf((east, north), time, channel;
                    mesh=(vertices=xy, cells=triangles), cor=Omega)
                sd(field, gmrf((east, north), time, channel)) ~ Exponential(0.7)
                length_scale(field, gmrf((east, north), time, channel), space) = 0.4
                length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(0.5, 0.7)
                y ~ Normal(field, 0.2)
            end
        else
            @brm data begin
                field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
                y ~ Normal(field, 0.2)
            end
        end
        plan = BRM._brm_rk_plan(model)
        emitted = BRM._rk_emit_ast(plan)
        source = string(emitted.main, emitted.defs)
        @test occursin("brm_gmrf_support", source)
        @test occursin("brm_gmrf_rows", source)
        law = only(first(p) for p in emitted.bindings if last(p) === BRM._BRMGMRFLaw)
        @test occursin(string(law, "("), source)
        @test occursin("weight1", source) && occursin("index3", source)
        @test !occursin("brm_gmrf_logdensity", source)
        @test !occursin("sampling_logdensity", source)
    end
end
