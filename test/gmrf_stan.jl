include("gmrf_preparation.jl")
using StanBlocks, Random

function gmrf_stan_model(; known=false)
    if known
        return @brm data begin
            field ~ 0 + gmrf((east, north), time, channel;
                mesh=(vertices=xy, cells=triangles), cor=Omega)
            sd(field, gmrf((east, north), time, channel)) = 1.2
            length_scale(field, gmrf((east, north), time, channel), space) = 0.4
            length_scale(field, gmrf((east, north), time, channel), time) = 1.7
            y ~ Normal(field, 0.2)
        end
    end
    @brm data begin
        field ~ 0 + gmrf((east, north), time, channel;
            mesh=(vertices=xy, cells=triangles), cor=Omega)
        sd(field, gmrf((east, north), time, channel)) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north), time, channel), space) ~ LogNormal(0.5, 0.8)
        length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(0.7, 0.6)
        y ~ Normal(field, 0.2)
    end
end

@testset "GMRF emits valid Stan with known and learned hyperparameters" begin
    for known in (false, true)
        sb = SBBRMI(gmrf_stan_model(; known); mod=@__MODULE__)
        code = StanBlocks.stan_code(sb.model)
        check = StanBlocks.stanc_check(code; warn_pedantic=false)
        check.ok || println(check)
        @test check.ok
        @test sb.data[:gmrf_field_east_north_time_channel_n_field] == 30
        @test occursin("brm_gmrf_lpdf", code)
        @test !occursin("multi_normal", code)
        @test isempty(popcoefnames(gmrf_stan_model(; known), :field))
    end
end

@testset "GMRF replay and declaration-owned coordinates" begin
    sb = SBBRMI(gmrf_stan_model(); mod=@__MODULE__)
    rows = [4, 2, 1]
    replay_data = (; (k => getproperty(data, k)[rows]
        for k in (:east, :north, :time, :channel, :y))...)
    replay = reprocess(sb, replay_data)
    stem = "gmrf_field_east_north_time_channel_"
    for suffix in ("index1", "index2", "index3", "weight1", "weight2", "weight3")
        key = Symbol(stem, suffix)
        @test replay.data[key] == sb.data[key][rows]
    end
    @test replay.data[Symbol(stem, "times")] == [0, 0.3, 1.7]
    @test replay.data[Symbol(stem, "n_field")] == 30
    @test_throws ArgumentError reprocess(sb, merge(replay_data, (; time=[2.0, 0.3, 0.0])))
    fresh = reprocess(sb, merge(data, (; time=[0, 0.3, 0.7, 2.0])); freeze_constants=false)
    @test fresh.data[Symbol(stem, "n_field")] == 40
    @test fresh.data[Symbol(stem, "times")] == [0, 0.3, 0.7, 2.0]
    for known in (false, true)
        model = gmrf_stan_model(; known)
        descriptor = brm_descriptor(SBBRMI(model; mod=@__MODULE__); highlights=())
        for language in (:en, :de)
            description = brm_description(descriptor; texts=description_texts(language))
            @test description.complete
            @test isempty(description.diagnostics)
            @test any(p -> occursin("Matérn", p) && occursin("Neumann", p), description.prose)
        end
        entry = only(BRM._brm_term_coordinate_entries(model, :field))
        @test entry.term == :gmrf_field_east_north_time_channel
        field = BRM._brm_term_parameter_output(descriptor, :field, entry, :field)
        @test prod(field.dims) == 30
        for role in (:sd, :spatial_length, :temporal_length)
            if known
                @test_throws "known value" BRM._brm_term_parameter_output(descriptor, :field, entry, role)
            else
                @test BRM._brm_term_parameter_output(descriptor, :field, entry, role).declaration !== nothing
            end
        end
    end
end
