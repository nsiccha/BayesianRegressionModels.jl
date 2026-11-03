include("gmrf_plate_models.jl")

@testset "Independent GMRF plate metadata and frozen replay" begin
    for known in (true, false)
        sb = SBBRMI(gmrf_plate_model(; known); mod=@__MODULE__)
        fit = only(e.const_ for e in values(sb.preproc) if e.kind === :cell_formulas)
        @test fit.labels == [10, 20]
        fields = [only(e.const_.prepared for e in values(c.preproc) if e.kind === :gmrf)
            for c in fit.cells]
        @test [p.state.layout.ncoordinates for p in fields] == [15, 5]
        @test [p.state.times for p in fields] == [[0.0, 0.3, 1.7], [0.3]]
        order = [2, 4, 1, 3]
        fresh = (; (k => getproperty(data, k)[order] for k in (:east, :north, :time, :y))...,
            unit=[20, 10, 10, 10])
        replay = reprocess(sb, fresh)
        replay_fit = only(e.const_ for e in values(replay.preproc) if e.kind === :cell_formulas)
        @test replay_fit.labels == [10, 20]
        replay_fields = [only(e.const_.prepared for e in values(c.preproc) if e.kind === :gmrf)
            for c in replay_fit.cells]
        @test replay_fields[1].state.indices == fields[1].state.indices[[3, 1, 2], :]
        @test replay_fields[2].state.indices == fields[2].state.indices
        subset = (; (k => getproperty(data, k)[[2]] for k in (:east, :north, :time, :y))...,
            unit=[20])
        missing_cell = reprocess(sb, subset)
        subset_fit = only(e.const_ for e in values(missing_cell.preproc) if e.kind === :cell_formulas)
        @test subset_fit.labels == [10, 20]
        subset_fields = [only(e.const_.prepared for e in values(c.preproc) if e.kind === :gmrf)
            for c in subset_fit.cells]
        @test [p.state.layout.ncoordinates for p in subset_fields] == [15, 5]
        @test size(subset_fields[1].state.indices) == (0, 3)
        @test_throws ArgumentError reprocess(sb, merge(subset, (; unit=[30])))
    end
end
