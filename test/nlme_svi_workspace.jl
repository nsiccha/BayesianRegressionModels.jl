using Test
include(joinpath(@__DIR__, "..", "src", "nlme_selection.jl"))
include(joinpath(@__DIR__, "..", "src", "nlme_svi_workspace.jl"))

@testset "a unit's places follow its footprint, not the globals" begin
    # Packed 1:12: globals 2, 4, 7; unit 1 owns 11, 1, 8; unit 2 owns 3, 9; the rest
    # (5, 6, 10, 12) are fixed values a footprint may read.
    globals, units = [2, 4, 7], [[11, 1, 8], [3, 9]]
    footprints = [[1, 2, 5, 8, 11], [3, 7, 9, 12]]
    places = _brm_svi_footprint_places(footprints, globals, units, 12)
    @test places == [[-2, 1, 0, -3, -1], [-1, 3, -2, 0]]
    # The packed-length table is temporary: no unit keeps it or another unit's entries.
    @test _brm_svi_footprint_places([[1], [1]], [1], [Int[], Int[]], 1) == [[1], [1]]
    small = _brm_svi_footprint_places(footprints, globals, units, 12)
    large = _brm_svi_footprint_places(footprints, vcat(globals, 13:100_000), units, 100_000)
    @test Base.summarysize(small) == Base.summarysize(large)
    # Touched globals: the footprint's, and the prior bucket's reads of the globals.
    @test _brm_svi_touched_globals(places, nothing) == [[1], [3]]
    prior = (; places=[[1], [3, 1, -1, 0, 3], Int[]])
    @test _brm_svi_touched_globals(places, prior) == [[1, 3], [3]]
    # A worker's buffers start at zero and are one pair per worker.
    buffers = _brm_svi_worker_buffers(globals)
    @test buffers.plus == buffers.minus == zeros(3) && buffers.plus !== buffers.minus
    @test _brm_svi_global_buffers(buffers, 3) === (buffers.plus, buffers.minus)
    @test _brm_svi_global_buffers(nothing, 2) == (zeros(2), zeros(2))
end

@testset "partition selections share globals and iterate the exact sorted union" begin
    globals = [2, 4, 7, 12]
    blocks = [[11, 1, 8], [3, 9, 6], Int[]]
    selections = [_brm_nlme_unit_selection(globals, b) for b in blocks]
    for (s, block) in zip(selections, blocks)
        expected = sort(vcat(globals, block))
        @test collect(s) == expected
        @test [s[i] for i in eachindex(s)] == expected
        @test all(j -> searchsortedfirst(s, j) == searchsortedfirst(expected, j), 0:14)
        @test s.globals === globals
        @test s.block === block
    end
    # Large shared coordinates add storage once, independent of unit count.
    g = collect(1:2:20_000)
    small = [_brm_nlme_unit_selection(g, [2i]) for i in 1:10]
    large = [_brm_nlme_unit_selection(g, [2i]) for i in 1:1000]
    @test Base.summarysize(large) - Base.summarysize(small) < 400_000
    @test globals == [2, 4, 7, 12] && blocks[1] == [11, 1, 8]
end
