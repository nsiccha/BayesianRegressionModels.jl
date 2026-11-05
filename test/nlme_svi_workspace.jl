using Test
include(joinpath(@__DIR__, "..", "src", "nlme_selection.jl"))
include(joinpath(@__DIR__, "..", "src", "nlme_svi_workspace.jl"))

@testset "SVI compact unit layouts" begin
    for globals in (Int[], [1, 4, 9]), block in (Int[], [6], [8, 2, 5])
        layout = _brm_svi_unit_layout(globals, block)
        indices = vcat(globals, block)
        order = sortperm(indices)
        g, z = 0.3 .* globals, -0.7 .* block
        # Independent old dense gather/scatter as a small-array oracle.
        oracle = vcat(g, z)[order]
        values = fill(NaN, length(indices))
        @test _brm_svi_joint!(values, layout, g, z) == oracle
        gradient = cos.(indices[order])
        joint_gradient = similar(gradient)
        joint_gradient[order] = gradient
        gg, gz = similar(g), similar(z)
        _brm_svi_split_gradient!(gg, gz, layout, gradient)
        @test vcat(gg, gz) == joint_gradient
    end
    small = _brm_svi_unit_layout(collect(1:100), [103, 101, 102])
    large = _brm_svi_unit_layout(collect(1:100_000), [103, 101, 102] .+ 100_000)
    @test Base.summarysize(small) == Base.summarysize(large)
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
