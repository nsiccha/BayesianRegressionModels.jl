using Test
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
