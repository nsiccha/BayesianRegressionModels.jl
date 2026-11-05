include("gmrf_plate_models.jl")

@testset "Native GMRF submodels inside independent plate cells" begin
    for known in (true, false)
        sb = SBBRMI(gmrf_plate_model(; known); mod=@__MODULE__)
        checked = StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false)
        checked.ok || println(checked)
        @test checked.ok
    end
end
