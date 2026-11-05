using Test

# An adaptive pilot of an RKBRMI model whose law is bound in the RK program's
# generated module (here a gmrf field) stops at initialization in one Julia
# process and resumes in another, from a serialized partial and from disk.
# Requires ReactiveKernelsPPL 0.1.98 (generated-module-safe SMC identities).
@testset "fresh-process pilot recovery with a generated-module law" begin
    mktempdir() do directory
        worker = joinpath(@__DIR__, "smc_generated_law_process.jl")
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $worker`
        @test success(pipeline(`$command write $directory`; stdout, stderr))
        @test success(pipeline(`$command read $directory`; stdout, stderr))
    end
end
