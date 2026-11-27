# The first `rkppl_model(name)` call of a session returns every catalogue body.
# RUN: julia --project=test test/rk_statistical_first_call.jl
#
# That call builds the module BRM's statistical bodies are defined in. Since
# Julia 1.12 a global is visible only to code running in a world at least as
# new as its definition, and one call runs in one world. So a call that defines
# the bodies and then reads one by name in its own world gets `UndefVarError`
# ("The binding may be too new") on Julia 1.12 and later, while the second call
# of the session succeeds. A constant read that way still works there, with a
# warning that it will become an error, so the warning fails this test too.
# Julia 1.10 has no binding worlds and passes either way.
#
# The check needs a session in which nothing has called `rkppl_model` yet, so
# it runs in a fresh process. Its one call to `catalogue` reads every body in
# the world that call started in, including the first, which builds the module.
using Test, BayesianRegressionModels
const BRM = BayesianRegressionModels

const NAMES = String.(collect(keys(BRM._BRM_STATISTICAL_MODELS)))

function fresh_session(script)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`
    out, err = IOBuffer(), IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout=out, stderr=err))
    (; exitcode=process.exitcode, out=String(take!(out)), err=String(take!(err)))
end

@testset "the first rkppl_model call of a session returns every body" begin
    session = fresh_session("""
        using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
        const BRM = BayesianRegressionModels
        catalogue(names) = [name => BRM.rkppl_model(name) for name in names]
        bodies = catalogue(keys(BRM._BRM_STATISTICAL_MODELS))
        for (name, body) in bodies
            println("BODY ", name, " ", BRM.rkppl_model(name) === body)
        end
        """)
    session.exitcode == 0 || print(stderr, session.err)
    @test session.exitcode == 0
    @test !occursin("prior to its definition world", session.err)
    lines = [split(line) for line in split(session.out, '\n') if startswith(line, "BODY ")]
    @test !isempty(NAMES)
    @test [String(line[2]) for line in lines] == NAMES
    # A later call returns the object the first call returned.
    @test all(line -> line[3] == "true", lines)
end
