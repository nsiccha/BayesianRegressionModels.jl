# Analyze retained reference chains; no sampler invocation and no new draws.
# julia --project=test research/shared_field_laplace/reference_diagnostics.jl 'saved reference diagnostic analysis'
include("operator_probe.jl")
using Serialization, Statistics, MCMCDiagnosticTools

@stestset "saved reference diagnostic analysis" begin
    (;rk) = shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2,
        prepare_model=false)
    names = brm_nlme_view(rk).coordinates
    root = joinpath(@__DIR__, "results")
    for label in ("posterior-v1","posterior-v2-accept99")
        directory = joinpath(root,label)
        signature = readchomp(joinpath(directory,"identity.txt"))
        chains = [deserialize(joinpath(directory,"reference-$i.jls")) for i in 1:4]
        @test all(c -> c.signature == signature,chains)
        @test all(c -> size(c.position,1) == length(names),chains)
        n = minimum(size(c.position,2) for c in chains)
        samples = permutedims(cat((c.position[:,end-n+1:end]' for c in chains)...;
            dims=3),(1,3,2))
        rhat = vec(MCMCDiagnosticTools.rhat(samples))
        bulk = vec(MCMCDiagnosticTools.ess(samples;kind=:bulk))
        tail = vec(MCMCDiagnosticTools.ess(samples;kind=:tail))
        println((;label,divergences=sum(c.divergences for c in chains),draws=4n))
        for i in sortperm(rhat;rev=true)[1:5]
            println((;coordinate=names[i],index=i,rhat=rhat[i],bulk=bulk[i],tail=tail[i],
                chain_means=[mean(c.position[i,:]) for c in chains],
                chain_sds=[std(c.position[i,:]) for c in chains]))
        end
        serialize(joinpath(directory,"coordinate-diagnostics.jls"),(;names,rhat,bulk,tail))
    end
end
