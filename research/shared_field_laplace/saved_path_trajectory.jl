# Saved-state diagnosis only: no model calls, new samples, optimization or
# endpoint selection. The last iterate is not a replacement candidate.
using Serialization, LinearAlgebra, Statistics, SHA, Test
import BayesianRegressionModels

@testset "saved path derivative trajectory" begin
    base=joinpath(@__DIR__, "results")
    path=joinpath(base,"native-path-derivative-fit-v1/fit.jls")
    saved=deserialize(path)
    audit=deserialize(joinpath(base,"native-path-derivative-endpoint-v1/audit.jls"))
    saved.identity.signature==audit.identity.fit_signature || error("fit/audit identity differs")
    saved.identity.julia==string(VERSION) || error("saved Julia version differs")
    q=saved.fit.q;last_q=saved.last_q
    covariance(q)=Matrix(Diagonal(exp.(2q.ρ))+q.Β*q.Β')
    C=covariance(q);L=covariance(last_q);scale=sqrt.(diag(C))
    window=50;patience=4;min_windows=8;polish_windows=10
    windows=[(;mean=mean(saved.fit.elbo[i:i+window-1]),var=var(saved.fit.elbo[i:i+window-1]))
        for i in 1:window:length(saved.fit.elbo)]
    stalls=map(max(min_windows,patience+1):length(windows)) do j
        a,b=windows[j-patience],windows[j]
        (;iteration=j*window,rise=b.mean-a.mean,threshold=2sqrt((a.var+b.var)/window))
    end
    first_stall=first(filter(x->x.rise<x.threshold,stalls))
    @test first_stall.iteration+window*polish_windows==saved.fit.iterations
    @test length(q.μ)==3 && size(C)==(3,3)
    @test all(s->s.diagnostic_phase==:pre_update && s.parameter_phase==:post_update,saved.snapshots)
    @test last(saved.snapshots).iteration==saved.fit.iterations
    @test last(saved.snapshots).mu==last_q.μ
    @test last(saved.snapshots).covariance≈L atol=1e-14
    trajectory=map(saved.snapshots) do s
        (;s.iteration,s.whiten,s.lr,mean_offset=(s.mu-q.μ)./scale,
            sd_ratio=sqrt.(diag(s.covariance))./scale,
            diagnostic_phase=s.diagnostic_phase,parameter_phase=s.parameter_phase)
    end
    # Only the retained 50-step post-update snapshots are used here. Their
    # covariance gap is descriptive, not the exact gap for all 500 iterates.
    polish=filter(s->s.whiten,saved.snapshots)
    Bbar=mean(s.B for s in polish);rhobar=mean(s.rho for s in polish)
    mean_covariance=mean(s.covariance for s in polish)
    parameter_covariance=Matrix(Diagonal(exp.(2rhobar))+Bbar*Bbar')
    gap=mean_covariance-parameter_covariance
    @test minimum(eigvals(Symmetric(gap)))>=-1e-12
    result=(;identity=(;source_sha256=bytes2hex(sha256(read(@__FILE__))),
            saved_sha256=bytes2hex(sha256(read(path))),fit_signature=saved.identity.signature,
            audit_signature=audit.identity.signature,julia=string(VERSION)),
        first_stall,stalls,windows,trajectory,returned_mean=copy(q.μ),returned_covariance=C,
        last_mean=copy(last_q.μ),last_covariance=L,last_mean_offset=(last_q.μ-q.μ)./scale,
        last_sd_ratio=sqrt.(diag(L))./scale,coarse_snapshot_covariance_gap=gap,
        coarse_snapshot_gap_fraction=diag(gap)./diag(mean_covariance),
        scope="saved trajectory and exact stopping-rule reconstruction only; coarse averaging diagnostic; no model evaluations, new draws, refit or endpoint selection")
    output=joinpath(base,"saved-path-trajectory-v1.jls")
    ispath(output) && error("preserve previous saved-trajectory diagnosis")
    serialize(output,result)
    println((;first_stall,returned_mean=result.returned_mean,last_mean_offset=result.last_mean_offset,
        last_sd_ratio=result.last_sd_ratio,coarse_snapshot_gap_fraction=result.coarse_snapshot_gap_fraction))
    foreach(println,trajectory)
end
