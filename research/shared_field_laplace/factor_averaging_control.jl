# Algebra and saved snapshots only. No fitting, target evaluation, new draws
# or endpoint selection. Dense matrices are confined to three globals.
using LinearAlgebra, Statistics, Serialization, SHA, Test
import BayesianRegressionModels

function factor_average_diagnostic(states)
    all(s->length(s.rho)==3 && size(s.B)==(3,3),states) || error("three-global control only")
    cov(s)=Matrix(Diagonal(exp.(2s.rho))+s.B*s.B')
    mean_covariance=mean(cov(s) for s in states)
    rhobar=mean(s.rho for s in states);Bbar=mean(s.B for s in states)
    parameter_covariance=Matrix(Diagonal(exp.(2rhobar))+Bbar*Bbar')
    scale_gap=Matrix(Diagonal(mean(exp.(2s.rho) for s in states)-exp.(2rhobar)))
    factor_gap=mean((s.B-Bbar)*(s.B-Bbar)' for s in states)
    reference=first(states).B
    aligned=map(states) do s
        S=svd(s.B'*reference)
        rotation=S.U*S.Vt
        (;B=s.B*rotation,rotation,covariance_error=norm((s.B*rotation)*(s.B*rotation)'-s.B*s.B',Inf))
    end
    aligned_mean=mean(s.B for s in aligned)
    aligned_covariance=Matrix(Diagonal(exp.(2rhobar))+aligned_mean*aligned_mean')
    (;mean_covariance,parameter_covariance,scale_gap,factor_gap,aligned_covariance,
        decomposition_error=norm(mean_covariance-parameter_covariance-scale_gap-factor_gap,Inf),
        alignment_covariance_error=maximum(s.covariance_error for s in aligned),
        scale_gap_fraction=diag(scale_gap)./diag(mean_covariance),
        factor_gap_fraction=diag(factor_gap)./diag(mean_covariance),
        aligned_gap_fraction=diag(mean_covariance-aligned_covariance)./diag(mean_covariance),
        relative_covariance_motion=sqrt(mean(norm(cov(s)-mean_covariance)^2 for s in states))/norm(mean_covariance))
end

@testset "factor averaging invariance" begin
    B=[.8 .1 -.2;.1 .5 .2;.2 -.1 .6];rho=log.([.3,.4,.5])
    rotations=[Matrix{Float64}(I,3,3),-Matrix{Float64}(I,3,3),
        [0. -1 0;1 0 0;0 0 1],[0. 1 0;-1 0 0;0 0 -1]]
    states=[(;rho,B=B*R) for R in rotations]
    control=factor_average_diagnostic(states)
    C=Matrix(Diagonal(exp.(2rho))+B*B')
    @test control.mean_covariance≈C atol=1e-14
    @test control.relative_covariance_motion<1e-14
    @test norm(control.parameter_covariance-C)>.1
    @test control.aligned_covariance≈C atol=1e-14
    @test control.decomposition_error<1e-14
    @test control.alignment_covariance_error<1e-14
    @test minimum(eigvals(Symmetric(control.factor_gap)))>=-1e-14
    base=joinpath(@__DIR__, "results")
    fit_path=joinpath(base,"native-path-derivative-fit-v1/fit.jls")
    saved=deserialize(fit_path)
    saved.identity.julia==string(VERSION) || error("saved Julia version differs")
    states=filter(s->s.whiten,saved.snapshots)
    diagnostic=factor_average_diagnostic(states)
    @test diagnostic.decomposition_error<1e-12
    @test diagnostic.alignment_covariance_error<1e-12
    @test minimum(eigvals(Symmetric(diagnostic.scale_gap)))>=-1e-12
    @test minimum(eigvals(Symmetric(diagnostic.factor_gap)))>=-1e-12
    result=(;identity=(;source_sha256=bytes2hex(sha256(read(@__FILE__))),
            fit_file_sha256=bytes2hex(sha256(read(fit_path))),fit_signature=saved.identity.signature),
        control,diagnostic,numeric_snapshots=[(;s.iteration,s.rho,s.B,s.mu) for s in states],
        scope="representation-invariance control and sparse snapshot diagnosis only; no exact all-iterate averaging effect or optimum-relative shrinkage claim; alignment is diagnostic, not a fitted replacement")
    output=joinpath(base,"factor-averaging-control-v1.jls")
    ispath(output) && error("preserve prior factor averaging control")
    serialize(output,result)
    println((;stage="fixed covariance control",control.factor_gap_fraction,control.aligned_gap_fraction))
    println((;stage="saved snapshots",diagnostic.scale_gap_fraction,diagnostic.factor_gap_fraction,
        diagnostic.aligned_gap_fraction,diagnostic.relative_covariance_motion))
end
