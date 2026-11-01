# Bounded research adaptation of FASO (Welandawe et al2024, Algorithm1).
# References: https://jmlr.org/papers/volume25/22-0327/22-0327.pdf
# VIABEL 46dd21afdcb994e45c4ed9e4ab56c68a0bf1a270, optimization.py:FASO.
# Both raw parameters and distribution summaries govern stopping. Unlike the
# author implementation, rechecks use deterministic doubling, the minimum ESS
# is50, and constants are refused. This is not RABVI bias extrapolation or a
# certificate of ELBO stationarity/posterior accuracy.
using LinearAlgebra, Statistics, MCMCDiagnosticTools

function iterate_windows(k;minimum_window=200)
    upper=2floor(Int,.95k/2)
    upper>=minimum_window || return Int[]
    unique(2 .* floor.(Int,range(minimum_window/2,upper/2;length=5)))
end

function iterate_rhat(history)
    # history is coordinates by iterations; one optimizer trajectory, split
    # into two halves. An entirely constant coordinate is not convergence.
    all(isfinite,history) || return (;valid=false,reason=:nonfinite,rhat=Float64[],maximum=Inf)
    all(>(0),vec(var(history;dims=2))) ||
        return (;valid=false,reason=:constant_coordinate,rhat=Float64[],maximum=Inf)
    samples=reshape(permutedims(history),size(history,2),1,size(history,1))
    rhat=vec(MCMCDiagnosticTools.rhat(samples;kind=:basic,split_chains=2))
    valid=all(isfinite,rhat)
    (;valid,reason=valid ? :ok : :nonfinite_diagnostic,rhat,maximum=valid ? maximum(rhat) : Inf)
end

function iterate_precision(history,scales)
    rh=iterate_rhat(history)
    rh.valid || return (;rh,ess=Float64[],mcse=Float64[],normalized_mcse=Float64[],
        maximum_mcse=Inf,minimum_ess=0.,passed=false)
    length(scales)==size(history,1) && all(x->isfinite(x)&&x>0,scales) ||
        error("invalid iterate precision normalization")
    samples=reshape(permutedims(history),size(history,2),1,size(history,1))
    # Basic ESS/MCSE of arithmetic averages, not rank-normalized bulk ESS.
    # Use all lags through the shorter split, with the package's public API.
    maxlag=max(1,div(size(history,2),2)-1)
    ess=vec(MCMCDiagnosticTools.ess(samples;kind=:basic,split_chains=2,maxlag))
    mcse=vec(MCMCDiagnosticTools.mcse(samples;kind=mean,split_chains=2,maxlag))
    normalized_mcse=mcse./scales
    valid=all(isfinite,ess)&&all(isfinite,normalized_mcse)
    maximum_mcse=valid ? maximum(normalized_mcse) : Inf
    minimum_ess=valid ? minimum(ess) : 0.
    passed=valid && rh.maximum<=1.1 && maximum_mcse<.1 && minimum_ess>=50
    (;rh,ess,mcse,normalized_mcse,maximum_mcse,minimum_ess,passed)
end

function iterate_distribution(q)
    length(q.μ)==3 && size(q.Β)==(3,3) || error("bounded three-global control only")
    C=Matrix(Diagonal(exp.(2q.ρ))+q.Β*q.Β')
    sd=sqrt.(diag(C))
    vcat(q.μ,log.(sd),[C[i,j]/(sd[i]*sd[j]) for j in 2:3 for i in 1:j-1])
end

function iterate_scales(q)
    sd=sqrt.(exp.(2q.ρ)+vec(sum(abs2,q.Β;dims=2)))
    # Means and each B row use the returned factor's marginal SD; rho,
    # log marginal SDs and correlations are dimensionless.
    (;raw=vcat(sd,ones(3),repeat(sd,3)),distribution=vcat(sd,ones(6)))
end

function exact_iterate_covariance_gap(raw)
    size(raw,1)==15 || error("bounded three-global history only")
    x=vec(mean(raw;dims=2));rho=x[4:6];B=reshape(x[7:15],3,3)
    C=Matrix(Diagonal(exp.(2rho))+B*B')
    covariances=map(eachcol(raw)) do v
        b=reshape(v[7:15],3,3)
        Matrix(Diagonal(exp.(2v[4:6]))+b*b')
    end
    mean_covariance=mean(covariances)
    (;returned_covariance=C,mean_covariance,gap=mean_covariance-C,
        fraction=diag(mean_covariance-C)./diag(mean_covariance))
end
