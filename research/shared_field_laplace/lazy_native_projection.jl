# First carried factor from the retained native fit, with a predetermined
# draw/rank/shrinkage budget. No fit, new posterior draws, or reference tuning.
include("lazy_core_probe.jl")
const LAZY_CARRY_RESEARCH=isfile(joinpath(@__DIR__,"block_projection_probe.jl")) ? (@__DIR__) :
    @__DIR__
include(joinpath(LAZY_CARRY_RESEARCH,"block_projection_probe.jl"))
using SHA

function auxiliary_active_carry(core,store,order)
    length(unique(order))==length(order) || error("duplicate active block")
    rows=Vector{Int}[];covariances=Matrix{Float64}[]
    means=Vector{Float64}[];loadings=Matrix{Float64}[];offset=0
    for key in order
        b=store[key];n=length(b.a)
        size(b.B,2)==length(core.m) || throw(DimensionMismatch("fixed auxiliary rank"))
        push!(rows,collect(offset+1:offset+n));offset+=n
        push!(covariances,Matrix(b.F.L*b.F.L'))
        push!(means,b.a+b.B*core.m)
        push!(loadings,b.B*core.F.L)
    end
    block_carry(vcat(means...),rows,covariances,vcat(loadings...))
end

@testset "fixed auxiliary representation of native fitted carry" begin
    base=joinpath(@__DIR__, "results")
    draw_record=deserialize(joinpath(base,"native-fixed-rate-fit-v1/draws.jls"))
    fit=deserialize(joinpath(base,"native-fixed-rate-fit-v1/comparison.jls"))
    reference=deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    fit.identity==draw_record.identity && reference.reference_ok &&
        fit.identity.reference_signature==reference.signature || error("fitted carry identity differs")
    directory=joinpath(base,get(ENV,"SHARED_CARRY_ARTIFACT","native-fixed-core-initial-v1"))
    ispath(directory) && error("preserve native fixed-core initial artifacts")
    mkpath(directory)
    M=128;rank=8;delta=.01
    indices=vcat([fit.globals,[fit.beta]],[[j] for j in vcat(fit.units,fit.field)])
    permutation=vcat(indices...)
    @test sort(permutation)==collect(eachindex(fit.coordinates))
    blocks=Vector{Int}[];offset=0
    for b in indices
        push!(blocks,collect(offset+1:offset+length(b)));offset+=length(b)
    end
    order=vcat([:auxiliary_hyper_block,fit.coordinates[fit.beta]],fit.coordinates[fit.units],fit.coordinates[fit.field])
    length(unique(order))==length(order) || error("ambiguous carry block identity")
    X=draw_record.draws[permutation,1:M]
    references=[Matrix{Float64}(I,length(b),length(b)) for b in blocks]
    projection=project_block_carry(X,blocks,references;rank,shrinkage=delta)
    c=projection.factor
    core=auxiliary_core(zeros(rank),Matrix{Float64}(I,rank,rank))
    store=Dict(key=>auxiliary_block(c.mu[b],c.B[b,:],Matrix(F.L*F.L'))
        for (key,b,F) in zip(order,c.blocks,c.factors))
    materialized=auxiliary_active_carry(core,store,order)
    rng=Xoshiro(715)
    directions=randn(rng,length(permutation),16)
    directions ./= sqrt.(sum(abs2,directions;dims=1))
    centered=(X.-vec(mean(X;dims=2)))/sqrt(M)
    variance_relative_error=Float64[]
    for v in eachcol(directions)
        @test block_covariance_action(materialized,v)≈block_covariance_action(c,v) atol=2e-12
        @test block_density_gradient(materialized,v).value≈block_density_gradient(c,v).value atol=2e-12
        empirical=sum(abs2,centered'*v)
        projected=dot(v,block_covariance_action(c,v))
        push!(variance_relative_error,(projected-empirical)/empirical)
    end
    variances=zeros(length(permutation))
    for (b,F) in zip(c.blocks,c.factors)
        variances[b]=diag(Matrix(F.L*F.L'))+vec(sum(abs2,c.B[b,:];dims=2))
    end
    mean_error=abs.(c.mu-reference.ref_mean[permutation])./reference.ref_sd[permutation]
    sd_ratio=sqrt.(variances)./reference.ref_sd[permutation]
    posterior_tolerance=maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio)
    identity=(;fit_signature=fit.identity.signature,reference_signature=reference.signature,
        draw_file_sha=bytes2hex(sha256(read(joinpath(base,"native-fixed-rate-fit-v1/draws.jls")))) ,
        source_sha=bytes2hex(sha256(reduce(vcat,read.([@__FILE__,joinpath(@__DIR__,"lazy_core_probe.jl"),
            joinpath(LAZY_CARRY_RESEARCH,"block_projection_probe.jl"),
            joinpath(LAZY_CARRY_RESEARCH,"block_carry_probe.jl")])))),
        julia=string(VERSION),julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
        draw_selection=1:M,rank,delta,reference_covariances=:identity,direction_seed=715)
    payload=(;identity,core,store,order,coordinate_blocks=[fit.coordinates[b] for b in indices],
        projection_metadata=projection.metadata,
        scope="data-only factor algebra; missing frozen preprocessing/transport and not a public checkpoint")
    serialize(joinpath(directory,"factor.jls"),payload)
    restored=deserialize(joinpath(directory,"factor.jls"))
    check=auxiliary_active_carry(restored.core,restored.store,restored.order)
    @test block_density_gradient(check,first(eachcol(directions)))==
        block_density_gradient(materialized,first(eachcol(directions)))
    summary=(;identity,posterior_tolerance,mean_error,sd_ratio,variance_relative_error,
        scope="fixed first128 retained draws; mean/draw noise, rank truncation and shrinkage remain; no sequential fit or accuracy guarantee")
    serialize(joinpath(directory,"comparison.jls"),summary)
    println((;posterior_tolerance,max_mean_error=maximum(mean_error),sd_ratio_extrema=extrema(sd_ratio),
        directional_variance_relative_error=extrema(variance_relative_error),rank,draws=M,delta))
end
