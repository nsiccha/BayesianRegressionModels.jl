# Public fit against the already accepted exact Gaussian-marginal reference.
# No reference resampling or post-hoc optimizer changes. Run twice in fresh
# processes, first `fit`, then `reproduce`; artifacts are never overwritten.
# `integrated` compares a later integrated candidate with the retained fit,
# allowing the source identity to change while keeping all numerical checks.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL,
    Enzyme, DifferentiationInterface, Random, LinearAlgebra, Statistics,
    Serialization, SHA, Test, Treebars, Distributions
const BRM=BayesianRegressionModels
include("public_fixture.jl")

function public_laplace_identity(reference)
    root=normpath(joinpath(@__DIR__,"../.."))
    files=sort(vcat([@__FILE__,joinpath(@__DIR__,"public_fixture.jl")],
        filter(f->startswith(basename(f),"nlme_svi"),readdir(joinpath(root,"src");join=true)),
        [joinpath(root,p) for p in ("src/BayesianRegressionModels.jl",
            "ext/rk_svi_laplace.jl","ext/BayesianRegressionModelsReactiveKernelsEnzymeExt.jl",
            "test/Manifest.toml")]))
    (;source_signature=bytes2hex(sha256(reduce(vcat,read.(files)))),
        reference_signature=reference.signature,julia=string(VERSION),
        repository_commit=readchomp(`git -C $root rev-parse HEAD`),
        fit_seed=700,draw_seed=701,iterations=5000,draws=1000,
        julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
        enzyme=pkgdir(Enzyme),libenzyme=Enzyme.Compiler.Enzyme_jll.libEnzyme_path)
end

function public_laplace_checkpoint_summary(state)
    order=sort(collect(keys(state.blocks));by=repr)
    (;state.format,state.source_id,state.globals,state.shared_coordinates,
        state.units,state.group,state.nested,state.shared,state.method,state.projection,state.updates,
        core=(;state.core.m,L=Matrix(state.core.F.L)),
        blocks=[(;key,a=state.blocks[key].a,B=state.blocks[key].B,
            L=Matrix(state.blocks[key].F.L)) for key in order])
end

@testset "public conditional Gaussian SVI acceptance" begin
    mode=only(ARGS);mode in ("fit","reproduce","integrated") || error("use fit, reproduce or integrated")
    base=joinpath(@__DIR__, "results")
    reference=deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    reference.reference_ok || error("reference gate is not green")
    directory=joinpath(base,"public-laplace-"*mode*"-v1")
    ispath(directory) && error("preserve existing public acceptance evidence")
    mkpath(directory)
    identity=public_laplace_identity(reference)
    serialize(joinpath(directory,"identity.jls"),identity)
    previous=mode=="fit" ? nothing : deserialize(joinpath(base,"public-laplace-fit-v1/result.jls"))
    mode!="reproduce" || previous.identity.source_signature==identity.source_signature ||
        error("public source/environment changed between fit and reproduction")
    if mode=="integrated"
        @test previous.identity.reference_signature==identity.reference_signature
        @test success(`git merge-base --is-ancestor $(previous.identity.repository_commit) $(identity.repository_commit)`)
    end
    fixture=public_laplace_fixture()
    seconds=@elapsed fit=with_progress(:state;description="Public conditional Gaussian SVI") do progress
        snapshot_progress(progress,joinpath(directory,"fit.progress")) do
            brm_svi_fit(fixture.builder(fixture.data);
                laplace=GaussianBlockLaplace(),family=LowRankGlobal(3),
                ad_backend=AutoEnzyme(;mode=Enzyme.Reverse),rng=Xoshiro(700),draw_rng=Xoshiro(701),
                max_iterations=5000,lr_global=.01,global_scale=.2,ndraws=1000,progress)
        end
    end
    state=brm_svi_checkpoint(fit)
    checkpoint_file=joinpath(directory,"checkpoint.jls")
    serialize(checkpoint_file,state)
    checkpoint_sha256=bytes2hex(sha256(read(checkpoint_file)))
    @test fit.coordinates==reference.coordinates
    mean_error=abs.(vec(mean(fit.draws;dims=2))-reference.ref_mean)./reference.ref_sd
    sd_ratio=vec(std(fit.draws;dims=2))./reference.ref_sd
    draw_accuracy=maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio)
    t=BRM._brm_svi_laplace_target(fit.diagnostics.model,GaussianBlockLaplace())
    layout=BRM._brm_svi_laplace_layout(t)
    packed=BRM._brm_svi_carry_active_marginal(state.core,state.blocks,layout.order)
    physical=reduce(vcat,layout.blocks;init=Int[])
    carried=BRM._brm_svi_carry_marginal(packed,invperm(physical))
    # Diagonal moments use bounded residual blocks and factor rows only.
    variances=vec(sum(abs2,carried.B;dims=2))
    for (indices,F) in zip(carried.blocks,carried.factors)
        variances[indices].+=vec(sum(abs2,Matrix(F.L);dims=2))
    end
    carry_mean_error=abs.(carried.mu-reference.ref_mean)./reference.ref_sd
    carry_sd_ratio=sqrt.(variances)./reference.ref_sd
    carry_accuracy=maximum(carry_mean_error)<=.3 && all(x->.8<=x<=1.2,carry_sd_ratio)
    d=fit.diagnostics
    summary=public_laplace_checkpoint_summary(state)
    result=(;identity,seconds,coordinates=fit.coordinates,draws=fit.draws,
        q=BRM._brm_svi_global_factor(d.q),elbo=d.elbo,mean_error,sd_ratio,
        carry_mean_error,carry_sd_ratio,draw_accuracy,carry_accuracy,checkpoint=summary,checkpoint_sha256,
        optimizer=(;d.iterations,d.converged,d.convergence_assessed,d.budget_exhausted,
            d.stop_reason,d.averaging_window),laplace=d.laplace)
    serialize(joinpath(directory,"result.jls"),result)
    println((;seconds,draw_accuracy,carry_accuracy,max_mean_error=maximum(mean_error),
        sd_ratio=extrema(sd_ratio),max_carry_mean_error=maximum(carry_mean_error),
        carry_sd_ratio=extrema(carry_sd_ratio),result.optimizer))
    @test draw_accuracy
    @test carry_accuracy
    @test !d.converged && !d.convergence_assessed && d.budget_exhausted
    @test d.iterations==5000 && d.averaging_window==4801:5000
    if previous!==nothing
        @test result.draws==previous.draws
        @test result.q==previous.q
        @test result.elbo==previous.elbo
        @test isequal(result.checkpoint,previous.checkpoint)
        @test result.optimizer==previous.optimizer
        @test result.laplace==previous.laplace
    end
end
