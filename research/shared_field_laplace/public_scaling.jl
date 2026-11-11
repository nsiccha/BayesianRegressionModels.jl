# Public implementation, fixed numerical limits, predeclared extreme cases.
# Run each case in a fresh process; no dense reference or posterior claim.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL,
    Enzyme, DifferentiationInterface, Random, LinearAlgebra, Statistics,
    Serialization, SHA, Test, Treebars, Distributions
const BRM=BayesianRegressionModels
include("public_fixture.jl")
const PUBLIC_SCALING_CASES=(
    (;name="base",groups=6,rows=12,temporal=4,spatial=2),
    (;name="rows16",groups=6,rows=192,temporal=4,spatial=2),
    (;name="field16",groups=6,rows=12,temporal=64,spatial=2),
    (;name="units16",groups=96,rows=12,temporal=4,spatial=2),
    (;name="rows16_field16",groups=6,rows=192,temporal=64,spatial=2))

function public_scaling_fixture(config)
    (;groups,rows,temporal,spatial)=config
    (;builder,data)=public_laplace_fixture(;groups,rows,temporal,spatial)
    m=brm_nlme_model(RKBRMI(builder(data));ad_backend=AutoEnzyme(;mode=Enzyme.Reverse))
    (;m,data,target=BRM._brm_svi_laplace_target(m,GaussianBlockLaplace()))
end

function public_scaling_measure(action,t;repetitions=5)
    function once()
        fill!(t.counts,0);empty!(t.numerical)
        result=@timed action()
        (;seconds=result.time,bytes=result.bytes,gctime=result.gctime,
            calls=copy(t.counts),numerical=copy(t.numerical))
    end
    first_use=once()
    records=[once() for _ in 1:repetitions]
    (;first_use,records,median_seconds=median(getproperty.(records,:seconds)),
        median_bytes=median(getproperty.(records,:bytes)))
end

@testset "public conditional Gaussian scaling" begin
    name=only(ARGS);config=only(c for c in PUBLIC_SCALING_CASES if c.name==name)
    root=normpath(joinpath(@__DIR__,"../.."))
    files=[@__FILE__,joinpath(@__DIR__,"public_fixture.jl"),joinpath(root,"test/Manifest.toml")]
    for folder in ("src","ext"), (directory,_,names) in walkdir(joinpath(root,folder))
        append!(files,[joinpath(directory,n) for n in names if endswith(n,".jl")])
    end
    identity=(;config,source_signature=bytes2hex(sha256(reduce(vcat,read.(sort(files))))),
        repository_commit=readchomp(`git -C $root rev-parse HEAD`),
        julia=string(VERSION),julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
        method=GaussianBlockLaplace(),repetitions=5,trajectory_updates=64,
        point_shifts=([0.,0.,0.],[.2,-.4,.1]))
    directory=joinpath(joinpath(@__DIR__, "results"),
        "public-scaling-v1",name)
    ispath(directory) && error("preserve existing public scaling evidence: $name")
    mkpath(directory);serialize(joinpath(directory,"identity.jls"),identity)
    stage="construction"
    try
        with_progress(:state;description="Public Gaussian scaling $name") do progress
            snapshot_progress(progress,joinpath(directory,"scaling.progress")) do
                setup=@timed public_scaling_fixture(config)
                (;m,data,target)=setup.value;t=target;c=t.certificate
                @test length(c.globals)==3
                @test length(c.field)==config.temporal*config.spatial^2
                @test length(c.unit_blocks)==config.groups
                start=first(BRM._brm_nlme_prior_median_point(m))
                results=NamedTuple[]
                with_progress(progress,2;description="Fixed-point public actions") do node
                    for (i,shift) in enumerate(identity.point_shifts)
                        point=copy(start);point[t.globals].+=shift
                        v=randn(Xoshiro(710),length(c.latent))
                        stage="point$i primitives"
                        value_gradient=public_scaling_measure(()->BRM._brm_svi_laplace_value_gradient(t,point),t)
                        hvp=public_scaling_measure(()->BRM._brm_svi_laplace_likelihood_hvp(t,point,v),t)
                        third=public_scaling_measure(()->BRM._brm_svi_laplace_third(t,point,v),t)
                        q=BRM._brm_svi_params(t,point[t.globals],LowRankGlobal(3);global_scale=.2,unit_scale=1.)
                        stage="point$i complete update"
                        update=public_scaling_measure(t) do
                            candidate=deepcopy(q)
                            opt=BRM._BRMSVIAdam(length(BRM._brm_svi_global_vector(candidate)),.01)
                            BRM._brm_svi_laplace_iteration!(Xoshiro(700),t,candidate,opt,.01)
                        end
                        stage="point$i joint draw"
                        draw=public_scaling_measure(()->BRM._brm_svi_draws(Xoshiro(701),t,q,1),t)
                        push!(results,(;shift,value_gradient,hvp,third,update,draw))
                        serialize(joinpath(directory,"partial.jls"),(;identity,results))
                        update_progress!(node,i)
                    end
                end
                stage="64 moving-factor updates"
                q=BRM._brm_svi_params(t,start[t.globals],LowRankGlobal(3);global_scale=.2,unit_scale=1.)
                fill!(t.counts,0);empty!(t.numerical)
                path=@timed BRM._brm_svi_laplace_optimize!(Xoshiro(700),t,q;
                    max_iterations=64,lr_global=.01,node=progress)
                path_calls=copy(t.counts);path_numerical=copy(t.numerical)
                trajectory=(;seconds=path.time,bytes=path.bytes,calls=path_calls,numerical=path_numerical)
                serialize(joinpath(directory,"trajectory.jls"),(;identity,trajectory,
                    q=BRM._brm_svi_global_factor(path.value.q),elbo=path.value.elbo))
                layout=BRM._brm_svi_laplace_layout(t)
                stage="128 conditional joint draws"
                fill!(t.counts,0);empty!(t.numerical)
                draws=@timed BRM._brm_svi_draws(Xoshiro(701),t,path.value.q,128)
                draw_summary=(;seconds=draws.time,bytes=draws.bytes,
                    calls=copy(t.counts),numerical=copy(t.numerical))
                serialize(joinpath(directory,"draws.jls"),(;identity,draws=draw_summary))
                stage="checkpoint projection"
                fill!(t.counts,0);empty!(t.numerical)
                projection=@timed BRM._brm_svi_laplace_checkpoint(Xoshiro(702),t,path.value.q,
                    draws.value,nothing,layout,nothing)
                result=(;identity,results,setup=(;seconds=setup.time,bytes=setup.bytes,gctime=setup.gctime),
                    trajectory,draws=draw_summary,
                    projection=(;seconds=projection.time,bytes=projection.bytes,metadata=projection.value.projection,
                        calls=copy(t.counts),numerical=copy(t.numerical)),
                    nrows=length(data.y),nfield=length(c.field),nlatent=length(c.latent),
                    reachable_julia_target_bytes=Base.summarysize(t),
                    checkpoint_bytes=Base.summarysize(projection.value),
                    process_memory=filter(l->startswith(l,"VmRSS:") || startswith(l,"VmHWM:"),readlines("/proc/self/status")),
                    scope="repeated fixed-point workloads and 64 moving-factor updates; no scaled posterior accuracy or convergence claim; per-axis scaling is not joint linear cost; process HWM includes construction and all earlier actions")
                serialize(joinpath(directory,"result.jls"),result)
                println((;config,nrows=result.nrows,nfield=result.nfield,
                    update_seconds=[r.update.median_seconds for r in results],
                    draw_seconds=[r.draw.median_seconds for r in results],
                    result.trajectory,result.draws,result.projection))
                @test all(isfinite,draws.value)
                @test !path.value.converged && path.value.budget_exhausted
            end
        end
    catch exception
        serialize(joinpath(directory,"failure.jls"),(;identity,stage,
            error=sprint(showerror,exception,catch_backtrace())))
        rethrow()
    end
end
