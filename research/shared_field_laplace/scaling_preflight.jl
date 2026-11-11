# Measure the actual native primitives and complete update/draw actions before
# launching larger fits. Per-axis scaling is not joint O(n+d), and fixed-point
# costs do not certify costs along an optimization path. No accuracy claim.
# SHARED_LAPLACE_SCALING_CASE=base julia --project=test research/shared_field_laplace/scaling_preflight.jl 'native scaling preflight'
include("scaling_target.jl")

@stestset "native scaling preflight" begin
    name=get(ENV,"SHARED_LAPLACE_SCALING_CASE","base")
    config=only(c for c in SCALING_CASES if c.name==name)
    base=joinpath(@__DIR__, "results")
    directory=joinpath(base,"native-scaling-preflight-v1",name)
    ispath(directory) && error("preserve existing scaling preflight: $name")
    mkpath(directory)
    files=sort(filter(f->endswith(f,".jl"),readdir(@__DIR__;join=true)))
    for (root,_,names) in walkdir(joinpath(@__DIR__,"../../src"))
        append!(files,[joinpath(root,n) for n in sort(names) if endswith(n,".jl")])
    end
    push!(files,joinpath(@__DIR__,"../../test/Manifest.toml"))
    identity=(;config,signature=bytes2hex(sha256(reduce(vcat,read.(files)))),
        repository_commit=readchomp(`git -C $(@__DIR__) rev-parse HEAD`),
        repository_status=readchomp(`git -C $(@__DIR__) status --porcelain`),
        julia=string(VERSION),julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
        trace_probes=8,logdet_probes=8,mode_krylovdim=128,trace_krylovdim=128,
        correction_krylovdim=64,logdet_krylovdim=32,draw_root_krylovdim=32,
        draw_solve_krylovdim=64,repetitions=7,point_shifts=([0.,0.,0.],[.2,-.4,.1]))
    serialize(joinpath(directory,"identity.jls"),identity)
    stage="construction"
    result=nothing
    try
        with_progress(:state;description="Native scaling preflight $name") do progress
            snapshot_progress(progress,joinpath(directory,"preflight.progress")) do
                measurement=@timed with_progress(progress;description="Construct fixture and native graph") do _
                    scaling_native_fixture(;groups=config.groups,rows=config.rows,
                        temporal=config.temporal,spatial=config.spatial)
                end
                fixture=measurement.value
                (;rk,m,data,target,field,units,beta,latent,globals)=fixture
                setup=(;seconds=measurement.time,bytes=measurement.bytes,gctime=measurement.gctime)
                serialize(joinpath(directory,"setup.jls"),(;identity,setup,coordinates=m.view.coordinates,
                    field,units,beta,latent,globals))
                @test sort(vcat(globals,latent))==collect(eachindex(m.unit))
                @test isempty(target.units) && length(globals)==3
                if name=="base"
                    stage="bounded selection control"
                    oracle=exact_gaussian_reference(rk,data)
                    block=affine_gaussian_block(rk,data,oracle.p,oracle.view)
                    @test globals==block.p3.globals && latent==block.p3.latent && beta==block.beta
                    globals==block.p3.globals && latent==block.p3.latent && beta==block.beta ||
                        error("fixture selection differs from validated bounded target")
                end
                counted=counted_scaling_target(target)
                op=counted.op
                start=first(BRM._brm_nlme_prior_median_point(m))
                results=NamedTuple[]
                with_progress(progress,2;description="Fixed point derivative/update/draw costs") do node
                    for (i,shift) in enumerate(identity.point_shifts)
                        u=copy(start);u[globals].+=shift
                        v=randn(Xoshiro(710),length(latent))
                        stage="point$i value/gradient"
                        value_gradient=scaling_measure(()->shared_value_gradient(op,u),op)
                        stage="point$i HVP"
                        hvp=scaling_measure(()->shared_likelihood_hvp(op,u,v),op)
                        stage="point$i third"
                        third=scaling_measure(()->shared_third(op,u,v),op)
                        q=BRM._brm_svi_params(counted,u[globals],BRM.LowRankGlobal(3);global_scale=.2,unit_scale=.5)
                        stage="point$i complete update"
                        function update()
                            t=PathDerivativeTarget(counted)
                            candidate=deepcopy(q)
                            opt=BRM._BRMSVIAdam(15,.01)
                            BRM._brm_svi_iteration!(Xoshiro(700),t,candidate,opt,Int[],.01,.1)
                        end
                        update_cost=scaling_measure(update,op)
                        stage="point$i conditional draw"
                        draw_cost=scaling_measure(()->BRM._brm_svi_draws(Xoshiro(701),counted,q,1),op)
                        stage="point$i solver diagnostics"
                        solver=scaling_solver_diagnostics(counted,u)
                        push!(results,(;shift,value_gradient,hvp,third,update_cost,draw_cost,solver))
                        serialize(joinpath(directory,"partial.jls"),(;identity,setup,results))
                        update_progress!(node,i)
                    end
                end
                result=(;identity,setup,results,nrows=length(data.y),nfield=length(field),
                    nunits=length(units),nlatent=length(latent),success=true,
                    reachable_julia_target_bytes=Base.summarysize(target),
                    process_memory=filter(l->startswith(l,"VmRSS:") || startswith(l,"VmHWM:"),
                        readlines("/proc/self/status")),
                    scope="fixed-point native actions only; repeated same workloads, not independent samples; updates include fresh optimizer/factor/first snapshot; reachable Julia memory excludes native/compiler storage; process memory includes all earlier work and the base-only oracle; no scaled fit, accuracy, or uniform iteration-cost bound")
                serialize(joinpath(directory,"result.jls"),result)
                println((;config,setup,nrows=result.nrows,nfield=result.nfield,nlatent=result.nlatent,
                    costs=[(;shift=r.shift,update=r.update_cost.median_seconds,
                        bytes=r.update_cost.median_bytes,calls=first(r.update_cost.records).calls,
                        draw=r.draw_cost.median_seconds) for r in results]))
            end
        end
    catch exception
        failure=(;identity,stage,success=false,error=sprint(showerror,exception,catch_backtrace()))
        serialize(joinpath(directory,"failure.jls"),failure)
        println((;config,stage,error=failure.error));flush(stdout)
        rethrow()
    end
    @test result.success
end
