# Fixed short trajectories exercise changing q; they do not establish a
# converged large-model posterior. No retries or changed numerical budgets.
const TRAJECTORY_RESEARCH=isfile(joinpath(@__DIR__,"scaling_target.jl")) ? (@__DIR__) :
    @__DIR__
include(joinpath(TRAJECTORY_RESEARCH,"scaling_target.jl"))

@stestset "native scaling trajectories" begin
    base=joinpath(@__DIR__, "results")
    directory=joinpath(base,"native-scaling-trajectory-v1")
    ispath(directory) && error("preserve scaling trajectory artifacts")
    names=("rows16","field16","units16","rows16_field16")
    # Inclusion and budgets are fixed before examining trajectory outcomes.
    all(name->isfile(joinpath(base,"native-scaling-preflight-v1",name,"result.jls")),names) ||
        error("finish all fixed-point cases before trajectory controls")
    preflights=map(names) do name
        p=deserialize(joinpath(base,"native-scaling-preflight-v1",name,"result.jls"))
        p.success && p.identity.config==only(c for c in SCALING_CASES if c.name==name) ||
            error("fixed-point prerequisite differs: $name")
        (;name,signature=p.identity.signature,commit=p.identity.repository_commit)
    end
    files=sort(filter(f->endswith(f,".jl"),readdir(TRAJECTORY_RESEARCH;join=true)))
    for (root,_,entries) in walkdir(joinpath(TRAJECTORY_RESEARCH,"../../src"))
        append!(files,[joinpath(root,n) for n in sort(entries) if endswith(n,".jl")])
    end
    append!(files,[@__FILE__,joinpath(TRAJECTORY_RESEARCH,"../../test/Manifest.toml")])
    identity=(;names,preflights,steps=64,seed=700,lr=.01,checkpoints=(0,16,64),
        signature=bytes2hex(sha256(reduce(vcat,read.(files)))),
        commit=readchomp(`git -C $TRAJECTORY_RESEARCH rev-parse HEAD`),
        julia=string(VERSION),julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
        trace_probes=8,logdet_probes=8,mode_krylovdim=128,trace_krylovdim=128,
        correction_krylovdim=64,logdet_krylovdim=32,draw_root_krylovdim=32,draw_solve_krylovdim=64)
    mkpath(directory);serialize(joinpath(directory,"identity.jls"),identity)
    results=NamedTuple[]
    with_progress(:state;description="Four fixed64-update native scaling trajectories") do progress
        snapshot_progress(progress,joinpath(directory,"trajectory.progress")) do
            for name in names
                config=only(c for c in SCALING_CASES if c.name==name)
                records=NamedTuple[];diagnostics=NamedTuple[];stage="construction";q=nothing;op=nothing
                case_directory=joinpath(directory,name);mkpath(case_directory)
                try
                    setup=@timed scaling_native_fixture(;groups=config.groups,rows=config.rows,
                        temporal=config.temporal,spatial=config.spatial)
                    f=setup.value;counted=counted_scaling_target(f.target);op=counted.op
                    construction_memory=(;reachable=Base.summarysize(f.target),
                        process=filter(l->startswith(l,"VmRSS:") || startswith(l,"VmHWM:"),readlines("/proc/self/status")))
                    t=PathDerivativeTarget(counted)
                    start=first(BRM._brm_nlme_prior_median_point(f.m))
                    q=BRM._brm_svi_params(t,start[t.globals],BRM.LowRankGlobal(3);global_scale=.2,unit_scale=.5)
                    opt=BRM._BRMSVIAdam(15,identity.lr);rng=Xoshiro(identity.seed)
                    function checkpoint(k)
                        u=copy(start);u[t.globals]=q.μ
                        measurement=@timed scaling_solver_diagnostics(counted,u)
                        push!(diagnostics,(;k,mu=copy(q.μ),solver=measurement.value,
                            seconds=measurement.time,bytes=measurement.bytes))
                    end
                    stage="initial solver diagnostic";checkpoint(0)
                    with_progress(progress,identity.steps;description="$name moving variational factor") do node
                        for k in 1:identity.steps
                            stage="update$k";fill!(op.counts,0)
                            measurement=@timed BRM._brm_svi_iteration!(rng,t,q,opt,Int[],identity.lr,.1;whiten=false)
                            raw=BRM._brm_svi_global_vector(q)
                            objective=measurement.value+BRM._brm_svi_entropy(q)
                            isfinite(objective) && all(isfinite,raw) || error("nonfinite trajectory update")
                            push!(records,(;k,seconds=measurement.time,bytes=measurement.bytes,
                                gctime=measurement.gctime,calls=(;value_gradient=op.counts[1],
                                    hvp=op.counts[2],third=op.counts[3]),raw=copy(raw),sampled_elbo=objective))
                            if k in identity.checkpoints
                                stage="checkpoint$k solver diagnostic";checkpoint(k)
                                serialize(joinpath(case_directory,"partial.jls"),(;identity,config,records,diagnostics,q))
                            end
                            k%8==0 && update_progress!(node,k)
                        end
                    end
                    t.iterations==identity.steps || error("trajectory count differs")
                    stage="endpoint conditional draw";fill!(op.counts,0)
                    draw=@timed BRM._brm_svi_draws(Xoshiro(701),counted,q,1)
                    all(isfinite,draw.value) || error("nonfinite endpoint draw")
                    endpoint_draw=(;value=draw.value,seconds=draw.time,bytes=draw.bytes,
                        calls=(;value_gradient=op.counts[1],hvp=op.counts[2],third=op.counts[3]))
                    result=(;identity,config,success=true,records,diagnostics,q,snapshots=t.snapshots,
                        construction_memory,endpoint_draw,
                        setup=(;seconds=setup.time,bytes=setup.bytes,gctime=setup.gctime),
                        reachable_julia_target_bytes=Base.summarysize(f.target),
                        process_memory=filter(l->startswith(l,"VmRSS:") || startswith(l,"VmHWM:"),readlines("/proc/self/status")),
                        first_update_seconds=first(records).seconds,
                        subsequent_update_seconds=extrema(getproperty.(records[2:end],:seconds)),
                        subsequent_update_median=median(getproperty.(records[2:end],:seconds)),
                        hvp_calls=extrema([r.calls.hvp for r in records]),
                        scope="one same-process case sequence; checkpoint0 warms solver before first optimizer update; process HWM is cumulative across cases;64 changing draws/factors, not independent timing samples or full-fit/posterior acceptance; all solve budgets unchanged")
                    serialize(joinpath(case_directory,"result.jls"),result);push!(results,result)
                    println((;case=name,result.success,result.first_update_seconds,
                        result.subsequent_update_seconds,result.subsequent_update_median,result.hvp_calls));flush(stdout)
                catch exception
                    exception isa InterruptException && rethrow()
                    result=(;identity,config,success=false,stage,records,diagnostics,q,
                        counts=op===nothing ? nothing : copy(op.counts),
                        last_completed_raw=isempty(records) ? nothing : copy(last(records).raw),
                        error=sprint(showerror,exception,catch_backtrace()))
                    serialize(joinpath(case_directory,"failure.jls"),result);push!(results,result)
                    println((;case=name,success=false,stage,completed=length(records),error=result.error));flush(stdout)
                    # Independent later shapes continue without retrying this case.
                end
            end
        end
    end
    serialize(joinpath(directory,"summary.jls"),(;identity,
        results=[(;name=r.config.name,success=r.success,completed=length(r.records)) for r in results]))
    for r in results
        @test r.success
    end
end
