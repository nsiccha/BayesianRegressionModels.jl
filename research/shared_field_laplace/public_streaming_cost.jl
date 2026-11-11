# Fixed active window, growing synthetic absent-unit state. This is a cost
# and marginalization control, not a multi-window posterior accuracy study.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL,
    Enzyme, DifferentiationInterface, Random, LinearAlgebra, Statistics,
    Serialization, SHA, Test, Treebars, Distributions
const BRM=BayesianRegressionModels
include("public_fixture.jl")

function public_cost_measure(action,t;repetitions=5)
    function once()
        fill!(t.counts,0);empty!(t.numerical)
        trial=@timed action()
        record=(;seconds=trial.time,bytes=trial.bytes,gctime=trial.gctime,
            calls=copy(t.counts),numerical=copy(t.numerical))
        record,trial.value
    end
    warmup,_=once();records=NamedTuple[];value=nothing
    for _ in 1:repetitions
        record,value=once();push!(records,record)
    end
    (;warmup,records,median_seconds=median(getproperty.(records,:seconds)),
        median_bytes=median(getproperty.(records,:bytes))),value
end

function public_cost_expanded(previous,n)
    state=deepcopy(previous)
    template=first(sort(collect(keys(previous.units))))
    for j in 1:n-length(previous.units)
        # A valid extension of the same Gaussian joint: each absent unit has
        # its own residual and the copied loading on the existing common core.
        # Integrating these extra units out leaves the active marginal intact.
        # Copied metadata does not establish replay under the synthetic label;
        # these added units remain absent throughout this cost experiment.
        label=1000+j
        state.units[label]=deepcopy(previous.units[template])
        state.blocks[(:unit,label)]=deepcopy(previous.blocks[(:unit,template)])
    end
    state
end

@testset "public Laplace SVI retained-state cost" begin
    base=joinpath(@__DIR__, "results")
    acceptance_file=joinpath(base,"public-laplace-fit-v1/result.jls")
    checkpoint_file=joinpath(base,"public-laplace-fit-v1/checkpoint.jls")
    accepted=deserialize(acceptance_file)
    accepted.draw_accuracy && accepted.carry_accuracy || error("public posterior gate must pass first")
    checkpoint_sha256=bytes2hex(sha256(read(checkpoint_file)))
    checkpoint_sha256==accepted.checkpoint_sha256 || error("checkpoint differs from accepted artifact")
    previous=deserialize(checkpoint_file)
    @test previous.source_id==accepted.checkpoint.source_id
    @test previous.method==accepted.checkpoint.method && previous.q==accepted.q
    directory=joinpath(base,"public-streaming-cost-v2")
    ispath(directory) && error("preserve existing retained-state cost evidence")
    mkpath(directory)
    root=normpath(joinpath(@__DIR__,"../.."))
    files=[@__FILE__,joinpath(@__DIR__,"public_fixture.jl"),joinpath(root,"test/Manifest.toml")]
    for folder in ("src","ext"), (path,_,names) in walkdir(joinpath(root,folder))
        append!(files,[joinpath(path,n) for n in names if endswith(n,".jl")])
    end
    identity=(;source_signature=bytes2hex(sha256(reduce(vcat,read.(sort(files))))),
        repository_commit=readchomp(`git -C $root rev-parse HEAD`),
        checkpoint_source=accepted.identity,checkpoint_sha256,
        acceptance_sha256=bytes2hex(sha256(read(acceptance_file))),retained_counts=(6,60,600,6000),
        repetitions=5,draws=128,julia=string(VERSION),julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads())
    serialize(joinpath(directory,"identity.jls"),identity)
    fixture=public_laplace_fixture()
    builder=fixture.builder
    labels=[3,4,5,6,7,8];rows=12
    # Preserve the original fitted coordinates bit for bit. Recomputing the
    # scalar trigonometric expressions differs from the range-based fixture
    # by a few ulps at some levels; frozen replay correctly rejects that.
    locations=Dict(i=>(fixture.data.sx[i],fixture.data.sy[i]) for i in 1:6)
    locations[7]=(0.,0.);locations[8]=(.1,.2)
    data=(;probe=repeat(labels;inner=rows),time=repeat(collect(range(.15,1.95;length=rows)),6),
        sx=[locations[i][1] for i in labels],sy=[locations[i][2] for i in labels],
        y=randn(Xoshiro(750),6rows))
    stage="frozen current-window construction"
    try
        with_progress(:state;description="Fixed-window carried-state costs") do progress
            snapshot_progress(progress,joinpath(directory,"cost.progress")) do
                m=brm_nlme_model(RKBRMI(builder(data),previous.preprocessing;resample_groups=(:probe,));
                    ad_backend=AutoEnzyme(;mode=Enzyme.Reverse))
                original=BRM._brm_svi_laplace_target(m,previous.method)
                layout=BRM._brm_svi_laplace_layout(original)
                initial=BRM._brm_svi_laplace_active(original,previous,layout)
                target=BRM._brm_svi_laplace_with_carry(original,initial.carry)
                q=BRM._brm_svi_params(target,previous.q.μ,previous.q.family;global_scale=1.,unit_scale=1.)
                q.ρ.=previous.q.ρ;q.Β.=previous.q.Β
                stage="fixed conditional draws"
                points=BRM._brm_svi_draws(Xoshiro(751),target,q,128)
                records=NamedTuple[];baseline=nothing;baseline_calls=nothing;baseline_checkpoint=nothing
                with_progress(progress,length(identity.retained_counts);description="Retained-unit counts") do node
                    for (index,n) in enumerate(identity.retained_counts)
                        stage="retained$n state construction"
                        state=public_cost_expanded(previous,n)
                        @test length(state.units)==n
                        # This is the same copy performed by public continuation
                        # at entry; its O(retained units) boundary cost is explicit.
                        copy_cost,_=public_cost_measure(()->brm_svi_checkpoint(state),original)
                        active_cost,active=public_cost_measure(
                            ()->BRM._brm_svi_laplace_active(original,state,layout),original)
                        @test active.carry.mu==initial.carry.mu
                        @test active.carry.B==initial.carry.B
                        @test active.carry.blocks==initial.carry.blocks
                        @test all(a.L==b.L for (a,b) in zip(active.carry.factors,initial.carry.factors))
                        current=BRM._brm_svi_laplace_with_carry(original,active.carry)
                        stage="retained$n update"
                        update,result=public_cost_measure(current) do
                            candidate=deepcopy(q)
                            opt=BRM._BRMSVIAdam(length(BRM._brm_svi_global_vector(candidate)),.01)
                            value=BRM._brm_svi_laplace_iteration!(Xoshiro(752),current,candidate,opt,.01)
                            (;value,q=BRM._brm_svi_global_vector(candidate))
                        end
                        baseline===nothing && (baseline=result;baseline_calls=first(update.records).calls)
                        @test result==baseline
                        @test all(r.calls==baseline_calls for r in update.records)
                        serialize(joinpath(directory,"retained$(n)-update.jls"),
                            (;identity,retained=n,copy_cost,active_cost,update,result))
                        stage="retained$n conditional projection and checkpoint"
                        checkpoint_build,checkpoint=public_cost_measure(
                            ()->BRM._brm_svi_laplace_checkpoint(Xoshiro(753),current,q,points,state,layout,active.active),current)
                        baseline_checkpoint===nothing && (baseline_checkpoint=checkpoint)
                        @test checkpoint.core.m==baseline_checkpoint.core.m
                        @test checkpoint.core.F.L==baseline_checkpoint.core.F.L
                        @test all(checkpoint.blocks[k].a==baseline_checkpoint.blocks[k].a &&
                            checkpoint.blocks[k].B==baseline_checkpoint.blocks[k].B &&
                            checkpoint.blocks[k].F.L==baseline_checkpoint.blocks[k].F.L for k in layout.order)
                        stage="retained$n serialization"
                        serialization,nbytes=public_cost_measure(current) do
                            io=IOBuffer();serialize(io,checkpoint);position(io)
                        end
                        @test length(checkpoint.units)==n+2
                        @test isempty(records) ||
                            all(r.calls==first(records).checkpoint_build.records[1].calls for r in checkpoint_build.records)
                        push!(records,(;retained=n,active_units=length(m.view.levels),nrows=length(data.y),
                            copy_cost,active_cost,update,checkpoint_build,serialization,serialized_bytes=nbytes,
                            checkpoint_julia_bytes=Base.summarysize(checkpoint)))
                        serialize(joinpath(directory,"partial.jls"),(;identity,records))
                        update_progress!(node,index)
                    end
                end
                result=(;identity,records,
                    scope="synthetic absent Gaussian conditional blocks, not replay-ready identities; added units remain absent; fixed active window and factor; exact action/call/projected-active-state equality, no streaming posterior accuracy claim; checkpoint construction includes retained dictionary copies; copying and serialization visit retained state; warmed repeats are the same workload, not independent samples")
                serialize(joinpath(directory,"result.jls"),result)
                println([(;r.retained,r.active_units,r.nrows,copy_seconds=r.copy_cost.median_seconds,
                    update_seconds=r.update.median_seconds,checkpoint_seconds=r.checkpoint_build.median_seconds,
                    serialization_seconds=r.serialization.median_seconds,r.serialized_bytes) for r in records])
            end
        end
    catch exception
        serialize(joinpath(directory,"failure.jls"),(;identity,stage,error=sprint(showerror,exception,catch_backtrace())))
        rethrow()
    end
end
