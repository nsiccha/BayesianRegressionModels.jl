# Native iterative embedded integration of the structurally selected Gaussian
# block, with three hyperparameters fit by the existing SVI optimizer. This
# remains a bounded research candidate, not a public streaming entry point.
# Uses the original fit budget/start rule and posterior tolerance, without a
# warm start from the saved successful conditional reconstruction/reference.
# julia --project=test research/shared_field_laplace/gaussian_block_fit.jl 'native Gaussian block SVI'
include("gaussian_block_comparison.jl")
include("whole_window_probe.jl")

function native_gaussian_block_target(m,block)
    # The existing operator adds an identity Gaussian precision. Verify the
    # authored population prior too; a general Normal needs its actual scale.
    block.prior.location==0 && block.prior.scale==1 ||
        error("native research target requires the authored standard Normal population prior")
    native=whole_window_operator(m)
    latent=block.p3.latent
    latent==sort(vcat(native.field,m.view.subject_coordinates...,[block.beta])) ||
        error("native and structural Gaussian selections differ")
    t=(;globals=block.p3.globals,units=Vector{Int}[])
    op=WholeWindowOperator(m,t,latent,native.full,native.hvp,native.third)
    # Previously two identical points each used four logdet probes. With no
    # unit factors, use one point with all eight; trace probes remain eight.
    JointWindowTarget(op,m,t.globals,t.units,8,8)
end

# Complete probes are confined to the <=32-coordinate deterministic control.
# The fitted target below uses fixed-count random trace and logdet probes.
function native_gaussian_block_control(t,u)
    op=t.op;d=length(op.field)
    d<=32 || error("complete-probe control exceeds dimension cap")
    basis=Matrix{Float64}(I,d,d)
    state=shared_poisson_mode(op,u;tol=1e-9)
    trace=shared_trace(op,state.mode,[sqrt(d)*basis[:,j] for j in 1:d]).trace
    H(v)=shared_precision(op,state.mode,v)
    direction,info=KrylovKit.linsolve(H,trace[op.field];issymmetric=true,
        isposdef=true,rtol=1e-11,atol=1e-13,krylovdim=64,maxiter=1)
    info.converged==1 || error("native Gaussian correction solve failed")
    gradient=state.gradient+(trace+shared_likelihood_hvp(op,state.mode,direction))/2
    determinant=[shared_logquadratic(H,basis[:,j];spectral_lower=1.,maxdim=32,
        atol=1e-10,rtol=1e-10) for j in 1:d]
    all(x->x.converged,determinant) || error("native control exhausted logdet budget")
    value=state.value+d*log(2pi)/2-sum(x.value for x in determinant)/2
    (;value,gradient,mode=state.mode)
end

@stestset "native Gaussian block SVI" begin
    base=joinpath(@__DIR__, "results")
    reference=deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    reference.reference_ok || error("reference diagnostic gate failed")
    directory=joinpath(base,"native-gaussian-block-fit-v1")
    ispath(directory) && error("preserve existing native Gaussian block fit")
    mkpath(directory)
    files=[joinpath(@__DIR__,name) for name in
        ("gaussian_block_fit.jl","gaussian_block_comparison.jl","stationarity_audit.jl",
         "exact_reference.jl","operator_probe.jl","whole_window_probe.jl",
         "joint_svi_probe.jl","nonlinear_probe.jl","trace_probe.jl","quadrature.jl")]
    append!(files,[joinpath(@__DIR__,"../../src",name) for name in
        ("nlme_view.jl","nlme_svi.jl","rk_backend.jl")])
    push!(files,joinpath(@__DIR__,"../../test/Manifest.toml"))
    identity=(;signature=bytes2hex(sha256(reduce(vcat,read.(files)))),
        reference_signature=reference.signature,julia=string(VERSION),
        enzyme=pkgdir(Enzyme),libenzyme=Enzyme.Compiler.Enzyme_jll.libEnzyme_path,
        fit_seed=700,draw_seed=701,max_iterations=5000,ndraws=1000,
        trace_probes=8,logdet_probes=8)
    serialize(joinpath(directory,"identity.jls"),identity)
    with_progress(:state;description="Native conditional Gaussian block SVI") do progress
        snapshot_progress(progress,joinpath(directory,"fit.progress")) do
            fixture=with_progress(progress;description="Prepare native and bounded reference targets") do _
                (;rk,m,data)=shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2)
                merge((;rk,m,data),exact_gaussian_reference(rk,data))
            end
            (;rk,m,data,p,view,field,units)=fixture
            view.coordinates==reference.coordinates || error("reference coordinate identity failed")
            block=affine_gaussian_block(rk,data,p,view)
            t=native_gaussian_block_target(m,block)
            @test isempty(t.units) && length(t.globals)==3 && length(t.op.field)==23
            @test sort(vcat(t.globals,t.op.field))==collect(eachindex(m.unit))
            checks=with_progress(progress,3;description="Native value, mode and marginal gradient") do node
                map(enumerate((zeros(3),[.2,-.4,.1],[-.3,.2,.4]))) do (i,shift)
                    h=p.start[t.globals]+shift
                    u=copy(p.start);u[t.globals]=h
                    native=native_gaussian_block_control(t,u)
                    exact=gaussian_reference_conditional(block.p3,h)
                    value,gradient=LogDensityProblems.logdensity_and_gradient(block.p3,h)
                    @test native.value≈value atol=2e-7
                    @test native.mode[t.op.field]≈exact.mu atol=2e-8
                    @test native.gradient[t.globals]≈gradient atol=2e-7 rtol=2e-7
                    update_progress!(node,i)
                    (;value_error=abs(native.value-value),
                        mode_error=maximum(abs.(native.mode[t.op.field]-exact.mu)),
                        gradient_error=maximum(abs.(native.gradient[t.globals]-gradient)))
                end
            end
            serialize(joinpath(directory,"controls.jls"),(;identity,checks,prior=block.prior,
                coordinates=view.coordinates,latent=t.op.field,globals=t.globals))
            all(c->c.value_error<2e-7 && c.mode_error<2e-8 && c.gradient_error<2e-7,checks) ||
                error("native Gaussian block control failed; no fit")
            println((;stage="native Gaussian control",checks,prior=block.prior));flush(stdout)
            start=first(BRM._brm_nlme_prior_median_point(m))
            q=BRM._brm_svi_params(t,start[t.globals],BRM.LowRankGlobal(3);
                global_scale=.2,unit_scale=.5)
            record=with_progress(progress;description="Original fixed SVI budget from prior medians") do node
                # Explicit window is required for zero remaining unit factors;
                # the ordinary default derives a window from the unit count.
                seconds=@elapsed fit=BRM._brm_svi_fit!(Xoshiro(700),t,q;
                    batch_size=3,max_iterations=5000,lr_global=.01,unit_step=.1,
                    polish_lr=.02,window=50,patience=4,min_windows=8,polish_windows=10,node)
                record=(;identity,fit,seconds,coordinates=view.coordinates,globals=t.globals,
                    latent=t.op.field,field,units,beta=block.beta)
                serialize(joinpath(directory,"fit.jls"),record)
                record
            end
            fit=record.fit
            @test fit.iterations<=5000
            @test all(isfinite,fit.elbo) && all(isfinite,fit.q.μ)
            println((;stage="native fit saved",fit.converged,fit.iterations,seconds=record.seconds));flush(stdout)
            draw_record=with_progress(progress,1000;description="Native joint conditional reconstruction") do node
                seconds=@elapsed draws=BRM._brm_svi_draws(Xoshiro(701),t,fit.q,1000;node)
                record=(;identity,draws,seconds)
                serialize(joinpath(directory,"draws.jls"),record)
                record
            end
            mean_error=abs.(vec(mean(draw_record.draws;dims=2))-reference.ref_mean)./reference.ref_sd
            sd_ratio=vec(std(draw_record.draws;dims=2))./reference.ref_sd
            accuracy_ok=fit.converged && maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio)
            digest=bytes2hex(sha256(reinterpret(UInt8,vcat(fit.q.μ,fit.q.ρ,vec(fit.q.Β),
                fit.elbo,vec(draw_record.draws)))))
            summary=(;identity,checks,accuracy_ok,fit.converged,fit.iterations,
                seconds=record.seconds,draw_seconds=draw_record.seconds,mean_error,sd_ratio,digest,
                coordinates=view.coordinates,globals=t.globals,latent=t.op.field,field,units,beta=block.beta,
                scope="native whole-window fit and fixed-draw comparison; no independent stationarity, streaming, scaling or fresh-process reproducibility claim")
            serialize(joinpath(directory,"comparison.jls"),summary)
            println((;stage="native Gaussian block accuracy",accuracy_ok,fit.converged,fit.iterations,
                max_mean_error=maximum(mean_error),original_global_sd_ratio=sd_ratio[p.globals],
                unit_sd_extrema=extrema(sd_ratio[units]),field_sd_extrema=extrema(sd_ratio[field]),digest))
            @test accuracy_ok
        end
    end
end
