# Private integration control. This deliberately costs full current-window
# sweeps; it does NOT claim batch-local work or implement streaming projection.
# Run: julia --project=test research/shared_field_laplace/joint_svi_probe.jl 'joint-window SVI'
include("nonlinear_probe.jl")
include("quadrature.jl")
using SHA

struct JointWindowTarget{O,M}
    op::O
    m::M
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    trace_probes::Int
    logdet_probes::Int
end

function joint_window_target(m; trace_probes=4, logdet_probes=2)
    op = shared_field_operator(m)
    globals = setdiff(op.t.globals, op.field)
    JointWindowTarget(op, m, globals, op.t.units, trace_probes, logdet_probes)
end

# For this Gaussian/Poisson affine-field control, H >= I analytically.
# A general model cannot borrow that bound without a concavity proof.
function joint_window_state(rng, t, point, probes)
    op = t.op
    state = shared_poisson_mode(op, point; tol=1e-8)
    trace = shared_trace(op, state.mode, probes).trace
    H(v) = shared_precision(op, state.mode, v)
    correction_direction, info = KrylovKit.linsolve(H, trace[op.field];
        issymmetric=true, isposdef=true, rtol=1e-10, atol=1e-12,
        krylovdim=64,maxiter=1)
    info.converged == 1 || error("implicit-mode CG did not converge")
    gradient = state.gradient + (trace +
        shared_likelihood_hvp(op,state.mode,correction_direction))/2
    determinant = shared_logdet(rng,H,length(op.field); probes=t.logdet_probes,
        spectral_lower=1.0,maxdim=32,atol=1e-8,rtol=1e-8)
    value = state.value + length(op.field)*log(2π)/2 - determinant.value/2
    (; value,gradient,mode=state.mode,determinant)
end

# This target is coupled across units after the field is marginalized.
# Draw ALL unit factors, evaluate the joint marginal, update only the batch.
# There is no N/B multiplication. Simultaneous antithetic unit draws retain
# the Stein identity for each unit's own epsilon (others are independent).
function BRM._brm_svi_iteration!(rng,t::JointWindowTarget,q,global_opt,batch,
        lr_global,beta;whiten=false)
    D = length(t.globals)
    epsilon, eta = randn(rng,D),randn(rng,size(q.Β,2))
    g = BRM._brm_svi_draw_global(q,epsilon,eta)
    plus,minus = copy(t.m.unit),copy(t.m.unit)
    plus[t.globals] .= g; minus[t.globals] .= g
    unit_eps = [randn(rng,length(block)) for block in t.units]
    for (i,block) in enumerate(t.units)
        offset = BRM._brm_svi_unit_offset(q,i,unit_eps[i])
        plus[block] .= q.M[i] .+ offset
        minus[block] .= q.M[i] .- offset
    end
    probes = [ifelse.(rand(rng,Bool,length(t.op.field)),1.0,-1.0)
        for _ in 1:t.trace_probes]
    a = joint_window_state(rng,t,plus,probes)
    b = joint_window_state(rng,t,minus,probes)
    w = (a.gradient[t.globals]+b.gradient[t.globals])/2
    x = BRM._brm_svi_global_vector(q)
    grad = BRM._brm_svi_global_gradient(q,w,epsilon,eta)
    whiten ? BRM._brm_svi_whitened_global_step!(global_opt,q,x,grad,lr_global) :
        BRM._brm_svi_step!(global_opt,x,grad,lr_global)
    BRM._brm_svi_set_global!(q,x)
    for i in batch
        block = t.units[i]
        mean_grad = (a.gradient[block]+b.gradient[block])/2
        diff_grad = (a.gradient[block]-b.gradient[block])/2
        BRM._brm_svi_unit_step!(q,i,mean_grad,diff_grad,unit_eps[i],beta)
    end
    println((;iteration=global_opt.t,value=(a.value+b.value)/2,
        scope="full-current-window joint collapse")); flush(stdout)
    (a.value+b.value)/2
end

function BRM._brm_svi_draws(rng,t::JointWindowTarget,q,n)
    U = repeat(t.m.unit,1,n)
    for j in 1:n
        U[t.globals,j] .= BRM._brm_svi_draw_global(q,randn(rng,length(q.μ)),
            randn(rng,size(q.Β,2)))
        for (i,block) in enumerate(t.units)
            U[block,j] .= BRM._brm_svi_draw_unit(q,i,randn(rng,length(block)))
        end
        state = shared_poisson_mode(t.op,U[:,j];tol=1e-8)
        H(v) = shared_precision(t.op,state.mode,v)
        function solve(v)
            x,info = KrylovKit.linsolve(H,v;issymmetric=true,isposdef=true,
                rtol=1e-11,atol=1e-12,krylovdim=64,maxiter=1)
            info.converged == 1 || error("conditional draw CG did not converge")
            x
        end
        root = BRM._brm_spl_covariance_root(solve,randn(rng,length(t.op.field));
            maxdim=32,rtol=1e-9)
        U[t.op.field,j] .= state.mode[t.op.field] + root.draw
    end
    U
end

@stestset "joint-window SVI optimizer and conditional field output" begin
    (;m) = shared_field_fixture()
    t = joint_window_target(m)
    start = first(BRM._brm_nlme_prior_median_point(m))
    q = BRM._brm_svi_params(t,start[t.globals],BRM.MeanFieldGlobal();
        global_scale=0.05,unit_scale=0.3)
    @test isempty(intersect(t.globals,t.op.field))
    @test sort(vcat(t.globals,t.op.field,t.units...)) == collect(eachindex(start))
    # Check batch-selection semantics on the coupled target before fitting.
    before = deepcopy(q)
    opt = BRM._BRMSVIAdam(length(BRM._brm_svi_global_vector(q)),0.002)
    rng = Xoshiro(20261101)
    BRM._brm_svi_iteration!(rng,t,q,opt,[1],0.002,0.05)
    @test q.M[2:end] == before.M[2:end]
    @test q.Λ[2:end] == before.Λ[2:end]
    @test q.M[1] != before.M[1]
    fit = BRM._brm_svi_fit!(rng,t,q;batch_size=2,max_iterations=12,
        lr_global=0.002,unit_step=0.05,polish_lr=0.01,window=4,patience=10,
        min_windows=20,polish_windows=2)
    @test fit.iterations == 12
    @test all(isfinite,fit.elbo)
    @test all(isfinite,fit.q.μ)
    @test all(isposdef,Symmetric.(fit.q.Λ))
    draws = BRM._brm_svi_draws(Xoshiro(84),t,fit.q,3)
    @test size(draws) == (length(start),3)
    @test all(isfinite,draws)
    @test any(!iszero,draws[t.op.field,:])
    digest = bytes2hex(sha256(reinterpret(UInt8,vcat(fit.q.μ,fit.q.ρ,
        fit.q.M...,fit.elbo,vec(draws)))))
    println((;digest,explicit_globals=length(t.globals),fields=length(t.op.field),
        units=length(t.units),iterations=fit.iterations,
        scope="short integration control; no posterior accuracy, streaming or scaling acceptance"))
end
