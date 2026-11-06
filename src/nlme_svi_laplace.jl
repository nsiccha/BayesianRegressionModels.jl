"""
    GaussianBlockLaplace(; trace_probes=8, logdet_probes=8,
        averaging_window=200, mode_tolerance=1e-8,
        mode_krylovdim=128, trace_krylovdim=128, correction_krylovdim=64,
        logdet_krylovdim=32, draw_krylovdim=32, draw_solve_krylovdim=64,
        carry_draws=128, carry_rank=8, carry_shrinkage=.01)

Opt in with `brm_svi_fit(model; laplace=GaussianBlockLaplace(), ...)`.
Jointly integrate the certified Gaussian population coefficients, unit effects
and shared HSGP weights conditional on the remaining variational globals.
The initial supported likelihood is Gaussian with an affine mean and a scale
independent of the integrated coordinates. Unsupported expressions are refused
from their model plan. This is conditional integration inside VI.

Each update evaluates the entire current window. Krylov bases and probe counts
have fixed caps; an exhausted numerical budget raises an error. Trace and
log-determinant probes add Monte Carlo error, distinct from iterative error.
Rounding is a third error: about `eps(Float64)*cond(H)*d` in a log-determinant
probe for a `d`-coordinate conditional precision `H`, which the recorded
`logdet_quadrature_width` does not include. A precision too ill-conditioned to
factor in double precision, possible from about `cond(H)=1e16`, raises an error.
Optimization runs the declared `max_iterations` at fixed `lr_global` and returns
the mean of its final `averaging_window` parameter iterates. Completing that
budget does not establish convergence or posterior accuracy.

The checkpoint projects the first `carry_draws` joint draws into a common
factor of fixed `carry_rank` plus residual blocks of at most 16 coordinates.
Unit conditional moments reduce projection noise. Positive `carry_shrinkage`
regularizes the residual covariance toward an explicit numerical reference.
This projection adds approximation error and drops residual cross-block
dependence. At least `carry_draws` joint draws are generated even when
`ndraws` requests fewer returned draws. Continue on a disjoint window with
`brm_svi_fit(previous, builder(new_window))`.
"""
struct GaussianBlockLaplace
    trace_probes::Int
    logdet_probes::Int
    averaging_window::Int
    mode_tolerance::Float64
    mode_krylovdim::Int
    trace_krylovdim::Int
    correction_krylovdim::Int
    logdet_krylovdim::Int
    draw_krylovdim::Int
    draw_solve_krylovdim::Int
    carry_draws::Int
    carry_rank::Int
    carry_shrinkage::Float64
    function GaussianBlockLaplace(; trace_probes=8, logdet_probes=8,
            averaging_window=200, mode_tolerance=1e-8,
            mode_krylovdim=128, trace_krylovdim=128, correction_krylovdim=64,
            logdet_krylovdim=32, draw_krylovdim=32, draw_solve_krylovdim=64,
            carry_draws=128, carry_rank=8, carry_shrinkage=.01)
        ints=(trace_probes,logdet_probes,averaging_window,mode_krylovdim,
            trace_krylovdim,correction_krylovdim,logdet_krylovdim,
            draw_krylovdim,draw_solve_krylovdim)
        all(x->x isa Integer && x>0,ints) || throw(ArgumentError(
            "GaussianBlockLaplace: probe, averaging and Krylov counts must be positive integers"))
        logdet_probes>=2 || throw(ArgumentError("GaussianBlockLaplace: at least two logdet probes are required"))
        isfinite(mode_tolerance) && mode_tolerance>0 || throw(ArgumentError(
            "GaussianBlockLaplace: mode_tolerance must be positive and finite"))
        carry_draws isa Integer && 2<=carry_draws<=256 || throw(ArgumentError(
            "GaussianBlockLaplace: carry_draws must be an integer in 2:256"))
        carry_rank isa Integer && 0<=carry_rank<=min(64,carry_draws-1) || throw(ArgumentError(
            "GaussianBlockLaplace: carry_rank must be in 0:min(64,carry_draws-1)"))
        isfinite(carry_shrinkage) && 0<carry_shrinkage<=1 || throw(ArgumentError(
            "GaussianBlockLaplace: carry_shrinkage must be in (0,1]"))
        new(trace_probes,logdet_probes,averaging_window,mode_tolerance,
            mode_krylovdim,trace_krylovdim,correction_krylovdim,
            logdet_krylovdim,draw_krylovdim,draw_solve_krylovdim,
            carry_draws,carry_rank,carry_shrinkage)
    end
end

# Public prepared RK queries are implemented in the Enzyme extension.
function _rk_svi_laplace_ad end
function _rk_svi_laplace_value_gradient end
function _rk_svi_laplace_hvp end
function _rk_svi_laplace_third end

struct _BRMSVILaplaceTarget{M,C,A,P}
    m::M
    certificate::C
    ad::A
    prior::P
    options::GaussianBlockLaplace
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    counts::Vector{Int} # full gradients, likelihood HVPs, thirds, modes, draws
    numerical::Dict{Symbol,Float64} # bounded accumulated maxima, not full history
end

function _brm_svi_laplace_target(m, options::GaussianBlockLaplace; prior=nothing)
    certificate=_brm_svi_gaussian_certificate(m.backend,m.view)
    isempty(certificate.globals) && throw(ArgumentError(
        "GaussianBlockLaplace: at least one remaining variational global is required"))
    _BRMSVILaplaceTarget(m,certificate,_rk_svi_laplace_ad(m),prior,options,
        certificate.globals,Vector{Int}[],zeros(Int,5),Dict{Symbol,Float64}())
end

function _brm_svi_laplace_record!(t,key,value)
    t.numerical[key]=max(get(t.numerical,key,0.),Float64(value))
end
function _brm_svi_laplace_value_gradient(t,u)
    t.counts[1]+=1
    _brm_svi_laplace_prior_value_gradient(t.prior,t.ad,u)
end
_brm_svi_laplace_prior_value_gradient(::Nothing,ad,u)=
    _rk_svi_laplace_value_gradient(ad,u,:sampler)

function _brm_svi_laplace_likelihood_hvp(t,u,v)
    direction=zeros(length(u));direction[t.certificate.latent]=v
    t.counts[2]+=1
    _rk_svi_laplace_hvp(t.ad,u,direction)
end
function _brm_svi_laplace_joint_hvp(::Nothing,t,u,v)
    out=_brm_svi_laplace_likelihood_hvp(t,u,v)
    out[t.certificate.latent].-=t.certificate.prior_precision.*v
    out
end
function _brm_svi_laplace_precision(t,u,v)
    _brm_svi_laplace_prior_precision(t.prior,t,v)-
        _brm_svi_laplace_likelihood_hvp(t,u,v)[t.certificate.latent]
end
_brm_svi_laplace_prior_precision(::Nothing,t,v)=t.certificate.prior_precision.*v
_brm_svi_laplace_lower_bound(::Nothing,t)=minimum(t.certificate.prior_precision)
function _brm_svi_laplace_third(t,u,v)
    direction=zeros(length(u));direction[t.certificate.latent]=v
    t.counts[3]+=1
    _rk_svi_laplace_third(t.ad,u,direction)
end

function _brm_svi_laplace_solve(t,H,b,operation,krylovdim;rtol,atol)
    x,info=KrylovKit.linsolve(H,b;issymmetric=true,isposdef=true,
        rtol,atol,krylovdim,maxiter=1)
    info.converged==1 || error("GaussianBlockLaplace: $operation exhausted its Krylov budget " *
        "($krylovdim directions; residual $(info.normres))")
    _brm_svi_laplace_record!(t,Symbol(operation,:_residual),info.normres)
    _brm_svi_laplace_record!(t,Symbol(operation,:_operations),info.numops)
    x
end

# Deterministic starts and a fixed iteration/backtracking cap; no warm history
# enters the target. A certified Gaussian block normally takes one Newton step.
function _brm_svi_laplace_mode(t,point)
    t.counts[4]+=1
    latent=t.certificate.latent;o=t.options
    u=copy(point);u[latent]=t.certificate.prior_mean
    for iteration in 0:40
        value,gradient=_brm_svi_laplace_value_gradient(t,u)
        isfinite(value) && all(isfinite,gradient) || error("GaussianBlockLaplace: nonfinite conditional mode point")
        g=gradient[latent]
        if norm(g)<=o.mode_tolerance
            _brm_svi_laplace_record!(t,:mode_iterations,iteration)
            _brm_svi_laplace_record!(t,:mode_gradient_norm,norm(g))
            return (;mode=u,gradient,value)
        end
        iteration==40 && break
        H(v)=_brm_svi_laplace_precision(t,u,v)
        step=_brm_svi_laplace_solve(t,H,g,:mode,o.mode_krylovdim;rtol=1e-12,atol=1e-13)
        decrement=dot(g,step)
        decrement>0 || error("GaussianBlockLaplace: nonpositive Newton decrement")
        accepted=false
        for half in 0:40
            scale=2.0^-half;candidate=copy(u);candidate[latent].+=scale.*step
            after,score=_brm_svi_laplace_value_gradient(t,candidate)
            if isfinite(after) && all(isfinite,score) &&
                    (decrement<1e-12 || after>=value+1e-4scale*decrement)
                u=candidate;accepted=true;break
            end
        end
        accepted || error("GaussianBlockLaplace: conditional mode backtracking failed")
    end
    error("GaussianBlockLaplace: conditional mode did not reach tolerance $(o.mode_tolerance) in 40 iterations")
end

function _brm_svi_laplace_state(rng,t,point)
    state=_brm_svi_laplace_mode(t,point);o=t.options
    latent=t.certificate.latent;d=length(latent)
    H(v)=_brm_svi_laplace_precision(t,state.mode,v)
    trace=zeros(length(point))
    for _ in 1:o.trace_probes
        r=ifelse.(rand(rng,Bool,d),1.,-1.)
        s=_brm_svi_laplace_solve(t,H,r,:trace,o.trace_krylovdim;rtol=1e-11,atol=1e-13)
        trace .+= (_brm_svi_laplace_third(t,state.mode,r+s)-
            _brm_svi_laplace_third(t,state.mode,r-s))/4
    end
    trace ./= o.trace_probes
    direction=_brm_svi_laplace_solve(t,H,trace[latent],:correction,
        o.correction_krylovdim;rtol=1e-10,atol=1e-12)
    gradient=state.gradient+(trace+_brm_svi_laplace_joint_hvp(t.prior,t,state.mode,direction))/2
    determinant=_brm_svi_laplace_logdet(rng,H,d;probes=o.logdet_probes,
        spectral_lower=_brm_svi_laplace_lower_bound(t.prior,t),
        maxdim=o.logdet_krylovdim,atol=1e-8,rtol=1e-8)
    _brm_svi_laplace_record!(t,:logdet_probe_mcse,determinant.mc_se)
    _brm_svi_laplace_record!(t,:logdet_quadrature_width,
        determinant.quadrature_upper-determinant.quadrature_lower)
    _brm_svi_laplace_record!(t,:logdet_iterations,determinant.max_iterations)
    value=state.value+d*log(2pi)/2-determinant.value/2
    (;value,gradient,mode=state.mode,determinant)
end

function _brm_svi_draws(rng,t::_BRMSVILaplaceTarget,q,n;node=nothing,offset=0)
    n>=0 || throw(ArgumentError("GaussianBlockLaplace: ndraws must be nonnegative"))
    U=repeat(t.m.unit,1,n);o=t.options
    for j in 1:n
        U[t.globals,j]=_brm_svi_draw_global(q,randn(rng,length(q.μ)),randn(rng,size(q.Β,2)))
        state=_brm_svi_laplace_mode(t,U[:,j])
        H(v)=_brm_svi_laplace_precision(t,state.mode,v)
        solve(v)=_brm_svi_laplace_solve(t,H,v,:draw,o.draw_solve_krylovdim;rtol=1e-11,atol=1e-12)
        root=_brm_spl_covariance_root(solve,randn(rng,length(t.certificate.latent));
            maxdim=o.draw_krylovdim,rtol=1e-9)
        U[t.certificate.latent,j]=state.mode[t.certificate.latent]+root.draw
        t.counts[5]+=1
        _brm_svi_laplace_record!(t,:draw_root_iterations,root.iterations)
        _brm_svi_laplace_record!(t,:draw_root_error_bound,root.error_bound)
        _brm_update_progress!(node,offset+j)
        _brm_throw_if_interrupted(node)
    end
    U
end
