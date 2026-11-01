# Stopped-q path derivative (Roeder et al 2017), using the existing SVI
# diagonal-plus-low-rank geometry and Adam. No dense global covariance.
function _brm_svi_laplace_iteration!(rng,t,q,opt,lr)
    D=length(t.globals)
    epsilon,eta=randn(rng,D),randn(rng,size(q.Β,2))
    u=copy(t.m.unit);u[t.globals]=_brm_svi_draw_global(q,epsilon,eta)
    state=_brm_svi_laplace_state(rng,t,u)
    geometry=_brm_svi_global_geometry(q)
    offset=geometry.s.*epsilon+q.Β*eta
    precision_offset=_brm_svi_global_precision(geometry,offset)
    w=state.gradient[t.globals]+precision_offset
    gradient=vcat(w,w.*epsilon.*geometry.s,vec(w*eta'))
    # Same pre-update q and same draw in both terms, up to the constant shared
    # with ordinary SVI. The score above already includes entropy in expectation.
    elbo=state.value+_brm_svi_global_logdet(geometry)/2+(dot(offset,precision_offset)-D)/2
    all(isfinite,gradient) && isfinite(elbo) || error("GaussianBlockLaplace: nonfinite variational update")
    x=_brm_svi_global_vector(q)
    _brm_svi_step!(opt,x,gradient,lr)
    all(isfinite,x) || error("GaussianBlockLaplace: nonfinite variational parameters")
    _brm_svi_set_global!(q,x)
    elbo
end

function _brm_svi_laplace_optimize!(rng,t,q;max_iterations,lr_global,node)
    max_iterations isa Integer && max_iterations>0 || throw(ArgumentError(
        "GaussianBlockLaplace: max_iterations must be a positive integer"))
    isfinite(lr_global) && lr_global>0 || throw(ArgumentError(
        "GaussianBlockLaplace: lr_global must be positive and finite"))
    width=min(max_iterations,t.options.averaging_window)
    initial=_brm_svi_global_vector(q)
    history=Matrix{Float64}(undef,length(initial),width)
    elbo=Vector{Float64}(undef,max_iterations)
    opt=_BRMSVIAdam(length(initial),lr_global)
    for k in 1:max_iterations
        elbo[k]=_brm_svi_laplace_iteration!(rng,t,q,opt,lr_global)
        history[:,mod1(k,width)]=_brm_svi_global_vector(q)
        if k%50==0 || k==max_iterations
            _brm_throw_if_interrupted(node)
            _brm_update_progress!(node,"conditional Gaussian SVI update $k/$max_iterations")
        end
    end
    result=deepcopy(q)
    _brm_svi_set_global!(result,vec(mean(history;dims=2)))
    (;q=result,elbo,elbo_windows=Float64[],iterations=max_iterations,converged=false,
        stop_reason=:fixed_update_budget,budget_exhausted=true,
        convergence_assessed=false,averaging_window=(max_iterations-width+1):max_iterations,
        averaging=:raw_parameter_mean,lr_global)
end

_brm_svi_laplace_fit(x,previous,method;kwargs...)=throw(ArgumentError(
    "brm_svi_fit: laplace must be nothing or GaussianBlockLaplace()"))

function _brm_svi_laplace_dispatch(x,previous,method;kwargs...)
    unsupported=intersect(keys(kwargs),(:batch_size,:unit_step,:unit_scale,
        :polish_lr,:window,:patience,:min_windows,:polish_windows))
    isempty(unsupported) || throw(ArgumentError(
        "GaussianBlockLaplace: whole-window fixed-budget optimization does not use " *
        join(string.(unsupported),", ") * "; omit these options"))
    _brm_svi_laplace_fit(x,previous,method;kwargs...)
end

function _brm_svi_laplace_fit(x,previous,method::GaussianBlockLaplace;
        rng=Random.default_rng(),draw_rng=rng,max_iterations=5000,family=nothing,
        lr_global=.01,start=nothing,inits=nothing,ad_backend=nothing,
        shared_point=nothing,subject_group=nothing,nested_groups=NamedTuple(),
        shared_groups=(),global_scale=.2,ndraws=1000,progress=nothing)
    previous===nothing || previous isa BRMSVILaplaceState || throw(ArgumentError(
        "GaussianBlockLaplace: continuation requires a BRMSVILaplaceState"))
    family=family===nothing ? (previous===nothing ? MeanFieldGlobal() : previous.q.family) : family
    family isa BRMSVIFamily || throw(ArgumentError("GaussianBlockLaplace: invalid variational family"))
    if previous!==nothing
        start===nothing && inits===nothing || throw(ArgumentError(
            "GaussianBlockLaplace: continuation starts from the previous factor; omit start and inits"))
        typeof(family)===typeof(previous.q.family) &&
            _brm_svi_rank(family)==_brm_svi_rank(previous.q.family) || throw(ArgumentError(
                "GaussianBlockLaplace: continuation keeps the global family and rank"))
        method.carry_rank==length(previous.core.m) || throw(ArgumentError(
            "GaussianBlockLaplace: continuation keeps the common-factor rank"))
    end
    max_iterations isa Integer && max_iterations>0 || throw(ArgumentError(
        "GaussianBlockLaplace: max_iterations must be a positive integer"))
    isfinite(lr_global) && lr_global>0 || throw(ArgumentError(
        "GaussianBlockLaplace: lr_global must be positive and finite"))
    isfinite(global_scale) && global_scale>0 || throw(ArgumentError(
        "GaussianBlockLaplace: global_scale must be positive and finite"))
    ndraws isa Integer && ndraws>=0 || throw(ArgumentError("GaussianBlockLaplace: ndraws must be a nonnegative integer"))
    _brm_with_progress(progress;description="BRM conditional Gaussian SVI") do node
        _brm_with_phases(node,_BRM_SVI_PHASES) do phases
            lowered=_brm_svi_lowered(x,phases.lower)
            # Refuse unsupported model structure before compiling AD for it.
            if lowered isa RKBRMI
                view=brm_nlme_view(lowered;subject_group,nested_groups,shared_groups)
                _brm_svi_gaussian_certificate(lowered,view)
            else
                _brm_svi_gaussian_certificate(lowered.backend,lowered.view)
            end
            m=_brm_svi_model(lowered,phases.prepare;
                ad_backend,shared_point,subject_group,nested_groups,shared_groups)
            t,q,layout,active=_brm_run_phase(phases.target) do _
                t=_brm_svi_laplace_target(m,method)
                layout=_brm_svi_laplace_layout(t)
                if previous===nothing
                    start_point=_brm_svi_start(m,start,inits)
                    q=_brm_svi_params(t,start_point[t.globals],family;global_scale,unit_scale=1.)
                    return t,q,layout,nothing
                end
                carried=_brm_svi_laplace_active(t,previous,layout)
                t=_brm_svi_laplace_with_carry(t,carried.carry)
                q=_brm_svi_params(t,previous.q.μ,family;global_scale=1.,unit_scale=1.)
                q.ρ.=previous.q.ρ;q.Β.=previous.q.Β
                t,q,layout,carried.active
            end
            seconds=@elapsed fit=_brm_enter_phase(phases.fit) do phase
                _brm_svi_laplace_optimize!(rng,t,q;max_iterations,lr_global,node=phase)
            end
            draws,checkpoint=_brm_run_phase(phases.draws) do phase
                # Construct the next prior from a fixed prefix before optional
                # output draws can advance the auxiliary-lift RNG stream.
                prefix=_brm_svi_draws(draw_rng,t,fit.q,method.carry_draws;node=phase)
                checkpoint=_brm_svi_laplace_checkpoint(draw_rng,t,fit.q,prefix,previous,layout,active)
                draws=ndraws<=method.carry_draws ? prefix[:,1:ndraws] :
                    hcat(prefix,_brm_svi_draws(draw_rng,t,fit.q,ndraws-method.carry_draws;
                        node=phase,offset=method.carry_draws))
                draws,checkpoint
            end
            (;draws,coordinates=copy(m.view.coordinates),diagnostics=(;
                fit...,globals=t.globals,seconds,model=m,source_id=_brm_svi_source_id(m.backend),
                streaming=checkpoint,laplace=(;method,scope=merge(t.certificate.scope,
                    (;streaming=previous!==nothing,
                      spectral_bound=previous===nothing ? :prior_precision : :conditional_carry_precision)),
                    integrated=copy(m.view.coordinates[t.certificate.latent]),
                    fields=copy(m.view.coordinates[t.certificate.field]),
                    authored_prior_precision=copy(t.certificate.prior_precision),
                    spectral_lower_bound=_brm_svi_laplace_lower_bound(t.prior,t),
                    calls=(;value_gradient=t.counts[1],likelihood_hvp=t.counts[2],
                        likelihood_third=t.counts[3],modes=t.counts[4],draws=t.counts[5]),
                    numerical=copy(t.numerical),projection=checkpoint.projection)))
        end
    end
end
