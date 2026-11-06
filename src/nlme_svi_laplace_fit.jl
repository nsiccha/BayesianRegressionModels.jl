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

# Everything a fixed-budget update reads and writes besides the target, so a
# run stops after any completed update and continues from a copy of it
# (`BRMSVIOptimizerState`) exactly as if it had not stopped: the random stream,
# the draw stream when it is a separate one (`nothing`: the draws continue
# `rng`), the global factor, Adam, the ELBO so far, the averaging window of
# global iterates, and the target's call counts and numerical records. During
# optimization `counts` and `numerical` are the target's own containers.
mutable struct _BRMSVILaplaceRun{Q}
    rng::AbstractRNG
    draw_rng::Union{Nothing,AbstractRNG}
    q::Q
    adam::_BRMSVIAdam
    iteration::Int
    elbo::Vector{Float64}
    history::Matrix{Float64}
    counts::Vector{Int}
    numerical::Dict{Symbol,Float64}
end

function _brm_svi_laplace_run(rng,draw_rng,t,q,c)
    n=length(_brm_svi_global_vector(q))
    _BRMSVILaplaceRun(rng,draw_rng===rng ? nothing : draw_rng,q,_BRMSVIAdam(n,c.lr_global),
        0,Float64[],zeros(n,c.width),t.counts,t.numerical)
end

_brm_svi_laplace_controls(method;max_iterations,lr_global)=
    (;method,max_iterations,lr_global,width=min(max_iterations,method.averaging_window))

_brm_svi_finished(run::_BRMSVILaplaceRun,c)=run.iteration>=c.max_iterations

# Advance `run` by up to `stop_after` updates, with checkpoints as for the
# minibatch engine (`_brm_svi_fit!`). The progress message and interrupt
# check come every 50 updates and at the last.
function _brm_svi_fit!(run::_BRMSVILaplaceRun,t,c;stop_after=typemax(Int),
        checkpoint_every=50,save=nothing,snapshot=identity,node=nothing)
    commit()=save===nothing || save(snapshot(run))
    taken=0
    while !_brm_svi_finished(run,c) && taken<stop_after
        k=run.iteration+1
        push!(run.elbo,_brm_svi_laplace_iteration!(run.rng,t,run.q,run.adam,c.lr_global))
        run.history[:,mod1(k,c.width)]=_brm_svi_global_vector(run.q)
        run.iteration=k
        taken+=1
        if k%50==0 || k==c.max_iterations
            try
                _brm_throw_if_interrupted(node)
            catch
                commit()
                rethrow()
            end
            _brm_update_progress!(node,"conditional Gaussian SVI update $k/$(c.max_iterations)")
        end
        save===nothing || k%checkpoint_every!=0 || _brm_svi_finished(run,c) ||
            taken>=stop_after || commit()
    end
    commit()
    run
end

# The result of a run so far: its global factor at the mean of the averaging
# window's iterates (those filled so far, for a run stopped early).
function _brm_svi_laplace_result(run,c)
    filled=min(run.iteration,c.width)
    q=deepcopy(run.q)
    filled>0 && _brm_svi_set_global!(q,vec(mean(view(run.history,:,1:filled);dims=2)))
    finished=_brm_svi_finished(run,c)
    (;q,elbo=run.elbo,elbo_windows=Float64[],iterations=run.iteration,converged=false,
        finished,stop_reason=finished ? :fixed_update_budget : :stopped,
        budget_exhausted=finished,convergence_assessed=false,
        averaging_window=(run.iteration-filled+1):run.iteration,
        averaging=:raw_parameter_mean,lr_global=c.lr_global)
end

_brm_svi_laplace_fit(save,x,previous,method;kwargs...)=throw(ArgumentError(
    "brm_svi_fit: laplace must be nothing or GaussianBlockLaplace()"))

function _brm_svi_laplace_dispatch(save,x,previous,method;kwargs...)
    unsupported=intersect(keys(kwargs),(:batch_size,:unit_step,:unit_scale,
        :polish_lr,:window,:patience,:min_windows,:polish_windows,:threads))
    isempty(unsupported) || throw(ArgumentError(
        "GaussianBlockLaplace: whole-window fixed-budget optimization does not use " *
        join(string.(unsupported),", ") * "; omit these options"))
    _brm_svi_laplace_fit(save,x,previous,method;kwargs...)
end

function _brm_svi_laplace_fit(save,x,previous,method::GaussianBlockLaplace;
        rng=Random.default_rng(),draw_rng=rng,max_iterations=5000,family=nothing,
        lr_global=.01,start=nothing,inits=nothing,global_scale=.2,kwargs...)
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
    c=_brm_svi_laplace_controls(method;max_iterations,lr_global)
    _brm_svi_laplace_drive(save,x,previous,method;caller="brm_svi_fit",
            description="BRM conditional Gaussian SVI",kwargs...) do t,layout
        if previous===nothing
            start_point=_brm_svi_start(t.m,start,inits)
            q=_brm_svi_params(t,start_point[t.globals],family;global_scale,unit_scale=1.)
        else
            q=_brm_svi_params(t,previous.q.μ,family;global_scale=1.,unit_scale=1.)
            q.ρ.=previous.q.ρ;q.Β.=previous.q.Β
        end
        _brm_svi_optimizer_state(t,previous,c,_brm_svi_laplace_run(rng,draw_rng,t,q,c))
    end
end

# A GaussianBlockLaplace fit or resume through its progress phases. After the
# target (with the carried prior of a window update) is prepared, `init(t,
# layout)` returns the `BRMSVIOptimizerState` to advance; it refuses an
# incompatible checkpoint before any update. The draws and the carry come from
# copies of the run's draw stream and leave its call counts as they were, so a
# stopped run's draws never change its continuation.
function _brm_svi_laplace_drive(init,save,x,previous,method;caller,description,
        ndraws=1000,stop_after=nothing,checkpoint_every=nothing,progress=nothing,
        ad_backend=nothing,shared_point=nothing,subject_group=nothing,
        nested_groups=NamedTuple(),shared_groups=(),unit_partition=nothing,check_lowered=Returns(nothing))
    ndraws isa Integer && ndraws>=0 || throw(ArgumentError("GaussianBlockLaplace: ndraws must be a nonnegative integer"))
    _brm_svi_check_interruption(caller,save,stop_after,checkpoint_every)
    _brm_with_progress(progress;description) do node
        _brm_with_phases(node,_BRM_SVI_PHASES) do phases
            lowered=_brm_svi_lowered(x,phases.lower)
            check_lowered(lowered)
            # Refuse unsupported model structure before compiling AD for it.
            if lowered isa RKBRMI
                view=brm_nlme_view(lowered;subject_group,nested_groups,shared_groups,unit_partition=something(unit_partition, :subjects))
                _brm_svi_gaussian_certificate(lowered,view)
            else
                _brm_svi_gaussian_certificate(lowered.backend,lowered.view)
            end
            m=_brm_svi_model(lowered,phases.prepare;
                ad_backend,shared_point,subject_group,nested_groups,shared_groups,unit_partition)
            t,layout,active,state=_brm_run_phase(phases.target) do _
                t=_brm_svi_laplace_target(m,method)
                layout=_brm_svi_laplace_layout(t)
                active=nothing
                if previous!==nothing
                    carried=_brm_svi_laplace_active(t,previous,layout)
                    t=_brm_svi_laplace_with_carry(t,carried.carry)
                    active=carried.active
                end
                t,layout,active,init(t,layout)
            end
            run,c=state.run,state.controls
            seconds=@elapsed _brm_enter_phase(phases.fit) do phase
                _brm_svi_fit!(run,t,c;stop_after=something(stop_after,typemax(Int)),
                    checkpoint_every=something(checkpoint_every,50),save,
                    snapshot=_->_brm_svi_detach(state),node=phase)
            end
            fit=_brm_svi_laplace_result(run,c)
            # The run's counts and records stop here; the draws count in the
            # target's own containers, which the diagnostics report.
            run.counts=copy(t.counts);run.numerical=copy(t.numerical)
            draw_rng=copy(something(run.draw_rng,run.rng))
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
                streaming=checkpoint,optimizer=state,laplace=(;method,scope=merge(t.certificate.scope,
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
