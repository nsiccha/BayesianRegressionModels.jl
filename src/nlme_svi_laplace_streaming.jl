# Data-only transport of the joint Gaussian approximation. Each density and
# update uses only current-window blocks. Producing a new checkpoint shallow-
# copies its retained-unit dictionary; serialization visits all retained units.
"""
    BRMSVILaplaceState

Data-only checkpoint for `brm_svi_fit(...; laplace=GaussianBlockLaplace())`.
It retains frozen preprocessing, a fixed-rank common Gaussian factor, small
conditional blocks keyed by coordinate or unit identity, and the last global
VI factor for initialization. It contains no observations or compiled queries.
Resume with `brm_svi_fit(state, builder(new_window); ...)`.

Projection preserves selected block moments approximately and drops residual
cross-block dependence. New units initially have zero common-factor loading;
fixed-rank continuation cannot preserve every new cross-unit dependence.
The checkpoint retains `unit_partition` and uses format 2. Connected units
must retain all their members in the same order across windows.
"""
struct BRMSVILaplaceState
    format::Int
    preprocessing::BRMFittedState
    source_id::String
    globals::Vector{Symbol}
    shared_coordinates::Vector{Symbol}
    core::NamedTuple
    blocks::Dict{Any,NamedTuple}
    units::Dict{Any,NamedTuple}
    q::NamedTuple
    group::Symbol
    nested::Tuple{Vararg{Symbol}}
    shared::Tuple{Vararg{Symbol}}
    method::GaussianBlockLaplace
    projection::NamedTuple
    unit_partition::Symbol
    updates::Int
end
brm_svi_checkpoint(state::BRMSVILaplaceState)=deepcopy(state)

function _brm_svi_laplace_locations(m)
    reg=_brm_nlme_regression(m.backend.plan)
    locations=Dict(label=>Dict{Any,Tuple}() for label in m.view.levels)
    subject_unit=Dict(label=>m.view.levels[m.view.subject_unit[s]]
        for (s,label) in enumerate(m.view.subject_levels))
    for predictor in reg.predictors, term in predictor.terms
        term.kind===:hsgp || continue
        o=term.options
        o.group_column===m.view.group || _brm_svi_carry_error(
            "shared-field grouping must match the streaming subject group")
        levels=o.group_levels;C=o.level_cor.coordinates
        size(C,1)==length(levels) || _brm_svi_carry_error("field location layout changed")
        for (i,label) in enumerate(levels)
            haskey(subject_unit,label) || continue
            unit=subject_unit[label]
            locations[unit][(o.id,label)]=Tuple(C[i,:])
        end
    end
    locations
end

function _brm_svi_laplace_layout(t)
    m,c=t.m,t.certificate
    blocks=Vector{Int}[];order=Any[];unit_keys=Any[]
    function shared(indices)
        push!(blocks,collect(indices))
        push!(order,(:coordinates,Tuple(m.view.coordinates[indices])))
    end
    # Fixed small residual blocks keep hyperparameters together where possible.
    for first in 1:16:length(c.globals)
        shared(c.globals[first:min(first+15,length(c.globals))])
    end
    unit_indices=Set(reduce(vcat,c.unit_blocks;init=Int[]))
    fields=Set(c.field)
    for i in c.latent
        i in unit_indices || i in fields || shared([i])
    end
    locations=_brm_svi_laplace_locations(m)
    units=Dict{Any,NamedTuple}()
    for (i,label) in enumerate(m.view.levels)
        indices=c.unit_blocks[i]
        1<=length(indices)<=16 || _brm_svi_carry_error("unit residual block exceeds 16 coordinates")
        key=(:unit,label);push!(unit_keys,key);push!(order,key);push!(blocks,copy(indices))
        units[label]=(;keys=copy(_brm_svi_unit_keys(m,i)),locations=locations[label])
    end
    for i in c.field
        shared([i])
    end
    sort(reduce(vcat,blocks;init=Int[]))==collect(eachindex(m.unit)) ||
        _brm_svi_carry_error("checkpoint blocks do not cover sampler coordinates exactly")
    shared_coordinates=m.view.coordinates[setdiff(eachindex(m.unit),collect(unit_indices))]
    (;blocks,order,unit_keys,units,shared_coordinates)
end

function _brm_svi_laplace_check_model(t,previous,layout)
    m=t.m
    previous.format==2 || _brm_svi_carry_error("unsupported checkpoint format")
    m.view.group===previous.group && Tuple(keys(m.view.nested_groups))==previous.nested &&
        m.view.shared_groups==previous.shared || _brm_svi_carry_error("grouping partition changed")
    _brm_svi_check_units(m.view,previous)
    _brm_svi_check_preprocessing(m.backend,previous)
    _brm_svi_source_id(m.backend)==previous.source_id || _brm_svi_carry_error("model source changed")
    m.view.coordinates[t.globals]==previous.globals || _brm_svi_carry_error("global coordinate layout changed")
    layout.shared_coordinates==previous.shared_coordinates || _brm_svi_carry_error("shared coordinate layout changed")
    for (label,current) in layout.units
        haskey(previous.units,label) || continue
        old=previous.units[label]
        Set(current.keys)==Set(old.keys) && length(current.keys)==length(old.keys) ||
            _brm_svi_carry_error("unit $(repr(label)) changed its coordinate identities")
        isequal(current.locations,old.locations) ||
            _brm_svi_carry_error("unit $(repr(label)) changed its retained field location")
    end
    nothing
end

function _brm_svi_laplace_active(t,previous,layout)
    _brm_svi_laplace_check_model(t,previous,layout)
    rank=length(previous.core.m)
    active=Dict{Any,NamedTuple}()
    latent_position=Dict(i=>j for (j,i) in enumerate(t.certificate.latent))
    for (key,indices) in zip(layout.order,layout.blocks)
        if haskey(previous.blocks,key)
            b=previous.blocks[key]
            if first(key)===:unit
                old=previous.units[last(key)].keys;current=layout.units[last(key)].keys
                p=[only(findall(==(k),old)) for k in current]
                b=_brm_svi_carry_block(b.a[p],b.B[p,:],Matrix(b.F.L*b.F.L')[p,p])
            end
            active[key]=b
        else
            first(key)===:unit || _brm_svi_carry_error("missing shared checkpoint block")
            pos=[latent_position[i] for i in indices]
            active[key]=_brm_svi_carry_block(t.certificate.prior_mean[pos],
                zeros(length(indices),rank),Matrix(Diagonal(inv.(t.certificate.prior_precision[pos]))))
        end
    end
    packed=_brm_svi_carry_active_marginal(previous.core,active,layout.order)
    physical=reduce(vcat,layout.blocks;init=Int[])
    carry=_brm_svi_carry_marginal(packed,invperm(physical))
    (;active,carry)
end

function _brm_svi_laplace_conditional_projection(rng,t,draws,layout,previous,active)
    o=t.options;M=o.carry_draws
    size(draws,2)>=M || _brm_svi_carry_error("insufficient checkpoint draws")
    points=draws[:,1:M];means=copy(points)
    conditional=[zeros(length(b),length(b)) for b in layout.blocks]
    block_position=Dict(key=>j for (j,key) in enumerate(layout.order))
    unit_positions=[block_position[key] for key in layout.unit_keys]
    H=previous===nothing ? nothing : _brm_svi_carry_lift(rng,previous.core,active,
        Dict(key=>points[b,:] for (key,b) in zip(layout.order,layout.blocks)))
    priors=previous===nothing ? nothing : [active[key] for key in layout.unit_keys]
    calls=(;value_gradient=0,likelihood_hvp=0)
    for j in 1:M
        moment=_brm_svi_laplace_unit_moments(t,points[:,j];priors,
            eta=H===nothing ? nothing : H[:,j])
        means[:,j]=moment.means
        for (pos,C) in zip(unit_positions,moment.covariances)
            conditional[pos].+=C/M
        end
        calls=(;value_gradient=calls.value_gradient+moment.calls.likelihood_gradients,
            likelihood_hvp=calls.likelihood_hvp+moment.calls.colored_likelihood_hvps)
    end
    if previous===nothing
        reference=[Matrix{Float64}(I,length(b),length(b)) for b in layout.blocks]
        projected=_brm_svi_carry_project(means,layout.blocks,reference;
            conditional,rank=o.carry_rank,shrinkage=o.carry_shrinkage)
        aux=_brm_svi_carry_auxiliary(projected.factor,layout.order)
        metadata=(;draw_count=M,rank=o.carry_rank,shrinkage=o.carry_shrinkage,
            projection=:means_gram_plus_unit_conditional_covariances,calls,
            numerical_reference=:identity,conditional_blocks=:units,
            residual_cross_blocks=:discarded)
        return (;core=aux.core,blocks=Dict{Any,NamedTuple}(aux.blocks),metadata)
    end
    moment_blocks=Dict(key=>means[b,:] for (key,b) in zip(layout.order,layout.blocks))
    covariances=Dict(key=>C for (key,C) in zip(layout.order,conditional))
    projected=_brm_svi_carry_project_active(previous.core,active,moment_blocks,H;
        conditional=covariances,shrinkage=o.carry_shrinkage)
    # O(retained units) dictionary copy at the window boundary only. Unchanged
    # absent conditional blocks are shared read-only, not reconstructed.
    blocks=copy(previous.blocks);merge!(blocks,projected.touched)
    (;core=projected.core,blocks,metadata=(;projected.metadata...,calls,
        numerical_reference=:previous_auxiliary_core_and_local_residual_covariances,
        conditional_blocks=:units,residual_cross_blocks=:discarded))
end

function _brm_svi_laplace_checkpoint(rng,t,q,draws,previous,layout,active)
    m=t.m
    projection=_brm_svi_laplace_conditional_projection(rng,t,draws,layout,previous,active)
    units=previous===nothing ? Dict{Any,NamedTuple}() : copy(previous.units)
    merge!(units,layout.units)
    preprocessing=previous===nothing ? brm_fitted_state(m.backend) : previous.preprocessing
    BRMSVILaplaceState(2,preprocessing,_brm_svi_source_id(m.backend),
        copy(m.view.coordinates[t.globals]),copy(layout.shared_coordinates),
        projection.core,projection.blocks,units,_brm_svi_global_factor(q),
        m.view.group,Tuple(keys(m.view.nested_groups)),m.view.shared_groups,
        t.options,projection.metadata,m.view.unit_partition,
        previous===nothing ? 0 : previous.updates+1)
end
