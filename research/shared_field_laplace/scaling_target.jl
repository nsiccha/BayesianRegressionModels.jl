# Scaling harness for the literal Gaussian fixture, not a general eligibility
# detector. Large cases never instantiate the bounded dense reference oracle.
include("fixed_rate_fit.jl")

function scaling_native_fixture(;groups,rows,temporal,spatial=2)
    fixture=shared_field_fixture(;groups,rows,temporal,spatial,poisson=false)
    (;rk,m,data)=fixture
    view=m.view
    population=findall(==(:population),view.role)
    length(population)==1 || error("scaling fixture requires one Gaussian population coefficient")
    beta=only(population)
    records=BRM._rk_coordinate_records(rk.plan)
    record=only(r for r in records if r.kind===:population &&
        BRM._brm_nlme_owned(view.coordinates[beta],r.declaration))
    plan=BRM._brm_nlme_regression(rk.plan)
    prior=only(r for r in plan.population_priors if
        r.predictor===record.predictor && r.addressee===record.coefficient)
    prior.family===:Normal && prior.args==(0,1) ||
        error("scaling fixture requires an authored standard Normal population prior")
    column=BRM._brm_nlme_population_column(plan,record.predictor,record.coefficient,length(data.y))
    column===nothing && error("scaling fixture population coefficient lacks an affine certificate")
    hsgp=only(r for r in records if r.kind===:hsgp)
    hsgp.grouped && hsgp.level_correlation || error("scaling fixture requires a multigroup HSGP")
    weights=Symbol(hsgp.declaration,".",hsgp.locals.weights)
    field=findall(n->BRM._brm_nlme_owned(n,weights),view.coordinates)
    length(field)==temporal*spatial^2 || error("scaling fixture field dimension differs")
    all(i->view.role[i]===:shared_latent,field) || error("scaling fixture field is not shared")
    length(view.subject_coordinates)==groups && all(x->length(x)==1,view.subject_coordinates) ||
        error("scaling fixture requires one standardized unit draw per group")
    units=sort(vcat(view.subject_coordinates...))
    latent=sort(vcat(field,units,[beta]))
    length(unique(latent))==length(latent) || error("scaling fixture block overlap")
    globals=setdiff(eachindex(view.coordinates),latent)
    length(globals)==3 || error("scaling fixture requires exactly three hyperparameters")
    # Only the selection and prior are consumed by the native constructor.
    # There is no ExactGaussianReference, dense precision, or dense covariance.
    block=(;p3=(;latent,globals),beta,prior=(;location=0.,scale=1.))
    target=native_gaussian_block_target(m,block)
    (;fixture...,target,field,units,beta,latent,globals)
end

struct CountedScalingOperator{O,M}
    inner::O
    m::M
    field::Vector{Int}
    counts::Vector{Int}
end
function counted_scaling_target(t)
    op=CountedScalingOperator(t.op,t.m,t.op.field,zeros(Int,3))
    JointWindowTarget(op,t.m,t.globals,t.units,t.trace_probes,t.logdet_probes)
end
function shared_value_gradient(op::CountedScalingOperator,u)
    op.counts[1]+=1
    shared_value_gradient(op.inner,u)
end
function shared_likelihood_hvp(op::CountedScalingOperator,u,v)
    op.counts[2]+=1
    shared_likelihood_hvp(op.inner,u,v)
end
function shared_third(op::CountedScalingOperator,u,v)
    op.counts[3]+=1
    shared_third(op.inner,u,v)
end

function scaling_measure(action,op;repetitions=7)
    # Warm exactly the same action first. Failures remain failures; no changed
    # tolerances, Krylov budgets, probes, or repeated seeds until success.
    fill!(op.counts,0)
    first_use=@timed action()
    warmup=(;seconds=first_use.time,bytes=first_use.bytes,gctime=first_use.gctime,
        calls=(;value_gradient=op.counts[1],hvp=op.counts[2],third=op.counts[3]))
    records=map(1:repetitions) do _
        fill!(op.counts,0)
        measurement=@timed action()
        (;seconds=measurement.time,bytes=measurement.bytes,gctime=measurement.gctime,
            calls=(;value_gradient=op.counts[1],hvp=op.counts[2],third=op.counts[3]))
    end
    (;warmup,records,median_seconds=median(getproperty.(records,:seconds)),
        median_bytes=median(getproperty.(records,:bytes)))
end

function scaling_solver_diagnostics(t,u)
    op=t.op
    fill!(op.counts,0)
    state=shared_poisson_mode(op,u;tol=1e-8)
    H(v)=shared_precision(op,state.mode,v)
    rng=Xoshiro(711)
    probes=[ifelse.(rand(rng,Bool,length(op.field)),1.,-1.) for _ in 1:8]
    trace=shared_trace(op,state.mode,probes)
    direction,correction=KrylovKit.linsolve(H,trace.trace[op.field];
        issymmetric=true,isposdef=true,rtol=1e-10,atol=1e-12,krylovdim=64,maxiter=1)
    correction.converged==1 || error("scaling diagnostic implicit solve exhausted budget")
    determinant=shared_logdet(rng,H,length(op.field);probes=8,
        spectral_lower=1.,maxdim=32,atol=1e-8,rtol=1e-8)
    solves=NamedTuple[]
    function solve(v)
        x,info=KrylovKit.linsolve(H,v;issymmetric=true,isposdef=true,
            rtol=1e-11,atol=1e-12,krylovdim=64,maxiter=1)
        push!(solves,(;operations=info.numops,residual=info.normres,converged=info.converged))
        info.converged==1 || error("scaling diagnostic draw solve exhausted budget")
        x
    end
    root=BRM._brm_spl_covariance_root(solve,randn(Xoshiro(712),length(op.field));maxdim=32,rtol=1e-9)
    (;mode_iterations=state.iteration,mode_operations=state.operations,
        mode_residual=norm(state.gradient[op.field]),trace_operations=trace.operations,
        trace_max_residual=trace.max_residual,
        correction_operations=correction.numops,correction_residual=correction.normres,
        determinant,root_iterations=root.iterations,root_error_bound=root.error_bound,
        draw_solves=solves,calls=(;value_gradient=op.counts[1],hvp=op.counts[2],third=op.counts[3]))
end

const SCALING_CASES=(
    (;name="base",groups=6,rows=12,temporal=4,spatial=2),
    (;name="rows4",groups=6,rows=48,temporal=4,spatial=2),
    (;name="rows16",groups=6,rows=192,temporal=4,spatial=2),
    (;name="field4",groups=6,rows=12,temporal=16,spatial=2),
    (;name="field16",groups=6,rows=12,temporal=64,spatial=2),
    (;name="units4",groups=24,rows=12,temporal=4,spatial=2),
    (;name="units16",groups=96,rows=12,temporal=4,spatial=2),
    (;name="rows16_field16",groups=6,rows=192,temporal=64,spatial=2))
