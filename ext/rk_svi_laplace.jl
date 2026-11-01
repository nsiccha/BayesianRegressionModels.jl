# Included inside BayesianRegressionModelsReactiveKernelsEnzymeExt.
# Whole-window derivative queries are public RKPPL/RK operations. The
# algorithm consumes vectors and capped Krylov bases, never a dense Hessian.
function BRM._rk_svi_laplace_ad(m::BRM.BRMNLMEModel)
    backend=m.backend
    ad_backend=m.eta_hvp.ad_backend
    ad_backend isa AutoEnzyme || throw(ArgumentError(
        "GaussianBlockLaplace: the prepared model must use AutoEnzyme"))
    rkext=Base.get_extension(BRM,:BayesianRegressionModelsReactiveKernelsExt)
    plan=rkext._rk_translated(backend)
    forward=AutoEnzyme(;mode=BRM._brm_enzyme_mode(Enzyme.Forward,ad_backend))
    reverse=AutoEnzyme(;mode=BRM._brm_enzyme_mode(Enzyme.Reverse,ad_backend))
    second=SecondOrder(reverse,forward)
    direction=zeros(length(m.unit))
    full=BRM.rk_logdensity_problem(backend;ad_backend=reverse)
    likelihood=prepare_query_ad(prepare_ad,backend.model,plan,:likelihood,reverse,m.unit)
    hvp=prepare_query_ad(prepare_ad_hvp,backend.model,plan,:likelihood,second,(direction,),m.unit)
    third=prepare_query_ad(prepare_ad_third,backend.model,plan,:likelihood,
        SecondOrder(second,forward),(direction,),m.unit)
    (;full,likelihood,hvp,third)
end

function BRM._rk_svi_laplace_value_gradient(ad,u,part)
    part===:sampler && return BRM.LogDensityProblems.logdensity_and_gradient(ad.full,u)
    part===:likelihood && return ad_value_and_gradient(ad.likelihood,u)
    throw(ArgumentError("GaussianBlockLaplace: unknown density query $part"))
end
BRM._rk_svi_laplace_hvp(ad,u,v)=only(last(ad_gradient_and_hvp(ad.hvp,(v,),u)))
BRM._rk_svi_laplace_third(ad,u,v)=only(last(ad_gradient_hvp_and_third(ad.third,(v,),u)))
