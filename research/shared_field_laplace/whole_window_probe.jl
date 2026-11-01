# A joint conditional field mode/trace consumes the whole current window.
# Use the public whole-query derivative graph for that joint operation.
# Compact-cell locality remains necessary for ordinary minibatch SVI and is
# still tracked upstream; this does not repair or bypass that cell contract.
# julia --project=test research/shared_field_laplace/whole_window_probe.jl 'joint whole-window query'
include("joint_svi_probe.jl")
using DifferentiationInterface: SecondOrder

struct WholeWindowOperator{M,T,P,H,D}
    m::M
    t::T
    field::Vector{Int}
    full::P
    hvp::H
    third::D
end

function whole_window_operator(m)
    record = only(r for r in BRM._rk_coordinate_records(m.backend.plan) if r.kind === :hsgp)
    record.grouped && record.level_correlation || error("requires shared multigroup HSGP")
    weights = Symbol(record.declaration,".",record.locals.weights)
    field = findall(n -> BRM._brm_nlme_owned(n,weights),m.view.coordinates)
    isempty(field) && error("no field weights found for $weights")
    all(i -> m.view.role[i] === :shared_latent,field) || error("field is not shared")
    t = (;globals=BRM._brm_svi_globals(m),units=m.view.subject_coordinates)
    forward = AutoEnzyme(;mode=Enzyme.Forward)
    second = SecondOrder(NLME_AD,forward)
    built,plan = m.backend.model,RKEXT._rk_translated(m.backend)
    direction = zeros(length(m.unit))
    hvp = prepare_query_ad(prepare_ad_hvp,built,plan,:likelihood,second,
        (direction,),m.unit)
    third = prepare_query_ad(prepare_ad_third,built,plan,:likelihood,
        SecondOrder(second,forward),(direction,),m.unit)
    full = rk_logdensity_problem(m.backend;ad_backend=NLME_AD)
    WholeWindowOperator(m,t,field,full,hvp,third)
end

shared_value_gradient(op::WholeWindowOperator,u) =
    LogDensityProblems.logdensity_and_gradient(op.full,u)
function shared_likelihood_hvp(op::WholeWindowOperator,u,v)
    direction = zeros(length(u)); direction[op.field] = v
    _,(hv,) = ad_gradient_and_hvp(op.hvp,(direction,),u)
    hv
end
function shared_third(op::WholeWindowOperator,u,v)
    direction = zeros(length(u)); direction[op.field] = v
    _,_,(third,) = ad_gradient_hvp_and_third(op.third,(direction,),u)
    third
end

@stestset "joint whole-window query" begin
    for poisson in (false,true)
        (;m) = shared_field_fixture(;poisson)
        joint,cell = whole_window_operator(m),shared_field_operator(m)
        rng = Xoshiro(845)
        u = first(BRM._brm_nlme_prior_median_point(m))+.05randn(rng,length(m.unit))
        v = randn(rng,length(joint.field))
        @test joint.field == cell.field
        value,gradient = shared_value_gradient(joint,u)
        cell_value,cell_gradient = shared_value_gradient(cell,u)
        @test value ≈ cell_value atol=1e-11
        @test gradient ≈ cell_gradient atol=1e-10
        @test shared_likelihood_hvp(joint,u,v) ≈ shared_likelihood_hvp(cell,u,v) atol=1e-10
        @test shared_third(joint,u,v) ≈ shared_third(cell,u,v) atol=1e-10
        a,b = shared_poisson_mode(joint,u),shared_poisson_mode(cell,u)
        @test a.mode ≈ b.mode atol=1e-9
        @test a.value ≈ b.value atol=1e-9
        println((;poisson,field=length(joint.field),
            scope="whole-window/cell equivalence; no scaling measurement"))
    end
end
