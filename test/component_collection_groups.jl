module CollectionGroupsTest

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using ReactiveKernelsDistributionKernels

@brm function panel(settings)
    if settings.shared
        shift ~ 0 + (1 | parentblock(:p) | settings.site)
    else
        shift ~ 0 + (1 | p | settings.site)
    end
    return sum(shift)
end

const panels = @brm begin
    root ~ 0 + (1 | p | site)
    for entry in records
        readings[entry.id] ~ panel(entry)
    end
    primary ~ Normal(root, 1)
    y ~ Normal(readings, 1)
end

const explicit = @brm begin
    root ~ 0 + (1 | p | site)
    a ~ 0 + (1 | p | a_site)
    b ~ 0 + (1 | p | b_site)
    primary ~ Normal(root, 1)
    ya ~ Normal(sum(a), 1)
    yb ~ Normal(sum(b), 1)
end

@testset "Explicit parent-block margins share the parent's factor" begin
    site = ["x","y"]
    records = [(id=:b, shared=true, site=["y","y","x"]),
               (id=:a, shared=true, site=["x","y"])]
    input = (; records, site, primary=[0.1,-0.1], y=[0.4,0.2])
    m = panels(input)
    twin = explicit((; site, input.primary, a_site=records[2].site,
        b_site=records[1].site, ya=[input.y[2]], yb=[input.y[1]]))
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(m, "collection-parent-block"; mod=@__MODULE__)
    ref = consumer_stan(twin, "explicit-parent-block"; mod=@__MODULE__)
    rk, problem = consumer_problem(m)
    names = BridgeStan.param_unc_names(stan.model)
    @test names == BridgeStan.param_unc_names(ref.model)
    @test only(ranef_blocks(sb)).n_terms == 3
    transport = brm_coordinate_transport(rk, sb, names)
    for u in (zeros(12), collect(range(-0.32,0.41; length=12)), [0.3sin(i) for i in 1:12])
        su = brm_rk_point_to_stan(transport, u)
        value, grad = BridgeStan.log_density_gradient(stan.model, su; propto=false)
        ref_value, ref_grad = BridgeStan.log_density_gradient(ref.model, su; propto=false)
        @test value ≈ ref_value
        @test grad ≈ ref_grad
        rk_value, rk_grad = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test rk_value ≈ brm_stan_logdensity_to_rk(transport,u,value)
        @test rk_grad ≈ brm_stan_gradient_to_rk(transport,u,grad)
    end
    more = vcat(records, [(; records[1]..., id=:c)])
    larger = panels((; input..., records=more, y=[input.y; 0.3]))
    @test stan_code(SBBRMI(larger; mod=@__MODULE__, total_groups=())) == stan_code(sb)
    small_program = BRM._rk_emit_ast(rk.plan)
    large_program = BRM._rk_emit_ast(RKBRMI(larger).plan)
    @test small_program.defs == large_program.defs
    @test small_program.main == large_program.main
end
end
