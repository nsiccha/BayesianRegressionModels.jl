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

@brm function independent(site)
    shift ~ 0 + (1 | p | site)
    return sum(shift)
end

const separate = @brm begin
    root ~ 0 + (1 | p | site)
    a ~ independent(a_site)
    b ~ independent(b_site)
    primary ~ Normal(root, 1)
    ya ~ Normal(a, 1)
    yb ~ Normal(b, 1)
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

const mixed = @brm begin
    root ~ 0 + (1 | p | site)
    a ~ 0 + (1 | p | a_site)
    b ~ independent(b_site)
    primary ~ Normal(root, 1)
    ya ~ Normal(sum(a), 1)
    yb ~ Normal(b, 1)
end

@stestset "Explicit parent-block margins share the parent's factor" begin
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
    names = BridgeStan.param_unc_names(stan.model)
    @test names == BridgeStan.param_unc_names(ref.model)
    @test only(ranef_blocks(sb)).n_terms == 3
    points = (zeros(12), collect(range(-0.32,0.41; length=12)), [0.3sin(i) for i in 1:12])
    probes = map(points) do u
        value, grad = BridgeStan.log_density_gradient(stan.model, u; propto=false)
        ref_value, ref_grad = BridgeStan.log_density_gradient(ref.model, u; propto=false)
        @test value ≈ ref_value
        @test grad ≈ ref_grad
        (; u, value, grad)
    end
    d = brm_descriptor(sb)
    cnames = BridgeStan.param_names(stan.model)
    scales = [only(brm_ranef_sd_coordinates(d, :shift, cnames; id=:p,
        submodel=(:readings, key)).coordinates) for key in (:a,:b)]
    @test scales == [findfirst(==("b_p_site_tau.$i"), cnames) for i in (2,3)]
    rk, problem = consumer_problem(m)
    transport = brm_coordinate_transport(rk, sb, names)
    for probe in probes
        u = brm_stan_point_to_rk(transport, probe.u)
        rk_value, rk_grad = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test rk_value ≈ brm_stan_logdensity_to_rk(transport,u,probe.value)
        @test rk_grad ≈ brm_stan_gradient_to_rk(transport,u,probe.grad)
    end
    more = vcat(records, [(; records[1]..., id=:c)])
    larger = panels((; input..., records=more, y=[input.y; 0.3]))
    @test stan_code(SBBRMI(larger; mod=@__MODULE__, total_groups=())) == stan_code(sb)
    small_program = BRM._rk_emit_ast(rk.plan)
    large_program = BRM._rk_emit_ast(RKBRMI(larger).plan)
    @test small_program.defs == large_program.defs
    @test small_program.main == large_program.main
end

@stestset "Local margins own one scale and two site innovations per fit" begin
    site = ["x","y"]
    records = [(id=:a, shared=false, site=["y","x"]),
               (id=:b, shared=false, site=["y","y","x"])]
    input = (; records, site, primary=[0.1,-0.1], y=[0.2,0.4])
    m = panels(input)
    reference = separate((; site, input.primary, a_site=records[1].site,
        b_site=records[2].site, ya=[input.y[1]], yb=[input.y[2]]))
    stan = consumer_stan(m, "collection-local-groups"; mod=@__MODULE__)
    twin = consumer_stan(reference, "explicit-local-groups"; mod=@__MODULE__)
    @test BridgeStan.param_unc_num(stan.model) == BridgeStan.param_unc_num(twin.model) == 9
    for u in (zeros(9), fill(0.27,9), fill(-0.34,9))
        value, grad = BridgeStan.log_density_gradient(stan.model, u; propto=false)
        expected, ref_grad = BridgeStan.log_density_gradient(twin.model, u; propto=false)
        @test value ≈ expected
        # A scoped call and a native plate pack the same local parameters in
        # different orders. A constant point is invariant under that permutation.
        @test sort(grad) ≈ sort(ref_grad)
    end
    rk, problem = consumer_problem(m)
    @test rk.model.layout.total == 9
    ref_rk, ref_problem = consumer_problem(reference)
    for u in (zeros(9), fill(0.27,9))
        value, grad = LogDensityProblems.logdensity_and_gradient(problem, u)
        expected, ref_grad = LogDensityProblems.logdensity_and_gradient(ref_problem, u)
        @test value ≈ expected
        @test sort(grad) ≈ sort(ref_grad)
    end
end

@stestset "One member shares the parent block while another fits its own" begin
    site = ["x","y"]
    a = (id=:a, shared=true, site=["y","x"])
    b = (id=:b, shared=false, site=["x","y","y"])
    primary = [0.1,-0.1]
    y = [0.4,0.2]
    m = panels((;site,primary,records=[b,a],y))
    ref = mixed((;site,primary,a_site=a.site,b_site=b.site,ya=[y[2]],yb=[y[1]]))
    stan = consumer_stan(m, "collection-mixed-groups"; mod=@__MODULE__).model
    twin = consumer_stan(ref, "explicit-mixed-groups"; mod=@__MODULE__).model
    @test BridgeStan.param_unc_num(stan) == BridgeStan.param_unc_num(twin) == 10
    native, rp = consumer_problem(m)
    native_ref, twin_rp = consumer_problem(ref)
    @test native.model.layout.total == native_ref.model.layout.total == 10
    for x in (0., 0.23)
        u = fill(x,10)
        value, gradient = BridgeStan.log_density_gradient(stan,u;propto=false)
        expected, ref_gradient = BridgeStan.log_density_gradient(twin,u;propto=false)
        @test value ≈ expected
        @test sort(gradient) ≈ sort(ref_gradient)
        value, gradient = LogDensityProblems.logdensity_and_gradient(rp,u)
        expected, ref_gradient = LogDensityProblems.logdensity_and_gradient(twin_rp,u)
        @test value ≈ expected
        @test sort(gradient) ≈ sort(ref_gradient)
    end
end
end
