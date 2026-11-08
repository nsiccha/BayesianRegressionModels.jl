# brm_nlme_view on random-effect blocks declared inside function-style
# components: `kin ~ kinetics(subject)` with `(1 | pk | subject)` in its body is
# read as the same block at model scope, over the column the call passes, so its
# grouping is one of the model's (snag nlme-view-does-n-4a5e64ce).
#
# Covered: the partition of a model whose subject block and crossed device
# block are declared in components equals its model-scope twin's under the
# calls' paths (the components' population coefficients stay shared latents);
# lockstep likelihoods equal the twin's at the same point and the full RK
# density, with η gradients; the SVI minibatch split; a component grouping
# named nowhere is refused by name, and in `shared_groups` its block is shared;
# a nested call through a renamed input; prior-median starts of the blocks'
# scales from the calls' own data; named estimates; `brm_svi_fit` with
# connected units; `brm_nlme_joint_map` and a checked full Stan start.
#
# RUN: julia --project=test test/nlme_component_blocks.jl [testset-filter...]

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using BridgeStan, LinearAlgebra, LogDensityProblems, Random

module ComponentBlocks
using BayesianRegressionModels, Distributions

@brm function kinetics(subject)
    log(CL) ~ 1 + (1 | pk | subject)
    log(V) ~ 1 + (1 | pk | subject)
    effect(CL, Intercept) ~ Normal(0, 1)
    effect(V, Intercept) ~ Normal(3, 1)
    sd(CL, pk) ~ Exponential(1)
    sd(V, pk) ~ Exponential(2)
    cor(:, pk) ~ LKJCholesky(2, 2)
    return (; CL, V)
end

@brm function devices(device, scale)
    gain ~ 1 + (1 | cal | device)
    effect(gain, Intercept) ~ Normal(0, 1)
    sd(gain, cal) ~ Exponential(scale)
    return gain
end

# A nested call, through an input named otherwise than the model's column.
@brm function instrument(unit, scale)
    inner ~ devices(unit, scale)
    return inner
end

const component = @brm begin
    sigma ~ Exponential(1)
    kin ~ kinetics(subject)
    dev ~ devices(device, cal_scale)
    CL = kin.CL
    V = kin.V
    m = dev + dose / V * exp(-CL / V * t)
    y ~ Normal(m, sigma)
end

const nested = @brm begin
    sigma ~ Exponential(1)
    kin ~ kinetics(subject)
    dev ~ instrument(device, cal_scale)
    CL = kin.CL
    V = kin.V
    m = dev + dose / V * exp(-CL / V * t)
    y ~ Normal(m, sigma)
end

const twin = @brm begin
    sigma ~ Exponential(1)
    log(CL) ~ 1 + (1 | pk | subject)
    log(V) ~ 1 + (1 | pk | subject)
    gain ~ 1 + (1 | cal | device)
    effect(CL, Intercept) ~ Normal(0, 1)
    effect(V, Intercept) ~ Normal(3, 1)
    effect(gain, Intercept) ~ Normal(0, 1)
    sd(CL, pk) ~ Exponential(1)
    sd(V, pk) ~ Exponential(2)
    sd(gain, cal) ~ Exponential(cal_scale)
    cor(:, pk) ~ LKJCholesky(2, 2)
    m = gain + dose / V * exp(-CL / V * t)
    y ~ Normal(m, sigma)
end
end

# The NLME fixture's four subjects, read by three devices crossed with them.
const CB_DATA = (; NLME_DATA...,
    device=["d1", "d2", "d3", "d2", "d1", "d3", "d1", "d2", "d3", "d1", "d2", "d3", "d1"],
    cal_scale=0.5)
const CB_OPTIONS = (; subject_group=:subject, shared_groups=(:device,))

# The twin's name of a component model's coordinate: the calls' path dropped.
cb_twin_name(name) = Symbol(replace(String(name), r"^(kin|dev)\.(inner\.)?" => ""))
cb_permutation(view, twin_view) =
    [findfirst(==(cb_twin_name(n)), twin_view.coordinates) for n in view.coordinates]

const CB_RK = RKBRMI(ComponentBlocks.component(CB_DATA))
const CB_TWIN = RKBRMI(ComponentBlocks.twin(CB_DATA))

# Lockstep likelihoods sum to the RK likelihood at the packed point, each
# subject's η gradient is the full RK density's on its block without the
# block's standard-normal prior, and a subject's term does not move with
# another subject's block.
function cb_check_lockstep(m, rng)
    θ = 0.2 .* randn(rng, length(m.theta))
    σ = 0.2 .* randn(rng, length(m.sigma))
    H = 0.5 .* randn(rng, length(m.view.block_margins), length(m.view.levels))
    values = brm_nlme_loglikelihoods(m, θ, σ, H)
    u = BRM._brm_nlme_point(m, θ, σ, H)
    @test sum(values) ≈ Base.invokelatest(likelihood_query(m.backend), u) rtol=1e-10
    _, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    full = rk_logdensity_problem(m.backend; ad_backend=NLME_AD)
    _, full_gradient = LogDensityProblems.logdensity_and_gradient(full, u)
    for (i, block) in enumerate(m.view.subject_coordinates)
        @test G[:, i] ≈ full_gradient[block] .+ u[block] atol=1e-8
    end
    other = copy(H)
    other[:, 2] .+= 0.7
    moved = brm_nlme_loglikelihoods(m, θ, σ, other)
    @test moved[2] != values[2]
    @test moved[[1; 3:end]] == values[[1; 3:end]]
    (; θ, σ, H, values, u)
end

@stestset "a component's blocks are the model's, under the call's path" begin
    v = brm_nlme_view(CB_RK; CB_OPTIONS...)
    vt = brm_nlme_view(CB_TWIN; CB_OPTIONS...)
    records = BRM._brm_nlme_records(CB_RK.plan)
    @test Set(BRM._brm_nlme_groupings(records)) == Set([:subject, :device])
    perm = cb_permutation(v, vt)
    @test sort(perm) == collect(eachindex(vt.coordinates))
    # Only the components' population coefficients differ: they stay shared.
    population = findall(i -> vt.role[perm[i]] === :population, eachindex(perm))
    @test Set(v.coordinates[population]) ==
        Set(Symbol.(["kin.CL_Intercept", "kin.V_Intercept", "dev.gain_Intercept"]))
    @test all(==(:shared_latent), v.role[population])
    rest = setdiff(eachindex(perm), population)
    @test v.role[rest] == vt.role[perm[rest]]
    @test v.subject == vt.subject[perm]
    @test v.levels == vt.levels
    @test [perm[block] for block in v.subject_coordinates] == vt.subject_coordinates
    @test v.family_sizes == vt.family_sizes == [2]
    @test only(v.families).block == (; group=:subject, id=Symbol("kin.pk"), predictor=nothing)
    @test [(m.predictor, m.coefficient) for m in v.block_margins] ==
        [(Symbol("kin.CL"), :Intercept), (Symbol("kin.V"), :Intercept)]
    # The device block, over a grouping crossed with the subjects: shared.
    device = findall(n -> startswith(String(n), "dev.b_cal_device."), v.coordinates)
    @test length(device) == 1 + 3
    @test all(==(:shared_latent), v.role[device])
    # A model whose components declare no block reads as before.
    @test BRM._brm_nlme_records(CB_TWIN.plan) == BRM._rk_coordinate_records(CB_TWIN.plan)
end

@stestset "component blocks: lockstep likelihoods equal the model-scope twin's" begin
    m = brm_nlme_model(CB_RK; ad_backend=NLME_AD, CB_OPTIONS...)
    mt = brm_nlme_model(CB_TWIN; ad_backend=NLME_AD, CB_OPTIONS...)
    perm = cb_permutation(m.view, mt.view)
    (; H, values, u) = cb_check_lockstep(m, Xoshiro(1))
    ut = similar(u)
    ut[perm] = u
    @test BRM._brm_nlme_point(mt, ut[mt.theta], ut[mt.sigma], H) ≈ ut atol=1e-12
    @test brm_nlme_loglikelihoods(mt, ut[mt.theta], ut[mt.sigma], H) ≈ values rtol=1e-12
    full = rk_logdensity_problem(CB_RK; ad_backend=NLME_AD)
    full_twin = rk_logdensity_problem(CB_TWIN; ad_backend=NLME_AD)
    value, gradient = LogDensityProblems.logdensity_and_gradient(full, u)
    value_twin, gradient_twin = LogDensityProblems.logdensity_and_gradient(full_twin, ut)
    @test value ≈ value_twin rtol=1e-12
    @test gradient ≈ gradient_twin[perm] rtol=1e-10
end

@stestset "component blocks: the SVI split is the RK target" begin
    m = brm_nlme_model(CB_RK; ad_backend=NLME_AD, CB_OPTIONS...)
    t = BRM._brm_svi_target(m)
    device = findall(n -> startswith(String(n), "dev.b_cal_device."), m.view.coordinates)
    @test device ⊆ t.globals
    @test sort(reduce(vcat, t.units)) ==
        sort(reduce(vcat, m.view.subject_coordinates))
    u = m.unit .+ 0.4 .* randn(Xoshiro(4), length(m.unit))
    g = u[t.globals]
    @test sort(vcat(t.globals, reduce(vcat, t.units))) == eachindex(u)
    P, gP = BRM._brm_svi_global_term(t, g)
    gg = zeros(length(g))
    terms = map(eachindex(t.units)) do i
        gz = zeros(length(t.units[i]))
        value = BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[t.units[i]])
        (; value, gg=copy(gg), gz)
    end
    full = rk_logdensity_problem(CB_RK; ad_backend=NLME_AD)
    value, gradient = LogDensityProblems.logdensity_and_gradient(full, u)
    @test P + sum(x -> x.value, terms) ≈ value rtol=1e-12
    @test gP .+ sum(x -> x.gg, terms) ≈ gradient[t.globals] rtol=1e-10
    for (i, block) in enumerate(t.units)
        @test terms[i].gz ≈ gradient[block] rtol=1e-10
    end
end

@stestset "a component's grouping is named like any other" begin
    # Named nowhere: refused by name, as at model scope.
    refusal = try
        brm_nlme_view(CB_RK; subject_group=:subject)
        nothing
    catch err
        err
    end
    @test refusal isa BRMNLMEViewError
    @test occursin("Named in neither: `device`", refusal.message)
    # Two groupings and no subject_group: the caller chooses.
    @test_throws BRMNLMEViewError brm_nlme_view(CB_RK)
    # Either component grouping may be the units: the devices, with the
    # subjects crossed with them.
    v = brm_nlme_view(CB_RK; subject_group=:device, shared_groups=(:subject,))
    @test v.group === :device && length(v.levels) == 3
    @test only(v.families).block.id === Symbol("dev.cal")
    pk = findall(n -> startswith(String(n), "kin.b_pk_subject."), v.coordinates)
    @test length(pk) == 2 + 1 + 4 * 2
    @test all(==(:shared_latent), v.role[pk])
end

@stestset "a nested component call through a renamed input" begin
    rk = RKBRMI(ComponentBlocks.nested(CB_DATA))
    v = brm_nlme_view(rk; CB_OPTIONS...)
    vc = brm_nlme_view(CB_RK; CB_OPTIONS...)
    @test [Symbol(replace(String(n), "dev.inner." => "dev.")) for n in v.coordinates] ==
        vc.coordinates
    @test v.role == vc.role && v.subject == vc.subject
    @test v.subject_coordinates == vc.subject_coordinates
    lifted = only(r for r in BRM._brm_nlme_records(rk.plan) if get(r, :scope, ()) == (:dev, :inner))
    @test lifted.group === :device
    @test lifted.id === Symbol("dev.inner.cal")
    @test lifted.z === Symbol("dev.inner.b_cal_device.z")
    @test lifted.call_block == (; group=:device, id=:cal, margins=((:gain, :Intercept),))
    @test lifted.margins == ((Symbol("dev.inner.gain"), :Intercept),)
    # The view reads the call's data scalar in its scale law, through both calls.
    @test lifted.scale_priors == (:(Exponential(0.5)),)
end

@stestset "component blocks: prior-median starts and named estimates" begin
    m = brm_nlme_model(CB_RK; ad_backend=NLME_AD, CB_OPTIONS...)
    mt = brm_nlme_model(CB_TWIN; ad_backend=NLME_AD, CB_OPTIONS...)
    perm = cb_permutation(m.view, mt.view)
    u, chosen = BRM._brm_nlme_prior_median_point(m)
    ut, chosen_twin = BRM._brm_nlme_prior_median_point(mt)
    # The blocks' scales start at their laws' medians, as the twin's do.
    scales = findall(==(:subject_scale), m.view.role)
    @test scales ⊆ chosen
    @test u[scales] ≈ ut[perm[scales]] atol=1e-12
    @test exp.(u[scales]) ≈ [median(Exponential(1)), median(Exponential(2))] atol=1e-12
    device_scale = findfirst(==(Symbol("dev.b_cal_device.tau.1")), m.view.coordinates)
    @test device_scale in chosen
    @test exp(u[device_scale]) ≈ median(Exponential(0.5)) atol=1e-12
    θ, σ = 0.3 .* randn(Xoshiro(6), length(m.theta)), 0.3 .* randn(Xoshiro(7), length(m.sigma))
    Ω = [0.36 0.105; 0.105 0.49]
    point = brm_nlme_estimates(m, θ, σ, Ω).point
    pk = point.var"kin.pk"
    @test pk.group === :subject
    @test pk.margins == [(; predictor=Symbol("kin.CL"), coefficient=:Intercept),
        (; predictor=Symbol("kin.V"), coefficient=:Intercept)]
    @test pk.covariance ≈ Ω
    cal = point.var"dev.cal"
    @test cal.group === :device && cal.crossed
    @test length(cal.effects) == 3
end

@stestset "brm_svi_fit on component blocks, with connected units" begin
    fit = brm_svi_fit(CB_RK; rng=Xoshiro(11), family=MeanFieldGlobal(),
        unit_partition=:connected, separator=:none, unit_sampling=:work,
        CB_OPTIONS..., stop_after=0, ndraws=2)
    m = fit.diagnostics.model
    @test m.view.group === :subject
    @test m.view.unit_partition === :connected
    @test m.view.subject_levels == ["s1", "s2", "s3", "s4"]
    fit = brm_svi_fit(m; rng=Xoshiro(12), batch_size=2, max_iterations=8, ndraws=4)
    @test all(isfinite, fit.diagnostics.elbo)
    @test size(fit.draws, 1) == length(m.view.coordinates)
end

@stestset "brm_nlme_joint_map on component blocks: a checked full Stan start" begin
    brmi = ComponentBlocks.component(CB_DATA)
    sb = SBBRMI(brmi; mod=ComponentBlocks, total_groups=())
    mkpath(joinpath(tempdir(), "brm-nlme-component-blocks"))
    map = brm_nlme_joint_map(CB_RK, sb; CB_OPTIONS...,
        path=joinpath(tempdir(), "brm-nlme-component-blocks", "component-blocks.stan"))
    view = map.view
    @test only(map.blocks).factor !== nothing
    @test length(only(map.blocks).scales) == 2
    rng = Xoshiro(8)
    θ = 0.3 .* randn(rng, count(r -> r in (:population, :shared_latent), view.role))
    σ = [-0.4]
    Ω = [0.36 0.105; 0.105 0.49]
    H = 0.3 .* randn(rng, 2, length(view.levels))
    start = brm_nlme_joint_start(map; theta=θ, sigma=σ, omega=Ω, eta=H,
        subjects=view.levels)
    @test start.checked.max_error <= 1e-9
    recovered = brm_nlme_joint_values(map, start.stan)
    # Every θ coordinate is shared here: the components' coefficients too.
    @test isempty(recovered.theta)
    @test recovered.shared.values ≈ θ atol=1e-10
    @test recovered.sigma ≈ σ atol=1e-10
    @test recovered.omega ≈ Ω atol=1e-10
    @test recovered.eta ≈ H atol=1e-10
end
