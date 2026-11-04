# test/rk_marginal_ranef.jl — the RK half of likelihood-inactive |ID| margins.
#
# Run: julia --project=test test/rk_marginal_ranef.jl
#
# RK plans a shared `|ID|` bucket over the reachable predictors only, so an
# omitted response already narrows it. That narrowed block must carry the
# exact marginal law — LKJ(a, eta + (K - a)/2) and the kept margins' own scale
# priors — exactly as SBBRMI samples it (test/marginal_ranef.jl).
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module RKMarginalRanef
using BayesianRegressionModels, Distributions, StanBlocks

const DF = (;
    subject = [1, 1, 2, 2, 3, 3, 4, 4],
    x = [-1.0, 0.5, -0.5, 1.0, 0.0, 1.5, -1.5, 0.25],
    ya = [0.3, 0.9, -0.2, 0.4, 0.1, 1.2, -0.8, 0.0],
    yb = [1.1, 0.7, 0.2, 0.5, -0.3, 0.6, 0.9, 1.4],
    yc = [-0.4, -0.1, 0.8, 0.3, 0.5, 0.2, -0.6, 0.1],
)
const OMIT_B = (; DF.subject, DF.x, DF.ya, DF.yc)

plain(data) = @brm data begin
    a_loc ~ 1 + x + (1 + x | p | subject)
    b_loc ~ 1 + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    ya ~ Normal(a_loc, 0.5)
    yb ~ Normal(b_loc, 0.5)
    yc ~ Normal(c_loc, 0.5)
end

configured(data) = @brm data begin
    a_loc ~ 1 + x + (1 + x | p | subject)
    b_loc ~ 1 + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    sd(:, p) ~ Exponential(0.5)
    sd(b_loc, p) ~ Exponential(0.2)
    cor(:, p) ~ LKJCholesky(4, 2.0)
    ya ~ Normal(a_loc, 0.5)
    yb ~ Normal(b_loc, 0.5)
    yc ~ Normal(c_loc, 0.5)
end

# One row per subject; a `@plate` cell observes `ya` (reading `a_loc` and
# `b_loc`) and `yc` (reading only `c_loc`). Without `yc`, `c_loc` is read only
# by an omitted in-cell observation (todo `1784wd0`).
const CELL = (; subject=[1, 2, 3, 4], x=[-1.0, 0.5, 0.0, 1.5],
    ya=[[0.3, 0.9], [-0.2], [0.1, 1.2], [-0.8, 0.0]],
    yc=[[-0.4], [0.8, 0.3], [0.5], [-0.6, 0.1]])
const CELL_OMIT_C = (; CELL.subject, CELL.x, CELL.ya)
cell(data) = @brm data begin
    a_loc ~ 1 + x + (1 | p | subject)
    b_loc ~ 1 + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    @plate for i in eachindex(a_loc)
        ya[i] ~ Normal(a_loc[i] .+ b_loc[i], 0.5)
        yc[i] ~ Normal(c_loc[i], 0.5)
    end
end

# The kept statements of a cell read what only an omitted observation needs: a
# local computed for it, an indexed output reporting that local, a scale, a
# population prior reading a constant, and a flat data column through a grouped
# read. `kept_out` reports only what the kept likelihood reaches. Each option
# below is one of the block laws SBBRMI keeps whole.
const LONG = (; subject=[1, 2, 3, 4], x=[-1.0, 0.5, 0.0, 1.5],
    obs_subject=[1, 1, 2, 3, 3, 4, 4], t=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ya=[[0.3, 0.9], [-0.2], [0.1, 1.2], [-0.8, 0.0]],
    yc=[0.4, 0.8, 0.5, -0.6, 0.1, 0.2, 0.3])
const LONG_OMIT_C = (; LONG.subject, LONG.x, LONG.obs_subject, LONG.t, LONG.ya)
reported(data) = @brm data begin
    a_loc ~ 1 + x + (1 | p | subject)
    b_loc ~ 1 + (1 | p | subject)
    c_loc ~ 1 + x + (1 | p | subject)
    d_loc ~ 1 + (1 + x | p | subject)
    c_scale ~ LogNormal(0.0, 1.0)
    c_prior = constant(0.7)
    effect(c_loc, Intercept) ~ Normal(0.0, c_prior)
    rows = groups(obs_subject => subject)
    @plate for i in eachindex(a_loc)
        shared = a_loc[i] .+ b_loc[i]
        c_mu = shared .+ c_loc[i] .* t[rows[i]] .+ d_loc[i]
        c_out[i] = c_mu
        yc[rows[i]] ~ Normal(c_mu, c_scale)
        kept_out[i] = shared .* 2.0
        ya_loc[i] = shared
        ya[i] ~ Normal(shared, 0.5)
    end
end

# The plate iterates a predictor only the omitted response reads.
inactive_axis(data) = @brm data begin
    a_loc ~ 1 + x + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    rows = groups(obs_subject => subject)
    @plate for i in eachindex(c_loc)
        c_mu = c_loc[i] .+ t[rows[i]]
        yc[rows[i]] ~ Normal(c_mu, 1.0)
        ya[i] ~ Normal(a_loc[i], 0.5)
    end
end

# A cell latent only the omitted response reads.
inactive_latent(data) = @brm data begin
    a_loc ~ 1 + x + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    rows = groups(obs_subject => subject)
    @plate for i in eachindex(a_loc)
        zc ~ Normal(0.0, 1.0)
        yc[rows[i]] ~ Normal(c_loc[i] .+ zc, 1.0)
        ya[i] ~ Normal(a_loc[i], 0.5)
    end
end

# A top-level omitted response reading a cell of its own.
inactive_cell(data) = @brm data begin
    a_loc ~ 1 + x + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    rows = groups(obs_subject => subject)
    @plate for i in eachindex(c_loc)
        c_cell[i] = c_loc[i] .+ t[rows[i]]
    end
    @plate for i in eachindex(a_loc)
        ya[i] ~ Normal(a_loc[i], 0.5)
    end
    yc ~ Normal(c_cell, 1.0)
end

StanBlocks.@deffun rk_marginal_helper(x::vector[n])::vector[n] = 2.0 .* x
helper_cell(data) = @brm data begin
    a_loc ~ 1 + x + (1 | p | subject)
    @plate for i in eachindex(a_loc)
        ya[i] ~ Normal(rk_marginal_helper(a_loc[i]), 0.5)
    end
end

const WHOLE = (; CELL.subject, CELL.x, arm=[1, 1, 2, 2], CELL.ya, CELL.yc)
const WHOLE_OMIT_C = (; WHOLE.subject, WHOLE.x, WHOLE.arm, WHOLE.ya)
macro whole_block(fname, statements...)
    esc(quote
        $fname(data) = @brm data begin
            a_loc ~ 1 + x + (1 | p | subject)
            b_loc ~ 1 + (1 | p | subject)
            c_loc ~ 1 + x + (1 | p | subject)
            $(statements...)
            @plate for i in eachindex(a_loc)
                c_mu = a_loc[i] .+ c_loc[i]
                c_out[i] = c_mu
                yc[i] ~ Normal(c_mu, 1.0)
                ya[i] ~ Normal(a_loc[i] .+ b_loc[i], 0.5)
            end
        end
    end)
end
@whole_block student_t ranef(:, p) ~ TDist(4.0)
@whole_block known_scales sd(a_loc, p, Intercept) = 0.5 sd(b_loc, p, Intercept) = 0.6 sd(c_loc, p, Intercept) = 0.7
@whole_block r2d2_scales sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2.0, reference_scale=0.5)
@whole_block r2d2_reference sc ~ Exponential(1.0) sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2.0, reference_scale=0.5) sd(c_loc, p) ~ r2d2(reference_scale=sc)
stratified(data) = @brm data begin
    a_loc ~ 1 + x + (1 | p | gr(subject, by=arm))
    c_loc ~ 1 + x + (1 | p | gr(subject, by=arm))
    @plate for i in eachindex(a_loc)
        c_mu = a_loc[i] .+ c_loc[i]
        c_out[i] = c_mu
        yc[i] ~ Normal(c_mu, 1.0)
        ya[i] ~ Normal(a_loc[i], 0.5)
    end
end
end

const RKM = RKMarginalRanef

rk_buckets(plan) = plan.ranef_buckets
rk_buckets(plan::BRM._RKValuePlan) = rk_buckets(plan.regression)
rk_bucket(brmi) = only(b for b in rk_buckets(BRM._brm_rk_plan(brmi)) if b.id === :p)
rk_margins(bucket) = [(m.predictor, m.coefficient) for m in bucket.margins]

@stestset "RK plans the exact marginal of an omitted response's shared block" begin
    full = rk_bucket(RKM.plain(RKM.DF))
    @test rk_margins(full) == [(:a_loc, :Intercept), (:a_loc, :x),
                               (:b_loc, :Intercept), (:c_loc, :Intercept)]
    @test full.lkj_eta == 1.0
    omitted = rk_bucket(RKM.plain(RKM.OMIT_B))
    @test rk_margins(omitted) == [(:a_loc, :Intercept), (:a_loc, :x), (:c_loc, :Intercept)]
    @test omitted.lkj_eta == 1.5

    # `cor(:, p) ~ LKJCholesky(4, ...)` sizes the shared block, not the
    # narrowed one; `sd(b_loc, p)` addresses the omitted margin.
    full = rk_bucket(RKM.configured(RKM.DF))
    @test full.lkj_eta == 2.0
    @test length(full.sd_priors) == 4
    omitted = rk_bucket(RKM.configured(RKM.OMIT_B))
    @test omitted.lkj_eta == 2.5
    @test omitted.sd_priors == full.sd_priors[[1, 2, 4]]
    @test omitted.sd_priors != full.sd_priors[1:3]
end

@stestset "RK and Stan sample the same reduced block" begin
    for (name, builder) in (("plain", RKM.plain), ("configured", RKM.configured)),
            centered_groups in ((), [:subject])
        brmi = builder(RKM.OMIT_B)
        rk, problem = consumer_problem(brmi)
        sb = SBBRMI(brmi; mod=RKM, total_groups=(), centered_groups)
        path = joinpath(tempdir(), "brm-rk-marginal-ranef",
            name * (isempty(centered_groups) ? "" : "_centered") * ".stan")
        mkpath(dirname(path))
        stan = BRM.stan_instantiate(sb; path)
        names = BridgeStan.param_unc_names(stan.model)
        @test any(startswith(isempty(centered_groups) ? "b_p_subject_active_z_flat." :
            "b_p_subject_active_b_cols_bc."), names)
        transport = brm_coordinate_transport(rk, sb, names)
        @test length(transport) == length(names) ==
              length(coordinate_names(rk.model.layout))
        @test length(transport.centered) == length(centered_groups)
        for u in (zeros(length(transport)),
                  collect(range(-0.43, 0.51; length=length(transport))),
                  [0.37sin(i) for i in 1:length(transport)])
            stan_u = brm_rk_point_to_stan(transport, u)
            @test brm_check_coordinate_transport(transport, rk, stan.model, u).max_error <=
                1e-12
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                stan_gradient; propto=false, jacobian=true)
            @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        end
    end
end

@stestset "an omitted in-cell response narrows the shared block" begin
    @test rk_margins(rk_bucket(RKM.cell(RKM.CELL))) ==
        [(:a_loc, :Intercept), (:b_loc, :Intercept), (:c_loc, :Intercept)]
    omitted = rk_bucket(RKM.cell(RKM.CELL_OMIT_C))
    @test rk_margins(omitted) == [(:a_loc, :Intercept), (:b_loc, :Intercept)]
    @test omitted.lkj_eta == 1.5
    brmi = RKM.cell(RKM.CELL_OMIT_C)
    rk, problem = consumer_problem(brmi)
    @test !any(n -> startswith(string(n), "c_loc"), coordinate_names(rk.model.layout))
    sb = SBBRMI(brmi; mod=RKM, total_groups=())
    path = joinpath(tempdir(), "brm-rk-marginal-ranef", "cell_omit_c.stan")
    mkpath(dirname(path))
    stan = BRM.stan_instantiate(sb; path)
    names = BridgeStan.param_unc_names(stan.model)
    transport = brm_coordinate_transport(rk, sb, names)
    @test length(transport) == length(names) == length(coordinate_names(rk.model.layout))
    for u in (zeros(length(transport)),
              collect(range(-0.43, 0.51; length=length(transport))))
        stan_u = brm_rk_point_to_stan(transport, u)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
            stan_gradient; propto=false, jacobian=true)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=2e-11 rtol=2e-11
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
    end
end

# RK and Stan sample one target on paired coordinates. `check_gradient=false`
# keeps a case to the primal: each Enzyme pass of a new model compiles for minutes.
function same_rk_stan_target(brmi, name; check_gradient=false)
    rk, problem = consumer_problem(brmi)
    sb = SBBRMI(brmi; mod=RKM, total_groups=())
    path = joinpath(tempdir(), "brm-rk-marginal-ranef", name * ".stan")
    mkpath(dirname(path))
    stan = BRM.stan_instantiate(sb; path)
    names = BridgeStan.param_unc_names(stan.model)
    transport = brm_coordinate_transport(rk, sb, names)
    @test length(transport) == length(names) == length(coordinate_names(rk.model.layout))
    for u in (zeros(length(transport)), collect(range(-0.43, 0.51; length=length(transport))))
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
            stan_gradient; propto=false, jacobian=true)
        if check_gradient
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        else
            value = LogDensityProblems.logdensity(problem, u)
        end
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=2e-11 rtol=2e-11
    end
    rk, sb
end

@stestset "kept cell statements read nothing only an omitted response needs" begin
    # Bound, the model keeps every margin and output.
    full = RKM.reported(RKM.LONG)
    @test rk_margins(rk_bucket(full)) == [(:a_loc, :Intercept), (:b_loc, :Intercept),
        (:c_loc, :Intercept), (:d_loc, :Intercept), (:d_loc, :x)]
    brmi = RKM.reported(RKM.LONG_OMIT_C)
    bucket = rk_bucket(brmi)
    @test rk_margins(bucket) == [(:a_loc, :Intercept), (:b_loc, :Intercept)]
    @test bucket.lkj_eta == 1.0 + 3 / 2
    rk, _ = same_rk_stan_target(brmi, "reported"; check_gradient=true)
    names = string.(coordinate_names(rk.model.layout))
    @test !any(n -> occursin("c_scale", n) || occursin("c_loc", n) || occursin("d_loc", n), names)
    # What no likelihood reaches but adds no coordinate stays readable, and the
    # reported output of the omitted branch is gone rather than prior-only.
    kernel = only(a for a in BRM._brm_rk_plan(brmi).assignments
                  if a isa BRM._RKPreparedKernelAssignment)
    @test :kept_out in kernel.outputs
    @test !(:c_out in kernel.outputs)
end

@stestset "an omitted response's plate axis, cell latent and cell leave the program" begin
    for (name, builder) in (("inactive_axis", RKM.inactive_axis),
            ("inactive_latent", RKM.inactive_latent), ("inactive_cell", RKM.inactive_cell))
        brmi = builder(RKM.LONG_OMIT_C)
        @test rk_margins(rk_bucket(brmi)) == [(:a_loc, :Intercept)]
        same_rk_stan_target(brmi, name)
    end
end

@stestset "a block SBBRMI keeps whole stays whole on RK" begin
    for (name, builder) in (("student_t", RKM.student_t), ("known_scales", RKM.known_scales),
            ("stratified", RKM.stratified))
        brmi = builder(RKM.WHOLE_OMIT_C)
        @test any(m -> m.predictor === :c_loc, rk_bucket(brmi).margins)
        same_rk_stan_target(brmi, name)
    end
end

# `brm_coordinate_transport` does not pair R2D2 blocks, omitted response or not, so
# these coordinates are matched by name here.
@stestset "an R2D2 block SBBRMI keeps whole has the same target on RK" begin
    for (name, builder) in (("r2d2_scales", RKM.r2d2_scales), ("r2d2_reference", RKM.r2d2_reference))
        brmi = builder(RKM.WHOLE_OMIT_C)
        K = length(rk_bucket(brmi).margins)
        @test K == 3
        rk, problem = consumer_problem(brmi)
        path = joinpath(tempdir(), "brm-rk-marginal-ranef", name * ".stan")
        mkpath(dirname(path))
        stan = BRM.stan_instantiate(SBBRMI(brmi; mod=RKM, total_groups=()); path)
        names = BridgeStan.param_unc_names(stan.model)
        function stan_name(rk_name)
            z = match(r"^b_p_subject\.z\.(\d+)\.(\d+)$", rk_name)
            z === nothing || return "b_p_subject_z_flat.$((parse(Int, z[1]) - 1) * K + parse(Int, z[2]))"
            rk_name == "b_loc_Intercept" && return "pop_b_loc_beta_pop.1"
            replace(rk_name, "b_p_subject.L." => "b_p_subject_L.",
                "pop_a_loc.beta_pop." => "pop_a_loc_beta_pop.")
        end
        rk_names = string.(coordinate_names(rk.model.layout))
        @test length(rk_names) == length(names)
        positions = [findfirst(==(stan_name(n)), names) for n in rk_names]
        @test !any(isnothing, positions)
        for u in (zeros(length(names)), collect(range(-0.43, 0.51; length=length(names))))
            stan_u = zeros(length(names))
            stan_u[Int.(positions)] = u
            @test LogDensityProblems.logdensity(problem, u) ≈
                BridgeStan.log_density(stan.model, stan_u; propto=false, jacobian=true) atol=1e-10 rtol=1e-10
        end
    end
end

# Every response omitted: the prior program is built from the cells' own module,
# which holds the functions they call (here, one `Main` does not see).
@stestset "a prior program finds the functions its cells call" begin
    plan = BRM._brm_rk_plan(RKM.helper_cell((; RKM.LONG.subject, RKM.LONG.x)))
    @test plan isa BRM._RKUnconditionedPlan
end

# Fully observed programs keep their cell statements: a replay design without
# responses reads them, and `reprocess` compares programs (test/rk_reprocess.jl).
@stestset "a fully observed program is unchanged" begin
    kernel = only(a for a in BRM._brm_rk_plan(RKM.reported(RKM.LONG)).assignments
                  if a isa BRM._RKPreparedKernelAssignment)
    @test :c_out in kernel.outputs
    @test any(s -> Meta.isexpr(s, :(=)) && first(s.args) === :c_mu, kernel.body)
end
