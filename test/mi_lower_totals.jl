# Lower support belongs to the declared law; the sum's density and the law's
# truncation normalizer both remain in the target. Run this file directly.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LinearAlgebra

struct LowerSharesRecordingVector <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end
Base.size(v::LowerSharesRecordingVector) = size(v.values)
Base.IndexStyle(::Type{LowerSharesRecordingVector}) = IndexLinear()
Base.getindex(v::LowerSharesRecordingVector, i::Int) = (push!(v.reads, i); v.values[i])

@stestset "bounded grouped map has linear work and storage" begin
    # Exercise the emitted plate composition: each group map must be shared
    # by its entries, rather than recomputed once for every entry in the group.
    body = BRM._BRM_STATISTICAL_VALUES.brm_lower_grouped_local
    spec = Core.eval(@__MODULE__, :(ReactiveKernels.@kernel $body))
    measurements = map((32, 64, 128)) do K
        ends, residuals = collect(K:K:4K), fill(2.0K, 4)
        layout = BRM._brm_grouped_shares_layout(ends, residuals)
        _, entry_slot, _, _, group_entries, group_coords, entry_ids, group_ids, entry_group = layout
        bound = (; entry_ids, entry_group, entry_slot, group_ids, group_coords,
            group_entries, residuals, lo=ones(4K))
        q = Base.invokelatest(ReactiveKernels.prepare, spec;
            have=(:z, keys(bound)...), want=:values, bound)
        point = LowerSharesRecordingVector(zeros(4(K-1)), Int[])
        x = Base.invokelatest(q, point)
        @test all(x .> 1.)
        @test all(isapprox(sum(x[rows]), residuals[k]) for (k, rows) in enumerate(group_entries))
        @test length(point.reads) <= 4length(point)
        u = point.values
        Base.invokelatest(q, u)
        bytes = minimum(@allocated(Base.invokelatest(q, u)) for _ in 1:3)
        (; K, reads=length(point.reads), bytes, layout_bytes=Base.summarysize(layout))
    end
    println("LOWER_SHARES_SCALING=", measurements)
    for (small, large) in zip(measurements, measurements[2:end])
        @test large.reads <= 2.2small.reads
        @test large.bytes <= 2.5small.bytes
        @test large.layout_bytes <= 2.5small.layout_bytes
    end
end

lower_data() = (; x=Union{Missing,Float64}[missing, missing, 3, missing, missing, 2],
    lo=[1., 2., 1., 2., 1., 0.], total=[9., 2.],
    g=[1, 1, 1, 2, 3, 3], cells=[1, 2], y=[2., 4., 3., 2., 2.5, 2.])

@stestset "bounded free rows retain strict locality and native gradients" begin
    data = (; x=fill(missing, 3), lo=[0.5, 1., 2.], y=ones(3))
    models = (
        @brm(data, begin
            mi(x) ~ truncated(Exponential(); lower=lo)
            y ~ Normal(x, 1)
        end),
        @brm(data, begin
            mi(x) ~ truncated(Exponential(2.); lower=lo)
            y ~ Normal(x, 1)
        end),
        @brm(data, begin
            mi(x) ~ truncated(Normal(0., 2.); lower=lo)
            y ~ Normal(x, 1)
        end))
    # At u=0, x=lo+1. Add the authored law's score, the response's score
    # (y-x), and the log-Jacobian derivative; the bounds are fixed data.
    gradients = (-data.lo, 0.5 .- data.lo, 0.75 .- 1.25 .* data.lo)
    for (model, expected_gradient) in zip(models, gradients)
        rk = RKBRMI(model; completions=:noncentered)
        @test length(coordinate_names(rk.model.layout)) == 3
        q = prepare_cell_query(rk.model, rk_bound_plan(rk), :y; locality=:error)
        for row in eachindex(data.y)
            point = LowerSharesRecordingVector(zeros(3), Int[])
            @test isfinite(q(point, row))
            @test unique(point.reads) == [row]
        end
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, zeros(3))
        @test isfinite(value)
        @test gradient ≈ expected_gradient atol=1e-10
    end
end

function lower_model(data; normalized=true)
    normalized ? (@brm data begin
        mu ~ Normal(0, 1)
        sigma ~ LogNormal(0, 0.3)
        rows = groups(g => cells)
        mi(x)::shares(total, rows) ~ truncated(LogNormal(mu, sigma); lower=lo)
        y ~ Normal(x, 0.7)
    end) : (@brm data begin
        mu ~ Normal(0, 1)
        sigma ~ LogNormal(0, 0.3)
        rows = groups(g => cells)
        mi(x)::shares(total, rows) ~ restricted(LogNormal(mu, sigma); lower=lo)
        y ~ Normal(x, 0.7)
    end)
end

const LOWER_CELLS = (; loc=[[1., 1.5], [1.]], lo=[[1., 2.], [2.]],
    total=[8., 2.], y=[[2.5, 5.5], [2.1]])
lower_cells(data) = @brm data begin
    sigma ~ LogNormal(0, 0.3)
    @plate for i in eachindex(total)
        T::shares(total[i]) ~ truncated(LogNormal(loc[i], sigma); lower=lo[i])
        y[i] ~ Normal(T, 0.7)
    end
end
lower_support_cells(data) = @brm data begin
    sigma ~ LogNormal(0, 0.3)
    @plate for i in eachindex(total)
        T::shares(total[i]) ~ truncated(Exponential(sigma); lower=lo[i])
        y[i] ~ Normal(T, 0.7)
    end
end

@stestset "bounded cell construction" begin
    m = lower_cells(LOWER_CELLS)
    @test occursin("brm_lower_cell", BRM.stan_code(SBBRMI(m; mod=@__MODULE__, total_groups=())))
    @test RKBRMI(m) isa RKBRMI
end

@stestset "bounded cell density and start round trips" for (build, label) in
        ((lower_cells, "lognormal"), (lower_support_cells, "exponential"))
    data = LOWER_CELLS
    m = build(data)
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(m, "lower_cells_$label")
    rk = check_rk_source_roundtrip(RKBRMI(m))
    p = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    t = brm_coordinate_transport(rk, sb, String.(BridgeStan.param_unc_names(stan.model)))
    function oracle(u)
        v = BRM._rk_constrained_values(rk, u)
        lp = logpdf(LogNormal(0, .3), v.sigma) + log(v.sigma)
        for k in eachindex(data.total)
            x, lo = v.T[k], data.lo[k]
            @test all(x .>= lo) && sum(x) ≈ data.total[k]
            law = build === lower_cells ? LogNormal.(data.loc[k], v.sigma) : Exponential(v.sigma)
            lp += sum(logpdf.(truncated.(law, lo, Inf), x))
            lp += sum(logpdf.(Normal.(x, .7), data.y[k]))
            length(x) == 1 || (lp += sum(log.(x .- lo)) - log(data.total[k]-sum(lo)))
        end
        lp
    end
    u = fill(.2, LogDensityProblems.dimension(p))
    rv, rg = check_consumer_point(p, u, oracle)
    sv, sg = BridgeStan.log_density_gradient(stan.model, brm_rk_point_to_stan(t, u); propto=false)
    @test rv ≈ brm_stan_logdensity_to_rk(t, u, sv) atol=1e-10
    @test rg ≈ brm_stan_gradient_to_rk(t, u, sg) atol=2e-8
    @test brm_check_coordinate_transport(t, rk, stan.model, u).max_error < 1e-10
    given = [[2.5, 5.5], [2.]]
    ru = brm_start(rk, (; T=given, sigma=.7))
    su = brm_start(sb, stan.model, (; T=given, sigma=.7))
    @test brm_rk_point_to_stan(t, ru) ≈ su atol=1e-10
    @test BRM._rk_constrained_values(rk, ru).T ≈ given
    @test_throws ArgumentError brm_start(rk, (; T=[[.5,7.5],[2.]]))
    @test_throws ArgumentError brm_start(sb, stan.model, (; T=[[.5,7.5],[2.]]))
end

@stestset "scalar bounds and ordinary bounded imputation" begin
    data = (; x=Union{Missing,Float64}[missing, missing, 3.], total=[7.], g=[1,1,1], cells=[1])
    grouped = @brm data begin
        mu ~ Normal(0, 1)
        rows = groups(g => cells)
        mi(x)::shares(total, rows) ~ truncated(LogNormal(mu, 1); lower=1.)
    end
    ordinary = @brm merge(data, (; y=[1.5,2.5,3.])) begin
        mu ~ Normal(0, 1)
        mi(x) ~ truncated(LogNormal(mu, 1); lower=1.)
        y ~ Normal(x, .7)
    end
    for (label, model) in (("scalar_unread", grouped), ("ordinary", ordinary))
        sb = SBBRMI(model; mod=@__MODULE__, total_groups=())
        stan = consumer_stan(model, "lower_$label")
        for mode in (:centered, :noncentered)
            rk = check_rk_source_roundtrip(RKBRMI(model; completions=mode))
            p = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            t = brm_coordinate_transport(rk, sb, String.(BridgeStan.param_unc_names(stan.model)))
            u = fill(-.3, LogDensityProblems.dimension(p))
            rv, rg = LogDensityProblems.logdensity_and_gradient(p, u)
            sv, sg = BridgeStan.log_density_gradient(stan.model, brm_rk_point_to_stan(t, u); propto=false)
            @test rv ≈ brm_stan_logdensity_to_rk(t, u, sv) atol=1e-10
            @test rg ≈ brm_stan_gradient_to_rk(t, u, sg) atol=2e-8
            @test brm_check_coordinate_transport(t, rk, stan.model, u).max_error < 1e-10
        end
    end
end
function lower_oracle(stan, data, u; normalized=true, family=LogNormal, live=false)
    values = Dict(zip(BridgeStan.param_names(stan; include_tp=true),
        BridgeStan.param_constrain(stan, u; include_tp=true)))
    mu, sigma = values["mu"], values["sigma"]
    x = [values["x.$j"] for j in eachindex(data.x)]
    law = family(mu, sigma)
    lower = live ? data.lo .+ values["bound"] : data.lo
    lp = logpdf(Normal(), mu) + logpdf(LogNormal(0, .3), sigma) + log(sigma)
    for j in eachindex(x)
        @test x[j] >= lower[j]
        lp += logpdf(law, x[j]) - (normalized ? logccdf(law, lower[j]) : 0.)
        lp += logpdf(Normal(x[j], .7), data.y[j])
    end
    @test sum(x[1:3]) ≈ data.total[1]
    @test x[4] == data.total[2]
    slack = data.total[1] - data.x[3] - sum(lower[1:2])
    live && (lp += logpdf(LogNormal(-2, .1), values["bound"]) + log(values["bound"]))
    lp + log(x[5]-lower[5]) + sum(log.(x[1:2] .- lower[1:2])) - log(slack)
end

@stestset "lower shares geometry and support" begin
    lo, total = [1., 2., 0.5], 9.
    z = [-.8, .4]
    x = BRM._brm_shares_values(z, total, lo)
    @test all(x .> lo) && sum(x) ≈ total
    @test BRM._brm_shares_coordinates(x, lo) ≈ z
    # Independent softmax in the explicit isometric basis.
    H = [1/sqrt(2) 1/sqrt(6); -1/sqrt(2) 1/sqrt(6); 0 -2/sqrt(6)]
    w = exp.(H*z); w ./= sum(w)
    @test x ≈ lo .+ (total-sum(lo)) .* w
    @test BRM._brm_lower_shares_logjac(z, total, lo) ≈
        2log(total-sum(lo)) + sum(log, w)
    @test BRM._brm_shares_values(Float64[], 2., 2.) == [2.]
    @test BRM._brm_shares_coordinates([2.], [2.]) == Float64[]
    @test_throws ArgumentError BRM._brm_shares_coordinates([1.], [2.])
    @test BRM._brm_lower_shares_logjac(Float64[], 2., 2.) == 0.
    @test BRM._brm_lower_shares_logjac([0.], 3., [1.,2.]) == -Inf
    @test BRM._brm_lower_shares_logjac([0.], 3., [Inf,2.]) == -Inf
    @test all(isfinite, BRM._brm_shares_values([0.], 3., [Inf,2.]))
    @test_throws ErrorException BRM._brm_lower_floor(Pareto)
    data = lower_data()
    for backend in (SBBRMI, RKBRMI)
        @test_throws ArgumentError backend(lower_model(merge(data, (; total=[6.,2.]))))
        @test_throws ArgumentError backend(lower_model(merge(data, (; lo=[1.,2.,4.,2.,1.,0.]))))
    end
end

@stestset "zero-coordinate singleton endpoint" begin
    data = (; x=Union{Missing,Float64}[missing,1.], total=[0.], g=[1,2], cells=[1])
    m = @brm data begin
        scale ~ LogNormal(0, .3)
        rows = groups(g => cells)
        mi(x)::shares(total, rows) ~ truncated(Exponential(scale); lower=0.)
    end
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(m, "lower_singleton_zero")
    for mode in (:centered, :noncentered)
        rk = RKBRMI(m; completions=mode)
        p = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        @test LogDensityProblems.dimension(p) == 1
        u = [.2]
        oracle(u) = logpdf(LogNormal(0,.3), exp(u[1])) + u[1] +
            logpdf(Exponential(exp(u[1])), 0.) + logpdf(Exponential(exp(u[1])), 1.)
        rv, rg = check_consumer_point(p, u, oracle)
        t = brm_coordinate_transport(rk, sb, String.(BridgeStan.param_unc_names(stan.model)))
        sv, sg = BridgeStan.log_density_gradient(stan.model, brm_rk_point_to_stan(t,u); propto=false)
        @test rv ≈ sv atol=1e-10
        @test rg ≈ brm_stan_gradient_to_rk(t,u,sg) atol=2e-8
    end
end

@stestset "bounded replay, outputs and complete groups" begin
    data = lower_data()
    m = lower_model(data)
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    d = brm_descriptor(sb)
    @test !isnothing(brm_output(d, :x))
    @test any(o -> o.name === :x_y_shares && o.kind === :parameter, d.outputs)
    @test any(o -> o.name === :x_y_free && o.kind === :parameter, d.outputs)
    fresh = merge(data, (; lo=data.lo .+ [.2,.1,0,0,.3,0], total=[10.,2.]))
    moved = reprocess(sb, fresh)
    stan = BRM.stan_instantiate(moved; path=joinpath(tempdir(), "brm-rk-consumer", "lower_completion_true.stan"))
    for mode in (:centered, :noncentered)
        rk = RKBRMI(m; completions=mode)
        replay = reprocess(rk, fresh)
        @test replay.model === rk.model
        p = rk_logdensity_problem(replay; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        t = brm_coordinate_transport(replay, moved, String.(BridgeStan.param_unc_names(stan.model)))
        u = brm_start(replay, (; mu=.2, sigma=.7))
        su = brm_start(moved, stan.model, (; mu=.2, sigma=.7))
        @test brm_rk_point_to_stan(t, u) ≈ su atol=1e-10
        check_consumer_point(p, u, point -> lower_oracle(stan.model, fresh, brm_rk_point_to_stan(t, point)))
        @test brm_check_coordinate_transport(t, replay, stan.model, u).max_error < 1e-10
        @test_throws ArgumentError reprocess(rk, merge(data, (; total=[6.,2.])))
        @test_throws ArgumentError reprocess(rk, merge(data, (; lo=[1.,2.,4.,2.,1.,0.])))
        changed_mask = copy(data.x); changed_mask[3] = missing
        @test_throws Exception reprocess(rk, merge(data, (; x=changed_mask)))
    end
    complete = merge(data, (; x=[2.,4.,3.,2.,2.5,2.]))
    for backend in (SBBRMI, RKBRMI)
        @test backend(lower_model(complete)) isa backend
        @test_throws ArgumentError backend(lower_model(merge(complete, (; lo=[3.,2.,1.,2.,1.,0.]))))
        @test_throws ErrorException backend(lower_model(merge(complete, (; total=[10.,2.]))))
        upper = @brm data begin
            rows = groups(g => cells)
            mi(x)::shares(total, rows) ~ truncated(LogNormal(0, 1); lower=lo, upper=10.)
        end
        @test_throws ErrorException backend(upper)
    end
end

lower_normal(data) = @brm data begin
    mu ~ Normal(0, 1)
    sigma ~ LogNormal(0, 0.3)
    rows = groups(g => cells)
    mi(x)::shares(total, rows) ~ truncated(Normal(mu, sigma); lower=lo)
    y ~ Normal(x, 0.7)
end
lower_live(data) = @brm data begin
    mu ~ Normal(0, 1)
    sigma ~ LogNormal(0, 0.3)
    bound ~ LogNormal(-2, 0.1)
    rows = groups(g => cells)
    mi(x)::shares(total, rows) ~ truncated(LogNormal(mu, sigma); lower=lo + bound)
    y ~ Normal(x, 0.7)
end
function lower_unread(data; normalized=true)
    normalized ? (@brm data begin
        mu ~ Normal(0, 1)
        sigma ~ LogNormal(0, 0.3)
        rows = groups(g => cells)
        mi(x)::shares(total, rows) ~ truncated(LogNormal(mu, sigma); lower=lo)
    end) : (@brm data begin
        mu ~ Normal(0, 1)
        sigma ~ LogNormal(0, 0.3)
        rows = groups(g => cells)
        mi(x)::shares(total, rows) ~ restricted(LogNormal(mu, sigma); lower=lo)
    end)
end
@stestset "unread restricted free rows retain their parameter-dependent mass" begin
    for normalized in (true, false)
        m = lower_unread(lower_data(); normalized)
        sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
        stan = consumer_stan(m, "lower_unread_$(normalized)")
        for mode in (:centered, :noncentered)
            rk = check_rk_source_roundtrip(RKBRMI(m; completions=mode))
            p = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            t = brm_coordinate_transport(rk, sb, String.(BridgeStan.param_unc_names(stan.model)))
            @test LogDensityProblems.dimension(p) == (normalized ? 3 : 4)
            u = fill(.15, LogDensityProblems.dimension(p))
            rv, rg = LogDensityProblems.logdensity_and_gradient(p, u)
            sv, sg = BridgeStan.log_density_gradient(stan.model, brm_rk_point_to_stan(t, u); propto=false)
            @test rv ≈ brm_stan_logdensity_to_rk(t, u, sv) atol=1e-10
            @test rg ≈ brm_stan_gradient_to_rk(t, u, sg) atol=2e-8
        end
    end
end
@stestset "another family and live lower support" begin
    data = lower_data()
    for (build, family, live) in ((lower_normal, Normal, false), (lower_live, LogNormal, true))
        fresh = live ? merge(data, (; lo=[1., 2., .1, .5, .1, 0.])) : data
        m = build(fresh)
        sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
        stan = consumer_stan(m, "lower_family_$(live)")
        for mode in (:centered, :noncentered)
            rk = check_rk_source_roundtrip(RKBRMI(m; completions=mode))
            problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            t = brm_coordinate_transport(rk, sb, String.(BridgeStan.param_unc_names(stan.model)))
            n = LogDensityProblems.dimension(problem)
            u = zeros(n)
            oracle = point -> lower_oracle(stan.model, fresh, brm_rk_point_to_stan(t, point); family, live)
            rv, rg = check_consumer_point(problem, u, oracle)
            sv, sg = BridgeStan.log_density_gradient(stan.model, brm_rk_point_to_stan(t, u); propto=false)
            @test rv ≈ brm_stan_logdensity_to_rk(t, u, sv) atol=1e-10
            @test rg ≈ brm_stan_gradient_to_rk(t, u, sg) atol=2e-8
            @test brm_check_coordinate_transport(t, rk, stan.model, u).max_error < 1e-10
            if live
                bad = copy(u)
                bad[coordinate(coordinate_names(rk.model.layout), :bound)] = 3.
                @test LogDensityProblems.logdensity(problem, bad) == -Inf
                @test BridgeStan.log_density(stan.model, brm_rk_point_to_stan(t, bad); propto=false) == -Inf
            end
        end
    end
end

@stestset "bounded completion construction" begin
    m = lower_model(lower_data())
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    @test occursin("brm_lower", BRM.stan_code(sb))
    for mode in (:centered, :noncentered)
        rk = RKBRMI(m; completions=mode)
        @test rk isa RKBRMI
    end
end

@stestset "bounded density gradient transport and source replay" begin
    data = lower_data()
    for normalized in (true, false)
        m = lower_model(data; normalized)
        sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
        stan = consumer_stan(m, "lower_completion_$(normalized)")
        for mode in (:centered, :noncentered)
            rk = check_rk_source_roundtrip(RKBRMI(m; completions=mode))
            problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            transport = brm_coordinate_transport(rk, sb, String.(BridgeStan.param_unc_names(stan.model)))
            n = LogDensityProblems.dimension(problem)
            for u in (zeros(n), collect(range(-.3, .4; length=n)))
                su = brm_rk_point_to_stan(transport, u)
                sv, sg = BridgeStan.log_density_gradient(stan.model, su; propto=false, jacobian=true)
                @test sv ≈ lower_oracle(stan.model, data, su; normalized) atol=1e-10
                oracle = point -> lower_oracle(stan.model, data,
                    brm_rk_point_to_stan(transport, point); normalized)
                rv, rg = check_consumer_point(problem, u, oracle)
                @test rv ≈ brm_stan_logdensity_to_rk(transport, u, sv) atol=1e-10
                @test rg ≈ brm_stan_gradient_to_rk(transport, u, sg) atol=2e-8
                @test brm_check_coordinate_transport(transport, rk, stan.model, u).max_error < 1e-10
            end
        end
        fresh = merge(data, (; lo=data.lo .+ [0.2, 0.1, 0, 0, .3, 0], total=[10.,2.]))
        @test reprocess(sb, fresh) isa SBBRMI
        @test_throws ArgumentError reprocess(sb, merge(data, (; total=[6.,2.])))
    end
end
