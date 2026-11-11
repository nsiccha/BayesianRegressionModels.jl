# Frozen shared coordinate fields across changing windows. Public synthetic
# data; a dense tensor-feature oracle checks values, density and native AD.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LinearAlgebra, Serialization

const TRAIN = (; time=repeat([0., 1., 2., 3.], 3),
    probe=repeat(["a", "b", "c"], inner=4), sx=[-1., 0., 1.], sy=[0., 1., -.5],
    y=zeros(12))
const WINDOW = (; time=[1., 2., 1., 2.], probe=["b", "b", "new", "new"],
    sx=[0., .25], sy=[1., .25], y=zeros(4))
const RETURNING = (; time=[.5, 2.5], probe=["c", "a"],
    sx=[-1., 1.], sy=[0., -.5], y=zeros(2))
const known = @brm begin
    mu ~ 0 + hsgp(time; k=4, by=probe, cor=multigroup((sx,sy),2.;k=2))
    length_scale(mu, hsgp(time)) = .8
    sd(mu, hsgp(time)) = .7
    y ~ Normal(mu, 1.)
end

const reporter = @brm begin
    mu ~ 1 + (1 | probe) + hsgp(time; k=4, by=probe, cor=multigroup((sx,sy),2.;k=2))
    y ~ Normal(mu, 1.)
end
const own_scale = @brm begin
    mu ~ 0 + hsgp(time; k=4, by=probe, cor=multigroup((sx,sy);k=2))
    y ~ Normal(mu, 1.)
end
const inducing = @brm begin
    mu ~ 0 + hsgp(time; k=4, by=probe,
        cor=multigroup((sx,sy),2.;k=2,method=:inducing))
    y ~ Normal(mu, 1.)
end
const free_positions = @brm begin
    mu ~ 0 + hsgp(time; k=4, by=probe,
        cor=multigroup((sx,sy),2.;method=:inducing,inducing=(px,py)))
    y ~ Normal(mu, 1.)
end

rknames(rk) = BRM._rk_layout_coordinate_names(rk)
function value(rk, u; bound=rk_bound_data(rk), effects=nothing)
    extra = effects === nothing ? () : (:b_probe,)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, extra..., keys(bound)...), want=(:mu,), bound)
    result = effects === nothing ? Base.invokelatest(query, u) :
        Base.invokelatest(query, u, effects)
    result isa Tuple ? only(result) : result
end
point(rk) = [.3sin(i) - .1cos(2i) for i in eachindex(rknames(rk))]
function roundtrip(state)
    io = IOBuffer()
    serialize(io, state)
    deserialize(seekstart(io))
end

# Form the full four-column tensor basis rather than the implementation's
# axis-by-axis contraction. Geometry is fixed from TRAIN by hand.
function oracle(u, design)
    levels = collect(levels_of(design.probe))
    center, half = [0., .25], [1.5, 1.125]
    z = reshape(u, 4, 4)
    result = zeros(length(design.time))
    for j in 1:4
        omega = pi*j/4.5
        r = 2/(.8omega)
        span = sqrt.(half.^2 .+ (3.2r)^2)
        modes = [(a,b) for b in 1:2 for a in 1:2]
        frequencies = [sum((pi .* [a,b] ./ (2 .* span)).^2) for (a,b) in modes]
        for i in eachindex(result)
            g = findfirst(==(design.probe[i]), levels)
            location = [design.sx[g], design.sy[g]]
            feature = [prod(sin.(pi .* [a,b] .* (location .- center .+ span) ./
                (2 .* span))) for (a,b) in modes] .* exp.(-r^2/4 .* frequencies)
            spatial = dot(feature/norm(feature), z[:,j])
            temporal = sin(omega*(design.time[i]-1.5+2.25))/sqrt(2.25)
            spectrum = .7sqrt(.8sqrt(2pi))*exp(-.8^2*omega^2/4)
            result[i] += temporal*spectrum*spatial
        end
    end
    result
end
levels_of(x) = BRM._brm_fit_levels(x)

@stestset "shared grid: frozen replay and dense field oracle" begin
    fit = check_rk_source_roundtrip(RKBRMI(known(TRAIN)))
    state = roundtrip(brm_fitted_state(fit))
    @test length(rknames(fit)) == 16
    @test rknames(fit) == [Symbol("hsgp_time.z.$p.$j") for j in 1:4 for p in 1:4]
    @test state.fitted.values.hsgp_time_coordinate_grid[1] == [0., .25]
    @test state.fitted.values.hsgp_time_coordinate_grid[2] == [1.5, 1.125]
    @test state.fitted.values.hsgp_time_fit == [1.5, 2.25]
    @test !any(v -> v isa AbstractArray && length(v) == 12,
        values(state.fitted.values))
    # Changing row order and categorical level order are distinct operations.
    reordered = merge(WINDOW, (; probe=categorical(WINDOW.probe; levels=["new","b"]),
        sx=reverse(WINDOW.sx), sy=reverse(WINDOW.sy)))
    single = (; time=[1.3], probe=["new"], sx=[.25], sy=[.25], y=[0.])
    binder = rk_design_binder(RKBRMI(known(WINDOW), state; resample_groups=(:probe,));
        resample_groups=(:probe,))
    for design in (TRAIN, WINDOW, RETURNING, single, reordered)
        replay = reprocess(fit, design; resample_groups=(:probe,))
        free = RKBRMI(known(design), state; resample_groups=(:probe,))
        @test rknames(free) == rknames(fit)
        @test BRM._rk_program_statements(BRM._rk_emit_ast(free.plan)) == state.program
        @test free.fitted.state.fits == state.fits
        bound = binder(design)
        for u in (zeros(16), point(fit), fill(.21,16))
            expected = oracle(u, design)
            @test value(replay, u) ≈ expected atol=2e-12
            @test value(free, u) ≈ expected atol=2e-12
            @test value(free, u; bound) ≈ expected atol=2e-12
        end
        # Each subsequent window also works from a data-free backend.
        @test value(reprocess(free, RETURNING; resample_groups=(:probe,)), point(fit)) ≈
            oracle(point(fit), RETURNING) atol=2e-12
    end
    @test all(c -> isempty(c.derived), values(binder.classes))
    free = RKBRMI(known(WINDOW), state; resample_groups=(:probe,))
    again = roundtrip(brm_fitted_state(free))
    @test again.fits[(:multigroup, :hsgp_time)] == state.fits[(:multigroup, :hsgp_time)]
    @test value(RKBRMI(known(RETURNING), again; resample_groups=(:probe,)), point(fit)) ≈
        oracle(point(fit), RETURNING) atol=2e-12
    problem = rk_logdensity_problem(free; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    density(u) = sum(logpdf.(Normal(),u)) + sum(logpdf.(Normal.(oracle(u,WINDOW),1.),WINDOW.y))
    for u in (point(fit), fill(.21,16))
        check_consumer_point(problem, u, density)
    end
end

@stestset "shared grid: identity checks include cached bindings" begin
    fit = RKBRMI(known(TRAIN))
    state = brm_fitted_state(fit)
    free = RKBRMI(known(WINDOW), state; resample_groups=(:probe,))
    binder = rk_design_binder(free; resample_groups=(:probe,))
    binder(WINDOW)
    for (design, message) in (
        (merge(WINDOW,(;sx=[.1,.25])), "different from its fitted location"),
        (merge(WINDOW,(;sx=[0.,9.])), "outside the region"),
        (merge(WINDOW,(;sy=[1.,NaN])), "finite"),
        (merge(WINDOW,(;sx=[0.])), "length"))
        @test_throws message reprocess(fit, design; resample_groups=(:probe,))
        @test_throws message RKBRMI(known(design), state; resample_groups=(:probe,))
        @test_throws message binder(design)
    end
    @test_throws "basis interval" RKBRMI(known(merge(WINDOW,(;time=fill(20.,4)))), state;
        resample_groups=(:probe,))
    # An ordinary replay keeps all fitted levels, even when its rows omit some.
    subset = merge(TRAIN, (;time=TRAIN.time[5:8],probe=TRAIN.probe[5:8],y=zeros(4)))
    @test value(reprocess(fit,subset),point(fit)) ≈ value(fit,point(fit))[5:8]
end

@stestset "shared grid: ordinary effects and spatial variants" begin
    for build in (reporter, own_scale, inducing, free_positions)
        train = merge(TRAIN,(;px=[-.5,.5],py=[0.,.5]))
        window = merge(WINDOW,(;px=[-.5,.5],py=[0.,.5]))
        fit = RKBRMI(build(train))
        state = roundtrip(brm_fitted_state(fit))
        replay = reprocess(fit,window;resample_groups=(:probe,))
        free = RKBRMI(build(window),state;resample_groups=(:probe,))
        shared(ns) = filter(n -> startswith(string(n),"hsgp_time.z."),ns)
        @test shared(rknames(free)) == shared(rknames(fit))
        @test BRM._rk_program_statements(BRM._rk_emit_ast(free.plan)) == state.program
        ufit = point(fit)
        u = [ufit[findfirst(==(n),rknames(fit))] for n in rknames(free)]
        effects = build === reporter ? zeros(2,1) : nothing
        # Both replays are models of the window's rows, at their own coordinates.
        @test value(replay,u;effects) ≈ value(free,u;effects) atol=2e-12
        # The binder's inputs read a draw of the fit in the fit's layout, the
        # window's probe effects in HAVE (snag `rk-replay-with-r-bf059724`).
        binder = rk_design_binder(fit;resample_groups=(:probe,))
        @test value(fit,ufit;bound=binder(window),effects) ≈ value(free,u;effects) atol=2e-12
        @test value(fit,ufit;bound=rk_design_binder(free;resample_groups=(:probe,))(window),
            effects) ≈ value(free,u;effects) atol=2e-12
        @test all(isfinite,value(free,u))
        single = merge(window,(;time=[1.3],probe=["new"],sx=[.25],sy=[.25],y=[0.]))
        singleton = RKBRMI(build(single),state;resample_groups=(:probe,))
        @test all(isfinite,value(singleton,point(singleton)))
        @test BRM._rk_program_statements(BRM._rk_emit_ast(singleton.plan)) == state.program
        if build === own_scale
            prior = state.fitted.values.hsgp_time_coordinate_prior
            @test prior[1] == log(2.)
            @test rk_bound_data(singleton).hsgp_time_coordinate_prior == prior
        elseif build === free_positions
            moved = RKBRMI(build(merge(window,(;px=[-.1,.1],py=[.1,.2]))),state;
                resample_groups=(:probe,))
            @test value(moved,u) ≈ value(free,u) atol=2e-12
        end
    end
end

@stestset "shared grid: per-level extensions remain refused" begin
    hyper = @brm begin
        mu ~ 0 + hsgp(time; k=4, by=probe, cor=multigroup((sx,sy),2.;k=2))
        log(sd(hsgp(time))) ~ 1 + (1 | probe)
        y ~ Normal(mu,1.)
    end
    projected = @brm begin
        mu ~ 0 + hsgp(time; k=4, by=probe, orthogonal_to=:intercept,
            cor=multigroup((sx,sy),2.;k=2))
        y ~ Normal(mu,1.)
    end
    for (build, message) in ((hyper,"per-level HSGP"),(projected,"per-level projection"))
        fit = RKBRMI(build(TRAIN))
        state = brm_fitted_state(fit)
        @test_throws message reprocess(fit,WINDOW;resample_groups=(:probe,))
        @test_throws message RKBRMI(build(WINDOW),state;resample_groups=(:probe,))
    end
end

@stestset "shared grid: fitted spatial prior agrees with Stan" begin
    fit = check_rk_source_roundtrip(RKBRMI(own_scale(TRAIN)))
    sb = SBBRMI(own_scale(TRAIN); mod=@__MODULE__)
    stan = BRM.stan_instantiate(sb;
        path=joinpath(tempdir(), "brm-rk-shared-grid-prior.stan"))
    transport = brm_coordinate_transport(fit,sb,BridgeStan.param_unc_names(stan.model))
    problem = rk_logdensity_problem(fit;ad_backend=AutoEnzyme(;mode=Enzyme.Reverse))
    for u in (zeros(length(rknames(fit))),point(fit))
        su = brm_rk_point_to_stan(transport,u)
        sg = similar(su)
        sv, _ = BridgeStan.log_density_gradient!(stan.model,su,sg;
            propto=false,jacobian=true)
        rv, rg = LogDensityProblems.logdensity_and_gradient(problem,u)
        @test rv ≈ brm_stan_logdensity_to_rk(transport,u,sv) atol=2e-10 rtol=2e-10
        @test rg ≈ brm_stan_gradient_to_rk(transport,u,sg) atol=2e-9 rtol=2e-9
    end
end
