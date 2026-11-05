using Test, BayesianRegressionModels, HTMXObjects
using ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels

module RKModelViewFixture
using BayesianRegressionModels
const inspecting = Ref(false)
"""Sum a toy cell's latent values. Display must not execute this helper."""
function cell_total(x)
    inspecting[] && error("model inspection evaluated cell_total")
    sum(x)
end
simplex = @brm begin
    sigma ~ Exponential(1)
    p ~ Dirichlet([1., 2., 3.])
    theta[1:n] .~ Normal.(0, 1)
    y ~ Normal(sum(theta) + sum(p), sigma)
end
cells = @brm begin
    sigma ~ Exponential(1)
    @plate for i in eachindex(total)
        latent ~ Normal(loc[i], 0.5)
        center = cell_total(latent)
        total[i] ~ Normal(center, sigma)
    end
end
grouped = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 + x | block | group)
    cor(:, block) ~ LKJCholesky(2, 2.0)
    y ~ Normal(mu, sigma)
end
end

@testset "RK model view uses constrained parameter metadata" begin
    @test Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt) !== nothing
    @test Base.get_extension(BRM, :BayesianRegressionModelsHTMXObjectsExt) !== nothing
    model = RKModelViewFixture.simplex((; n=2, y=[0.1, 0.2]))
    rk = RKBRMI(model)
    rows = Dict(row.path => row for row in BRM._brm_view_rk_parameters(rk))
    @test rows[(:sigma,)].rank == 0
    @test rows[(:sigma,)].shape == ()
    @test rows[(:p,)].shape == (3,)
    @test rows[(:p,)].ncoordinates == 2
    @test rows[(:theta,)].shape == (2,)
    before_bound = rk.bound.value
    for mime in ("text/html", "text/markdown", "text/plain")
        rendered = repr(mime, brm_model_view(rk))
        @test occursin("Prepared RK parameters", rendered)
        @test occursin("Vector · 3", rendered)
        @test occursin("free coordinates", lowercase(rendered))
    end
    @test rk.bound.value === before_bound
    @test model === brm_source(rk)
    @test BRM._brm_view_rk_parameters(deepcopy(rk)) !== nothing

    # Wrapping an existing artifact must not lazily translate or bind it just
    # to display metadata already present in the build.
    wrapped = RKBRMI(model, rk.plan, rk.model)
    @test wrapped.bound.value === nothing
    @test wrapped.program.value === nothing
    @test occursin("Vector · 3", repr("text/plain", brm_model_view(wrapped)))
    @test wrapped.bound.value === nothing
    @test wrapped.program.value === nothing
end

@testset "RK model view retains ragged cells and helper documentation" begin
    data = (; loc=[[0., 1.], Float64[], [2.]], total=[0.1, 0.2, 0.3])
    rk = RKBRMI(RKModelViewFixture.cells(data))
    rows = BRM._brm_view_rk_parameters(rk)
    latent = only(row for row in rows if row.path == (:latent,))
    @test latent.rank == 1
    @test latent.shape == (3,)
    @test latent.element_shapes == [(2,), (0,), (1,)]
    @test latent.ncoordinates == 3
    RKModelViewFixture.inspecting[] = true
    try
        html = repr("text/html", brm_model_view(rk))
        @test occursin("Sum a toy cell", html)
        @test occursin("Element shapes", html)
        @test occursin("Vector · 0", html)
        @test occursin("<details", html)
        @test !occursin("Base.~", html)
        if haskey(ENV, "BRM_RK_MODEL_VIEW_HTML")
            page = htmx(HTMXObjects.h.main(
                brm_model_view(rk; title="RK model with ragged cell parameters"); class="container");
                pico_version="2", extra_head=(htmxo_syntax_head()...,))
            write(ENV["BRM_RK_MODEL_VIEW_HTML"], repr("text/html", page))
        end
    finally
        RKModelViewFixture.inspecting[] = false
    end
end

@testset "RK replay does not claim the training build's shapes are current" begin
    training = (; x=[-1., 0., 1., 2.], group=[1, 1, 2, 2], y=[0.1, 0.2, 0.3, 0.4])
    design = (; x=[0., 1., 2.], group=[3, 4, 5], y=[0.2, 0.3, 0.4])
    rk = RKBRMI(RKModelViewFixture.grouped(training))
    rows = BRM._brm_view_rk_parameters(rk)
    @test any(row -> row.rank == 2 && row.shape == (2, 2) && row.ncoordinates == 1, rows)
    @test occursin("Matrix · 2 × 2", repr("text/plain", brm_model_view(rk)))
    replay = reprocess(rk, design; resample_groups=(:group,))
    @test replay.model === rk.model
    before = replay.bound.value
    @test BRM._brm_view_rk_parameters(replay) === nothing
    rendered = repr("text/plain", brm_model_view(replay))
    @test occursin("Current parameter dimensions are unresolved", rendered)
    @test occursin("Declaration data", rendered)
    @test replay.bound.value === before

    # Data-free replays can also reuse an earlier graph/layout.
    state = brm_fitted_state(rk)
    fresh = RKBRMI(RKModelViewFixture.grouped(training), state)
    @test BRM._brm_view_rk_parameters(fresh) !== nothing
    replay = reprocess(fresh, design; resample_groups=(:group,))
    @test replay.model === fresh.model
    @test BRM._brm_view_rk_parameters(replay) === nothing
    @test occursin("Current parameter dimensions are unresolved",
        repr("text/plain", brm_model_view(replay)))
end
