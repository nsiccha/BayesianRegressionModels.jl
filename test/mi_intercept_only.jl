# test/mi_intercept_only.jl — an intercept-only predictor that only an `mi(...)`
# likelihood sizes (snag `intercept-only-l-163ef7a4`).
#
# Run: julia --startup-file=no --project=test test/mi_intercept_only.jl
#
# `log(sigma) ~ 1` read by `mi(y) ~ LogNormal(mu, sigma)` has no column of its
# own, so its row axis is `y`'s. The `mi` column never reaches the data as
# itself (it becomes observed values plus missing coordinates), so SBBRMI used
# to emit `num_elements(y)` and StanBlocks failed with "Could not find y in
# model", and RKBRMI refused the row axis. Both now take `y`'s row count.
#
# The density checks compare each model with its scalar twin, which declares
# the intercept as an ordinary parameter, `log_sigma ~ Normal(m, s);
# sigma = exp(log_sigma)`: one unconstrained real with the same prior and the
# same likelihood, so the two log densities and gradients agree at every point.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const COLS = (; x=[-1.0, 0.5, 2.0, 0.25, 1.0, -0.3],
                g=["a", "b", "a", "c", "b", "c"],
                y=Union{Missing,Float64}[1.2, missing, 0.4, missing, 2.0, 0.7])

lognormal_scale(data) = @brm data begin
    log(sigma) ~ 1
    effect(sigma, Intercept) ~ Normal(-0.5, 0.5)
    mu ~ 1 + x + (1 | g)
    mi(y) ~ LogNormal(mu, sigma)
end
lognormal_scale_twin(data) = @brm data begin
    log_sigma ~ Normal(-0.5, 0.5)
    sigma = exp(log_sigma)
    mu ~ 1 + x + (1 | g)
    mi(y) ~ LogNormal(mu, sigma)
end

normal_scale(data) = @brm data begin
    log(sigma) ~ 1
    effect(sigma, Intercept) ~ Normal(-0.5, 0.5)
    mu ~ 1 + x
    mi(y) ~ Normal(mu, sigma)
end
normal_scale_twin(data) = @brm data begin
    log_sigma ~ Normal(-0.5, 0.5)
    sigma = exp(log_sigma)
    mu ~ 1 + x
    mi(y) ~ Normal(mu, sigma)
end

normal_location(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1
    effect(mu, Intercept) ~ Normal(1, 2)
    mi(y) ~ Normal(mu, sigma)
end
normal_location_twin(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ Normal(1, 2)
    mi(y) ~ Normal(mu, sigma)
end

# A covariate model's scale, read by an outcome that is fully observed.
const COVARIATE = (; x=Union{Missing,Float64}[1.0, missing, 2.0, 0.5, missing, 1.5],
                     y=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
covariate_scale(data) = @brm data begin
    mx ~ Normal(0, 1)
    log(sx) ~ 1
    effect(sx, Intercept) ~ Normal(-0.5, 0.5)
    mi(x) ~ LogNormal(mx, sx)
    mu ~ 1 + x
    y ~ Normal(mu, 1)
end
covariate_scale_twin(data) = @brm data begin
    mx ~ Normal(0, 1)
    log_sx ~ Normal(-0.5, 0.5)
    sx = exp(log_sx)
    mi(x) ~ LogNormal(mx, sx)
    mu ~ 1 + x
    y ~ Normal(mu, 1)
end

const STAN_DIR = mkpath(joinpath(tempdir(), "brm-mi-intercept-only"))

build(brmi) = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
function stan_problem(brmi, name)
    sb = build(brmi)
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    sb, BRM.stan_instantiate(sb; path=joinpath(STAN_DIR, name * ".stan"))
end

points(n) = (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))

# `renames` maps each of the model's coordinate names that the twin spells
# differently to the twin's name; every other coordinate shares its name.
function check_twin(names, twin_names, density, twin_density, renames)
    twin_name(n) = get(renames, n, n)
    @test sort(twin_name.(names)) == sort(twin_names)
    order = [findfirst(==(twin_name(n)), twin_names) for n in names]
    for u in points(length(names))
        twin_u = similar(u)
        twin_u[order] = u
        value, gradient = density(u)
        twin_value, twin_gradient = twin_density(twin_u)
        @test isfinite(value)
        @test value ≈ twin_value atol=1e-12
        @test gradient ≈ twin_gradient[order] atol=1e-10
    end
end

check_stan_twin(stan, twin, renames) = check_twin(
    BridgeStan.param_unc_names(stan.model), BridgeStan.param_unc_names(twin.model),
    u -> BridgeStan.log_density_gradient(stan.model, u; propto=false),
    u -> BridgeStan.log_density_gradient(twin.model, u; propto=false), renames)

function check_rk_twin(brmi, twin_brmi, renames)
    backend, problem = consumer_problem(brmi)
    twin_backend, twin_problem = consumer_problem(twin_brmi)
    check_twin(coordinate_names(backend.model.layout),
        coordinate_names(twin_backend.model.layout),
        u -> LogDensityProblems.logdensity_and_gradient(problem, u),
        u -> LogDensityProblems.logdensity_and_gradient(twin_problem, u), renames)
end

@stestset "SBBRMI sizes the intercept by the mi column's row count" begin
    for (model, source) in ((lognormal_scale(COLS), :y), (normal_scale(COLS), :y),
                            (normal_location(COLS), :y),
                            (covariate_scale(COVARIATE), :x))
        sb = build(model)
        code = BRM.stan_code(sb)
        extent = Symbol(:n_rows_, source)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test sb.data[extent] == 6
        @test sb.preproc[extent].kind === :axis_extent
        @test sb.preproc[extent].raw_ref === source
        @test occursin("rep_vector(1.0, $extent)", code)
        @test !occursin("num_elements($source)", code)
    end
    # One extent serves every intercept-only predictor of the same column.
    both = build(@brm COLS begin
        log(sigma) ~ 1
        mu ~ 1
        mi(y) ~ Normal(mu, sigma)
    end)
    code = BRM.stan_code(both)
    @test count("rep_vector(1.0, n_rows_y)", code) == 2
    @test both.data[:n_rows_y] == 6
    # A predictor with a covariate keeps its own axis and binds no extent.
    with_covariate = build(@brm COLS begin
        log(sigma) ~ 1 + x
        mu ~ 1 + x
        mi(y) ~ Normal(mu, sigma)
    end)
    @test !haskey(with_covariate.data, :n_rows_y)
    @test !haskey(with_covariate.preproc, :n_rows_y)
end

@stestset "a data column with the extent's name is refused" begin
    # refused: the extent would overwrite the caller's `n_rows_y` column, which
    # the model reads as a covariate.
    clash = merge(COLS, (; n_rows_y=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    @test_throws "data column `n_rows_y` collides with the row count" build(@brm clash begin
        log(sigma) ~ 1
        mu ~ 1 + n_rows_y
        mi(y) ~ Normal(mu, sigma)
    end)
end

@stestset "SBBRMI: the density of the scalar twin" begin
    for (model, twin, data, renames, name) in (
            (lognormal_scale, lognormal_scale_twin, COLS,
             Dict("pop_log_sigma_beta_pop.1" => "log_sigma"), "lognormal-scale"),
            (normal_scale, normal_scale_twin, COLS,
             Dict("pop_log_sigma_beta_pop.1" => "log_sigma"), "normal-scale"),
            (normal_location, normal_location_twin, COLS,
             Dict("pop_mu_beta_pop.1" => "mu"), "normal-location"),
            (covariate_scale, covariate_scale_twin, COVARIATE,
             Dict("pop_log_sx_beta_pop.1" => "log_sx"), "covariate-scale"))
        _, stan = stan_problem(model(data), name)
        _, stan_twin = stan_problem(twin(data), name * "-twin")
        check_stan_twin(stan, stan_twin, renames)
    end
end

@stestset "RKBRMI: the density of the scalar twin" begin
    # RK admits a modeled Normal scale; a modeled LogNormal scale is outside its
    # current slice whatever the response, observed or `mi`.
    check_rk_twin(normal_scale(COLS), normal_scale_twin(COLS),
        Dict(:sigma_Intercept => :log_sigma))
    check_rk_twin(normal_location(COLS), normal_location_twin(COLS),
        Dict(:mu_Intercept => :mu))
end

@stestset "replay recomputes the extent from the column's length" begin
    sb = build(normal_scale(COLS))
    longer = (; x=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                y=Union{Missing,Float64}[missing, 1.0, 1.1, missing, 0.9, 1.3, missing, 0.8])
    direct = build(normal_scale(longer))
    replayed = reprocess(sb, longer; freeze_constants=false)
    @test replayed.data[:n_rows_y] == 8
    @test BRM.stan_code(replayed) == BRM.stan_code(direct)
    @test replayed.data == direct.data
    @test restan_data(sb, longer; freeze_constants=false)[:n_rows_y] == 8
    # A frozen replay keeps the fitted mask and refreshes the observed values.
    refreshed = merge(COLS, (; y=Union{Missing,Float64}[1.0, missing, 0.5, missing, 1.9, 0.8]))
    frozen = reprocess(sb, refreshed; freeze_constants=true)
    @test frozen.data[:n_rows_y] == 6
    @test frozen.data[:y_obs] == [1.0, 0.5, 1.9, 0.8]
    # RKBRMI replays onto the longer frame as it builds on it.
    rk_replayed = reprocess(RKBRMI(normal_scale(COLS)), longer; freeze_constants=false)
    rk_direct = RKBRMI(normal_scale(longer))
    @test coordinate_names(rk_replayed.model.layout) == coordinate_names(rk_direct.model.layout)
    replayed_problem = rk_logdensity_problem(rk_replayed; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    direct_problem = rk_logdensity_problem(rk_direct; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in points(length(coordinate_names(rk_direct.model.layout)))
        @test LogDensityProblems.logdensity(replayed_problem, u) ≈
              LogDensityProblems.logdensity(direct_problem, u) atol=1e-12
    end
end
