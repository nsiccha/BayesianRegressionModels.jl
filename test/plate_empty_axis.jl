# test/plate_empty_axis.jl — a secondary axis with no rows (todo `0jp0opk`).
#
# Run: julia --startup-file=no --project=test test/plate_empty_axis.jl
#
# A design can leave a whole secondary axis empty: a placebo-only arm with no
# placebo rows, a schedule with no observations yet. `ragged(x, group)` over a
# zero-row `group` then gives every cell an empty slice, for a formula LP, a
# raw data column and an observed response alike. An empty slice contributes
# nothing, so each model must equal a twin that never reads the empty axis —
# density and gradient, on both backends — and the same program must replay
# onto data that has rows (and back) with `reprocess`, including the row
# count of a declared axis that sizes a predictor with no data column.
#
# A plate can also have no cells at all, when the column naming its cells has no
# rows. Its density is then exactly its priors, and it keeps the same program.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Unsorted subjects; the placebo axis has no rows.
const EMPTY_PLACEBO = (;
    subject=["b", "a"],
    weight=[0.6, 0.8],
    p_time=Float64[],
    p_subject=String[],
    obs_subject=["a", "b", "a"],
    y=[0.1, 0.2, 0.3],
)
# The same subjects with placebo rows: b owns row 2, a rows 1 and 3.
const PLACEBO_ROWS = merge(EMPTY_PLACEBO, (;
    p_time=[0.5, 1.0, 2.0],
    p_subject=["a", "b", "a"],
))

lp_gather(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    placebo ~ 0 + p_time
    placebo_rows = groups(p_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) + sum(placebo[placebo_rows[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

# The placebo predictor and its coefficient exist, but no cell reads them.
lp_twin(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    placebo ~ 0 + p_time
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

# A predictor with no data column takes its length from the declared axis.
intercept_gather(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    p_offset ~ 1
    placebo_rows = groups(p_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) + sum(p_offset[placebo_rows[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

column_gather(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    placebo_rows = groups(p_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) + sum(p_time[placebo_rows[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

column_twin(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

# A response observed on the empty placebo axis, beside `y` on a populated one.
const PLACEBO_RESPONSE = merge(EMPTY_PLACEBO, (; py=Float64[]))

observed_in_cell(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    placebo_rows = groups(p_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i])
        y[obs[i]] ~ normal(loc[i], sigma)
        py[placebo_rows[i]] ~ normal(loc[i] .* p_time[placebo_rows[i]], sigma)
    end
end

# The same response observed against a collected plate output.
observed_output(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    placebo_rows = groups(p_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        ploc[i] = exp(log_CL[i]) .* p_time[placebo_rows[i]]
        y[obs[i]] ~ normal(exp(log_CL[i]), sigma)
        py[placebo_rows[i]] ~ Normal(ploc[i], sigma)
    end
end

const STAN_DIR = mkpath(joinpath(tempdir(), "brm-plate-empty-axis"))

function stan_problem(brmi, name)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    sb, BRM.stan_instantiate(sb; path=joinpath(STAN_DIR, name * ".stan"))
end

points(n) = (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))

# The twin never reads the empty axis, so a backend may drop the placebo
# predictor from it, and with it the predictor's coefficient. `extra` names
# those coordinates: they must be the only ones the twin lacks, and they add
# exactly their own standard-normal prior to the density and gradient.
function check_twin(names, twin_names, density, twin_density, extra)
    @test sort(setdiff(names, twin_names)) == sort(extra)
    @test setdiff(twin_names, names) == []
    shared = [findfirst(==(n), names) for n in twin_names]
    added = [findfirst(==(n), names) for n in extra]
    for u in points(length(names))
        value, gradient = density(u)
        twin_value, twin_gradient = twin_density(u[shared])
        @test isfinite(value)
        @test value ≈ twin_value + sum(logpdf.(Normal(), u[added]); init=0.0) atol=1e-12
        @test gradient[shared] ≈ twin_gradient atol=1e-10
        @test gradient[added] ≈ -u[added] atol=1e-10
    end
end

function check_stan_twin(stan, twin; extra=String[])
    check_twin(BridgeStan.param_unc_names(stan.model),
        BridgeStan.param_unc_names(twin.model),
        u -> BridgeStan.log_density_gradient(stan.model, u; propto=false),
        u -> BridgeStan.log_density_gradient(twin.model, u; propto=false), extra)
end

function check_rk_twin(brmi, twin_brmi; extra=Symbol[])
    backend, problem = consumer_problem(brmi)
    twin_backend, twin_problem = consumer_problem(twin_brmi)
    check_twin(coordinate_names(backend.model.layout),
        coordinate_names(twin_backend.model.layout),
        u -> LogDensityProblems.logdensity_and_gradient(problem, u),
        u -> LogDensityProblems.logdensity_and_gradient(twin_problem, u), extra)
end

@stestset "an empty axis gathers an empty slice per cell on Stan" begin
    saved = deepcopy(EMPTY_PLACEBO)
    sb, stan = stan_problem(lp_gather(EMPTY_PLACEBO), "lp-gather")
    @test sb.data[:kernel_loc_placebo_ragged] == [Int[], Int[]]
    _, twin = stan_problem(lp_twin(EMPTY_PLACEBO), "lp-twin")
    check_stan_twin(stan, twin; extra=["pop_placebo_beta_pop.1"])

    sb, stan = stan_problem(column_gather(EMPTY_PLACEBO), "column-gather")
    @test sb.data[:kernel_loc_p_time_ragged] == [Float64[], Float64[]]
    _, twin = stan_problem(column_twin(EMPTY_PLACEBO), "column-twin")
    check_stan_twin(stan, twin)

    sb, stan = stan_problem(intercept_gather(EMPTY_PLACEBO), "intercept-gather")
    @test sb.data[:n_rows_p_offset_p_subject] == 0
    check_stan_twin(stan, twin; extra=["pop_p_offset_beta_pop.1"])
    @test isequal(EMPTY_PLACEBO, saved)
end

@stestset "an empty axis gathers an empty slice per cell on RK" begin
    check_rk_twin(lp_gather(EMPTY_PLACEBO), lp_twin(EMPTY_PLACEBO);
        extra=[Symbol("pop_placebo.beta_pop.1")])
    check_rk_twin(column_gather(EMPTY_PLACEBO), column_twin(EMPTY_PLACEBO))
    check_rk_twin(intercept_gather(EMPTY_PLACEBO), column_twin(EMPTY_PLACEBO);
        extra=[:p_offset_Intercept])
end

@stestset "an empty observed axis observes nothing" begin
    # The twin observes only `y`.
    _, twin = stan_problem(column_twin(PLACEBO_RESPONSE), "observed-twin")
    for (model, name) in ((observed_in_cell, "observed-in-cell"),
                          (observed_output, "observed-output"))
        _, stan = stan_problem(model(PLACEBO_RESPONSE), name)
        check_stan_twin(stan, twin)
        check_rk_twin(model(PLACEBO_RESPONSE), column_twin(PLACEBO_RESPONSE))
    end
end

@stestset "reprocess moves a program between an empty and a populated axis" begin
    for (model, key, populated) in (
            (lp_gather, :kernel_loc_placebo_ragged, [[2], [1, 3]]),
            (column_gather, :kernel_loc_p_time_ragged, [[1.0], [0.5, 2.0]]),
            (intercept_gather, :kernel_loc_p_offset_ragged, [[2], [1, 3]]))
        empty = SBBRMI(model(EMPTY_PLACEBO); mod=@__MODULE__, total_groups=())
        direct = SBBRMI(model(PLACEBO_ROWS); mod=@__MODULE__, total_groups=())
        @test direct.data[key] == populated
        # The program does not depend on whether the axis has rows.
        @test BRM.stan_code(direct) == BRM.stan_code(empty)

        filled = reprocess(empty, PLACEBO_ROWS; freeze_constants=true)
        @test BRM.stan_code(filled) == BRM.stan_code(empty)
        @test filled.data == direct.data

        emptied = reprocess(direct, EMPTY_PLACEBO; freeze_constants=true)
        @test emptied.data[key] == empty.data[key]
        @test emptied.data == empty.data
    end
    # The replayed populated program evaluates like the one built on rows. The
    # intercept-only predictor is sized by the declared axis's row count, which
    # the replay recomputes (`:axis_extent`) rather than keeping the fit's 0.
    for (model, name) in ((lp_gather, "lp"), (intercept_gather, "intercept"))
        filled = reprocess(SBBRMI(model(EMPTY_PLACEBO); mod=@__MODULE__,
            total_groups=()), PLACEBO_ROWS; freeze_constants=true)
        replayed = BRM.stan_instantiate(filled;
            path=joinpath(STAN_DIR, name * "-replayed.stan"))
        _, direct = stan_problem(model(PLACEBO_ROWS), name * "-direct")
        for u in points(BridgeStan.param_unc_num(direct.model))
            @test BridgeStan.log_density(replayed.model, u; propto=false) ≈
                  BridgeStan.log_density(direct.model, u; propto=false) atol=1e-12
        end
    end
end

# A plate with no cells at all: the column that names its cells has no rows, as
# a schedule with no observations yet. The program must not depend on that, and
# its density must be exactly its priors.
const NO_CELLS = (;
    block=String[], cell=String[], weight=Float64[], day=Int[], y=Float64[], x=Float64[],
    level_sd=2.0)
const TWO_SERIES = merge(NO_CELLS, (;
    block=["b", "a", "a"], cell=["a", "b"], weight=[0.5, -0.5], day=[4, 3, 1],
    y=[0.2, 0.3, 0.1], x=[1.0, -0.5, 0.5]))

series_model(data) = @brm data begin
    level ~ Normal(0.0, level_sd)
    slope ~ Normal(0.1, 0.5)
    log_sigma ~ Normal(0.0, 1.0)
    sigma = exp(log_sigma)
    phi ~ Uniform(-0.8, 0.9)
    @plate for p in groups(block; order = day)
        m = level .+ slope .* x[p]
        y[p] ~ AR1Normal(m, sigma, phi; time = day[p])
    end
end

# The same law with its data-supplied prior scale written as a literal.
literal_series_model(data) = @brm data begin
    level ~ Normal(0.0, 2.0)
    slope ~ Normal(0.1, 0.5)
    log_sigma ~ Normal(0.0, 1.0)
    sigma = exp(log_sigma)
    phi ~ Uniform(-0.8, 0.9)
    @plate for p in groups(block; order = day)
        m = level .+ slope .* x[p]
        y[p] ~ AR1Normal(m, sigma, phi; time = day[p])
    end
end

# Cells named by their own label column, rather than by the grouping's labels.
labelled_cells_model(data) = @brm data begin
    level ~ Normal(0.0, level_sd)
    rows = groups(block => cell)
    @plate for i in eachindex(weight)
        y[rows[i]] ~ normal(level .+ weight[i] .+ x[rows[i]], 1.0)
    end
end

# A random effect over the empty label column has no levels, and its scale and
# the population intercept stay. With an explicit `total_groups` its program is
# the populated twin's; `total_groups=:auto` picks a parameterization from the data.
ranef_cells_model(data) = @brm data begin
    log_rate ~ 1 + (1 | p | cell)
    sigma ~ Exponential(1)
    rows = groups(block => cell)
    @plate for i in eachindex(log_rate)
        y[rows[i]] ~ normal(exp(log_rate[i]) .* x[rows[i]], sigma)
    end
end

no_cells_sb(model, data) = SBBRMI(model(data); mod=@__MODULE__, total_groups=:auto)

# The independent density and gradient of `series_model`'s priors, in the
# coordinate order `names` gives. Stan's transform of `phi` is checked through
# `uniform_value` and its log Jacobian.
function series_prior(names, q)
    at(name) = coordinate(string.(names), name)
    level, slope, log_sigma, u = q[at("level")], q[at("slope")], q[at("log_sigma")], q[at("phi")]
    value = logpdf(Normal(0.0, 2.0), level) + logpdf(Normal(0.1, 0.5), slope) +
        logpdf(Normal(0.0, 1.0), log_sigma) +
        logpdf(Uniform(-0.8, 0.9), uniform_value(u, -0.8, 0.9)) + uniform_logjac(u, -0.8, 0.9)
    gradient = zero(q)
    gradient[at("level")] = -level / 4
    gradient[at("slope")] = -(slope - 0.1) / 0.25
    gradient[at("log_sigma")] = -log_sigma
    gradient[at("phi")] = 1 - 2 * LogExpFunctions.logistic(u)
    value, gradient
end

function check_series_prior(names, density)
    @test sort(string.(names)) == ["level", "log_sigma", "phi", "slope"]
    for q in points(4)
        value, gradient = density(q)
        expected, expected_gradient = series_prior(names, q)
        @test value ≈ expected atol=1e-12
        @test gradient ≈ expected_gradient atol=1e-10
    end
end
check_series_prior(stan) = check_series_prior(BridgeStan.param_unc_names(stan.model),
    q -> BridgeStan.log_density_gradient(stan.model, q; propto=false))

@stestset "a plate with no cells is its priors on Stan" begin
    saved = deepcopy(NO_CELLS)
    sb = no_cells_sb(series_model, NO_CELLS)
    # Typed as its populated twin's inputs, so the program cannot differ.
    @test sb.data[:kernel_y_cell_x_ragged] isa Vector{Vector{Float64}}
    @test sb.data[:kernel_y_cell_day_ragged] isa Vector{Vector{Int}}
    @test isempty(sb.data[:kernel_y_cell_x_ragged])
    @test sb.data[:kernel_nsub_y_cell] == 0
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    check_series_prior(BRM.stan_instantiate(sb; path=joinpath(STAN_DIR, "no-cells.stan")))

    # The literal twin: the same law, whichever way its prior scale is written.
    literal = no_cells_sb(literal_series_model, NO_CELLS)
    check_series_prior(BRM.stan_instantiate(literal;
        path=joinpath(STAN_DIR, "no-cells-literal.stan")))
    @test isequal(NO_CELLS, saved)
end

@stestset "a plate's program does not depend on having cells" begin
    for model in (series_model, literal_series_model)
        empty = no_cells_sb(model, NO_CELLS)
        direct = no_cells_sb(model, TWO_SERIES)
        @test BRM.stan_code(empty) == BRM.stan_code(direct)

        emptied = reprocess(direct, NO_CELLS)
        @test BRM.stan_code(emptied) == BRM.stan_code(direct)
        @test stan_data(emptied) == stan_data(empty)

        filled = reprocess(empty, TWO_SERIES)
        @test BRM.stan_code(filled) == BRM.stan_code(direct)
        @test stan_data(filled) == stan_data(direct)
    end
    # The replayed programs evaluate like the ones built on their own data.
    direct = no_cells_sb(series_model, TWO_SERIES)
    check_series_prior(BRM.stan_instantiate(reprocess(direct, NO_CELLS);
        path=joinpath(STAN_DIR, "no-cells-emptied.stan")))
    filled = BRM.stan_instantiate(reprocess(no_cells_sb(series_model, NO_CELLS), TWO_SERIES);
        path=joinpath(STAN_DIR, "no-cells-filled.stan"))
    built = BRM.stan_instantiate(direct; path=joinpath(STAN_DIR, "two-series.stan"))
    for q in points(4)
        value = BridgeStan.log_density(built.model, q; propto=false)
        @test isfinite(value)
        @test BridgeStan.log_density(filled.model, q; propto=false) ≈ value atol=1e-12
    end
end

@stestset "cells named by an empty label column are no cells on Stan" begin
    sb = no_cells_sb(labelled_cells_model, NO_CELLS)
    @test BRM.stan_code(sb) == BRM.stan_code(no_cells_sb(labelled_cells_model, TWO_SERIES))
    stan = BRM.stan_instantiate(sb; path=joinpath(STAN_DIR, "no-labelled-cells.stan"))
    @test BridgeStan.param_unc_names(stan.model) == ["level"]
    for q in ([0.0], [0.7], [-1.3])
        value, gradient = BridgeStan.log_density_gradient(stan.model, q; propto=false)
        @test value ≈ logpdf(Normal(0.0, 2.0), only(q)) atol=1e-12
        @test gradient ≈ [-only(q) / 4] atol=1e-10
    end
end

@stestset "a plate with no cells is its priors on RK" begin
    for model in (series_model, literal_series_model)
        backend, problem = consumer_problem(model(NO_CELLS))
        check_series_prior(coordinate_names(backend.model.layout),
            q -> LogDensityProblems.logdensity_and_gradient(problem, q))
    end
    backend, problem = consumer_problem(labelled_cells_model(NO_CELLS))
    @test coordinate_names(backend.model.layout) == [:level]
    for q in ([0.0], [0.7], [-1.3])
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test value ≈ logpdf(Normal(0.0, 2.0), only(q)) atol=1e-12
        @test gradient ≈ [-only(q) / 4] atol=1e-10
    end
end

@stestset "a random effect with no levels keeps its priors on both backends" begin
    sb = SBBRMI(ranef_cells_model(NO_CELLS); mod=@__MODULE__, total_groups=())
    direct = SBBRMI(ranef_cells_model(TWO_SERIES); mod=@__MODULE__, total_groups=())
    @test BRM.stan_code(sb) == BRM.stan_code(direct)
    # A replay keeps the fitted levels' coordinates, unobserved.
    @test BRM.stan_code(reprocess(direct, NO_CELLS)) == BRM.stan_code(direct)
    stan = BRM.stan_instantiate(sb; path=joinpath(STAN_DIR, "no-levels.stan"))
    stan_names = BridgeStan.param_unc_names(stan.model)
    @test sort(stan_names) == ["b_p_cell_tau.1", "pop_log_rate_beta_pop.1", "sigma"]

    # The same three priors on RK, under its own coordinate names.
    backend, problem = consumer_problem(ranef_cells_model(NO_CELLS))
    rk_names = coordinate_names(backend.model.layout)
    @test sort(string.(rk_names)) == ["b_p_cell.tau.1", "log_rate_Intercept", "sigma"]
    to_stan = Dict("b_p_cell.tau.1" => "b_p_cell_tau.1",
        "log_rate_Intercept" => "pop_log_rate_beta_pop.1", "sigma" => "sigma")
    order = [coordinate(stan_names, to_stan[string(n)]) for n in rk_names]
    for q in points(3)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        stan_q = zero(q)
        stan_q[order] = q
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, stan_q;
            propto=false)
        @test isfinite(value)
        @test value ≈ stan_value atol=1e-10
        @test gradient ≈ stan_gradient[order] atol=1e-8
    end
end

@stestset "missing grouping labels are still refused" begin
    # refused: a missing label names no subject, so its row would be dropped
    # from every cell without notice. The generic data-column check refuses it
    # before the gather's own missing-label check is reached.
    unlabelled = merge(PLACEBO_ROWS, (; p_subject=["a", missing, "a"]))
    @test_throws "data column `p_subject` contains `missing` values" SBBRMI(
        lp_gather(unlabelled); mod=@__MODULE__, total_groups=())
    @test_throws "data column `p_subject` contains `missing` values" RKBRMI(
        lp_gather(unlabelled))
end
