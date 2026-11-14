# RKBRMI formula responses read an infinite entry of a per-row response bound
# column as a row without that bound: `-Inf` below, `Inf` above (also an
# interval's upper end), as SBBRMI does (`infinite_censoring_bounds.jl`).
# Snag rkbrmi-censored-995e238d: the formula route refused such a column
# ("column `lo` must be finite") while the value route accepted it.
#
# Each accepted case checks, on the formula route itself:
# - an independent Distributions.jl oracle for the value, and its central
#   differences for the native Enzyme Reverse gradient;
# - the same model on finite stand-ins (±1e300 for every infinity), whose
#   bound no uncensored row reads;
# - SBBRMI's compiled Stan density and gradient.
#
# Run: julia --startup-file=no --project=test test/rk_infinite_censoring_bounds.jl [filter...]
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const INF_BOUND_X = [-1.0, -0.5, 0.0, 0.4, 0.9, 1.3]
const INF_BOUND_Y = [0.1, -0.3, 0.2, 0.8, 0.5, 1.4]
# Rows 1 and 3 sit on their lower limit (left-censored), row 6 is detected above
# a finite limit, rows 2, 4 and 5 have no lower limit.
const INF_BOUND_LO = [0.1, -Inf, 0.2, -Inf, -Inf, 0.0]
# Row 4 sits on its upper limit (right-censored); rows 1, 2 and 5 have none.
const INF_BOUND_HI = [Inf, Inf, 0.9, 0.8, Inf, 2.0]
# Interval upper ends: rows 2 and 4 are the right-censored events (y, Inf).
const INF_BOUND_INTERVAL = [0.5, Inf, 0.9, Inf, 1.0, 2.0]

# (`upper` is a Stan reserved word, so SBBRMI's interval column is `ceiling`.)
inf_bound_data(; kwargs...) = merge((; x=INF_BOUND_X, y=INF_BOUND_Y,
    lo=INF_BOUND_LO, hi=INF_BOUND_HI, ceiling=INF_BOUND_INTERVAL), (; kwargs...))
# Every infinity replaced by a finite value beyond every observation.
finite_stand_in(v) = map(b -> isinf(b) ? copysign(1e300, b) : b, v)
inf_bound_finite_data() = inf_bound_data(; lo=finite_stand_in(INF_BOUND_LO),
    hi=finite_stand_in(INF_BOUND_HI), ceiling=finite_stand_in(INF_BOUND_INTERVAL))

# The rows' evidence categories, written out independently of BRM and RKPPL.
function censored_rows(d, lo, hi)
    sum(eachindex(INF_BOUND_Y)) do i
        y = INF_BOUND_Y[i]
        y <= lo[i] ? logcdf(d[i], lo[i]) :
            y >= hi[i] ? logccdf(d[i], hi[i]) : logpdf(d[i], y)
    end
end
truncated_rows(d, lo) = sum(eachindex(INF_BOUND_Y)) do i
    logpdf(d[i], INF_BOUND_Y[i]) - (isinf(lo[i]) ? 0.0 : logccdf(d[i], lo[i]))
end
interval_rows(d, upper) = sum(eachindex(INF_BOUND_Y)) do i
    isinf(upper[i]) ? logccdf(d[i], INF_BOUND_Y[i]) :
        log(cdf(d[i], upper[i]) - cdf(d[i], INF_BOUND_Y[i]))
end

const NO_HI = fill(Inf, length(INF_BOUND_Y))
const NO_LO = fill(-Inf, length(INF_BOUND_Y))

const INF_BOUND_LOWER = @brm begin
    mu ~ 1 + x
    y ~ censored(Normal(mu, 1); lower=lo)
end
const INF_BOUND_UPPER = @brm begin
    mu ~ 1 + x
    y ~ censored(Normal(mu, 1); upper=hi)
end
const INF_BOUND_BOTH = @brm begin
    mu ~ 1 + x
    y ~ censored(Normal(mu, 1); lower=lo, upper=hi)
end
const INF_BOUND_TRUNCATED = @brm begin
    mu ~ 1 + x
    y ~ truncated(Normal(mu, 1); lower=lo)
end
const INF_BOUND_INTERVAL_MODEL = @brm begin
    mu ~ 1 + x
    y ~ interval_censored(Normal(mu, 1); upper=ceiling)
end
# The reporter's second case: sampled scale and degrees of freedom.
const INF_BOUND_STUDENT = @brm begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    nu ~ Gamma(2, 10)
    y ~ censored(LocationScale(mu, sigma, TDist(nu)); lower=lo)
end

# The formula's population coefficients (intercept, then `x`) carry the default
# Normal(0, 1) prior.
const INF_BOUND_BETA = [Symbol("pop_mu.beta_pop.1"), Symbol("pop_mu.beta_pop.2")]
location_oracle(rows) = u -> begin
    a, b = u[1], u[2]
    rows(Normal.(a .+ b .* INF_BOUND_X, 1)) + logpdf(Normal(), a) + logpdf(Normal(), b)
end

const INF_BOUND_CASES = (
    ("lower", INF_BOUND_LOWER, location_oracle(d -> censored_rows(d, INF_BOUND_LO, NO_HI))),
    ("upper", INF_BOUND_UPPER, location_oracle(d -> censored_rows(d, NO_LO, INF_BOUND_HI))),
    ("both", INF_BOUND_BOTH,
        location_oracle(d -> censored_rows(d, INF_BOUND_LO, INF_BOUND_HI))),
    ("truncated", INF_BOUND_TRUNCATED, location_oracle(d -> truncated_rows(d, INF_BOUND_LO))),
    ("interval", INF_BOUND_INTERVAL_MODEL,
        location_oracle(d -> interval_rows(d, INF_BOUND_INTERVAL))),
    ("student", INF_BOUND_STUDENT, nothing))

function inf_bound_points(n)
    (zeros(n), fill(0.2, n), collect(range(-0.3, 0.4; length=n)))
end

# RK's `pop_mu.beta_pop.k` is Stan's `pop_mu_beta_pop.k`; scalars keep their name.
inf_bound_mapping(names) =
    [name => replace(string(name), "pop_mu.beta_pop" => "pop_mu_beta_pop") for name in names]

@stestset "infinite bound entries on the formula route" begin
    for (label, builder, oracle) in INF_BOUND_CASES
        @testset "$label" begin
            data = inf_bound_data()
            before = deepcopy(data)
            brmi = builder(data)
            backend, problem = consumer_problem(brmi)
            # The formula (structural) route, which used to refuse the column.
            @test backend.plan isa BRM._RKStructuralPlan
            names = coordinate_names(backend.model.layout)
            label == "student" ?
                @test(Set(names) == Set([INF_BOUND_BETA; :sigma; :nu])) :
                @test(names == INF_BOUND_BETA)
            finite = RKBRMI(builder(inf_bound_finite_data()))
            @test finite.plan isa BRM._RKStructuralPlan
            @test coordinate_names(finite.model.layout) == names
            finite_problem = rk_logdensity_problem(finite;
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            stan = consumer_stan(brmi, "infinite-bounds-" * label)
            mapping = inf_bound_mapping(names)
            for u in inf_bound_points(length(names))
                oracle === nothing || check_consumer_point(problem, u, oracle)
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                @test isfinite(value) && all(isfinite, gradient)
                v, g = LogDensityProblems.logdensity_and_gradient(finite_problem, u)
                @test value ≈ v atol=1e-12 rtol=1e-12
                @test gradient ≈ g atol=1e-12 rtol=1e-12
                check_consumer_stan(problem, stan, mapping, backend, u)
            end
            @test isequal(data, before)
        end
    end
end

@stestset "infinite bound entries that are not an absent bound stay refused" begin
    refused(builder, data) = try
        RKBRMI(builder(data)); ""
    catch err
        sprint(showerror, err)
    end
    # NaN, and the opposite side's infinity, are no absent bound.
    for (column, value) in ((:lo, NaN), (:lo, Inf), (:hi, NaN), (:hi, -Inf))
        values = copy(getfield(inf_bound_data(), column))
        values[5] = value
        builder = column === :lo ? INF_BOUND_LOWER : INF_BOUND_UPPER
        message = refused(builder, inf_bound_data(; (column => values,)...))
        side = column === :lo ? "lower" : "upper"
        open = column === :lo ? "-Inf" : "Inf"
        @test occursin("column `$column` is a response's $side bound: each entry " *
            "must be finite, or `$open` on a row without that bound", message)
    end
    # A bound column the model also reads as a covariate keeps the finite rule.
    covariate = @brm begin
        mu ~ 1 + x + lo
        y ~ censored(Normal(mu, 1); lower=lo)
    end
    message = refused(covariate, inf_bound_data())
    @test occursin("column `lo` must be finite: the model reads it as a " *
        "response's lower bound and elsewhere", message)
    # So does a column read as a lower bound and as an upper bound.
    both_sides = @brm begin
        mu ~ 1 + x
        y ~ censored(Normal(mu, 1); lower=lo)
        z ~ censored(Normal(mu, 1); upper=lo)
    end
    message = refused(both_sides, inf_bound_data(; z=fill(-2.0, 6)))
    @test occursin("column `lo` must be finite: the model reads it as both " *
        "a lower and an upper bound", message)
    # The finite control builds.
    @test refused(covariate, inf_bound_finite_data()) == ""
end
