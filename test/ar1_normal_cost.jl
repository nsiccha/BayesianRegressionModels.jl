# test/ar1_normal_cost.jl — one RK log-density value of a per-group recurrence
# costs memory in proportion to the rows.
#
# Run: julia --startup-file=no --project=test test/ar1_normal_cost.jl [filter...]
#
# `y[p] ~ AR1Normal(mu[p], sigma, phi; time = t[p])` in a
# `@plate for p in groups(...)` cell, and the `@scan` it states written in the
# cell, read a group's rows through gathers (`mu[p]`, `t[p]`). The recurrence
# reads each row once, so doubling every series doubles the bytes one value
# allocates. While each scan step repeated the whole-series gathers, the bytes
# quadrupled: 47.5, 186, 738 and 2,934 MB at 500, 1,000, 2,000 and 4,000 rows
# per series (snag `ar1normal-plate-a2960527`; ReactiveKernels snag
# `scan-step-re-eva-8952ed66`).
#
# The check is on allocated bytes, which do not depend on the host's load. The
# value is taken through the sampler query without a derivative backend.
using Test, BayesianRegressionModels, Distributions, Random
import ReactiveKernelsPPL
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

# `series` groups of `n` consecutive hourly rows each.
panel(n; series=2, rng=Xoshiro(20261128)) = (;
    serie=repeat(["s$k" for k in 1:series]; inner=n),
    t=repeat(1:n, series),
    x=randn(rng, n * series),
    y=randn(rng, n * series),
)

shared_ar(data) = @brm data begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    mu ~ 1 + x
    @plate for p in groups(serie; order = t)
        y[p] ~ AR1Normal(mu[p], sigma, phi; time = t[p])
    end
end

# Every argument read per row.
varying_ar(data) = @brm data begin
    atanh(phi) ~ 1 + (1 | serie)
    log(sigma) ~ 1 + (1 | serie)
    mu ~ 1 + x
    @plate for p in groups(serie; order = t)
        y[p] ~ AR1Normal(mu[p], sigma[p], phi[p]; time = t[p])
    end
end

# Without `time`: the plain recursion.
stepped_ar(data) = @brm data begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    mu ~ 1 + x
    @plate for p in groups(serie; order = t)
        y[p] ~ AR1Normal(mu[p], sigma, phi)
    end
end

# The recurrence written in the cell, as brm-use shows it for other
# innovation laws.
written_ar(data) = @brm data begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    mu ~ 1 + x
    @plate for p in groups(serie; order = t)
        @scan begin
            y[p][1] ~ Normal(mu[p][1], sigma / sqrt(1 - phi ^ 2))
            for s in 2:length(t[p])
                k = t[p][s] - t[p][s - 1]
                y[p][s] ~ Normal(mu[p][s] + phi ^ k * (y[p][s - 1] - mu[p][s - 1]),
                    sigma * sqrt((1 - phi ^ (2 * k)) / (1 - phi ^ 2)))
            end
        end
    end
end

# The bytes one warm log-density value allocates at `n` rows per series.
function value_bytes(model, n)
    backend = RKBRMI(model(panel(n)))
    query = ReactiveKernelsPPL.prepare_query(backend.model, rk_bound_plan(backend), :sampler)
    u = fill(0.1, backend.model.layout.total)
    @test isfinite(Base.invokelatest(query, u))
    Base.invokelatest(query, u)
    @allocated Base.invokelatest(query, u)
end

const COST_SIZES = (250, 500, 1000)
# Linear growth doubles the bytes with the rows; the repeated gathers gave 3.9
# to 4.0. A fixed part of the allocation only lowers the ratio.
const COST_GROWTH_BOUND = 2.6

function check_linear_bytes(label, model)
    bytes = [value_bytes(model, n) for n in COST_SIZES]
    growth = [b / a for (a, b) in zip(bytes, bytes[2:end])]
    println("AR1_COST ", (; model=label, rows_per_series=COST_SIZES, bytes,
        growth=round.(growth; digits=2)))
    flush(stdout)
    for g in growth
        @test g <= COST_GROWTH_BOUND
    end
end

@testset "RK value of a per-group recurrence allocates linearly in the rows" begin
    @stestset "AR1Normal, shared scale and persistence" begin
        check_linear_bytes("shared", shared_ar)
    end
    @stestset "AR1Normal, every argument per row" begin
        check_linear_bytes("varying", varying_ar)
    end
    @stestset "AR1Normal without time" begin
        check_linear_bytes("stepped", stepped_ar)
    end
    @stestset "the recurrence written in the cell" begin
        check_linear_bytes("written", written_ar)
    end
end
