# test/plate_matrix_rows.jl — whole rows of a matrix read in a `@plate` cell:
# `G[p, :]` in a `groups(g)` plate, `G[rows[i], :]` through a binding (snag
# `mixed-membership-1a1930a6`, design decision `0b9jgl5`).
#
# The cell reads its rows of a model-scope matrix (an assignment such as
# `reshape(g, N, n)`, the n curves of a grouped HSGP side by side) or of a data
# matrix, as the Julia loop does. Checked: the density and gradient equal the
# same model read column by column (`g1[p]`, `g2[p]`) on RKBRMI and SBBRMI,
# RKBRMI against Stan through the coordinate transport, a data matrix, the
# binding form, and the refusals.
#
# Run: julia --startup-file=no --project=test test/plate_matrix_rows.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

function rows_data(; S=3, T=4, n=2, seed=4)
    rng = Xoshiro(seed)
    t = repeat(collect(range(0.0, 1.0; length=T)), S)
    series = repeat(1:S, inner=T)
    N = length(t)
    (; y=round.(randn(rng, N); digits=3), t, series, n, N, labels=collect(1:S),
        tg=repeat(t, n), cg=repeat(1:n, inner=N), p1=collect(1:N), p2=collect(N+1:2N),
        X=round.(randn(rng, N, n); digits=3))
end
const D = rows_data()

# The n curves of a grouped HSGP side by side, read as whole rows per series.
by_rows(data) = @brm data begin
    sigma ~ Exponential(1.0)
    mu ~ 0 + (1 | series)
    g ~ 0 + hsgp(tg; by = cg, k = 3)
    G = reshape(g, N, n)
    @plate for p in groups(series)
        theta ~ Dirichlet(n, 1.0)
        y[p] ~ Normal(mu[p] .+ G[p, :] * theta, sigma)
    end
end
# The same model, one column at a time.
by_columns(data) = @brm data begin
    sigma ~ Exponential(1.0)
    mu ~ 0 + (1 | series)
    g ~ 0 + hsgp(tg; by = cg, k = 3)
    g1 = g[p1]
    g2 = g[p2]
    @plate for p in groups(series)
        theta ~ Dirichlet(n, 1.0)
        y[p] ~ Normal(mu[p] .+ theta[1] .* g1[p] .+ theta[2] .* g2[p], sigma)
    end
end
# Through a binding, in a plate over the series labels.
by_binding(data) = @brm data begin
    sigma ~ Exponential(1.0)
    mu ~ 0 + (1 | series)
    g ~ 0 + hsgp(tg; by = cg, k = 3)
    G = reshape(g, N, n)
    rows = groups(series => labels)
    @plate for i in eachindex(labels)
        theta ~ Dirichlet(n, 1.0)
        y[rows[i]] ~ Normal(mu[rows[i]] .+ G[rows[i], :] * theta, sigma)
    end
end
# Rows of a data matrix, against its columns read one at a time.
data_rows(data) = @brm data begin
    sigma ~ Exponential(1.0)
    b ~ Normal(0.0, 1.0)
    @plate for p in groups(series)
        w ~ Normal(0.0, 1.0)
        y[p] ~ Normal(b .+ X[p, :] * [w, 1.0], sigma)
    end
end
data_columns(data) = @brm merge(data, (; x1=data.X[:, 1], x2=data.X[:, 2])) begin
    sigma ~ Exponential(1.0)
    b ~ Normal(0.0, 1.0)
    @plate for p in groups(series)
        w ~ Normal(0.0, 1.0)
        y[p] ~ Normal(b .+ w .* x1[p] .+ x2[p], sigma)
    end
end

const AD = AutoEnzyme(; mode=Enzyme.Reverse)
const FOLDER = mkpath(joinpath(tempdir(), "brm-plate-matrix-rows"))

# The two spellings give the same coordinates, density and gradient on both
# backends, and RKBRMI agrees with Stan through the transport.
function check_same(a, b, label; points=3)
    ra, rb = RKBRMI(a), RKBRMI(b)
    names = coordinate_names(ra.model.layout)
    @test names == coordinate_names(rb.model.layout)
    pa = rk_logdensity_problem(check_rk_source_roundtrip(ra); ad_backend=AD)
    pb = rk_logdensity_problem(rb; ad_backend=AD)
    sa = SBBRMI(a; mod=@__MODULE__, total_groups=())
    sta = BRM.stan_instantiate(sa; path=joinpath(FOLDER, label * "_a.stan"))
    stb = BRM.stan_instantiate(SBBRMI(b; mod=@__MODULE__, total_groups=());
        path=joinpath(FOLDER, label * "_b.stan"))
    stan_names = BridgeStan.param_unc_names(sta.model)
    @test stan_names == BridgeStan.param_unc_names(stb.model)
    t = brm_coordinate_transport(ra, sa, String.(stan_names))
    rng = Xoshiro(17)
    for _ in 1:points
        u = 0.3 .* randn(rng, length(names))
        va, ga = LogDensityProblems.logdensity_and_gradient(pa, u)
        vb, gb = LogDensityProblems.logdensity_and_gradient(pb, u)
        @test va ≈ vb atol=1e-12 rtol=1e-13
        @test ga ≈ gb atol=1e-11 rtol=1e-12
        v = brm_rk_to_stan(t, u)
        sva, sga = BridgeStan.log_density_gradient(sta.model, v; propto=false, jacobian=true)
        svb, sgb = BridgeStan.log_density_gradient(stb.model, v; propto=false, jacobian=true)
        @test sva ≈ svb atol=1e-12 rtol=1e-13
        @test sga ≈ sgb atol=1e-11 rtol=1e-12
        @test va ≈ sva atol=1e-10 rtol=1e-11
        @test ga ≈ brm_stan_to_rk(t, sga) atol=1e-9 rtol=1e-10
    end
end

@stestset "G[p, :] reads a series' rows of the curves, as their columns do" begin
    check_same(by_rows(D), by_columns(D), "groups")
end

@stestset "G[rows[i], :] through a binding" begin
    check_same(by_binding(D), by_columns(D), "binding")
end

@stestset "rows of a data matrix" begin
    check_same(data_rows(D), data_columns(D), "data")
end

@stestset "the description reads the rows and the reshape as written" begin
    r = brm_description(SBBRMI(by_rows(D); total_groups=()))
    @test r.complete
    @test "\\mathrm{G}=\\operatorname{reshape}_{\\mathrm{N}\\times \\mathrm{n},\\mathrm{col}}\\left(\\mathrm{g}\\right)" in r.equations
    @test any(e -> occursin("{\\mathrm{G}}_{J_{p},\\cdot} \\cdot \\mathrm{theta}", e), r.equations)
end

@stestset "refusals name the spelling" begin
    message(f) = try f(); "" catch e; sprint(showerror, e) end
    @test occursin("through a binding", message(() -> @eval @brm begin
        @plate for i in eachindex(y)
            y[i] ~ Normal(sum(G[i, :]), 1.0)
        end
    end))
    @test occursin("reads rows of the plate output", message(() -> @eval @brm begin
        rows = groups(series => labels)
        @plate for i in eachindex(labels)
            out[i] = 1.0
            y[rows[i]] ~ Normal(sum(out[rows[i], :]), 1.0)
        end
    end))
end
