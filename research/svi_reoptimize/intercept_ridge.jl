# The global–unit ridge of todo 122p9l3 on an exactly Gaussian toy: an intercept
# beside per-unit effects `u[g] ~ Normal(0, 0.5)` (known SD) that strong per-unit
# data pin only in `a + u[g]`, plus population slopes; noise sd 0.1 known. The
# posterior is Gaussian, so its means are exact: the fit's global means are
# reported in exact posterior sds, for `brm_svi_fit`'s stock step and for
# `unit_reoptimize = k`.
#
#     julia --project=test research/svi_reoptimize/intercept_ridge.jl [G] [iterations] [k...]
include(joinpath(@__DIR__, "..", "..", "test", "fitting_packages.jl"))
using BayesianRegressionModels, Random, Statistics, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
const BRM = BayesianRegressionModels

function ridge_data(; G=200, n=20, P=4, seed=1)
    rng = Xoshiro(seed)
    u = 0.5 .* randn(rng, G)
    X = randn(rng, G * n, P)
    beta = randn(rng, P)
    g = repeat(1:G; inner=n)
    y = 1.0 .+ u[g] .+ X * beta .+ 0.1 .* randn(rng, G * n)
    (; g, y, x1=X[:, 1], x2=X[:, 2], x3=X[:, 3], x4=X[:, 4], known_sd=0.5, noise=0.1,
       prior_sd=10.0)
end

ridge_model(data) = @brm data begin
    mu ~ 1 + x1 + x2 + x3 + x4 + (1 | g)
    effect(mu, :) ~ Normal(0, prior_sd)
    sd(mu, g) = known_sd
    y ~ Normal(mu, noise)
end

# The exact posterior of (a, beta) of the Gaussian linear model.
function exact_globals(data)
    G = maximum(data.g)
    Z = hcat(ones(length(data.y)), data.x1, data.x2, data.x3, data.x4,
             [data.g[r] == j ? 1.0 : 0.0 for r in eachindex(data.y), j in 1:G])
    prior = vcat(fill(1 / data.prior_sd^2, 5), fill(1 / data.known_sd^2, G))
    H = Symmetric(Z' * Z ./ data.noise^2 + Diagonal(prior))
    m = H \ (Z' * data.y ./ data.noise^2)
    S = inv(H)
    m[1:5], sqrt.(diag(S)[1:5])
end

function main(G, iterations, ks)
    data = ridge_data(; G)
    rk = RKBRMI(ridge_model(data))
    names = String.(BRM._rk_layout_coordinate_names(rk))
    globals = [findfirst(==("pop_mu.beta_pop.$j"), names) for j in 1:5]
    any(isnothing, globals) && error("population names: ", names[1:min(end, 10)])
    m, s = exact_globals(data)
    println("G=$G iterations=$iterations; exact means ", round.(m; digits=4),
            " sds ", round.(s; sigdigits=3))
    for k in ks
        t = @elapsed fit = brm_svi_fit(rk; subject_group=:g, rng=Xoshiro(11),
            max_iterations=iterations, ndraws=2000, unit_reoptimize=k)
        mu = vec(mean(fit.draws[globals, :]; dims=2))
        err = abs.(mu .- m) ./ s
        sdr = vec(std(fit.draws[globals, :]; dims=2)) ./ s
        println(rpad("unit_reoptimize=$k", 20), "errors (exact sds) ", round.(err; digits=2),
                "  sd ratio ", round.(sdr; digits=2),
                "  iterations ", fit.diagnostics.iterations,
                "  unit calls ", fit.diagnostics.unit_gradient_calls,
                "  ", round(t; digits=1), " s")
        flush(stdout)
    end
end

main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200,
     length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20_000,
     length(ARGS) >= 3 ? parse.(Int, ARGS[3:end]) : [0, 1])
