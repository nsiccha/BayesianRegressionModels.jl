# A seeded fit's exact trace, draws and factors, to compare two trees byte for byte.
#
#     julia --project=test research/svi_field_units/identity.jl
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, Random, SHA, Serialization
include(joinpath(@__DIR__, "..", "..", "test", "nlme_svi_owned_fixture.jl"))

digest(x) = bytes2hex(sha256(reinterpret(UInt8, collect(Float64, x))))
function fit_digest(label, model; kwargs...)
    fit = brm_svi_fit(model; rng=Xoshiro(21), batch_size=3, max_iterations=120, ndraws=8, threads=1, kwargs...)
    d = fit.diagnostics
    println("IDENTITY ", label, " elbo=", digest(d.elbo), " draws=", digest(vec(fit.draws)),
        " mu=", digest(d.q.μ), " rho=", digest(d.q.ρ),
        " M=", digest(reduce(vcat, d.q.M; init=Float64[])),
        " gradient_calls=", d.unit_gradient_calls)
end
data = svi_owned_data(collect(1:8); shared=true)
ad = AutoEnzyme(; mode=Enzyme.Reverse)
fit_digest("strict", RKBRMI(svi_owned_model(data)); ad_backend=ad)
fit_digest("strict_lowrank", RKBRMI(svi_owned_model(data)); ad_backend=ad, family=LowRankGlobal(2))
fit_digest("connected_uniform", RKBRMI(svi_owned_model(data)); ad_backend=ad, unit_partition=:connected, unit_sampling=:uniform, separator=:none)
fit_digest("connected_work", RKBRMI(svi_owned_model(data)); ad_backend=ad, unit_partition=:connected, unit_sampling=:work, separator=:none)
