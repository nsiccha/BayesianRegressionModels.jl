#!/usr/bin/env julia

# Acceptance harness for a reviewed BRM family-surface candidate. Run with the
# candidate checkout as the active project and pass its exact reviewed SHA.

using BayesianRegressionModels
using Distributions: LocationScale, TDist
using LogDensityProblems
using StanBlocks

length(ARGS) >= 1 || error("usage: candidate_family_controls.jl <expected-brm-sha> [output.tsv]")
expected_sha = ARGS[1]
output = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "candidate_family_controls.tsv")
project_root = dirname(Base.active_project())
brm_sha = readchomp(`git -C $project_root rev-parse HEAD`)
brm_sha == expected_sha || error("active BRM SHA $brm_sha != reviewed $expected_sha")
stanblocks_root = dirname(dirname(pathof(StanBlocks)))
stanblocks_sha = readchomp(`git -C $stanblocks_root rev-parse HEAD`)

data = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    y_zip=[0, 1, 0, 2, 3, 0],
    y_nb=[0, 1, 2, 4, 3, 6],
    y_t=[-0.8, -0.2, 0.1, 0.7, 1.0, 1.4],
)

builder = @brm begin
    log(lambda) ~ 1 + x
    y_zip ~ ZeroInflatedPoisson(lambda, 0.25)

    log(mu) ~ 1 + x
    log(phi) ~ 1
    y_nb ~ NegativeBinomial2(mu, phi)

    loc ~ 1 + x
    log(scale) ~ 1
    y_t ~ LocationScale(loc, scale, TDist(4.0))
end

descriptor = brm_descriptor(builder, data; mod=@__MODULE__, name=:family_surfaces)
code = brm_execute(descriptor, :transpile)
stanc = StanBlocks.stanc_check(code; warn_pedantic=false)
stanc.ok || error("stanc failed: $(stanc.output)")
operations = Symbol[operation.name for operation in descriptor.operations]
all(operation -> operation in operations, (:fit, :predict, :pointwise_loglik)) ||
    error("missing descriptor operations; offered=$operations")

problem = brm_execute(descriptor, :fit)
dimension = LogDensityProblems.dimension(problem)
q = zeros(dimension)
log_density, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
finite = isfinite(log_density) && all(isfinite, gradient)
finite || error("non-finite density/gradient at zero")

prediction = brm_execute(descriptor, :predict; problem, draws=q, seed=20260728)
pointwise = brm_execute(descriptor, :pointwise_loglik; problem, draws=q, seed=20260728)
expected_prediction = Set([:y_zip_gen, :y_nb_gen, :y_t_gen])
expected_pointwise = Set([:y_zip_likelihood, :y_nb_likelihood, :y_t_likelihood])
Set(keys(prediction)) == expected_prediction ||
    error("unexpected prediction outputs: $(keys(prediction))")
Set(keys(pointwise)) == expected_pointwise ||
    error("unexpected pointwise outputs: $(keys(pointwise))")
all(value -> length(value) == length(data.x), values(prediction)) ||
    error("prediction output length mismatch")
all(value -> length(value) == length(data.x), values(pointwise)) ||
    error("pointwise output length mismatch")

columns = [
    "candidate_sha", "stanblocks_sha", "julia_version", "stanc",
    "dimension", "log_density_zero", "gradient_finite",
    "descriptor_operations", "prediction_outputs", "pointwise_outputs",
    "rows_per_output",
]
values_row = [
    brm_sha, stanblocks_sha, string(VERSION), "pass", string(dimension),
    string(log_density), string(finite), join(string.(operations), ','),
    join(sort!(string.(collect(keys(prediction)))), ','),
    join(sort!(string.(collect(keys(pointwise)))), ','),
    string(length(data.x)),
]
open(output, "w") do io
    println(io, join(columns, '\t'))
    println(io, join(values_row, '\t'))
end

println("candidate_sha=$brm_sha")
println("stanblocks_sha=$stanblocks_sha")
println("stanc=pass")
println("dimension=$dimension")
println("log_density_zero=$log_density")
println("gradient_finite=$finite")
println("prediction_outputs=$(join(sort!(string.(collect(keys(prediction)))), ','))")
println("pointwise_outputs=$(join(sort!(string.(collect(keys(pointwise)))), ','))")
println("output=$(abspath(output))")
