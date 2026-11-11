# Run with the canonical test environment and a compute token:
# julia --project=test research/distribution_graphs/hoisting.jl OUTPUT_DIRECTORY [--varying-bound]
# Each row times the likelihood query's value and the whole log density's
# Enzyme Reverse value and gradient, for the native (owned graph) and the
# opaque (caller-defined Distributions call) route.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Distributions, Statistics, TOML, Test
using ADTypes, Enzyme, LogDensityProblems

opaque_lower_mean(mu, sigma, lo) = mean(truncated(LogNormal(mu, sigma); lower=lo))
opaque_lower_var(mu, sigma, lo) = var(truncated(LogNormal(mu, sigma); lower=lo))
opaque_upper_mean(mu, sigma, hi) = mean(truncated(LogNormal(mu, sigma); upper=hi))
opaque_upper_var(mu, sigma, hi) = var(truncated(LogNormal(mu, sigma); upper=hi))

function moment_model(data, native, upper)
    if native && upper
        @brm data begin
            mu ~ Normal(0, 1)
            log_sigma ~ Normal(0, 1)
            log_bound ~ Normal(0, 1)
            @plate for i in eachindex(y)
                m[i] = mean(truncated(LogNormal(mu, exp(log_sigma)); upper=exp(log_bound)))
                v[i] = var(truncated(LogNormal(mu, exp(log_sigma)); upper=exp(log_bound)))
                y[i] ~ Normal(m[i], sqrt(v[i] + 0.01))
            end
        end
    elseif native
        @brm data begin
            mu ~ Normal(0, 1)
            log_sigma ~ Normal(0, 1)
            log_bound ~ Normal(0, 1)
            @plate for i in eachindex(y)
                m[i] = mean(truncated(LogNormal(mu, exp(log_sigma)); lower=exp(log_bound)))
                v[i] = var(truncated(LogNormal(mu, exp(log_sigma)); lower=exp(log_bound)))
                y[i] ~ Normal(m[i], sqrt(v[i] + 0.01))
            end
        end
    elseif upper
        @brm data begin
            mu ~ Normal(0, 1)
            log_sigma ~ Normal(0, 1)
            log_bound ~ Normal(0, 1)
            @plate for i in eachindex(y)
                m[i] = opaque_upper_mean(mu, exp(log_sigma), exp(log_bound))
                v[i] = opaque_upper_var(mu, exp(log_sigma), exp(log_bound))
                y[i] ~ Normal(m[i], sqrt(v[i] + 0.01))
            end
        end
    else
        @brm data begin
            mu ~ Normal(0, 1)
            log_sigma ~ Normal(0, 1)
            log_bound ~ Normal(0, 1)
            @plate for i in eachindex(y)
                m[i] = opaque_lower_mean(mu, exp(log_sigma), exp(log_bound))
                v[i] = opaque_lower_var(mu, exp(log_sigma), exp(log_bound))
                y[i] ~ Normal(m[i], sqrt(v[i] + 0.01))
            end
        end
    end
end

function varying_moment_model(data, native, upper)
    if native && upper
        @brm data begin
            mu ~ Normal(0, 1)
            log_sigma ~ Normal(0, 1)
            log_bound ~ Normal(0, 1)
            @plate for i in eachindex(y)
                m[i] = mean(truncated(LogNormal(mu, exp(log_sigma)); upper=exp(log_bound + offset[i])))
                v[i] = var(truncated(LogNormal(mu, exp(log_sigma)); upper=exp(log_bound + offset[i])))
                y[i] ~ Normal(m[i], sqrt(v[i] + 0.01))
            end
        end
    elseif native
        @brm data begin
            mu ~ Normal(0, 1)
            log_sigma ~ Normal(0, 1)
            log_bound ~ Normal(0, 1)
            @plate for i in eachindex(y)
                m[i] = mean(truncated(LogNormal(mu, exp(log_sigma)); lower=exp(log_bound + offset[i])))
                v[i] = var(truncated(LogNormal(mu, exp(log_sigma)); lower=exp(log_bound + offset[i])))
                y[i] ~ Normal(m[i], sqrt(v[i] + 0.01))
            end
        end
    elseif upper
        @brm data begin
            mu ~ Normal(0, 1)
            log_sigma ~ Normal(0, 1)
            log_bound ~ Normal(0, 1)
            @plate for i in eachindex(y)
                m[i] = opaque_upper_mean(mu, exp(log_sigma), exp(log_bound + offset[i]))
                v[i] = opaque_upper_var(mu, exp(log_sigma), exp(log_bound + offset[i]))
                y[i] ~ Normal(m[i], sqrt(v[i] + 0.01))
            end
        end
    else
        @brm data begin
            mu ~ Normal(0, 1)
            log_sigma ~ Normal(0, 1)
            log_bound ~ Normal(0, 1)
            @plate for i in eachindex(y)
                m[i] = opaque_lower_mean(mu, exp(log_sigma), exp(log_bound + offset[i]))
                v[i] = opaque_lower_var(mu, exp(log_sigma), exp(log_bound + offset[i]))
                y[i] ~ Normal(m[i], sqrt(v[i] + 0.01))
            end
        end
    end
end

function measure(q, point, location_index)
    q(point); q(point)
    active = copy(point)
    checksum = 0.0
    samples = Float64[]
    for round in 1:7
        seconds = @elapsed for i in 1:100
            active[location_index] = point[location_index] + 1e-6*(i + 100round)
            checksum += q(active)
        end
        push!(samples, seconds/100)
    end
    @test isfinite(checksum)
    (; seconds=median(samples), bytes=@allocated(q(point)), checksum)
end

# The same changed-input protocol for `logdensity_and_gradient`.
function measure_gradient(problem, point, location_index)
    call(u) = LogDensityProblems.logdensity_and_gradient(problem, u)
    call(point); call(point)
    active = copy(point)
    checksum = 0.0
    samples = Float64[]
    for round in 1:7
        seconds = @elapsed for i in 1:20
            active[location_index] = point[location_index] + 1e-6*(i + 20round)
            value, gradient = call(active)
            checksum += value + sum(gradient)
        end
        push!(samples, seconds/20)
    end
    @test isfinite(checksum)
    value, gradient = call(point)
    (; seconds=median(samples), bytes=@allocated(call(point)), checksum, value, gradient)
end

function run(output; varying=false)
    mkpath(output)
    rows = []
    for upper in (false, true), n in (0, 1, 64, 1024)
        data = (;y=[0.8 + 0.01mod(i, 13) for i in 1:n],
            offset=[0.001mod(i, 7) for i in 1:n])
        saved = deepcopy(data)
        checksums = Float64[]
        values = Vector{Float64}[]
        gradients = Vector{Float64}[]
        for native in (true, false)
            model = varying ? varying_moment_model(data, native, upper) : moment_model(data, native, upper)
            rk = RKBRMI(model)
            q = prepare_query(rk.model, rk_bound_plan(rk), :likelihood)
            names = coordinate_names(rk.model.layout)
            point = zeros(length(names))
            location_index = only(findall(==(:mu), names))
            point[location_index] = upper ? 3.0 : -3.0
            point[only(findall(==(:log_sigma), names))] = log(0.2)
            push!(values, [Base.invokelatest(q, point), Base.invokelatest(q, point .+ 0.001)])
            label = "$(upper ? "upper" : "lower")-$(native ? "owned" : "opaque")-$n"
            codefile = joinpath(output, label*".jl")
            open(codefile, "w") do io
                show(io, code_expr(q)); println(io)
            end
            result = Base.invokelatest(measure, q, point, location_index)
            problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            reverse = Base.invokelatest(measure_gradient, problem, point, location_index)
            push!(checksums, result.checksum)
            push!(gradients, [reverse.value; reverse.gradient])
            push!(rows, Dict("side"=>string(upper ? :upper : :lower), "native"=>native,
                "n"=>n, "varying_bound"=>varying, "code"=>codefile, "seconds"=>result.seconds,
                "bytes"=>result.bytes, "checksum"=>result.checksum,
                "gradient_seconds"=>reverse.seconds, "gradient_bytes"=>reverse.bytes))
            println("BRM_MOMENT_HOISTING ", label, " ", result,
                " gradient=", (; seconds=reverse.seconds, bytes=reverse.bytes))
            flush(stdout)
        end
        @test values[1] ≈ values[2] rtol=2e-12 atol=2e-12
        @test checksums[1] ≈ checksums[2] rtol=2e-12 atol=2e-12
        @test gradients[1] ≈ gradients[2] rtol=1e-10 atol=1e-10
        @test isequal(data, saved)
    end
    open(joinpath(output, "measurements.toml"), "w") do io
        TOML.print(io, Dict("rows"=>rows, "julia"=>string(VERSION),
            "brm"=>string(Base.pkgversion(BayesianRegressionModels)),
            "rk"=>string(Base.pkgversion(ReactiveKernels))))
    end
end

length(ARGS) in (1, 2) || error("provide the output directory and optional --varying-bound")
length(ARGS) == 1 || ARGS[2] == "--varying-bound" || error("unknown option")
run(first(ARGS); varying=length(ARGS) == 2)
