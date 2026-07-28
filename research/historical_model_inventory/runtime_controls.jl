#!/usr/bin/env julia

using BayesianRegressionModels
using Distributions: Exponential, Normal
using Downloads
using LogDensityProblems
using Random
using SHA
using StanBlocks
using WarmupHMC

const OUT = joinpath(@__DIR__, "runtime_controls.tsv")
const CACHE = joinpath(tempdir(), "brm-historical-runtime-controls")
const BRM_SHA = "784712998ea67f6429d0a3b5a3241fe9cb690e64"
const STANBLOCKS_SHA = "329a178a7ad7877da0b58ad2c360d417ddd663f9"
const WARMUPHMC_SHA = "b185eedbbeeef6fb3327afb30dc995c98591af02"
const PEER_WARMUPHMC_SHA = "38398527e0406ad31aeaec3efe24d581a18a269e"

tsv_escape(value) = replace(string(value),
    '\\' => "\\\\", '\t' => "\\t", '\n' => "\\n", '\r' => "\\r")

function compact_error(err, bt=catch_backtrace())
    text = replace(sprint(showerror, err, bt), '\t' => ' ', '\r' => ' ', '\n' => ' ')
    length(text) <= 1600 ? text : first(text, 1600) * "…"
end

function write_tsv(rows)
    columns = [
        "case", "route", "fidelity", "source_reference", "data_assumptions",
        "descriptor", "stanc", "stan_code_sha256", "bridgestan_instantiate",
        "dimension", "log_density_zero", "gradient_finite", "warmuphmc",
        "warmuphmc_draws", "warmuphmc_divergences", "expected_control",
        "error", "brm_sha", "stanblocks_sha", "warmuphmc_sha",
    ]
    open(OUT, "w") do io
        println(io, join(columns, '\t'))
        for row in rows
            println(io, join((tsv_escape(get(row, c, "")) for c in columns), '\t'))
        end
    end
end

function run_model(case_name, route, model;
                   fidelity, source_reference, data_assumptions,
                   descriptor="not-applicable", sample_draws=50,
                   expected_control="")
    row = Dict{String,String}(
        "case" => case_name,
        "route" => route,
        "fidelity" => fidelity,
        "source_reference" => source_reference,
        "data_assumptions" => data_assumptions,
        "descriptor" => descriptor,
        "stanc" => "not-run",
        "stan_code_sha256" => "",
        "bridgestan_instantiate" => "not-run",
        "dimension" => "",
        "log_density_zero" => "",
        "gradient_finite" => "not-run",
        "warmuphmc" => "not-run",
        "warmuphmc_draws" => "",
        "warmuphmc_divergences" => "",
        "expected_control" => expected_control,
        "error" => "",
        "brm_sha" => BRM_SHA,
        "stanblocks_sha" => STANBLOCKS_SHA,
        "warmuphmc_sha" => WARMUPHMC_SHA,
    )

    code = try
        value = StanBlocks.stan_code(model)
        row["stan_code_sha256"] = bytes2hex(sha256(value))
        value
    catch err
        row["error"] = compact_error(err)
        return row
    end
    stanc = try
        StanBlocks.stanc_check(code; warn_pedantic=false)
    catch err
        row["stanc"] = "fail"
        row["error"] = compact_error(err)
        return row
    end
    row["stanc"] = stanc.ok ? "pass" : "fail"
    if !stanc.ok
        row["error"] = replace(stanc.output, '\n' => ' ')
        return row
    end

    mkpath(CACHE)
    problem = try
        path = joinpath(CACHE, row["stan_code_sha256"] * ".stan")
        value = StanBlocks.stan_instantiate(model; path, make_args=["O=0", "STAN_THREADS=true"])
        row["bridgestan_instantiate"] = "pass"
        value
    catch err
        row["bridgestan_instantiate"] = "fail"
        row["error"] = compact_error(err)
        return row
    end

    dim = LogDensityProblems.dimension(problem)
    row["dimension"] = string(dim)
    q = zeros(dim)
    try
        lp = LogDensityProblems.logdensity(problem, q)
        lp2, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        finite = isfinite(lp) && isfinite(lp2) && all(isfinite, gradient)
        row["log_density_zero"] = repr(lp)
        row["gradient_finite"] = string(finite)
        finite || return row
    catch err
        row["gradient_finite"] = "false"
        row["error"] = compact_error(err)
        return row
    end

    try
        fit = WarmupHMC.adaptive_warmup_mcmc(
            Xoshiro(0x20260728), problem; init=q, n_draws=sample_draws,
        )
        row["warmuphmc"] = "pass"
        row["warmuphmc_draws"] = string(size(fit.posterior_position, 2))
        row["warmuphmc_divergences"] = string(fit.n_divergent_samples)
    catch err
        row["warmuphmc"] = "fail"
        row["error"] = compact_error(err)
    end
    row
end

function sleepstudy_data()
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv"
    path = Downloads.download(url)
    lines = readlines(path)
    header = split(first(lines), ',')
    index = Dict(name => i for (i, name) in enumerate(header))
    table = [split(line, ',') for line in Iterators.drop(lines, 1) if !isempty(line)]
    reaction = [parse(Float64, row[index["Reaction"]]) / 100 for row in table]
    days = [parse(Float64, row[index["Days"]]) for row in table]
    raw_subject = [parse(Int, row[index["Subject"]]) for row in table]
    levels = sort(unique(raw_subject))
    codes = Dict(subject => i for (i, subject) in enumerate(levels))
    data = (; Reaction=reaction, Days=days, Subject=[codes[s] for s in raw_subject])
    data, bytes2hex(sha256(read(path))), length(table), length(levels)
end

function sleepstudy_model()
    data, data_sha, n, n_subject = sleepstudy_data()
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + Days + (1 + Days | Subject)
        Reaction ~ Normal(mu, sigma)
    end
    sb = SBBRMI(builder(data); mod=@__MODULE__)
    descriptor = brm_descriptor(sb; name=:sleepstudy_real_control)
    row = run_model(
        "sleepstudy_real_data", "ordinary_brm", sb.model;
        fidelity="capability control; historical response rescaled, not a catalog-faithful fit",
        source_reference="WarmupHMC peer control; Rdatasets lme4/sleepstudy; BRM historical bambi/sleepstudy",
        data_assumptions="n=$n; subjects=$n_subject; Reaction divided by 100; raw Subject ids densely recoded; data_sha256=$data_sha",
        descriptor="pass; operations=$(join(getproperty.(descriptor.operations, :name), ','))",
        sample_draws=100,
        expected_control="dimension=42; peer logdensity(zeros)=-832.3603659550055; peer 500 draws × 12 seeds, zero divergences; peer WarmupHMC=$PEER_WARMUPHMC_SHA",
    )
    row
end

const kernel_builder = @brm begin
    sigma_a ~ Exponential(1)
    sigma_p ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    log_V ~ 1 + (1 | p | subject)
    pred ~ kernel(t, dose, dv, log_CL, log_V) do ts, dd, yy, lCL, lV
        CL = exp(lCL)
        V = exp(lV)
        mu = dd / V * exp(-(CL / V) * ts)
        yy ~ normal(mu, addprop(mu, sigma_a, sigma_p))
        mu
    end
end

kernel_data(n=4) = (;
    t=[collect(1.0:3.0) for _ in 1:n],
    dose=fill(100.0, n),
    dv=[collect(1.0:3.0) ./ 10 for _ in 1:n],
    subject=collect(1:n),
)

function kernel_model()
    data = kernel_data()
    sb = SBBRMI(kernel_builder(data); mod=@__MODULE__)
    descriptor = brm_descriptor(sb; name=:kernel_route_control)
    run_model(
        "kernel_ragged_group_local", "brm_kernel", sb.model;
        fidelity="executable route control; faithful for ragged group-local deterministic cells, not fanned to any historical row",
        source_reference="test/descriptor.jl kernel_builder at BRM $BRM_SHA",
        data_assumptions="4 subjects; each has three time/observation values; deterministic dose 100",
        descriptor="pass; columns=$(join(descriptor.columns, ',')); operations=$(join(getproperty.(descriptor.operations, :name), ','))",
        sample_draws=50,
    )
end

StanBlocks.@slic historical_plate_cell(
    k::int,
    L::matrix[k, k],
    tau::vector[k],
) = begin
    z::vector[k] ~ std_normal()
    return diag_pre_multiply(tau, L) * z
end

const plate_model = StanBlocks.@slic (;
    n_groups=5,
    k=3,
    y=[0.2, -0.1, 0.3, 0.0, 0.4],
) begin
    L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2.0)
    tau::vector[k] ~ normal(0.0, 1.0; lower=0.0)
    b::vector[k] ~ plate(y; outer=(n_groups,)) do yi
        cell ~ historical_plate_cell(k, L, tau)
        yi ~ normal(cell[1], 0.5)
        cell
    end
end

function plate_control()
    run_model(
        "stanblocks_fixed_correlated_plate", "stanblocks_plate", plate_model;
        fidelity="executable substrate analogue for fixed-width correlated group-local cells; not a translation of one catalog row",
        source_reference="test/plate_stress.jl fixed_correlated at BRM $BRM_SHA",
        data_assumptions="5 fixed-width groups; three-dimensional correlated latent effect; scalar Gaussian observation per group",
        sample_draws=50,
    )
end

rows = Dict{String,String}[]
for (name, build) in (
    "sleepstudy_real_data" => sleepstudy_model,
    "kernel_ragged_group_local" => kernel_model,
    "stanblocks_fixed_correlated_plate" => plate_control,
)
    println("CONTROL\t$name")
    row = try
        build()
    catch err
        Dict{String,String}(
            "case" => name,
            "route" => "build",
            "fidelity" => "",
            "source_reference" => "",
            "data_assumptions" => "",
            "descriptor" => "fail",
            "error" => compact_error(err),
            "brm_sha" => BRM_SHA,
            "stanblocks_sha" => STANBLOCKS_SHA,
            "warmuphmc_sha" => WARMUPHMC_SHA,
        )
    end
    push!(rows, row)
    write_tsv(rows)
end

println("output=$(abspath(OUT))")
