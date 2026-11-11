# test/fitted_state_version_models.jl — the models whose fitted-state programs
# `test/fitted_state_programs/` holds as earlier BRM versions exported them, and
# the reader and writer of those files (`test/rk_fitted_state_versions.jl`).
#
# Ragged joins as `test/rk_jax.jl` writes them (unordered, two columns per join,
# ordered) and fitted preprocessing as `test/rk_fitted_state.jl` writes it
# (`zscale`, factor and `mo` levels, a `constant(...)` with a grouping, an HSGP
# interval, R2D2). Public synthetic models only. Needs `BayesianRegressionModels`,
# `Distributions`, `ReactiveKernels` and `ReactiveKernelsPPL` loaded.

module FittedStateVersionModels
using BayesianRegressionModels, Distributions, Statistics
flat_pk(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    log_V ~ 1 + (1 | p | subject)
    log_F ~ 0 + log_dose
    effect(log_CL, Intercept) ~ Normal(0, 1)
    effect(log_V, Intercept) ~ Normal(0, 1)
    effect(log_F, :) ~ Normal(0, 0.5)
    sd(:, p) ~ Exponential(1)
    doses = groups(dose_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        dose = sum(exp.(log_F[doses[i]]))
        loc[i] = dose / exp(log_V[i]) .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end
flat_pk_two(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    log_V ~ 1 + (1 | p | subject)
    log_F ~ 0 + log_dose
    effect(log_CL, Intercept) ~ Normal(0, 1)
    effect(log_V, Intercept) ~ Normal(0, 1)
    effect(log_F, :) ~ Normal(0, 0.5)
    sd(:, p) ~ Exponential(1)
    doses = groups(dose_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        dose = sum(dose_amount[doses[i]] .* exp.(log_F[doses[i]]))
        loc[i] = dose / exp(log_V[i]) .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]]) .*
            obs_scale[obs[i]]
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end
flat_pk_ordered(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | subject)
    log_F ~ 0 + log_dose
    effect(log_CL, Intercept) ~ Normal(0, 1)
    effect(log_F, :) ~ Normal(0, 0.5)
    doses = groups(dose_subject => subject; order=dose_rank)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        first_dose = exp(first(log_F[doses[i]]))
        loc[i] = first_dose .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end
zscaled(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x)
    y ~ Normal(mu, sigma)
end
factors(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + g + gs + mo(r)
    y ~ Normal(mu, sigma)
end
grouped(d) = @brm d begin
    sigma ~ Exponential(1)
    xbar = constant(mean(x))
    mu ~ 1 + zscale(x) + (1 + z | subject)
    effect(mu, Intercept) ~ Normal(xbar, 1)
    y ~ Normal(mu, sigma)
end
smooth(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=5)
    y ~ Normal(mu, sigma)
end
shrunk(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + gs + (1 | subject)
    effect(mu, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5, alpha=1)
    y ~ Normal(mu, sigma)
end

# Subjects with uneven dose and observation rows, listed out of subject order;
# `s4` has no dose row (`test/rk_jax.jl`).
const FLAT_PK = (; subject=["s1", "s2", "s3", "s4"], weight=[61.0, 75.0, 68.0, 82.0],
    dose_subject=["s2", "s1", "s2", "s3", "s1", "s2"],
    log_dose=log.([1.4, 1.1, 1.7, 1.2, 1.9, 1.3]), dose_rank=[3, 2, 1, 1, 1, 2],
    obs_subject=["s1", "s3", "s2", "s1", "s4", "s2", "s3", "s4", "s1"],
    obs_time=[0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0],
    y=[0.9, 0.6, 1.4, 0.7, 0.2, 0.8, 0.3, 0.1, 0.4])
const FLAT_PK_TWO = merge(FLAT_PK, (; dose_amount=[100.0, 80.0, 120.0, 90.0, 60.0, 110.0],
    obs_scale=[1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95]))
# The ordered join reads each subject's first dose, so every subject has one.
const FLAT_PK_ORDERED = merge(FLAT_PK, (; dose_subject=[FLAT_PK.dose_subject; "s4"],
    log_dose=[FLAT_PK.log_dose; log(1.6)], dose_rank=[FLAT_PK.dose_rank; 1]))
# Fixed numbers, so every BRM version builds the same data.
const N = 24
const TRAIN = (; y=[sin(1.7i) for i in 1:N], x=[1 + 4 * abs(sin(0.9i)) for i in 1:N],
    z=[cos(1.3i) for i in 1:N], subject=repeat(1:6, inner=4),
    g=repeat(["a", "b", "c"], 8), gs=repeat(["lo", "mid", "hi"], outer=8),
    r=repeat(1:4, outer=6))

# Each model by the name its stored program is filed under, with its data.
const MODELS = (
    flat_pk=(flat_pk, FLAT_PK), flat_pk_two=(flat_pk_two, FLAT_PK_TWO),
    flat_pk_ordered=(flat_pk_ordered, FLAT_PK_ORDERED), zscaled=(zscaled, TRAIN),
    factors=(factors, TRAIN), grouped=(grouped, TRAIN), smooth=(smooth, TRAIN),
    shrunk=(shrunk, TRAIN))
end

# The stored programs of one file: `## <model>` starts a model, and each line
# under it is one statement as a Julia string literal (statements span lines).
function read_fitted_programs(path)
    programs = Dict{Symbol,Vector{String}}()
    current = nothing
    for line in eachline(path)
        (isempty(line) || startswith(line, "# ")) && continue
        if startswith(line, "## ")
            current = programs[Symbol(line[4:end])] = String[]
        else
            push!(current, Meta.parse(line)::String)
        end
    end
    programs
end

# This BRM's fitted-state program of every model, written as `read_fitted_programs`
# reads it.
function write_fitted_programs(path)
    BRM = FittedStateVersionModels.BayesianRegressionModels
    open(path, "w") do io
        println(io, "# The fitted-state program (`brm_fitted_state(rk).program`) of each model of")
        println(io, "# test/fitted_state_version_models.jl, as BayesianRegressionModels ",
            pkgversion(BRM), " exports it.")
        for (name, (build, data)) in pairs(FittedStateVersionModels.MODELS)
            println(io, "## ", name)
            foreach(s -> println(io, repr(s)), BRM.brm_fitted_state(BRM.RKBRMI(build(data))).program)
        end
    end
    path
end
