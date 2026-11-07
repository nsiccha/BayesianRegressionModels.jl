# Synthetic model of the shape connected SVI's `cell_units` exists for.
#
# - `nsite` scientific subjects, each with one vector-valued cell of unequal
#   length: all of a site's rows are one cell, and that cell reads every
#   coefficient of a shared field, so its footprint does not grow with its rows.
# - `nsensor` sensors, each with a gain and a noise effect as ordinary coordinates
#   of plain vectors, not a grouping. Their rows are elementwise cells that read
#   the field and their own sensor's two effects, and no subject's effect. The
#   response indexes its rows, so a unit with no rows demands none of them.
#
# Purely synthetic: no consumer's data, sizes or parameter values.
using BayesianRegressionModels, Random

function cell_toy_data(rng; nsite, nsensor, nfield=32, nread=6, site_rows=48:48:384,
        sensor_rows=60)
    sensor = shuffle(rng, repeat(1:nsensor; inner=sensor_rows))
    state = [rand(rng, 1:nfield, nread) for _ in sensor]
    weight = [randn(rng, nread) ./ sqrt(nread) for _ in sensor]
    truth = (; field=randn(rng, nfield), gain=0.1 .* randn(rng, nsensor),
        scale=-1.0 .+ 0.2 .* randn(rng, nsensor))
    row_y = [exp(truth.gain[s]) * sum(w .* truth.field[r]) + exp(truth.scale[s]) * randn(rng)
        for (s, r, w) in zip(sensor, state, weight)]
    counts = [site_rows[mod1(g, length(site_rows))] for g in 1:nsite]
    site_x = [collect(range(-1.0, 1.0; length=n)) for n in counts]
    # Each row reads every coefficient through a smooth feature of its position.
    site_phi = [[cos(k * x) / sqrt(nfield) for x in xs, k in 1:nfield] for xs in site_x]
    a, b = 0.5 .* randn(rng, nsite), 0.3 .* randn(rng, nsite)
    site_y = [a[g] .+ b[g] .* site_x[g] .+ site_phi[g] * truth.field .+ 0.4 .* randn(rng, counts[g])
        for g in 1:nsite]
    (; site=["site $g" for g in 1:nsite], site_x, site_phi, site_y,
       row_sensor=sensor, row_state=state, row_weight=weight, row_y,
       row_index=collect(eachindex(row_y)), nfield, nsensor)
end

cell_toy_model(data) = @brm data begin
    noise ~ Exponential(1)
    field[1:nfield] .~ Normal(0, 1)
    gain_mean ~ Normal(0, 0.5)
    gain_sd ~ Exponential(0.2)
    scale_mean ~ Normal(-1, 0.5)
    scale_sd ~ Exponential(0.2)
    gain_z[1:nsensor] .~ Normal(0, 1)
    scale_z[1:nsensor] .~ Normal(0, 1)
    log_gain = gain_mean + gain_sd * gain_z
    log_scale = scale_mean + scale_sd * scale_z
    row_gain = exp(log_gain[row_sensor])
    row_scale = exp(log_scale[row_sensor])
    level ~ 1 + (1 | effect | site)
    slope ~ 1 + (1 | effect | site)
    @plate for g in eachindex(level)
        site_y[g] ~ Normal(level[g] .+ slope[g] .* site_x[g] .+ site_phi[g] * field, noise)
    end
    @plate for i in eachindex(row_state)
        row_mean[i] = row_gain[i] * sum(row_weight[i] .* field[row_state[i]])
    end
    row_y ~ Normal(row_mean[row_index], row_scale[row_index])
end

# What a consumer keeps global so sensors can separate: every sensor reads these.
cell_toy_globals(m) = [name for name in m.view.coordinates
    if any(prefix -> startswith(String(name), prefix),
        ("gain_mean", "gain_sd", "scale_mean", "scale_sd"))]
