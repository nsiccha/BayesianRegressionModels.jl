# What brm_svi_fit does with a `multiresolution=` field, which it has never run: learned
# lengths, then lengths fixed by `=`. Synthetic stations only.
#
#     julia --project=test research/svi_field_units/multiresolution_probe.jl
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, Random, Statistics, Printf
const BRM = BayesianRegressionModels

function probe_data(; stations=24, times=6, reads=3, rows=3, seed=1)
    rng = Xoshiro(seed)
    px, py = 10 .* rand(rng, stations), 10 .* rand(rng, stations)
    pairs = [(s, t) for s in 1:stations for t in sort!(randperm(rng, times)[1:reads])]
    row = repeat(eachindex(pairs); inner=rows)
    station = [pairs[k][1] for k in row]
    (; east=px[station], north=py[station], time=[10.0 * pairs[k][2] for k in row],
       x=randn(rng, length(row)), y=randn(rng, length(row)), site=string.("s", station))
end

learned(data) = @brm data begin
    sigma ~ Exponential(1)
    field ~ 0 + gmrf((east, north), time; multiresolution=3)
    sd(field, gmrf((east, north), time)) ~ Exponential(1)
    length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(3.0), 0.4)
    length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(13.0), 0.4)
    mu ~ 1 + (0 + x | site)
    y ~ Normal(mu .+ field, sigma)
end

fixed(data) = @brm data begin
    sigma ~ Exponential(1)
    field ~ 0 + gmrf((east, north), time; multiresolution=3)
    sd(field, gmrf((east, north), time)) = 1.0
    length_scale(field, gmrf((east, north), time), space) = 3.0
    length_scale(field, gmrf((east, north), time), time) = 13.0
    mu ~ 1 + (0 + x | site)
    y ~ Normal(mu .+ field, sigma)
end

function probe(label, build)
    data = probe_data()
    println("MULTIRES == $label: $(length(data.y)) rows")
    flush(stdout)
    try
        m = brm_nlme_model(RKBRMI(build(data)); ad_backend=AutoEnzyme(; mode=Enzyme.Reverse),
            subject_group=:site, unit_partition=:connected)
        println("MULTIRES   view roles: ", sort!(collect(Set(m.view.role))))
        timed = @timed brm_svi_fit(m; batch_size=4, max_iterations=200, ndraws=0, threads=1,
            rng=Xoshiro(3), separator=:none)
        d = timed.value.diagnostics
        @printf("MULTIRES   ran: %d iterations, %.4f s/step, %.1f s in all, %d globals, final ELBO %.1f\n",
            d.iterations, d.seconds / max(d.iterations, 1), timed.time, length(d.globals),
            mean(d.elbo[max(1, end - 49):end]))
    catch err
        text = sprint(showerror, err)
        println("MULTIRES   threw ", typeof(err), ": ", first(text, 1200))
    end
    flush(stdout)
end

probe("learned lengths", learned)
probe("fixed lengths", fixed)
