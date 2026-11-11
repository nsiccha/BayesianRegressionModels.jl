# julia --project=research/adaptive_centering/plots research/jax/figures.jl
using AlgebraOfVega, CairoMakie
import AlgebraOfGraphics
include("presentation.jl")
using .JAXExampleResults: table, POPULATION

const METHOD_LABELS = Dict("native" => "WarmupHMC · Julia", "jax" => "NumPyro · JAX")
const COLORS = ["WarmupHMC · Julia" => "#2a78d6", "NumPyro · JAX" => "#eb6834"]

function main(outdir=normpath(joinpath(@__DIR__, "../../docs/src/assets/jax")))
    mkpath(outdir)
    draws = table("population_draws")
    truth = Dict(r.quantity => r.value for r in table("generating_values"))
    fig = Figure(; size=(1160, 750), fontsize=17, backgroundcolor=:white)
    Label(fig[0, 1:3], "One model, sampled in Julia and Python";
        fontsize=25, font=:bold, tellwidth=false)
    Label(fig[1, 1:3], "Posterior marginal densities · dotted lines mark the generating values";
        fontsize=16, tellwidth=false)
    for (i, (quantity, label, generating)) in enumerate(POPULATION)
        rows = [(; value=r.value, method=METHOD_LABELS[r.method])
                for r in draws if r.quantity == quantity]
        curves = reduce(+, [
            data(filter(r -> r.method == METHOD_LABELS[method], rows)) *
            mapping(:value => label; color=:method => "Sampler") *
            AlgebraOfGraphics.density() *
            AlgebraOfGraphics.subvisual(:line; linewidth=2.5, linestyle) *
            AlgebraOfGraphics.subvisual(:area; visible=false, legend=(; visible=false))
            for (method, linestyle) in (("native", :solid), ("jax", :dash))])
        reference = mapping([truth[generating]]) * visual(VLines; color=:black, linestyle=:dot, linewidth=1.5)
        # Draw the analysis directly: its named sublayers are resolved by AoG.
        grid = AlgebraOfGraphics.draw!(fig[2 + div(i-1, 3), 1 + mod(i-1, 3)],
            curves + reference, AlgebraOfGraphics.scales(Color=(; palette=COLORS));
            axis=(; ylabel="Density", xgridvisible=false, ygridvisible=false))
        i == 1 && AlgebraOfGraphics.legend!(fig[4, 1:3], grid; orientation=:horizontal, titleposition=:left)
    end
    save(joinpath(outdir, "posterior-marginals.png"), fig; px_per_unit=1.5)
    println("FIGURE posterior-marginals.png")

    summaries = table("group_effects")
    fig = Figure(; size=(1160, 840), fontsize=16, backgroundcolor=:white)
    Label(fig[0, 1:2], "Group effects on the response scale";
        fontsize=25, font=:bold, tellwidth=false)
    Label(fig[1, 1:2], "Posterior median · thick intervals 50% · thin intervals 95%";
        fontsize=16, tellwidth=false)
    for (i, quantity) in enumerate(("intercept", "slope"))
        native = sort([r for r in summaries if r.quantity == quantity && r.method == "native"]; by=r->r.q50)
        labels = [r.group for r in native]
        rows = [(; r..., method=METHOD_LABELS[r.method],
                    position=findfirst(==(r.group), labels) + (r.method == "native" ? -0.15 : 0.15))
                for r in summaries if r.quantity == quantity]
        intervals = data(rows) * mapping(:position => "Group", :q025 => "Deviation", :q975 => "Deviation";
            color=:method => "Sampler") * visual(Rangebars; direction=:x, linewidth=1.6, whiskerwidth=0)
        inner = data(rows) * mapping(:position, :q25, :q75; color=:method => "Sampler") *
            visual(Rangebars; direction=:x, linewidth=5, whiskerwidth=0)
        medians = data(rows) * mapping(:q50 => "Deviation", :position => "Group"; color=:method => "Sampler") *
            visual(Scatter; markersize=6, strokecolor=:white, strokewidth=1.2)
        zero = mapping([0.0]) * visual(VLines; color=(:black, 0.35), linestyle=:dot)
        grid = sdraw!(fig[2, i], (intervals + inner + medians + zero) *
            config(scales=scales(Color=(; palette=COLORS)),
                axis=(; title=uppercasefirst(quantity) * " deviation", yticks=(1:length(labels), labels),
                    xgridvisible=false, ygridvisible=false)))
        i == 1 && AlgebraOfGraphics.legend!(fig[3,1:2], grid; orientation=:horizontal, titleposition=:left)
    end
    save(joinpath(outdir, "group-effects.png"), fig; px_per_unit=1.5)
    println("FIGURE group-effects.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
