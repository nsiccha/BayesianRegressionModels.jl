module JAXExampleFigures

using AlgebraOfVega
include("presentation.jl")
using .JAXExampleResults: table, POPULATION

const METHOD_LABELS = Dict("native" => "WarmupHMC · Julia", "jax" => "NumPyro · JAX")
const METHOD_ORDER = [METHOD_LABELS["native"], METHOD_LABELS["jax"]]
const PALETTE = ["#2a78d6", "#eb6834"]

"""Population medians and central 50%/95% intervals from the cached fit summaries."""
function population_figure()
    posterior = table("posterior")
    rows = [(; r..., label, sampler=METHOD_LABELS[r.method])
            for (key, label, _) in POPULATION
            for r in posterior if r.quantity == key]
    data(rows) * mapping(:q50 => "Parameter value";
        y=:sampler => "", row=:label => sorter(getindex.(POPULATION, 2)) => "",
        color=:sampler => "Sampler") *
        pointinterval(bands=[:q025 => :q975, :q25 => :q75]) *
        config(width=470, height=55, facet=(; linkxaxes=:none),
            scales=scales(Color=(; categories=METHOD_ORDER, palette=PALETTE)))
end

"""Group effects from the cached, response-scale posterior summaries."""
function group_figure()
    rows = [(; r..., label=uppercasefirst(r.quantity), sampler=METHOD_LABELS[r.method])
            for r in table("group_effects")]
    data(rows) * mapping(:q50 => "Deviation"; y=:group => "Group",
        col=:label => "", color=:sampler => "Sampler") *
        pointinterval(bands=[:q025 => :q975, :q25 => :q75]) *
        config(width=240, height=460, facet=(; linkxaxes=:none),
            scales=scales(Color=(; categories=METHOD_ORDER, palette=PALETTE)))
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    # Optional standalone specs, outside the tracked source tree.
    outdir = isempty(ARGS) ? mktempdir() : only(ARGS)
    mkpath(outdir)
    for (name, spec) in (("population", JAXExampleFigures.population_figure()),
                         ("groups", JAXExampleFigures.group_figure()))
        path = joinpath(outdir, name * ".json")
        write(path, JAXExampleFigures.AlgebraOfVega.to_json(spec))
        println(path)
    end
end
