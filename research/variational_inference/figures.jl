# AlgebraOfVega figures of docs/src/variational-inference.md, built from cached tables.
# `figure_specs(read_table)` supplies the build-time Vega-Lite plots; `main` writes
# optional local PNG previews from exactly the same specifications.
#
# Run (any environment with AlgebraOfVega; the test environment has it):
#   julia --project=test research/variational_inference/figures.jl [outdir]
# `outdir` defaults to docs/src/assets/variational-inference. The PNGs are written with
# AlgebraOfVega's `vsave`, which renders through the vl-convert command-line tool.

using AlgebraOfVega
using AlgebraOfGraphics: sorter

results(name) = joinpath(@__DIR__, "results", name)

function table(name)
    lines = readlines(results(name))
    header = Symbol.(split(lines[1], '\t'))
    cell(s) = something(tryparse(Int, s), tryparse(Float64, s), String(s))
    [NamedTuple{Tuple(header)}(Tuple(cell.(split(l, '\t'; keepempty=true)))) for l in lines[2:end]]
end

# One colour per method, the same in every figure.
const METHODS = ["NUTS", "SVI", "streaming SVI"]
const COLOURS = ["#2a78d6", "#eb6834", "#eb6834"]
method_scale(methods) = scales(Color=(; categories=methods,
    palette=[COLOURS[findfirst(==(m), METHODS)] for m in methods]))

const PARAMETERS = ["intercept a", "slope b", "sd of site intercepts", "sd of site slopes",
                    "correlation of intercepts and slopes"]

function save_figure(outdir, name, spec, sc)
    path = joinpath(outdir, name * ".png")
    vsave(spec, sc, path; scale=2)
    println("FIGURE ", path)
end

# Population medians and intervals, with the generating value as a dashed rule.
function marginals(read_table=table)
    rows = read_table("population.tsv")
    truth = unique([(; parameter=r.parameter, generating=r.generating) for r in rows])
    order = sorter(PARAMETERS)
    spec = data(rows) * mapping(:median => "parameter value", :method => "";
                                 color=:method => "method", layout=:parameter => order) *
               pointinterval(bands=[:q025 => :q975, :q25 => :q75]) +
           data(truth) * mapping(:generating; layout=:parameter => order) *
               visual(VLines; linestyle=:dash, color=:ink)
    spec = spec * config(width=240, height=100, facet=(; linkxaxes=:none),
                         scales=scales(Layout=(; palette=wrapped(cols=2))),
                         title="Population parameters")
    spec * config(scales=method_scale(["NUTS", "SVI"]))
end

# Each site's intercept and slope deviation: SVI posterior mean against NUTS posterior
# mean, with the identity line.
function site_means(read_table=table)
    rows = read_table("site_effects.tsv")
    lo = minimum(min(r.nuts_mean, r.svi_mean) for r in rows)
    hi = maximum(max(r.nuts_mean, r.svi_mean) for r in rows)
    identity_line = [(; effect=e, x=v, y=v) for e in unique(r.effect for r in rows), v in (lo, hi)]
    spec = data(rows) * mapping(:nuts_mean => "NUTS posterior mean", :svi_mean => "SVI posterior mean";
                                layout=:effect) * visual(Scatter; markersize=5, alpha=0.6, color="#eb6834") +
           data(vec(identity_line)) * mapping(:x => "NUTS posterior mean", :y => "SVI posterior mean";
                                              layout=:effect) * visual(Lines; linestyle=:dash, color=:ink)
    spec * config(width=240, height=240, title="Site effects")
end

# The streaming estimates per window beside NUTS and batch SVI on the data seen so far:
# median, 50% and 95% intervals.
function streaming(read_table=table)
    rows = [(; r..., window=string("window ", r.window)) for r in read_table("streaming.tsv")]
    methods = ["NUTS", "streaming SVI"]
    spec = data(filter(r -> r.method != "batch SVI", rows)) *
           mapping(:window => "data seen up to", :median => "posterior"; color=:method => "method",
                                layout=:parameter => sorter(PARAMETERS)) *
           pointinterval(bands=[:q025 => :q975, :q25 => :q75], orientation=:vertical) +
           data(filter(r -> r.method == "batch SVI", rows)) *
           mapping(:window => "data seen up to", :median => "posterior";
                   layout=:parameter => sorter(PARAMETERS)) *
           pointinterval(bands=[:q025 => :q975, :q25 => :q75], orientation=:vertical) * visual(; color=:ink)
    spec = spec * config(width=240, height=180, facet=(; linkyaxes=:none),
                         scales=scales(Layout=(; palette=wrapped(cols=2)), Color=(; categories=methods)),
                         title="Population estimates as sites and visits arrive")
    spec * config(scales=method_scale(methods))
end

# The ELBO trace of the full fit: each window's trimmed mean, coloured by stage.
function elbo(read_table=table)
    rows = read_table("elbo.tsv")
    stages = ["exploring", "settling"]
    spec = data(filter(r -> r.stage != "polishing", rows)) *
           mapping(:iteration, :elbo_trimmed_mean => "ELBO (10% trimmed window mean)";
                                color=:stage => sorter(stages)) * visual(Scatter; markersize=6) +
           data(filter(r -> r.stage == "polishing", rows)) *
           mapping(:iteration, :elbo_trimmed_mean => "ELBO (10% trimmed window mean)") *
               visual(Scatter; markersize=6, color=:ink) +
           data(rows) * mapping(:iteration, :elbo_trimmed_mean => "ELBO (10% trimmed window mean)") *
               visual(Lines; color=:ink, alpha=0.4)
    spec * config(width=540, height=220, title="Optimization windows",
        scales=scales(Color=(; categories=stages, palette=["#2a78d6", "#eb6834"])))
end

function figure_specs(read_table=table)
    ("population-marginals" => marginals(read_table),
     "site-effects" => site_means(read_table),
     "streaming" => streaming(read_table),
     "elbo" => elbo(read_table))
end

function main(outdir=normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "variational-inference")))
    mkpath(outdir)
    for (name, spec) in figure_specs()
        save_figure(outdir, name, spec, scales())
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
