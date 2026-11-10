# Figures of docs/src/variational-inference.md, drawn from the tables in results/.
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
const METHODS = ["NUTS", "SVI", "streaming SVI", "batch SVI"]
const COLOURS = ["#2a78d6", "#eb6834", "#eb6834", "#1baf7a"]
method_scale(methods) = scales(Color=(; categories=methods,
    palette=[COLOURS[findfirst(==(m), METHODS)] for m in methods]))

const PARAMETERS = ["intercept a", "slope b", "sd of site intercepts", "sd of site slopes",
                    "correlation of intercepts and slopes"]

function save_figure(outdir, name, spec, sc)
    path = joinpath(outdir, name * ".png")
    vsave(spec, sc, path; scale=2)
    println("FIGURE ", path)
end

# The marginal posteriors of the population parameters under NUTS and SVI, with the
# generating value as a dashed rule.
function marginals(outdir)
    draws = table("population_draws.tsv")
    truth = unique([(; parameter=r.parameter, generating=r.generating) for r in table("population.tsv")])
    order = sorter(PARAMETERS)
    spec = data(draws) * mapping(:value => "posterior draw"; color=:method => "method",
                                 layout=:parameter => order) * density() +
           data(truth) * mapping(:generating; layout=:parameter => order) *
               visual(VLines; linestyle=:dash, color=:ink)
    spec = spec * config(width=260, height=180, facet=(; linkxaxes=:none, linkyaxes=:none),
                         scales=scales(Layout=(; palette=wrapped(cols=3))),
                         title="Population parameters: NUTS and SVI (dashed: generating value)")
    save_figure(outdir, "population-marginals", spec, method_scale(["NUTS", "SVI"]))
end

# Each site's intercept and slope deviation: SVI posterior mean against NUTS posterior
# mean, with the identity line.
function site_means(outdir)
    rows = table("site_effects.tsv")
    lo = minimum(min(r.nuts_mean, r.svi_mean) for r in rows)
    hi = maximum(max(r.nuts_mean, r.svi_mean) for r in rows)
    identity_line = [(; effect=e, x=v, y=v) for e in unique(r.effect for r in rows), v in (lo, hi)]
    spec = data(rows) * mapping(:nuts_mean => "NUTS posterior mean", :svi_mean => "SVI posterior mean";
                                layout=:effect) * visual(Scatter; markersize=5, alpha=0.6, color="#eb6834") +
           data(vec(identity_line)) * mapping(:x => "NUTS posterior mean", :y => "SVI posterior mean";
                                              layout=:effect) * visual(Lines; linestyle=:dash, color=:ink)
    spec = spec * config(width=300, height=300, title="Site effects: SVI against NUTS, one point per site")
    save_figure(outdir, "site-effects", spec, scales())
end

# The streaming estimates per window beside NUTS and batch SVI on the data seen so far:
# median, 50% and 95% intervals.
function streaming(outdir)
    rows = [(; r..., window=string("window ", r.window)) for r in table("streaming.tsv")]
    methods = ["NUTS", "streaming SVI", "batch SVI"]
    spec = data(rows) * mapping(:window => "data seen up to", :median => "posterior"; color=:method => "method",
                                layout=:parameter => sorter(PARAMETERS)) *
           pointinterval(bands=[:q025 => :q975, :q25 => :q75], orientation=:vertical)
    spec = spec * config(width=260, height=200, facet=(; linkyaxes=:none),
                         scales=scales(Layout=(; palette=wrapped(cols=3)), Color=(; categories=methods)),
                         title="Streaming SVI against NUTS and batch SVI on the accumulated data (median, 50% and 95% intervals)")
    save_figure(outdir, "streaming", spec, method_scale(methods))
end

# The ELBO trace of the full fit: each window's trimmed mean, coloured by stage.
function elbo(outdir)
    rows = table("elbo.tsv")
    stages = ["exploring", "settling", "polishing"]
    spec = data(rows) * mapping(:iteration, :elbo_trimmed_mean => "ELBO (10% trimmed window mean)";
                                color=:stage => sorter(stages)) * visual(Scatter; markersize=6) +
           data(rows) * mapping(:iteration, :elbo_trimmed_mean => "ELBO (10% trimmed window mean)") *
               visual(Lines; color=:ink, alpha=0.4)
    spec = spec * config(width=640, height=240, title="How the fit stops: the ELBO by window and stage")
    save_figure(outdir, "elbo", spec, scales(Color=(; categories=stages, palette=["#2a78d6", "#eda100", "#1baf7a"])))
end

function main(outdir=normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "variational-inference")))
    mkpath(outdir)
    marginals(outdir); site_means(outdir); streaming(outdir); elbo(outdir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
