using Documenter, DocumenterVitepress, BayesianRegressionModels

include("backend_comparisons.jl")
include("backend_exports.jl")
include("centering_examples.jl")

const DOCS_PAGES = [
    "Start" => ["Overview" => "index.md"],
    "Guides" => [
        "Model syntax" => "model-syntax.md",
        "Terms and priors" => "terms-and-priors.md",
        "Response families" => "response-families.md",
        "Space-time fields" => "space-time-fields.md",
        "Backends" => "backends.md",
        "Adaptive centering" => "adaptive-centering.md",
    ],
    "Examples" => [
        "Warfarin PK/PD" => "warfarin.md",
        "Wastewater renewal" => "wastewater.md",
        "Renewal and reporting delays" => "renewal.md",
        "Continuous-time states" => "state-space.md",
        "Whole-body pharmacokinetics" => "monster.md",
        "Python/JAX sampling" => "jax.md",
        "Streaming variational inference" => "variational-inference.md",
        "Sequential filtering" => "filtering.md",
        "Spatial field reconstruction" => "spatial-fields.md",
    ],
    "Reference" => [
        "Public API" => "api.md",
        "Formula terms" => "terms-reference.md",
    ],
]
const DOCS_MARKDOWN_PAGES = [joinpath(@__DIR__, "src", last(page))
                            for (_, pages) in DOCS_PAGES for page in pages]
const GENERATED_EXAMPLE_PAGES = filter(DOCS_MARKDOWN_PAGES) do path
    occursin("Main.BRMDocsComparisons.comparison(", read(path, String))
end
BRMDocsComparisons.validate_generated_templates(GENERATED_EXAMPLE_PAGES)
BRMDocsComparisons.validate_no_bypasses(DOCS_MARKDOWN_PAGES)

# The docs have one publication home. Keep the target literal: Documenter's
# GitHub Actions deploy criterion requires the workflow repository to match
# `repo`, and the CI-equivalent docs wrapper validates the same static target.
# A legacy JuliaBayes workflow therefore cannot create a drifting second site.
const DOCS_DEVBRANCH = "ns/devibe"


makedocs(
    sitename = "BayesianRegressionModels.jl",
    repo = Documenter.Remotes.GitHub("nsiccha", "BayesianRegressionModels.jl"),
    modules  = [BayesianRegressionModels],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/BayesianRegressionModels.jl",
        devurl = "dev",
        devbranch = DOCS_DEVBRANCH,
        build_vitepress = false,
    ),
    pages = DOCS_PAGES,
    checkdocs = :exports,
    warnonly = false,
)

# Required panes must contain emitted Stan, never a backend-error placeholder.
for (page, names) in (
    ("index.md", (:intro_model,)),
    ("warfarin.md", (:warfarin_pk_brmi, :warfarin_pd_brmi, :warfarin_joint_brmi)),
    ("wastewater.md", (:cdc_ww_brm_model,)),
    ("monster.md", (:monster_brmi,)),
    ("jax.md", (:varying_slopes_model,)),
    ("variational-inference.md", (:site_counts_model,)),
)
    BRMDocsComparisons.validate_required_stan_outputs(
        joinpath(@__DIR__, "build", ".documenter", page), names)
end

# Keep local version metadata beside the generated downloads and vendor files.
let source = joinpath(@__DIR__, "src", "public"),
    destination = joinpath(@__DIR__, "build", ".documenter", "public")
    mkpath(destination)
    for name in readdir(source)
        cp(joinpath(source, name), joinpath(destination, name); force=true)
    end
end

# The Monster page links its generated Stan program and data for download;
# write both from the same reproduction source the page renders.
let mod = Module(:MonsterDownloads)
    Core.eval(mod, :(using BayesianRegressionModels, Distributions, StanBlocks))
    Base.include(mod, joinpath(@__DIR__, "..", "research", "monster", "reproduce.jl"))
    Base.invokelatest(getfield(mod, :write_monster_stan_files),
        joinpath(@__DIR__, "build", ".documenter", "public", "downloads"))
end

# Documenter validates the JAX module in its build directory before VitePress
# publishes the same generated file at the site's download URL.
let source = joinpath(@__DIR__, "build", ".documenter", "downloads", "jax"),
    destination = joinpath(@__DIR__, "build", ".documenter", "public", "downloads", "jax")
    mkpath(dirname(destination))
    cp(source, destination; force=true)
end

# Publish the pinned plotting runtime as build output, never as tracked assets.
let vendor = joinpath(@__DIR__, "build", ".documenter", "public", "vendor")
    mkpath(dirname(vendor))
    cp(BRMDocsComparisons.AlgebraOfVega.vega_vendor_dir(), vendor; force=true)
end

DocumenterVitepress.build_docs(joinpath(@__DIR__, "build"))

# The version picker requests this at the repository root. Deployment writes
# its real version list; a local preview uses the empty list from public/.
let versions = joinpath(@__DIR__, "build", "versions.js")
    isfile(versions) || cp(joinpath(@__DIR__, "src", "public", "versions.js"), versions)
end

let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head>
    <meta http-equiv="refresh" content="0; url=dev/">
    </head><body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/BayesianRegressionModels.jl",
    devbranch = DOCS_DEVBRANCH,
    push_preview = true,
)
