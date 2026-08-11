using Documenter, DocumenterVitepress, BayesianRegressionModels

# The docs have one publication home. Hard-coding it is deliberate:
# Documenter's GitHub Actions deploy criterion will refuse to publish a build
# running in a different fork, so an accidental JuliaBayes workflow run cannot
# create a second, drifting docs site.
const DOCS_REPO = "nsiccha/BayesianRegressionModels.jl"
const DOCS_DEVBRANCH = "ns/devibe"

# Note: BRM's docs must build without the private HTMXObjects.jl repo — the
# CI cannot clone it without repository-specific credentials. So `htmxo-embed.ts`
# is committed in-tree (see docs/src/.vitepress/theme/) rather than synced from
# HTMXObjects via `HTMXObjects.vitepress_theme_install` at build time. Local
# edits to the canonical upstream file should be hand-copied; see
# /docs-vitepress §8.
@static if Base.find_package("HTMXObjects") !== nothing
    @eval import HTMXObjects
    @eval HTMXObjects.vitepress_theme_install(joinpath(@__DIR__, "src", ".vitepress", "theme"))
end

makedocs(
    sitename = "BayesianRegressionModels.jl",
    modules  = [BayesianRegressionModels],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/$DOCS_REPO",
        devurl = "dev",
        devbranch = DOCS_DEVBRANCH,
        build_vitepress = false,
    ),
    pages = [
        "Home" => "index.md",
        "Formula terms" => "formula-terms.md",
        "Likelihoods" => "likelihoods.md",
        "Turing backend" => "turing-backend.md",
        "Gallery" => "gallery.md",
        "Complete-PLATE blueprint" => "plate-building-blocks.md",
        "API" => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

# Copy committed live-brm recordings into the VitePress build tree so
# the gallery embed resolves the same URLs in production as in dev
# (where vite proxies /live-brm to the running app on :8121).
let src = joinpath(@__DIR__, "src", "public")
    dst = joinpath(@__DIR__, "build", ".documenter", "public")
    if isdir(src)
        cp(src, dst; force=true)
        @info "Copied public assets to $dst"
    end
end

DocumenterVitepress.build_docs(joinpath(@__DIR__, "build"))

let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head>
    <meta http-equiv="refresh" content="0; url=dev/">
    </head><body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/$DOCS_REPO",
    devbranch = DOCS_DEVBRANCH,
    push_preview = true,
)
