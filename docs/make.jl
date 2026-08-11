using Documenter, DocumenterVitepress, BayesianRegressionModels

# BRM documentation is published only from the current nsiccha home. Keep the
# target literal: Documenter's GitHub Actions deploy config requires the
# workflow repository to match `repo`, so a legacy JuliaBayes mirror cannot
# accidentally update a second `gh-pages` site.
const DOCS_DEVBRANCH = "ns/devibe"

# Keep the docs build independent of the private HTMXObjects.jl repository:
# `htmxo-embed.ts` is committed in-tree rather than fetched in CI. Local edits
# to the canonical upstream file should be hand-copied; see /docs-vitepress §8.
@static if Base.find_package("HTMXObjects") !== nothing
    @eval import HTMXObjects
    @eval HTMXObjects.vitepress_theme_install(joinpath(@__DIR__, "src", ".vitepress", "theme"))
end

makedocs(
    sitename = "BayesianRegressionModels.jl",
    modules  = [BayesianRegressionModels],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/BayesianRegressionModels.jl",
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
    repo = "github.com/nsiccha/BayesianRegressionModels.jl",
    devbranch = DOCS_DEVBRANCH,
    push_preview = true,
)
