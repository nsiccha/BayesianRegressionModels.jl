using Documenter, DocumenterVitepress, BayesianRegressionModels

# Which GitHub repository these docs are built FOR. Documenter's
# `GitHubActions` deploy config requires `ENV["GITHUB_REPOSITORY"]` to occur in
# `deploydocs(repo=…)` — it is a hard deployment criterion, not a hint — so a
# workflow can only ever publish to its OWN repository's `gh-pages`. Reading
# the environment therefore lets one `make.jl` serve both homes:
#
#   * `nsiccha/BayesianRegressionModels.jl`  -> nsiccha.github.io/…   (current)
#   * `JuliaBayes/BayesianRegressionModels.jl` -> juliabayes.github.io/… (legacy)
#
# whichever runs it, with no per-fork edit and no cross-repo token. Locally the
# fallback picks the current home, which is what edit links should point at.
const DOCS_REPO = get(ENV, "GITHUB_REPOSITORY", "nsiccha/BayesianRegressionModels.jl")

# The branch whose build lands under `dev/`. `ns/devibe` is the active line
# (dev §11.5); the legacy JuliaBayes home only ever published from `main`.
const DOCS_DEVBRANCH = startswith(DOCS_REPO, "JuliaBayes/") ? "main" : "ns/devibe"

# Note: BRM's docs must build without the private HTMXObjects.jl repo — the
# legacy JuliaBayes home could not clone it from CI at all. So `htmxo-embed.ts`
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
