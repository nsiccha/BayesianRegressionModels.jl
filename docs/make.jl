using Documenter, DocumenterVitepress, BayesianRegressionModels
import HTMXObjects

# Sync the canonical `htmxo-embed.ts` into our theme dir before
# DocumenterVitepress runs. The theme's `index.ts` imports from it.
HTMXObjects.vitepress_theme_install(joinpath(@__DIR__, "src", ".vitepress", "theme"))

makedocs(
    sitename = "BayesianRegressionModels.jl",
    modules  = [BayesianRegressionModels],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/JuliaBayes/BayesianRegressionModels.jl",
        devurl = "dev",
        devbranch = "dev",
        build_vitepress = false,
    ),
    pages = [
        "Home" => "index.md",
        "Gallery" => "gallery.md",
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
    repo = "github.com/JuliaBayes/BayesianRegressionModels.jl",
    devbranch = "dev",
    push_preview = true,
)
