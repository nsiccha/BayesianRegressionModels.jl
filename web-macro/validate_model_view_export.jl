#!/usr/bin/env julia

# Export acceptance with public synthetic data. No listener or fit is needed.
# julia --project=web-macro web-macro/validate_model_view_export.jl <output-dir>
# The output directory must be new or empty. It contains model.pdf, model.md
# and a self-contained site/ bundle suitable for a static server.

using Test, BayesianRegressionModels, HTMXObjects

module ExportFixture
using BayesianRegressionModels

"""Center the toy predictor. Inspecting or exporting the model must not call it."""
center_predictor(x) = error("export evaluated center_predictor")

"""Reusable centered curve; each call has its own intercept."""
@brm function centered_curve(x)
    """Local intercept of the centered curve."""
    intercept ~ Normal(0, 1)
    return intercept + center_predictor(x)
end

"""Nested curve definition, retained in every complete export."""
@brm function shifted_curve(x)
    inner ~ centered_curve(x)
    shift ~ Normal(0, 0.5)
    return inner + shift
end

builder = @brm begin
    """Residual **response scale** for the synthetic regression."""
    sigma ~ Exponential(1)
    beta ~ Normal(0, 1)
    left ~ shifted_curve(x)
    right ~ shifted_curve(x)
    mu = beta * (left + right)
    """Response likelihood. This authored prose must survive HTML and PDF export."""
    y ~ Normal(mu, sigma)
end
model = builder((; x=[-1., 0., 1.], y=[0.1, 0.2, 0.3]))
end

@htmx struct ModelViewExportPage
    model = ExportFixture.model
    view = brm_model_view(model; title="Regression model")
    __page__(content) = HTMXObjects.pico_page(content; assets=:vendor, runtime=:linked,
        lang="en-GB", title="Regression model")
    @get index() = HTMXObjects.h.div(view,
        HTMXObjects.h.p(HTMXObjects.h.a("Download the complete model PDF";
            href=__self__/"files/model.pdf")))
end

output = isempty(ARGS) ? mktempdir() : abspath(only(ARGS))
isdir(output) && !isempty(readdir(output)) &&
    error("Export acceptance requires a new or empty output directory: $output")
mkpath(output)
page = ModelViewExportPage()
view = page.view
markdown = repr("text/markdown", view)
write(joinpath(output, "model.md"), markdown)
pdf_report(view, joinpath(output, "model.pdf"); title="Regression model",
    lang="en-GB", metadata=(toc=true, fontsize="10pt"))

route!(page)
route_markdown = String(dispatch(:GET, "/"; headers=["Accept" => "text/markdown"]).body)
route_plain = String(dispatch(:GET, "/"; headers=["Accept" => "text/plain"]).body)
vendorfiles()
staticfiles(output, "files")
site = joinpath(output, "site")
result = export_site(site; pages=["/"], parallel=false)

@testset "model view PDF and static HTML export" begin
    @test occursin("Model quantities", markdown)
    @test occursin("Center the toy predictor", markdown)
    @test occursin("Response likelihood. This authored prose", markdown)
    @test occursin("Authored documentation", route_markdown)
    @test occursin("Model quantities", route_plain)
    for source in (markdown, route_markdown, route_plain)
        @test occursin("function centered_curve(x)", source)
        @test occursin("function shifted_curve(x)", source)
        @test occursin("(:left, :inner)", source)
        @test occursin("(:right, :inner)", source)
        @test occursin("Local intercept of the centered curve", source)
        @test !occursin("Core.@doc", source)
    end
    pdf = read(joinpath(output, "model.pdf"))
    @test startswith(String(pdf[1:5]), "%PDF-")
    @test read(joinpath(site, "files", "model.pdf")) == pdf
    html = read(joinpath(site, "index.html"), String)
    @test occursin("Center the toy predictor", html)
    @test occursin("<details", html)
    @test occursin("files/model.pdf", html)
    @test isdir(joinpath(site, "vendor"))
end
println("MODEL_VIEW_EXPORT_COMPLETE output=", output, " result=", result)
