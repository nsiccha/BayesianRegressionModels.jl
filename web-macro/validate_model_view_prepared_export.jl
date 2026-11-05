#!/usr/bin/env julia

# Run after validate_model_view_export.jl, with the same output directory,
# in an environment containing ReactiveKernels and ReactiveKernelsPPL.
# Preparation is explicit here; viewing/exporting the supplied objects does
# not prepare, fit, or evaluate their helpers.
using Test, BayesianRegressionModels, HTMXObjects
using ReactiveKernels, ReactiveKernelsPPL

module PreparedExportFixture
using BayesianRegressionModels
const inspecting = Ref(false)
function cell_total(x)
    inspecting[] && error("inspection evaluated cell_total")
    sum(x)
end
default_scale() = error("inspection evaluated a component default")

cells = @brm begin
    sigma ~ Exponential(1)
    @plate for i in eachindex(total)
        latent ~ Normal(loc[i], 0.5)
        center = cell_total(latent)
        total[i] ~ Normal(center, sigma)
    end
end
simplex = @brm begin
    p ~ Dirichlet([1., 2., 3.])
    sigma ~ Exponential(1)
    mu ~ 1 + x
    y ~ Normal(mu + sum(p), sigma)
end
@brm function optional_inputs(x, shift=2; scale=default_scale())
    beta ~ Normal(0, scale)
    return x + beta + shift
end
end

@htmx struct PreparedModelViewExportPage
    rk_view
    descriptor_view
    component_view
    __page__(content) = HTMXObjects.pico_page(content; assets=:vendor, runtime=:linked,
        lang="en-GB", title="Prepared model inspection")
    @get rk() = rk_view
    @get descriptor() = descriptor_view
    @get component() = component_view
end

let output = abspath(only(ARGS))
rk = RKBRMI(PreparedExportFixture.cells((;
    loc=[[0., 1.], Float64[], [2.]], total=[0.1, 0.2, 0.3])))
descriptor = brm_descriptor(PreparedExportFixture.simplex,
    (; x=[-1., 0., 1.], y=[0.1, 0.2, 0.3]))
PreparedExportFixture.inspecting[] = true
try
    page = PreparedModelViewExportPage(;
        rk_view=brm_model_view(rk; title="RK model with ragged parameters"),
        descriptor_view=brm_model_view(descriptor; title="Regression with simplex weights"),
        component_view=brm_model_view(PreparedExportFixture.optional_inputs; title="Component with defaults"))
    route!(page)
    vendorfiles()
    result = export_site(joinpath(output, "site", "prepared");
        pages=["/rk", "/descriptor", "/component"], parallel=false)
    @testset "prepared model static export" begin
        for (name, required) in (("rk", "Element shapes"),
                ("descriptor", "Carrier details"), ("component", "Optional"))
            html = read(joinpath(output, "site", "prepared", name * ".html"), String)
            @test occursin(required, html)
            @test occursin("role=\"tab\"", html)
            markdown = String(dispatch(:GET, "/" * name;
                headers=["Accept" => "text/markdown"]).body)
            @test occursin(required, markdown)
            write(joinpath(output, name * ".md"), markdown)
        end
    end
    println("PREPARED_MODEL_VIEW_EXPORT_COMPLETE result=", result)
finally
    PreparedExportFixture.inspecting[] = false
end
end
