#!/usr/bin/env julia

# Offline validation: no listener is opened and no external service is used.

using Test
using DynamicObjects
using HTMXObjects

include(joinpath(@__DIR__, "HistoricalInventoryGallery.jl"))
using .HistoricalInventoryGallery

@testset "historical inventory gallery" begin
    graph = build_gallery()
    surface = graph.surface
    physical_rows = length(readlines(DEFAULT_MATRIX_PATH)) - 1

    @test length(surface.rows) == physical_rows == 359
    @test length(unique(row["row_key"] for row in surface.rows)) == physical_rows
    @test length(filtered_rows(graph)) == physical_rows

    for property in (
        :source_fidelity,
        :family_provenance,
        :translation_route,
        :validation_tier,
    )
        domain = property_options(surface, property)
        records = option_records(domain)
        @test records !== nothing
        @test first(records).value == "all"
        @test first(records).label == "All (359)"
        @test getproperty(surface, property) in domain
    end

    @test Set(getproperty.(option_records(
        property_options(surface, :translation_route)), :value)) ==
        Set(["all", "ordinary_brm", "stanblocks_plate", "brm_kernel"])
    @test Set(getproperty.(option_records(
        property_options(surface, :validation_tier)), :value)) ==
        union(Set(["all"]), Set(row["inferred_capability_tier"] for row in surface.rows))

    confirmed = HistoricalInventory(;
        matrix_path=DEFAULT_MATRIX_PATH, source_fidelity="confirmed")
    @test length(filtered_rows(confirmed)) == 154
    exact = HistoricalInventory(;
        matrix_path=DEFAULT_MATRIX_PATH,
        validation_tier="bridgestan-finite-density-gradient")
    @test length(filtered_rows(exact)) == count(
        row -> row["inferred_capability_tier"] == "bridgestan-finite-density-gradient",
        surface.rows,
    )
    combined = HistoricalInventory(;
        matrix_path=DEFAULT_MATRIX_PATH,
        translation_route="ordinary_brm", validation_tier="unsupported")
    @test all(row["semantic_route"] == "ordinary_brm" &&
              row["inferred_capability_tier"] == "unsupported"
              for row in filtered_rows(combined))

    cards = [row_card(row) for row in surface.rows]
    @test length(cards) == physical_rows
    @test all(card -> card isa SemanticCard, cards)
    @test all(card -> any(child -> child isa SemanticFields, card.children), cards)
    @test all(card -> any(child -> child isa SemanticStatus, card.children), cards)
    @test all(card -> any(child -> child isa SemanticDisclosure, card.children), cards)

    unresolved_indices = findall(row ->
        row["inferred_translation_status"] != "ready", surface.rows)
    @test length(unresolved_indices) == count(
        row -> row["inferred_translation_status"] != "ready", surface.rows)
    @test all(index -> any(child -> child isa SemanticUnavailable,
                          cards[index].children), unresolved_indices)

    first_card_html = repr("text/html", first(cards))
    first_card_md = repr("text/markdown", first(cards))
    @test occursin(first(surface.rows)["row_key"], first_card_html)
    @test occursin(first(surface.rows)["row_key"], first_card_md)
    @test occursin("experimental", lowercase(first_card_md))
    @test occursin("```julia", first_card_md)

    inventory = semantic_inventory(surface.rows, length(surface.rows))
    inventory_md = repr("text/markdown", inventory)
    @test all(row -> occursin(row["row_key"], inventory_md), surface.rows)
    @test length(findall("### ", inventory_md)) == physical_rows

    semantic_html = repr("text/html", semantic_app(
        surface;
        title="Historical Bayesian regression models",
        render_operation=HistoricalInventoryGallery._gallery_semantic_operation,
    ))
    for name in ("source_fidelity", "family_provenance",
                 "translation_route", "validation_tier")
        @test occursin("name=\"$name\"", semantic_html)
    end
    @test occursin("hx-get=\"/surface/\"", semantic_html)
    @test !occursin("AppData", inventory_md)
    @test !occursin("AppContext", inventory_md)

    # Exercise the registered HTTP route entirely in process. This opens no
    # socket, but proves that the page wrapper retains every semantic card and
    # that query context reaches the same authoritative graph.
    route!(graph)
    drive(path) = begin
        request = HTTP.Request("GET", path)
        handler = first(HTTP.Handlers.gethandler(
            HTMXObjects.CONTEXT[].service.router, request))
        handler(request)
    end
    try
        full = drive("/")
        full_body = String(full.body)
        @test full.status == 200
        @test occursin("text/html", HTTP.header(full, "Content-Type", ""))
        @test length(findall(
            "<article class=\"htmxo-semantic-card-body\"", full_body)) == 359
        @test all(name -> occursin("name=\"$name\"", full_body),
                  ("source_fidelity", "family_provenance",
                   "translation_route", "validation_tier"))

        fidelity = drive("/surface/?source_fidelity=confirmed&family_provenance=all" *
                         "&translation_route=all&validation_tier=all")
        fidelity_body = String(fidelity.body)
        @test fidelity.status == 200
        @test length(findall(
            "<article class=\"htmxo-semantic-card-body\"",
            fidelity_body)) == 154
    finally
        terminate()
    end
end

println("offline gallery validation: ok")
