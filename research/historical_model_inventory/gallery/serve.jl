#!/usr/bin/env julia

# Run from the repository root with the existing web runtime environment:
#   julia --project=web-macro research/historical_model_inventory/gallery/serve.jl [port] [host] [matrix]

include(joinpath(@__DIR__, "HistoricalInventoryGallery.jl"))
using .HistoricalInventoryGallery
using HTMXObjects

port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8127
host = length(ARGS) >= 2 ? ARGS[2] : "127.0.0.1"
matrix = length(ARGS) >= 3 ? ARGS[3] : DEFAULT_MATRIX_PATH

app = build_gallery(matrix)
route!(app)

println("Historical inventory: $(length(app.surface.rows)) rows from $(abspath(matrix))")
println("Serving on http://$host:$port")
serve(; host, port, async=false)
