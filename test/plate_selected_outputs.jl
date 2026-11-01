# Selected cells retain their inner axes across distinct producer/reader plates.
# Run: julia --project=test test/plate_selected_outputs.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@stestset "selected output rewriting respects cell bindings" begin
    select(name, index) = Expr(:ref, Symbol(name, :_cells), index)
    lower(ex) = BRM._brm_plate_cell_refs(select, ex, Set((:path,)))
    @test lower(:(path[which[g]][hours[g]])) == :(path_cells[which[g]][hours[g]])
    @test lower(:(path -> path[1])) == :(path -> path[1])
    local_cell = :(x -> begin
        path = x
        path[1]
    end)
    @test isequal(lower(local_cell), local_cell)
    @test lower(:(sum(path[1] for path in catalog))) ==
        :(sum(path[1] for path in catalog))
    @test lower(:(sum(path[k] for k in positions))) ==
        :(sum(path_cells[k] for k in positions))
    @test lower(:(sum(path[k] for path in catalog for k in path[1]))) ==
        :(sum(path[k] for path in catalog for k in path[1]))
    # Selection rewrites value positions, leaving a field/keyword named path
    # and the caller's original source intact.
    source = :(f(object.path; path=path[1]))
    before = deepcopy(source)
    @test lower(source) == :(f(object.path; path=path_cells[1]))
    @test isequal(source, before)
end

function selected_output_builder(; reverse_outputs=false)
    outputs = Any[:(background[p] = 2.0 .* paths[p]),
                  :(live[p] = a .* paths[p])]
    reverse_outputs && reverse!(outputs)
    # Reader-first composition must discover both collected and side-output
    # dependencies; a capture need not have its own operation entry.
    brm(quote
        @plate for g in eachindex(y)
            selected[g] = background[trajectory[g]][hours[g]]
            selected_live[g] = live[trajectory[g]][hours[g]]
            y[g] ~ Normal(selected[g] .+ selected_live[g], 0.6)
        end
        @plate for p in eachindex(paths)
            $(outputs...)
        end
        a ~ Normal(0, 1)
    end; mod=@__MODULE__)
end

@stestset "selected shared plate outputs retain each cell's vector" begin
    for (label, paths) in (("ragged", [[0.2, 0.7, -0.4], [1.3, -0.1]]),
                           ("fixed", [[0.2, 0.7, -0.4], [1.3, -0.1, 0.8]])),
            reverse_outputs in (false, true)
        data = (; paths, trajectory=[2, 1, 2],
            hours=[[2, 1], [3], [1]], y=[[0.0, 0.2], [0.1], [0.4]])
        before = deepcopy(data)
        model = selected_output_builder(; reverse_outputs)(data)
        rk, problem = consumer_problem(model)
        stan = consumer_stan(model, "selected-outputs-$label-$reverse_outputs")
        oracle(u) = logpdf(Normal(0, 1), only(u)) + sum(
            sum(logpdf.(Normal.((2 + only(u)) .* paths[data.trajectory[g]][data.hours[g]],
                0.6), data.y[g])) for g in eachindex(data.y))
        for u in ([0.0], [-0.3], [0.7])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, [:a => "a"], rk, u)
        end
        @test isequal(data, before)
    end
end
