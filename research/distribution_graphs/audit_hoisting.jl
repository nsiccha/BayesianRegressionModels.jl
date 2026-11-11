# Audit the saved public code_expr output; never evaluate or inspect the
# opaque captured objects printed within it. Run after hoisting.jl.
using Test, TOML

function source_walk(f, value)
    value isa QuoteNode && return
    f(value)
    value isa Expr || return
    value.head in (:quote, :$) && return
    foreach(x -> source_walk(f, x), value.args)
end

function source_symbols(value)
    names = Set{Symbol}()
    source_walk(value) do node
        node isa Symbol && push!(names, node)
    end
    names
end

function has_loop(value)
    found = false
    source_walk(value) do node
        node isa Expr && node.head in (:for, :while) && (found = true)
    end
    found
end

function audit(path, n)
    expression = Meta.parse(read(path, String))
    @test Meta.isexpr(expression, :quote)
    definition = only(expression.args)
    @test Meta.isexpr(definition, :function)
    body = last(definition.args)
    statements = filter(x -> !(x isa LineNumberNode), body.args)
    assignments = Dict{Symbol,Any}()
    positions = Dict{Symbol,Int}()
    for (i, statement) in enumerate(statements)
        if Meta.isexpr(statement, :(=)) && first(statement.args) isa Symbol
            name = first(statement.args)
            @test !haskey(assignments, name)
            assignments[name] = last(statement.args)
            positions[name] = i
        end
    end
    loop_positions = findall(has_loop, statements)
    # RK emits one size-generic program, so an empty plate keeps the moment
    # assignments and the (zero-trip) loop; the same placement checks apply.
    @test !isempty(loop_positions)
    rows = []
    for endpoint in (:m_cell, :v_cell)
        @test haskey(assignments, endpoint)
        @test positions[endpoint] < minimum(loop_positions)
        @test !has_loop(assignments[endpoint])
        occurrences = 0
        source_walk(body) do node
            Meta.isexpr(node, :(=)) && first(node.args) === endpoint && (occurrences += 1)
        end
        @test occurrences == 1
        # Follow public source assignments to prove these were not frozen
        # when the data was bound. All three model parameters remain live.
        dependencies = source_symbols(assignments[endpoint])
        pending = collect(dependencies)
        while !isempty(pending)
            name = pop!(pending)
            haskey(assignments, name) || continue
            for dependency in source_symbols(assignments[name])
                dependency in dependencies && continue
                push!(dependencies, dependency)
                push!(pending, dependency)
            end
        end
        @test all(in(dependencies), (:mu, :log_sigma, :log_bound, :unconstrained))
        push!(rows, Dict("endpoint"=>string(endpoint), "assignment_position"=>positions[endpoint],
            "first_loop_position"=>minimum(loop_positions), "evaluations_per_call"=>1,
            "active_inputs"=>["mu", "log_sigma", "log_bound"]))
    end
    Dict("file"=>path, "n"=>n, "endpoints"=>rows)
end

length(ARGS) == 1 || error("provide the hoisting output directory")
output = only(ARGS)
rows = []
@testset "public emitted moment placement and active inputs" begin
    for side in ("lower", "upper"), n in (0, 1, 64, 1024)
        push!(rows, audit(joinpath(output, "$side-owned-$n.jl"), n))
    end
end
open(joinpath(output, "source-audit.toml"), "w") do io
    TOML.print(io, Dict("rows"=>rows))
end
