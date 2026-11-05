# Whole-value helpers read one value per plate cell, including every inner axis.
# Run: julia --project=test test/plate_output_helpers.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

cellpath(rain, a) = hcat(a .* rain, (a + 1) .* rain, (a + 2) .* rain)
function secondcolumn(paths, shift)
    paths isa AbstractVector{<:AbstractMatrix} ||
        error("expected one matrix per cell, got $(typeof(paths))")
    [shift[g] .+ paths[g][:, 2] for g in eachindex(paths)]
end
secondcolumn_kw(shift; paths) = secondcolumn(paths, shift)
column_total(paths, shift) = sum(sum, secondcolumn(paths, shift))

function matrix_output_builder(; side_output=false, reader_first=false,
        keyword=false, inline=false, data_only=false)
    read = keyword ? :(secondcolumn_kw(shift; paths=path)) : :(secondcolumn(path, shift))
    readers = inline ? [:(y ~ Normal(column_total(path, shift), 1.0))] :
        [:(columns = $read), :(y ~ Normal(sum($sum, columns), 1.0))]
    outputs = [:(path[g] = cellpath(rain[rain_index[g]], $(data_only ? :fixed_a : :a)[g]))]
    side_output && push!(outputs, :(extra[g] = a[g] + 1.0))
    writer = :(@plate for g in eachindex(rain_index)
        $(outputs...)
    end)
    body = reader_first ? [readers..., writer] : [writer, readers...]
    brm(quote
        a[1:2] .~ Normal(0.0, 1.0)
        shift[1:2] .~ Normal(0.0, 1.0)
        $(body...)
    end; mod=@__MODULE__)
end

const MATRIX_DATA = (; rain=[0.1, 0.2, 0.3, 0.4, 0.5],
    rain_index=[[1, 2, 3], [4, 5]], fixed_a=[0.3, -0.2], y=1.0)

function matrix_oracle(names, data; data_only=false)
    ia = data_only ? Int[] : [coordinate(names, Symbol("a.", g)) for g in 1:2]
    ishift = [coordinate(names, Symbol("shift.", g)) for g in 1:2]
    u -> begin
        a = data_only ? data.fixed_a : u[ia]
        # Independent column calculation, without the matrix-producing helper.
        location = sum(length(data.rain_index[g]) * u[ishift[g]] +
            (a[g] + 1) * sum(data.rain[data.rain_index[g]]) for g in 1:2)
        sum(logpdf.(Normal(), u)) + logpdf(Normal(location, 1), data.y)
    end
end

@stestset "RKBRMI matrix plate outputs in whole-value helpers" begin
    @test column_total([cellpath(MATRIX_DATA.rain[idx], MATRIX_DATA.fixed_a[g])
        for (g, idx) in enumerate(MATRIX_DATA.rain_index)], [0.1, 0.2]) ≈ 2.2
    for options in ((;), (; side_output=true, reader_first=true), (; keyword=true),
                    (; inline=true), (; data_only=true))
        builder = matrix_output_builder(; options...)
        backend, problem = consumer_problem(builder(MATRIX_DATA))
        names = coordinate_names(backend.model.layout)
        data_only = get(options, :data_only, false)
        oracle = matrix_oracle(names, MATRIX_DATA; data_only)
        # An unused `a ~ Normal` is an omitted response in the data-only case.
        points = data_only ? (zeros(2), [0.1, 0.2], [-0.3, 0.1]) :
            (zeros(4), [0.3, -0.2, 0.1, 0.2], [-0.4, 0.5, -0.3, 0.1])
        for u in points
            check_consumer_point(problem, u, oracle)
        end
        # Changed cell lengths, an empty matrix cell, and a new total value.
        data = merge(MATRIX_DATA, (; rain=[0.7, -0.2, 0.4], rain_index=[Int[], [3, 1, 2]]))
        replay = reprocess(backend, data)
        replay_problem = rk_logdensity_problem(replay; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        replay_oracle = matrix_oracle(coordinate_names(replay.model.layout), data;
            data_only=get(options, :data_only, false))
        check_consumer_point(replay_problem, points[2], replay_oracle)
    end
end

vectorcell(x, a) = a .* x
tensorcell(x, a) = reshape(vcat(a .* x, (a + 1) .* x), length(x), 1, 2)
firstentry_total(cells::AbstractVector{<:AbstractArray}) = sum(first, cells)

@stestset "RKBRMI whole-value helpers preserve vector and tensor cells" begin
    for (makecell, location) in ((:vectorcell, :(firstentry_total(out))),
                                 (:tensorcell, :(firstentry_total(out))),
                                 (:vectorcell, :(sum($first, out))))
        builder = brm(quote
            a ~ Normal(0.0, 1.0)
            @plate for g in eachindex(xs)
                out[g] = $makecell(xs[g], a)
            end
            y ~ Normal($location, 1.0)
        end; mod=@__MODULE__)
        data = (; xs=[[0.2, 0.7], [-0.4, 0.8, 0.9]], y=0.3)
        backend, problem = consumer_problem(builder(data))
        oracle(u) = logpdf(Normal(), only(u)) + logpdf(Normal(-0.2 * only(u), 1), data.y)
        for u in ([0.0], [0.6], [-0.7])
            check_consumer_point(problem, u, oracle)
        end
    end
end

@stestset "RKBRMI rowwise plate arithmetic keeps its flattened inputs" begin
    for location in (:out, :(out + 0.2))
        builder = brm(quote
            a ~ Normal(0.0, 1.0)
            @plate for g in eachindex(xs)
                out[g] = a .* xs[g]
            end
            y ~ Normal($location, 1.0)
        end; mod=@__MODULE__)
        data = (; xs=[[0.2, 0.7], [-0.4]], y=[0.1, -0.2, 0.4])
        backend, problem = consumer_problem(builder(data))
        offset = location === :out ? 0.0 : 0.2
        oracle(u) = logpdf(Normal(), only(u)) +
            sum(logpdf.(Normal.(only(u) .* [0.2, 0.7, -0.4] .+ offset, 1), data.y))
        for u in ([0.0], [0.6], [-0.7])
            check_consumer_point(problem, u, oracle)
        end
    end
end

# `addprop` is row-wise, so it reads a plate output's rows as arithmetic does:
# its law squares each location, which a whole cell cannot do. Assigned first or
# written in the law, with a shared or a per-row additive scale.
@stestset "RKBRMI addprop over a plate output keeps its flattened inputs" begin
    data = (; xs=[[0.2, 0.7], [-0.4]], w=[1.0, 2.0, 0.5], y=[0.1, -0.2, 0.4])
    rows = reduce(vcat, data.xs)
    for (assigned, add, additive) in ((true, :a, (a, j) -> a), (false, :a, (a, j) -> a),
                                      (true, :(a * w), (a, j) -> a * data.w[j]),
                                      (false, :(a * w), (a, j) -> a * data.w[j]))
        scale = :(addprop(out, $add, b))
        readers = assigned ? [:(scales = $scale), :(y ~ Normal(out, scales))] :
            [:(y ~ Normal(out, $scale))]
        builder = brm(quote
            m ~ Normal(0.0, 1.0)
            a ~ Exponential(1.0)
            b ~ Exponential(1.0)
            @plate for g in eachindex(xs)
                out[g] = m .* xs[g]
            end
            $(readers...)
        end; mod=@__MODULE__)
        backend, problem = consumer_problem(builder(data))
        names = coordinate_names(backend.model.layout)
        im, ia, ib = (coordinate(names, n) for n in (:m, :a, :b))
        oracle(u) = begin
            a, b = exp(u[ia]), exp(u[ib])
            # Independent scalar law; no BRM helper is called here.
            logpdf(Normal(), u[im]) - a - b + u[ia] + u[ib] + sum(eachindex(rows)) do j
                loc = u[im] * rows[j]
                logpdf(Normal(loc, sqrt(additive(a, j)^2 + (loc * b)^2)), data.y[j])
            end
        end
        for u in (zeros(3), [0.6, -0.3, 0.2], [-0.7, 0.4, -0.1])
            check_consumer_point(problem, u, oracle)
        end
    end
end
