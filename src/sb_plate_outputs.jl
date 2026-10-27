# A plate output read at model scope, as the Julia loop leaves it: after
# `@plate for s in … ; cnt[s] = v; end`, `cnt` holds one value per cell, so
# `cnt[g]` is cell `g`'s value, `cnt[idx]` the cells `idx`, and `sum(cnt)` /
# `sum(cnt[idx])` add the cells' values (elementwise for vector cells).
#
# StanBlocks stores the plate's cells in one of three shapes: a scalar per cell
# as `vector[N]`, a vector of a fixed width as `matrix[K, N]` (one column per
# cell), any other vector as a ragged value. Its own `M[g]` of the matrix is a
# row and its `sum` of a matrix or a ragged value is the total of every entry,
# so a model-scope read of a plate output goes through these helpers, one
# method per stored shape. The scalar methods are inlined, so a scalar output
# keeps its Stan source (snag `sum-a-vector-pla-812e04d2`, decision `09e0gc5`).
StanBlocks.@deffun begin
    brm_ragged_cells_sum(mem::vector[k], ends::int[n])::vector[ragged_length(ends, 1)] = begin
        @stan_assert n > 0 "sum of a plate output: the plate has no cells"
        total = mem[ragged_start(ends, 1):ragged_end(ends, 1)]
        for g in 2:n
            @stan_assert ragged_length(ends, g) == ragged_length(ends, 1) "sum of a plate output adds its cells elementwise, but the cells have different lengths"
            total = total + mem[ragged_start(ends, g):ragged_end(ends, g)]
        end
        total
    end
    brm_ragged_cells_sum(mem::vector[k], ends::int[n], idx::int[m])::vector[ragged_length(ends, idx[1])] = begin
        @stan_assert m > 0 "sum of a plate output's cells: no cells selected"
        total = mem[ragged_start(ends, idx[1]):ragged_end(ends, idx[1])]
        for j in 2:m
            @stan_assert ragged_length(ends, idx[j]) == ragged_length(ends, idx[1]) "sum of a plate output adds its cells elementwise, but the cells have different lengths"
            total = total + mem[ragged_start(ends, idx[j]):ragged_end(ends, idx[j])]
        end
        total
    end
end

# An inlined body is traced where it is called, so it reaches the ragged
# helper by value.
@eval StanBlocks.@deffun begin
    @inline brm_plate_sum(x::vector[n]) = sum(x)
    @inline brm_plate_sum(x::int[n]) = sum(x)
    @inline brm_plate_sum(x::matrix[k, n]) = x * rep_vector(1.0, cols(x))
    # A ragged value goes in as its memory and its (data) ends, so that the
    # caller's declaration is sized by the ends alone; a size naming the whole
    # ragged value would name its parameter memory, which stanc refuses.
    @inline brm_plate_sum(x::RaggedVector) = $(brm_ragged_cells_sum)(x.mem, x.ends)
    @inline brm_plate_sum(x::vector[n], idx::int[m]) = sum(x[idx])
    @inline brm_plate_sum(x::int[n], idx::int[m]) = sum(x[idx])
    @inline brm_plate_sum(x::matrix[k, n], idx::int[m]) = x[:, idx] * rep_vector(1.0, size(idx))
    @inline brm_plate_sum(x::RaggedVector, idx::int[m]) =
        $(brm_ragged_cells_sum)(x.mem, x.ends, idx)
    @inline brm_plate_cell(x::vector[n], i) = x[i]
    @inline brm_plate_cell(x::int[n], i) = x[i]
    @inline brm_plate_cell(x::matrix[k, n], i::int) = col(x, i)
    @inline brm_plate_cell(x::matrix[k, n], idx::int[m]) = x[:, idx]
    @inline brm_plate_cell(x::RaggedVector, i) = x[i]
end

# The Stan call for a model-scope read of a plate output, or `nothing` when
# `x` is not one: `out[i]`, `sum(out)` and `sum(out[idx])`.
_sb_plate_output_read(x::ExprColumn, data) = _sb_plate_output_read(getf(x), x, data)
_sb_plate_output_read(_f, _x, _data) = nothing
function _sb_plate_output_read(::typeof(getindex), x, data)
    args = getargs(x)
    (length(args) == 2 && isempty(getkwargs(x)) &&
        _sb_kernel_producer(first(args)) !== nothing) || return nothing
    Expr(:call, brm_plate_cell, (_sb_scalar_expr(arg, data) for arg in args)...)
end
function _sb_plate_output_read(::typeof(sum), x, data)
    args = getargs(x)
    (length(args) == 1 && isempty(getkwargs(x))) || return nothing
    cells = _sb_plate_output_cells(only(args))
    cells === nothing && return nothing
    Expr(:call, brm_plate_sum, (_sb_scalar_expr(arg, data) for arg in cells)...)
end
# `(out,)` for a plate output `out`, `(out, idx)` for `out[idx]`.
_sb_plate_output_cells(x) = _sb_kernel_producer(x) === nothing ? nothing : (x,)
_sb_plate_output_cells(x::ExprColumn) = _sb_plate_output_cells(getf(x), x)
_sb_plate_output_cells(_f, _x) = nothing
function _sb_plate_output_cells(::typeof(getindex), x)
    args = getargs(x)
    (length(args) == 2 && isempty(getkwargs(x)) &&
        _sb_kernel_producer(first(args)) !== nothing) || return nothing
    Tuple(args)
end
