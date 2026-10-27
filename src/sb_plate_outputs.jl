# A plate output read at model scope, as the Julia loop leaves it: after
# `@plate for s in … ; cnt[s] = v; end`, `cnt` holds one value per cell, so
# `cnt[g]` is cell `g`'s value, `cnt[idx]` the cells `idx`, and `sum(cnt)` /
# `sum(cnt[idx])` add the cells' values (elementwise for vector cells).
#
# StanBlocks stores the plate's cells in one of three shapes: a scalar per cell
# as `vector[N]`, a vector of a fixed width as `matrix[K, N]` (one column per
# cell), any other vector as a ragged value. Stan's `M[g]` of the matrix is a
# row and its `sum` of a matrix is the total of every entry, so a model-scope
# read of a plate output goes through these helpers, one method per stored
# shape; a ragged value already reads by group in StanBlocks. The scalar
# methods are inlined, so a scalar output keeps its Stan source (snag
# `sum-a-vector-pla-812e04d2`, decision `09e0gc5`).
StanBlocks.@deffun begin
    @inline brm_plate_sum(x::vector[n]) = sum(x)
    @inline brm_plate_sum(x::int[n]) = sum(x)
    @inline brm_plate_sum(x::matrix[k, n]) = x * rep_vector(1.0, cols(x))
    # StanBlocks sums a ragged value's groups elementwise and selects several
    # groups with an index array (StanBlocks 0.2.2, snag selecting-severa-027a8191).
    @inline brm_plate_sum(x::RaggedVector) = sum(x)
    @inline brm_plate_sum(x::vector[n], idx::int[m]) = sum(x[idx])
    @inline brm_plate_sum(x::int[n], idx::int[m]) = sum(x[idx])
    @inline brm_plate_sum(x::matrix[k, n], idx::int[m]) = x[:, idx] * rep_vector(1.0, size(idx))
    @inline brm_plate_sum(x::RaggedVector, idx::int[m]) = sum(x[idx])
    @inline brm_plate_cell(x::vector[n], i) = x[i]
    @inline brm_plate_cell(x::int[n], i) = x[i]
    @inline brm_plate_cell(x::matrix[k, n], i::int) = col(x, i)
    @inline brm_plate_cell(x::matrix[k, n], idx::int[m]) = x[:, idx]
    @inline brm_plate_cell(x::RaggedVector, i) = x[i]
end

# A plate output: a cell's collected one, or one of its other indexed outputs
# the model reads (`_brm_kernel_output_reads`; its Stan name is set afterwards).
_sb_plate_output(x, data) = _sb_kernel_producer(x) !== nothing ||
    x isa NamedColumn && parent(x) isa MissingColumn &&
    name(x) in get(data, _SB_KERNEL_OUTPUT_READS_KEY, ())

# The Stan call for a model-scope read of a plate output, or `nothing` when
# `x` is not one: `out[i]`, `sum(out)` and `sum(out[idx])`.
_sb_plate_output_read(x::ExprColumn, data) = _sb_plate_output_read(getf(x), x, data)
_sb_plate_output_read(_f, _x, _data) = nothing
function _sb_plate_output_read(::typeof(getindex), x, data)
    args = getargs(x)
    (length(args) == 2 && isempty(getkwargs(x)) &&
        _sb_plate_output(first(args), data)) || return nothing
    Expr(:call, brm_plate_cell, (_sb_scalar_expr(arg, data) for arg in args)...)
end
function _sb_plate_output_read(::typeof(sum), x, data)
    args = getargs(x)
    (length(args) == 1 && isempty(getkwargs(x))) || return nothing
    cells = _sb_plate_output_cells(only(args), data)
    cells === nothing && return nothing
    Expr(:call, brm_plate_sum, (_sb_scalar_expr(arg, data) for arg in cells)...)
end
# `(out,)` for a plate output `out`, `(out, idx)` for `out[idx]`.
_sb_plate_output_cells(x, data) = _sb_plate_output(x, data) ? (x,) : nothing
_sb_plate_output_cells(x::ExprColumn, data) = _sb_plate_output_cells(getf(x), x, data)
_sb_plate_output_cells(_f, _x, _data) = nothing
function _sb_plate_output_cells(::typeof(getindex), x, data)
    args = getargs(x)
    (length(args) == 2 && isempty(getkwargs(x)) &&
        _sb_plate_output(first(args), data)) || return nothing
    Tuple(args)
end
