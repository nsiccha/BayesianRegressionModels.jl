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
    # A downstream plate iterates producer cells, not the storage's first
    # dimension. Fixed-width vector cells occupy matrix columns; all other
    # carriers already iterate one value per cell. EachCol is a compiler view.
    @inline brm_plate_cells(x) = x
    @inline brm_plate_cells(x::matrix[k, n]) = EachCol(x)
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
    # Several selected vector cells remain a collection at helper boundaries.
    @inline brm_plate_cell(x::matrix[k, n], idx::int[m]) = EachCol(x[:, idx])
    @inline brm_plate_cell(x::RaggedVector, i) = x[i]
    @inline brm_plate_cell(x::NestedRaggedVector, i) = x[i]

    # Preserve the member axis when concatenating declaration variants.
    # A fixed-width collection stores columns, a ragged one stores flat
    # memory and cumulative ends. Different variants may have either shape.
    brm_column_ends(k::int, n::int)::int[n] = begin
        ends = rep_array(0, n)
        for i in 1:n
            ends[i] = i*k
        end
        ends
    end
    brm_shifted_ends(x::int[n], shift::int)::int[n] = begin
        ends = rep_array(0, n)
        for i in 1:n
            ends[i] = x[i] + shift
        end
        ends
    end
    # The length of a ragged value's memory, read from its ends: a size that is
    # data when the ends are. `num_elements(mem)` of a model value is not, and
    # Stan refuses it in a top-level size declaration.
    brm_ends_total(ends::int[n])::int = begin
        total = 0
        if n > 0
            total = ends[n]
        end
        total
    end
    @inline brm_member_vectors(x::matrix[k, n]) =
        RaggedVector(to_vector(x), brm_column_ends(k, n))
    @inline brm_member_vectors(x::RaggedVector) = x
    @inline brm_concat_members(a::vector[n], b::vector[m]) = append_row(a, b)
    @inline brm_concat_members(a::int[n], b::int[m]) = append_array(a, b)
    @inline brm_concat_members(a::RaggedVector, b::RaggedVector) =
        RaggedVector(append_row(a.mem, b.mem),
            append_array(a.ends, brm_shifted_ends(b.ends, brm_ends_total(a.ends))))
    # A member returning a subject plate has two sets of cumulative ends.
    # Shift reading ends by the left reading count and group ends by its subject count.
    @inline brm_concat_members(a::NestedRaggedVector, b::NestedRaggedVector) =
        NestedRaggedVector(append_row(a.mem, b.mem),
            append_array(a.ends, brm_shifted_ends(b.ends, brm_ends_total(a.ends))),
            append_array(a.groups, brm_shifted_ends(b.groups, size(a.ends))))
    @inline brm_concat_members(a::matrix[k,n], b::matrix[j,m]) =
        brm_concat_members(brm_member_vectors(a), brm_member_vectors(b))
    @inline brm_concat_members(a::matrix[k,n], b::RaggedVector) =
        brm_concat_members(brm_member_vectors(a), b)
    @inline brm_concat_members(a::RaggedVector, b::matrix[k,n]) =
        brm_concat_members(a, brm_member_vectors(b))
    brm_plate_row_pairs(x::matrix[k, n], members::int[m], rows::int[m])::vector[m] = begin
        rv = rep_vector(0.0, m)
        for i in 1:m
            rv[i] = x[rows[i], members[i]]
        end
        rv
    end
    brm_plate_row_pairs(x::RaggedVector, members::int[m], rows::int[m])::vector[m] = begin
        rv = rep_vector(0.0, m)
        for i in 1:m
            rv[i] = x[members[i]][rows[i]]
        end
        rv
    end
end

_sb_julia_stan_call(::typeof(_brm_concat_members), args) = Expr(:call, brm_concat_members, args...)
_sb_julia_stan_call(::typeof(_brm_observed_member_rows), args) =
    Expr(:call, brm_plate_row_pairs, args...)

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
