# The caller owns this joint law. RK's extension supplies its visible density
# graph and identity geometry; neither changes the vertex/channel/time law.
struct _BRMGMRFLaw{S,T,L,G}
    spatial_length::S
    temporal_length::T
    correlation_factor::L
    geometry::G
end

# Symbolic sparse lookup only: no sampled values enter this operation. Zero
# denotes an absent entry. Keeping the lookup avoids storing every pair in
# every sparse elimination dot product (which can exceed the factor storage).
function _brm_gmrf_sparse_slot(rows, pointers, column, row)
    first, last = pointers[column], pointers[column + 1] - 1
    while first <= last
        middle = first + (last - first) ÷ 2
        candidate = rows[middle]
        candidate == row && return middle
        if candidate < row
            first = middle + 1
        else
            last = middle - 1
        end
    end
    0
end

# This is a data-only recipe of the emitted model, evaluated from its raw mesh
# and fitted axes. The host formula planner must not supply a numerical factor.
function brm_gmrf_geometry(vertices, cells, times, nchannels)
    mesh = _brm_gmrf_mesh(vertices, cells)
    layout = _BRMGMRFStateLayout(size(vertices, 1), collect(1:nchannels), times)
    factor = _brm_gmrf_factor_data(mesh)
    columns = Vector{Int}(undef, length(factor.lrow))
    stiffness = Vector{Float64}(undef, length(factor.lrow))
    for j in eachindex(factor.mass), p in factor.lptr[j]:(factor.lptr[j + 1] - 1)
        columns[p] = j
        source = _brm_gmrf_sparse_slot(factor.grow, factor.gptr, j, factor.lrow[p])
        stiffness[p] = source == 0 ? 0.0 : factor.gval[source]
    end
    (; factor..., columns, stiffness, times=layout.times,
       nchannels, ncoordinates=layout.ncoordinates)
end
