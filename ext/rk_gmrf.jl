# A sparse LDL recurrence over stored entries. A diagonal history entry is D;
# an off-diagonal entry is L. Every learned-length operation stays in the
# retained scan. Only the sparse integer lookup is an ordinary data function.
ReactiveKernels.@kernel function _gmrf_factor_graph(kappa2, geometry)
    factor = scan(eachindex(geometry.lrow); init=0, history=0.0) do carry, p, earlier
        i = geometry.lrow[p]
        j = geometry.columns[p]
        corrections = plate(geometry.previous_ptr[j]:(geometry.previous_ptr[j + 1] - 1)) do q
            k = geometry.previous_col[q]
            slot = BRM._brm_gmrf_sparse_slot(geometry.lrow, geometry.lptr, k, i)
            value = slot == 0 ? 0.0 : earlier[slot] *
                earlier[geometry.previous_pos[q]] * earlier[geometry.lptr[k]]
            return value
        end
        raw = geometry.stiffness[p] + (i == j ? kappa2 * geometry.mass[j] : 0.0)
        numerator = raw - sum(corrections)
        entry = i == j ? numerator : numerator / earlier[geometry.lptr[j]]
        (carry, entry)
    end
    return factor
end

# K = kappa² C + G commutes with the channel operation. Apply it before the
# small channel solve, so each vertex/time cell needs only a C-entry history.
# No dense spatial matrix or space-time factor is constructed.
ReactiveKernels.@kernel function _gmrf_density_graph(field, spatial_length,
        temporal_length, L, geometry)
    N = length(geometry.mass)
    C = geometry.nchannels
    T = length(geometry.times)
    kappa2 = 2.0 / spatial_length^2
    factor = _gmrf_factor_graph(kappa2, geometry)
    logdet_space = 2.0 * sum(log.(factor[geometry.lptr[1:N]])) -
        sum(log.(geometry.mass)) - N * log(4.0 * pi * kappa2)
    channel_logs = plate(1:C) do c
        return log(L[c, c])
    end
    normalizer = T * C * logdet_space / 2.0 -
        N * C * T * log(2.0 * pi) / 2.0 - N * T * sum(channel_logs)
    transitions = plate(1:T) do t
        gap = t == 1 ? 0.0 : geometry.times[t] - geometry.times[t - 1]
        decay = t == 1 ? 0.0 : exp(-gap / temporal_length)
        variance = t == 1 ? 1.0 : -expm1(-2.0 * gap / temporal_length)
        vertices = plate(1:N) do i
            residual = scan(1:C; init=0, history=0.0) do carry, c, earlier
                offset = N * ((c - 1) + C * (t - 1))
                index = geometry.permutation[i] + offset
                innovation = field[index] - (t == 1 ? 0.0 : decay * field[index - N * C])
                neighbours = plate(geometry.gptr[i]:(geometry.gptr[i + 1] - 1)) do p
                    neighbour = geometry.permutation[geometry.grow[p]] + offset
                    difference = field[neighbour] -
                        (t == 1 ? 0.0 : decay * field[neighbour - N * C])
                    return geometry.gval[p] * difference
                end
                spatial = kappa2 * geometry.mass[i] * innovation + sum(neighbours)
                previous = sum((earlier[k] * L[c, k] for k in 1:(c - 1)); init=0.0)
                white = (spatial - previous) / L[c, c]
                (carry, white)
            end
            return sum(abs2, residual) / geometry.mass[i]
        end
        return (sum(vertices) / (4.0 * pi * kappa2 * variance) + N * C * log(variance)) / 2.0
    end
    return normalizer - sum(transitions)
end

ReactiveKernels.@kernel function _gmrf_coordinates(u, shape, spatial_length,
        temporal_length, L, geometry)
    return u
end
ReactiveKernels.@kernel function _gmrf_logjac(u, shape, spatial_length,
        temporal_length, L, geometry)
    return 0.0
end

function ReactiveKernelsPPL.sampling_geometry(::Type{BRM._BRMGMRFLaw}, args, shape)
    dimension = :($(args[4]).ncoordinates)
    dimensions = isempty(shape) ? (dimension,) : shape
    ParameterGeometry(dimensions, dimension; support=:real,
        constrain=_gmrf_coordinates, unconstrain=_gmrf_coordinates, logjac=_gmrf_logjac)
end
ReactiveKernelsPPL.sampling_distribution_graph(::Type{BRM._BRMGMRFLaw}, args::Tuple) =
    _gmrf_density_graph
