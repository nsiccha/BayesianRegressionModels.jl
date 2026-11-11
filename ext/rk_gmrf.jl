# A sparse LDL recurrence over stored entries. A diagonal history entry is D;
# an off-diagonal entry is L. Every learned-length operation stays in the
# retained scan. Only the sparse integer lookup is an ordinary data function.
ReactiveKernels.@kernel function _gmrf_factor_graph(mass_scale, stiffness_scale, geometry)
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
        raw = stiffness_scale * geometry.stiffness[p] +
            (i == j ? mass_scale * geometry.mass[j] : 0.0)
        numerator = raw - sum(corrections)
        entry = i == j ? numerator : numerator / earlier[geometry.lptr[j]]
        (carry, entry)
    end
    return factor
end

# B = C/ell + (ell/2) G = K/kappa/sqrt(2) gives Q = B C^-1 B/(2pi).
# Unlike K, B never squares or inverse-squares a length before cancellation.
# It commutes with the channel operation. Apply it before the
# small channel solve, so each vertex/time cell needs only a C-entry history.
# No dense spatial matrix or space-time factor is constructed.
ReactiveKernels.@kernel function _gmrf_density_graph(field, spatial_length,
        temporal_length, L, geometry)
    N = length(geometry.mass)
    C = geometry.nchannels
    T = length(geometry.times)
    factor = _gmrf_factor_graph(1.0 / spatial_length, spatial_length / 2.0, geometry)
    # Outside the support the density is -Inf, never an error (WarmupHMC's density
    # contract): lengths so long that B's mass term rounds away against its
    # stiffness term lose a pivot, and a zero or infinite length has none finite.
    pivot_logs = plate(1:N) do i
        d = factor[geometry.lptr[i]]
        return (d > 0.0) & (d < Inf) ? log(d) : -Inf
    end
    logdet_space = 2.0 * sum(pivot_logs) -
        sum(log.(geometry.mass)) - N * log(2.0 * pi)
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
                spatial = geometry.mass[i] * (innovation / spatial_length) +
                    (spatial_length / 2.0) * sum(neighbours)
                previous = sum((earlier[k] * L[c, k] for k in 1:(c - 1)); init=0.0)
                white = (spatial - previous) / L[c, c]
                (carry, white)
            end
            # Normalize before squaring: an unscaled square can overflow
            # even when the final log density is representable.
            scale = sqrt(4.0 * pi * geometry.mass[i]) * sqrt(variance)
            return sum(abs2, residual ./ scale)
        end
        return sum(vertices) + N * C * log(variance) / 2.0
    end
    inside = (spatial_length > 0.0) & (spatial_length < Inf) &
        (temporal_length > 0.0) & (temporal_length < Inf)
    return inside ? normalizer - sum(transitions) : -Inf
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

# A footprint field's density and its reverse derivative are one data-only pass
# per distinct conditional geometry over plain arrays. The derivative is
# authored in the rule graph, so reverse mode keeps no tape over the states.
ReactiveKernels.@kernel function _gmrf_vecchia_rule_graph(field::Vector{Float64},
        spatial_length::Float64, temporal_length::Float64, L::Matrix{Float64},
        geometry, density_bar::Float64)
    density::Float64 = BRM._brm_vecchia_logdensity(field, spatial_length, temporal_length, L, geometry)
    gradient = BRM._brm_vecchia_gradient(field, spatial_length, temporal_length, L, geometry)
    field_bar::Vector{Float64} = density_bar .* gradient.field
    spatial_length_bar::Float64 = density_bar * gradient.spatial_length
    temporal_length_bar::Float64 = density_bar * gradient.temporal_length
    L_bar::Matrix{Float64} = density_bar .* gradient.L
    # The footprint is data: its cotangent port is empty and never read.
    geometry_bar = nothing
    return density, field_bar, spatial_length_bar, temporal_length_bar, L_bar, geometry_bar
end
const _gmrf_vecchia_density = ReactiveKernels.derivative_rule(_gmrf_vecchia_rule_graph;
    primal=:density, covector=:density_bar,
    cotangents=(field=:field_bar, spatial_length=:spatial_length_bar,
        temporal_length=:temporal_length_bar, L=:L_bar, geometry=:geometry_bar),
    name=:brm_gmrf_vecchia_density)

ReactiveKernels.@kernel function _gmrf_vecchia_density_graph(field, spatial_length,
        temporal_length, L, geometry)
    return _gmrf_vecchia_density(field, spatial_length, temporal_length, L, geometry)
end

function ReactiveKernelsPPL.sampling_geometry(::Type{BRM._BRMGMRFVecchiaLaw}, args, shape)
    dimension = :($(args[4]).ncoordinates)
    dimensions = isempty(shape) ? (dimension,) : shape
    ParameterGeometry(dimensions, dimension; support=:real,
        constrain=_gmrf_coordinates, unconstrain=_gmrf_coordinates, logjac=_gmrf_logjac)
end
ReactiveKernelsPPL.sampling_distribution_graph(::Type{BRM._BRMGMRFVecchiaLaw}, args::Tuple) =
    _gmrf_vecchia_density_graph

ReactiveKernels.@kernel function _gmrf_multiresolution_rule_graph(field::Vector{Float64},
        spatial_length::Float64, temporal_length::Float64, L::Matrix{Float64},
        geometry, density_bar::Float64)
    density::Float64 = BRM._brm_multiresolution_density(field, temporal_length, L, geometry).value
    gradient = BRM._brm_multiresolution_density(field, temporal_length, L, geometry; derivative=true)
    field_bar::Vector{Float64} = density_bar .* gradient.field
    spatial_length_bar::Float64 = 0.0
    temporal_length_bar::Float64 = density_bar * gradient.temporal_length
    L_bar::Matrix{Float64} = density_bar .* gradient.L
    geometry_bar = nothing
    return density, field_bar, spatial_length_bar, temporal_length_bar, L_bar, geometry_bar
end
const _gmrf_multiresolution_density = ReactiveKernels.derivative_rule(
    _gmrf_multiresolution_rule_graph; primal=:density, covector=:density_bar,
    cotangents=(field=:field_bar, spatial_length=:spatial_length_bar,
        temporal_length=:temporal_length_bar, L=:L_bar, geometry=:geometry_bar),
    name=:brm_gmrf_multiresolution_density)
ReactiveKernels.@kernel function _gmrf_multiresolution_density_graph(field, spatial_length,
        temporal_length, L, geometry)
    return _gmrf_multiresolution_density(field, spatial_length, temporal_length, L, geometry)
end
function ReactiveKernelsPPL.sampling_geometry(::Type{BRM._BRMGMRFMultiresolutionLaw}, args, shape)
    dimension = :($(args[4]).ncoordinates)
    dimensions = isempty(shape) ? (dimension,) : shape
    ParameterGeometry(dimensions, dimension; support=:real,
        constrain=_gmrf_coordinates, unconstrain=_gmrf_coordinates, logjac=_gmrf_logjac)
end
ReactiveKernelsPPL.sampling_distribution_graph(::Type{BRM._BRMGMRFMultiresolutionLaw}, args::Tuple) =
    _gmrf_multiresolution_density_graph

# Only the spatial length is differentiated here. The sparse triangular
# recurrence's analytic derivative avoids an archive-sized reverse tape.
ReactiveKernels.@kernel function _gmrf_multiresolution_basis_rule(spatial_length::Float64,
        geometry, weights_bar::Vector{Float64})
    weights::Vector{Float64} = BRM._brm_multiresolution_basis(geometry.spatial, spatial_length).weights
    partial::Vector{Float64} = BRM._brm_multiresolution_basis(
        geometry.spatial, spatial_length; derivative=true).derivative
    spatial_length_bar::Float64 = sum(weights_bar .* partial)
    geometry_bar = nothing
    return weights, spatial_length_bar, geometry_bar
end
const _gmrf_multiresolution_basis = ReactiveKernels.derivative_rule(
    _gmrf_multiresolution_basis_rule; primal=:weights, covector=:weights_bar,
    cotangents=(spatial_length=:spatial_length_bar, geometry=:geometry_bar),
    name=:brm_gmrf_multiresolution_basis)

BRM._rk_gmrf_multiresolution_basis() = _gmrf_multiresolution_basis
