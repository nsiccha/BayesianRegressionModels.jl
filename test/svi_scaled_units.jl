# An analytic SVI target for the standalone arithmetic test modules
# (nlme_svi_unit_curvature, nlme_svi_reoptimize_step): include it inside a module
# that has included src/nlme_svi.jl.
#
# Units whose terms are Gaussian in their coordinates: unit i is
# log N(z | 0, σ² C) + log N(y_i | z, I) with σ = exp(g[1]), the one global, under a
# N(0, 1) prior on log σ. C is an AR(1) correlation, so the scale's optimum reads
# the units' widths along C⁻¹'s rough, high-precision directions, as a chain's
# length scale and innovation sd do.
struct ScaledUnits
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    plan::NamedTuple
    plans::Vector{NamedTuple}
    work::Vector{Nothing}
    Ci::Matrix{Float64}
    y::Vector{Vector{Float64}}
end
function ScaledUnits(Ci, y)
    d, N = size(Ci, 1), length(y)
    units = [collect(1+(i-1)*d+1:1+i*d) for i in 1:N]
    plan = (; footprints=[vcat(1, u) for u in units])
    ScaledUnits([1], units, plan, [plan], [nothing], Ci, y)
end

_brm_svi_global_term(::ScaledUnits, g) = (-g[1]^2 / 2, [-g[1]])
function _brm_svi_unit_term!(gg, gz, t::ScaledUnits, i, g, z, plan, work)
    d, s2 = length(z), exp(2g[1])
    Cz = t.Ci * z
    quad = dot(z, Cz)
    r = t.y[i] .- z
    fill!(gg, 0.0)
    gg[1] = quad / s2 - d
    gz .= .-Cz ./ s2 .+ r
    -quad / (2s2) - d * g[1] - sum(abs2, r) / 2
end
