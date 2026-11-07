# src/nlme_laplace_coloring.jl — a structural coloring of the whole-model
# Laplace's η Hessian (`BRMLaplaceProblem`, src/nlme_laplace.jl).
#
# Without it, the units' Hessian blocks take one η Hessian-vector product per
# within-unit position (`_brm_nlme_sampler_hessian_blocks`), and the gradient's
# third-order sum one contraction per position: both grow with the largest
# unit. A connected unit (`unit_partition = :connected`) joins every subject a
# likelihood cell couples, so on data where a few effects are read by many
# subjects (one site effect read by every path through it) the largest unit
# grows with the data, although its Hessian is sparse: each cell couples only
# the coordinates it reads.
#
# The connected view records each cell's structural packed-coordinate support
# (`view.factor_supports`). With the sampler density in the units' standardized
# effects z equal to the N(0, I) prior plus the sum of the cells (the same
# split `_brm_laplace_unit_values` uses), the z–z Hessian is structurally zero
# outside the diagonal and the cells' supports. A star coloring of that pattern
# (SparseMatrixColorings.jl) recovers every structural entry directly from one
# HVP per color: for a stored entry (a, b) either b is the only neighbour of a
# with b's color, or a the only neighbour of b with a's color, and the entry is
# that row of that color's HVP. Units never share a cell, so one coloring of all
# units serves them all, as the position coloring does.
#
# The gradient's third-order sum is ∇ tr(S H(u)), S = blockdiag((−H_i)⁻¹)
# (Σ_m v_m v_mᵀ for the columns v_m of each C_i⁻ᵀ). Only structural entries of H
# carry a derivative, and each is read from one (row, color) of the compressed
# Hessian, so tr(S H) = Σ_c R_cᵀ H d_c with R_c[r] the sum of S over the entries
# read at (r, c), and the sum is Σ_c D³f[R_c, d_c, ·], each a polarization of
# two symmetric contractions.

import SparseMatrixColorings

# Each color's packed coordinates (`directions`), and per unit every structural
# entry of its block, both triangles, as (row position, column position, packed
# row of the HVP it is read from, color). A coloring exists only where its
# colors are fewer than the largest unit's positions, so the Hessian always
# takes it; `third` says whether the gradient's sum, two contractions per color
# against one per position, does too. `checked` records the first assembly's
# check of the pattern (`_brm_laplace_coloring_error`).
struct _BRMLaplaceColoring
    directions::Vector{Vector{Int}}
    entries::Vector{Vector{NTuple{4,Int}}}
    third::Bool
    checked::Base.RefValue{Bool}
end

Base.show(io::IO, c::_BRMLaplaceColoring) = print(io, "_BRMLaplaceColoring(",
    length(c.directions), " colors; third-order sum ",
    c.third ? "colored" : "by position", ")")

"""
    _brm_laplace_coloring(view::BRMNLMEView) -> Union{Nothing,_BRMLaplaceColoring}

A structural coloring of the units' η Hessian from the connected view's cell
supports, or `nothing` where none is cheaper than the position coloring: a view
without supports (`unit_partition = :subjects`), a cell reading the whole
inventory, or a pattern needing as many colors as the largest unit has
positions.
"""
function _brm_laplace_coloring(view::BRMNLMEView)
    isempty(view.factor_supports) && return nothing
    blocks = view.subject_coordinates
    width = maximum(length, blocks; init=0)
    width > 1 || return nothing
    # Each cell's reads of its unit's effects, by unit. A cell reading the whole
    # inventory, or as many of one unit's effects as the largest unit has, leaves
    # no coloring with fewer colors than positions.
    reads = [Vector{Int}[] for _ in blocks]
    for supports in values(view.factor_supports), support in supports
        support === :whole && return nothing
        read = unique!([c for c in support if view.subject[c] != 0])
        isempty(read) && continue
        length(read) >= width && return nothing
        i = view.subject[first(read)]
        all(c -> view.subject[c] == i, read) ||
            error("_brm_laplace_coloring: a cell support spans connected units")
        push!(reads[i], read)
    end
    # The units' η patterns as the nested NLME route declares them
    # (`_brm_nlme_eta_patterns`), block-diagonal over the units in their order,
    # with the N(0, I) prior's diagonal.
    packed = reduce(vcat, blocks; init=Int[])
    unit = reduce(vcat, [fill(i, length(b)) for (i, b) in enumerate(blocks)]; init=Int[])
    position = reduce(vcat, [collect(eachindex(b)) for b in blocks]; init=Int[])
    n = length(packed)
    pattern = blockdiag(_brm_nlme_eta_patterns(blocks, [[r] for r in reads])...) .|
        sparse(I, n, n)
    result = SparseMatrixColorings.coloring(pattern,
        SparseMatrixColorings.ColoringProblem(; structure=:symmetric, partition=:column),
        SparseMatrixColorings.GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst();
            decompression=:direct))
    colors = SparseMatrixColorings.column_colors(result)
    ncolors = maximum(colors)
    ncolors < width || return nothing
    # How many neighbours (the entry itself included) of each coordinate carry
    # each color: an entry is read where its column's color is unique.
    multiplicity = [Dict{Int,Int}() for _ in 1:n]
    for a in 1:n, k in nzrange(pattern, a)
        color = colors[rowvals(pattern)[k]]
        multiplicity[a][color] = get(multiplicity[a], color, 0) + 1
    end
    unit_entries = [NTuple{4,Int}[] for _ in blocks]
    for b in 1:n, k in nzrange(pattern, b)
        a = rowvals(pattern)[k]
        source = if multiplicity[a][colors[b]] == 1
            (packed[a], colors[b])
        elseif multiplicity[b][colors[a]] == 1
            (packed[b], colors[a])
        else
            error("_brm_laplace_coloring: SparseMatrixColorings returned a coloring " *
                "that is not a star coloring; entry ($a, $b) has no direct source")
        end
        push!(unit_entries[unit[a]], (position[a], position[b], source...))
    end
    directions = [packed[findall(==(c), colors)] for c in 1:ncolors]
    _BRMLaplaceColoring(directions, unit_entries, 2ncolors < width, Ref(false))
end

_brm_laplace_direction(coordinates, n) = (d = zeros(Float64, n); d[coordinates] .= 1.0; d)

# The units' Hessian blocks of the sampler density at `u`, as
# `_brm_nlme_sampler_hessian_blocks` returns them, from one HVP per color.
function _brm_laplace_colored_hessian_blocks(m::BRMNLMEModel, c::_BRMLaplaceColoring,
        u::Vector{Float64})
    products = [_rk_eta_hvp(m.eta_hvp, _brm_laplace_direction(d, length(u)), u)
        for d in c.directions]
    blocks = [zeros(Float64, length(b), length(b)) for b in m.view.subject_coordinates]
    for (i, entries) in enumerate(c.entries), (p, q, r, k) in entries
        blocks[i][p, q] = products[k][r]
    end
    blocks
end

# The seed of the direction the first colored assembly is checked along.
const _BRM_LAP_COLORING_CHECK_SEED = 0x20261107

"""
    _brm_laplace_coloring_error(m, blocks, u) -> Float64

The largest relative difference, over the units, between one η HVP at `u` along
a fixed pseudo-random direction in the units' effects and the colored `blocks`
times that direction. The pattern assumes the units' curvature is the N(0, I)
prior's plus the cells' (`_brm_laplace_unit_values` makes the same split);
curvature from any other term, or a support that misses a read, leaves an entry
outside the pattern, and that entry shows here with probability one.
"""
function _brm_laplace_coloring_error(m::BRMNLMEModel, blocks, u::Vector{Float64})
    rng = Random.Xoshiro(_BRM_LAP_COLORING_CHECK_SEED)
    direction = zeros(Float64, length(u))
    for coordinates in m.view.subject_coordinates
        direction[coordinates] .= randn(rng, length(coordinates))
    end
    product = _rk_eta_hvp(m.eta_hvp, direction, u)
    worst = 0.0
    for (block, coordinates) in zip(blocks, m.view.subject_coordinates)
        expected = product[coordinates]
        assembled = block * direction[coordinates]
        worst = max(worst, norm(expected - assembled) /
            max(norm(expected), norm(assembled), floatmin(Float64)))
    end
    worst
end

# Σ_m D³f[v_m, v_m, ·] (`_brm_laplace_third_sum`) as Σ_c D³f[R_c, d_c, ·]: each
# by polarization, with R_c and d_c scaled to equal maximum norms so neither
# square swamps the other.
function _brm_laplace_colored_third_sum(c::_BRMLaplaceColoring, query, factors,
        u::Vector{Float64})
    inverses = [inv(F) for F in factors]
    weights = [zeros(Float64, length(u)) for _ in c.directions]
    for (i, entries) in enumerate(c.entries), (p, q, r, k) in entries
        weights[k][r] += inverses[i][p, q]
    end
    total = zeros(Float64, length(u))
    for (k, coordinates) in enumerate(c.directions)
        scale = maximum(abs, weights[k])
        scale > 0 || continue
        α = sqrt(1 / scale)
        w = α .* weights[k]
        d = _brm_laplace_direction(coordinates, length(u)) ./ α
        total .+= (_rk_laplace_third(query, w .+ d, u) .-
            _rk_laplace_third(query, w .- d, u)) ./ 4
    end
    total
end
