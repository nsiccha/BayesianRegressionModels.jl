module WrenHSGPFixture

using BayesianRegressionModels, Distributions, Random

# The reporting-rate case study's full model. Its three row axes are cells,
# cell-years and observed cell-months; the gathers connect those axes.
# All basis sizes are fixed, including in the small simulated fixture below.
const model = @brm begin
    place ~ 0 + hsgp(east, north; k=(24, 36),
        domain=((-5.1, 5.1), (-7.7, 7.7))) + (1 | cell)
    season_cos ~ 0 + hsgp(east, north; k=(10, 15),
        domain=((-5.1, 5.1), (-7.7, 7.7)))
    season_sin ~ 0 + hsgp(east, north; k=(10, 15),
        domain=((-5.1, 5.1), (-7.7, 7.7)))
    place_year ~ 0 + hsgp(cy_east, cy_north, cy_decade; iso=false,
        k=(10, 15, 6), domain=((-5.1, 5.1), (-7.7, 7.7), (-0.7, 0.7)))
    when ~ 1 + hsgp(month; k=4, cov=:periodic, period=12.0) + hsgp(decade; k=8)
    eta = when + place[cell_pos] + season_cos[cell_pos] .* cos_m +
        season_sin[cell_pos] .* sin_m + place_year[cy_pos]
    p = logistic(eta)
    phi ~ LogNormal(3.0, 1.5)
    wren ~ BetaBinomial2(checklists, p, phi)
end

# Independent squared-exponential HSGP simulation: tensor Dirichlet
# eigenfunctions times the square root of the kernel's spectral density.
function field(rng, axes, modes, widths, rhos; sigma=0.3)
    result = zeros(length(first(axes)))
    for mode in CartesianIndices(modes)
        omega = [mode[j] * pi / (2widths[j]) for j in eachindex(axes)]
        scale = sigma * prod(sqrt(sqrt(2pi) * rhos[j]) *
            exp(-rhos[j]^2 * omega[j]^2 / 4) for j in eachindex(axes))
        weight = scale * randn(rng)
        for i in eachindex(result)
            result[i] += weight * prod(sin(omega[j] * (axes[j][i] + widths[j])) /
                sqrt(widths[j]) for j in eachindex(axes))
        end
    end
    result
end

"Simulated beta-binomial counts; no observed bird data or private model inputs."
function data(; seed=20261127)
    rng = Xoshiro(seed)
    east, north = [-2.0, 1.0, -1.5, 1.5], [-3.0, -2.0, 2.0, 4.0]
    cell = string.("cell", 1:4)
    cy_cell = repeat(1:4, 2)
    cy_decade = repeat([-0.25, 0.25]; inner=4)
    cy_east, cy_north = east[cy_cell], north[cy_cell]
    cell_pos = repeat(cy_cell; inner=4)
    cy_pos = repeat(1:8; inner=4)
    month = repeat([1.0, 4.0, 7.0, 10.0], 8)
    decade = cy_decade[cy_pos]
    cos_m, sin_m = cos.(2pi .* month ./ 12), sin.(2pi .* month ./ 12)
    spatial = field(rng, (east, north), (24, 36), (5.1, 7.7), (1.0, 1.0))
    cosine = field(rng, (east, north), (10, 15), (5.1, 7.7), (2.0, 2.0))
    sine = field(rng, (east, north), (10, 15), (5.1, 7.7), (2.0, 2.0))
    annual = field(rng, (cy_east, cy_north, cy_decade), (10, 15, 6),
        (5.1, 7.7, 0.7), (2.0, 2.0, 0.4))
    trend = field(rng, (decade,), (8,), (0.375,), (0.2,))
    # Any fixed harmonic coefficients are an admissible point in the periodic
    # term's full-support Gaussian prior; retain all four harmonics here.
    seasonal = sum(0.2randn(rng) .* cos.(2pi * j .* month ./ 12) +
        0.2randn(rng) .* sin.(2pi * j .* month ./ 12) for j in 1:4)
    cell_effect = 0.25randn(rng, 4)
    eta = -1.5 .+ seasonal .+ trend .+ (spatial .+ cell_effect)[cell_pos] .+
        cosine[cell_pos] .* cos_m .+ sine[cell_pos] .* sin_m .+ annual[cy_pos]
    checklists = rand(rng, 25:60, length(month))
    wren = [rand(rng, BetaBinomial2(n, logistic(e), 20.0))
        for (n, e) in zip(checklists, eta)]
    (; east, north, cell, cy_east, cy_north, cy_decade, cell_pos, cy_pos,
       month, decade, cos_m, sin_m, checklists, wren)
end

end
