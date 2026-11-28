module WrenHSGPFixture

using BayesianRegressionModels, Distributions, Random

# The reporting-rate case study's full model. Its three row axes are cells,
# cell-years and observed cell-months; the gathers connect those axes.
# All basis sizes are fixed, including in the small simulated fixture below.
const model = @brm begin
    "Where wrens are reported: a smooth field over the UK plus each cell's own deviation (logit scale)."
    place ~ 0 + hsgp(east, north; k = (20, 36), domain = ((-4.0, 4.0), (-7.4, 7.4)),
                     orthogonal_to = :intercept) + (1 | cell)
    "How the seasonal cycle's first harmonic (cosine part) varies across the UK."
    season_cos ~ 0 + hsgp(east, north; k = (8, 15), domain = ((-4.0, 4.0), (-7.4, 7.4)),
                          orthogonal_to = :intercept)
    "How the seasonal cycle's first harmonic (sine part) varies across the UK."
    season_sin ~ 0 + hsgp(east, north; k = (8, 15), domain = ((-4.0, 4.0), (-7.4, 7.4)),
                          orthogonal_to = :intercept)
    "Regional year-to-year change: a smooth field over space and years."
    place_year ~ 0 + hsgp(cy_east, cy_north, cy_decade; iso = false, k = (8, 15, 6),
                          domain = ((-4.0, 4.0), (-7.4, 7.4), (-0.7, 0.7)),
                          orthogonal_to = :intercept)
    "When wrens are reported nationally: the seasonal cycle and the trend over the years."
    when ~ 1 + hsgp(month; k = 4, cov = :periodic, period = 12.0) + hsgp(decade; k = 8)
    # each cell-month reads its cell's and its cell-year's values by position
    place_i = place[cell_pos]
    cos_i = season_cos[cell_pos]
    sin_i = season_sin[cell_pos]
    year_i = place_year[cy_pos]
    eta = when + place_i + cos_i .* cos_m + sin_i .* sin_m + year_i
    p = logistic(eta)
    "Precision of the beta-binomial: how far checklists in one cell and month scatter beyond the model."
    phi ~ LogNormal(3.0, 1.5)
    wren ~ BetaBinomial2(checklists, p, phi)
end

# Independent squared-exponential HSGP simulation: tensor Dirichlet
# eigenfunctions times the square root of the kernel's spectral density.
function field(rng, axes, modes, widths, rhos; sigma=0.3, demean=false)
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
    demean && (result .-= sum(result) / length(result))
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
    spatial = field(rng, (east, north), (20, 36), (4.0, 7.4), (1.0, 1.0); demean=true)
    cosine = field(rng, (east, north), (8, 15), (4.0, 7.4), (2.0, 2.0); demean=true)
    sine = field(rng, (east, north), (8, 15), (4.0, 7.4), (2.0, 2.0); demean=true)
    annual = field(rng, (cy_east, cy_north, cy_decade), (8, 15, 6),
        (4.0, 7.4, 0.7), (2.0, 2.0, 0.4); demean=true)
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
