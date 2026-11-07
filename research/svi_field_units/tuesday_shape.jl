# Should a spatial slow field use random Fourier features or a gmrf, under minibatch SVI?
#
# A toy of the shape of a stand-in fit: one field read per site with no time axis, at
# 64 or 184 synthetic sites, a quarter of them held out. The truth is drawn from the
# Matérn ν = 1 mesh prior, the adopted target's law, and everything is scored on the
# held-out sites' latent values with hyperparameters learned from each family's own
# ELBO (flat on log θ, within a factor 30 of the truth). Normal rows, so every optimum
# is closed form and there is no optimizer noise. No ODL data or model content.
#
#     julia --startup-file=no research/svi_field_units/tuesday_shape.jl
#
# Families, all with an independent Gaussian per coordinate:
#   exact        the mesh gmrf with its conjugate posterior (the reference)
#   mesh_diag    the mesh gmrf, one factor per mesh state: today's global factor
#   site_diag    the same law on one state per site, as a read-pair field has
#   eigen_K      the K lowest eigenmodes of the prior, every other mode at its prior
#   features_M   M paired Fourier features with frozen Gaussian frequencies: a squared
#                exponential field, as the stand-in uses, so also misspecified here
include("family_oracle.jl")

function shape_data(side, sites, rows, θ, rng)
    mesh = grid_mesh(side)
    x, y = rand(rng, sites) .* 0.9 .+ 0.05, rand(rng, sites) .* 0.9 .+ 0.05
    (; indices, weights) = GM._brm_gmrf_interpolation(mesh, x, y)
    N = length(mesh.mass)
    read = sparse(repeat(1:sites, 3), vec(indices), vec(weights), sites, N)
    train = 1:round(Int, 0.75sites)
    held = last(train)+1:sites
    A = reduce(vcat, [read[train, :] for _ in 1:rows])
    modes = spatial_modes(mesh)
    z = spatial_spectral_root(modes, θ.ℓs) * randn(rng, N)
    truth = θ.σ .* (read * z)
    y_rows = θ.σ .* (A * z) .+ θ.τ .* randn(rng, size(A, 1))
    (; mesh, modes, N, A, y=y_rows, Ah=read[held, :], As=read[train, :], latent=truth[held],
       coordinates=hcat(x, y), train, held, rows)
end

logdet_q(d, ℓ) = sum(2log(1 / ℓ + ℓ * μ / 2) for μ in d.modes.μ) + sum(log, d.mesh.mass) - d.N * log(2π)

function gmrf_posterior(d, θ)
    λ = 1 / θ.τ^2
    Λ = spatial_q(d.mesh, θ.ℓs) + θ.σ^2 * λ * (d.A' * d.A)
    F = cholesky(Symmetric(Λ))
    b = θ.σ * λ .* (d.A' * d.y)
    μ = F \ b
    evidence = -length(d.y) * log(2π / λ) / 2 - λ * dot(d.y, d.y) / 2 + dot(b, μ) / 2 - logdet(F) / 2 + logdet_q(d, θ.ℓs) / 2
    (; λ, Λ, F, b, μ, evidence)
end

# Each family returns its ELBO at θ and, on request, the held-out mean and SD of the
# latent σ A z.
function family_elbo(name, d, θ; predict=false)
    if name === :exact || name === :mesh_diag
        p = gmrf_posterior(d, θ)
        gap = name === :exact ? 0.0 : (sum(log, diag(p.Λ)) - logdet(p.F)) / 2
        predict || return p.evidence - gap
        mean = θ.σ .* (d.Ah * p.μ)
        variance = name === :exact ?
            vec(sum(d.Ah' .* (p.F \ Matrix(d.Ah')); dims=1)) :
            [dot(d.Ah[r, :] .^ 2, 1 ./ diag(p.Λ)) for r in axes(d.Ah, 1)]
        return mean, θ.σ .* sqrt.(variance)
    elseif name === :site_diag
        p = gmrf_posterior(d, θ)
        D = sqrt(2π) ./ (1 / θ.ℓs .+ θ.ℓs .* d.modes.μ ./ 2)
        S = Matrix(d.As * d.modes.U) .* D'
        K = S * S'
        K += 1e-10 * tr(K) / size(K, 1) * I
        Kf = cholesky(Symmetric(K))
        n = size(K, 1)
        P = inv(Kf) + (θ.σ^2 * p.λ * d.rows) * I
        gap = (sum(log, diag(P)) - logdet(cholesky(Symmetric(P)))) / 2
        predict || return p.evidence - gap
        # The mean of the site states is exact; the rest follows the prior's conditional.
        fS = d.As * p.μ
        H = Matrix(d.Ah * d.modes.U) .* D'
        Kh = H * S'
        weights = (Kf \ Kh')'
        variance = vec(sum(abs2, H; dims=2)) .- vec(sum(weights .* Kh; dims=2)) .+
            vec(sum(weights .^ 2 ./ diag(P)'; dims=2))
        return θ.σ .* (weights * fS), θ.σ .* sqrt.(max.(variance, 0.0))
    elseif startswith(String(name), "eigen_")
        K = parse(Int, split(String(name), "_")[2])
        p = gmrf_posterior(d, θ)
        D = sqrt(2π) ./ (1 / θ.ℓs .+ θ.ℓs .* d.modes.μ ./ 2)
        AT = Matrix(d.A * d.modes.U) .* D' .* (θ.σ * sqrt(p.λ))
        Pd = 1 .+ vec(sum(abs2, AT; dims=1))
        keep, tail = 1:K, K+1:d.N
        AS = AT[:, keep]
        PS = Matrix{Float64}(I, K, K) + AS' * AS
        bS = sqrt(p.λ) .* (AS' * d.y)
        m = cholesky(Symmetric(PS)) \ bS
        quadratic = dot(p.μ, p.b) - dot(m, bS)
        logP = logdet(p.F) - logdet_q(d, θ.ℓs)
        gap = (sum(Pd[tail] .- 1) - logP + sum(log, Pd[keep]) + quadratic) / 2
        predict || return p.evidence - gap
        H = Matrix(d.Ah * d.modes.U) .* D'
        variance = vec(sum(H[:, keep] .^ 2 ./ Pd[keep]'; dims=2)) .+ vec(sum(H[:, tail] .^ 2; dims=2))
        return θ.σ .* (H[:, keep] * m), θ.σ .* sqrt.(variance)
    else
        M = parse(Int, split(String(name), "_")[2])
        frequency = randn(MersenneTwister(918), 2, M)
        features(x) = begin
            phase = (x * frequency) ./ θ.ℓs
            hcat(cos.(phase), sin.(phase)) ./ sqrt(M)
        end
        Φ = reduce(vcat, [features(d.coordinates[d.train, :]) for _ in 1:d.rows]) .* θ.σ
        λ = 1 / θ.τ^2
        P = Symmetric(Matrix{Float64}(I, 2M, 2M) + λ .* (Φ' * Φ))
        F = cholesky(P)
        b = λ .* (Φ' * d.y)
        μ = F \ b
        evidence = -length(d.y) * log(2π / λ) / 2 - λ * dot(d.y, d.y) / 2 + dot(b, μ) / 2 - logdet(F) / 2
        gap = (sum(log, diag(P)) - logdet(F)) / 2
        predict || return evidence - gap
        Φh = features(d.coordinates[d.held, :]) .* θ.σ
        return Φh * μ, sqrt.(vec(sum(Φh .^ 2 ./ diag(P)'; dims=2)))
    end
end

score(d, mean, sd) = (; rmse=sqrt(Statistics.mean(abs2, mean .- d.latent)),
    cover=Statistics.mean(abs.(mean .- d.latent) .<= 1.645 .* sd), sd=Statistics.median(sd))

const SHAPE_FAMILIES = (:exact, :mesh_diag, :site_diag, :eigen_32, :eigen_64, :eigen_128, :features_64, :features_256)

function main_shape()
    regimes = (
        ("64 sites, field long against the domain", (; side=41, sites=64, rows=8), (; σ=1.0, ℓs=0.5, τ=0.3)),
        ("64 sites, shorter field, noisier", (; side=41, sites=64, rows=4), (; σ=1.0, ℓs=0.2, τ=1.0)),
        ("184 sites, field short against the domain", (; side=41, sites=184, rows=8), (; σ=1.0, ℓs=0.15, τ=0.3)),
    )
    for (label, shape, truth) in regimes
        rows = Dict(name => NamedTuple[] for name in SHAPE_FAMILIES)
        for seed in 1:4
            d = shape_data(shape.side, shape.sites, shape.rows, truth, Xoshiro(seed))
            pack(θ) = log.([θ.σ, θ.ℓs, θ.τ])
            unpack(x) = (; σ=exp(x[1]), ℓs=exp(x[2]), τ=exp(x[3]))
            for name in SHAPE_FAMILIES
                objective(x) = any(abs.(x .- pack(truth)) .> log(30)) ? 1e12 : -family_elbo(name, d, unpack(x))
                x, _ = nelder_mead(objective, pack(truth); iterations=120)
                θ = unpack(x)
                s = score(d, family_elbo(name, d, θ; predict=true)...)
                push!(rows[name], (; s..., θ.σ, θ.ℓs, θ.τ))
            end
        end
        @printf("== %s (truth σ=%.2f ℓ=%.2f τ=%.2f; held-out latent SD about %.2f)\n", label, truth.σ, truth.ℓs, truth.τ, truth.σ)
        @printf("  %-14s %18s %18s %8s %8s %16s\n", "family", "rmse median [range]", "90% cover median", "σ̂", "τ̂", "ℓ̂ median [range]")
        for name in SHAPE_FAMILIES
            r = rows[name]
            rmse, cover = getfield.(r, :rmse), getfield.(r, :cover)
            ℓ = getfield.(r, :ℓs)
            @printf("  %-14s %6.3f [%.3f,%.3f] %12.2f %14.2f %8.2f %8.3f [%.3f,%.3f]\n", name, median(rmse), minimum(rmse), maximum(rmse),
                median(cover), median(getfield.(r, :σ)), median(getfield.(r, :τ)), median(ℓ), minimum(ℓ), maximum(ℓ))
        end
        flush(stdout)
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && main_shape()
