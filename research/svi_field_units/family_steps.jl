# How many stochastic steps a field family needs to reach its own optimum.
#
# Same toy as family_oracle.jl, hyperparameters fixed at the truth and station
# effects known, so only the field factor moves. A step draws `batch` stations,
# scales their rows, and takes the Adam step SVI's global factor uses. The optimum
# of each family and its KL are closed form, so the excess KL is exact.
#
#     julia --startup-file=no research/svi_field_units/family_steps.jl [family...]
include("family_oracle.jl")

function steps_target(name, mesh, times, reads, θ, y, roots)
    A, λ = reads.A, 1 / θ.τ^2
    Q = Matrix(kron(temporal_q(times, θ.ℓt), spatial_q(mesh, θ.ℓs)))
    R = name === :diagonal ? Matrix{Float64}(I, size(Q)...) : roots[name](θ)
    AR = Matrix(A) * R .* θ.σ
    # The target's precision and information in the family's coordinates.
    prior = name === :diagonal ? Q : Matrix{Float64}(I, size(Q)...)
    Π = Symmetric(prior + λ .* (AR' * AR))
    F = cholesky(Π)
    μ = F \ (λ .* (AR' * y))
    (; AR, R, prior, Π, μ, λ, logdetΠ=logdet(F), diagΠ=diag(Π),
        klopt=(sum(log, diag(Π)) - logdet(F)) / 2,
        zsd=sqrt.(diag(inv(cholesky(Symmetric(Q + λ * θ.σ^2 * Matrix(A' * A)))))))
end

# KL(q ‖ N(μ, Π⁻¹)) = ½(tr(ΠΣ) − n − log|Σ| − log|Π| + dᵀΠd).
function family_kl_exact(tg, m, s)
    d = m .- tg.μ
    (sum(tg.diagΠ .* s .^ 2) - length(m) - 2sum(log, s) - tg.logdetΠ + dot(d, tg.Π * d)) / 2
end

function run_steps(name, tg, stations, S, y; lr, iterations, batch, seed=7, marks)
    n = size(tg.AR, 2)
    rng = Xoshiro(seed)
    m, ρ = zeros(n), zeros(n)
    name === :diagonal && (ρ .= log.(0.1))
    M, V = zeros(2n), zeros(2n)
    rows = [findall(==(s), stations) for s in 1:S]
    out = NamedTuple[]
    next = 1
    for it in 1:iterations
        chosen = randperm(rng, S)[1:batch]
        ε = randn(rng, n)
        s = exp.(ρ)
        x = m .+ s .* ε
        g = -(tg.prior * x)
        for c in chosen, r in rows[c]
            row = view(tg.AR, r, :)
            g .+= (S / batch) * tg.λ * (y[r] - dot(row, x)) .* row
        end
        grad = vcat(g, g .* s .* ε .+ 1)
        M .= 0.9 .* M .+ 0.1 .* grad
        V .= 0.999 .* V .+ 0.001 .* grad .^ 2
        step = lr .* (M ./ (1 - 0.9^it)) ./ (sqrt.(V ./ (1 - 0.999^it)) .+ 1e-8)
        m .+= view(step, 1:n)
        ρ .+= view(step, n+1:2n)
        if next <= length(marks) && it == marks[next]
            zerr = abs.(tg.R * (m .- tg.μ)) ./ tg.zsd
            push!(out, (; it, excess=family_kl_exact(tg, m, exp.(ρ)) - tg.klopt,
                median_mean_error=median(zerr), p95_mean_error=quantile(zerr, 0.95)))
            next += 1
        end
    end
    out
end

function main_steps()
    v, nt, stations_n, rows_n = 11, 16, 12, 8
    θ = (; σ=1.0, ℓs=0.5, ℓt=10.0, τ=0.2, σα=0.5)
    mesh = grid_mesh(v)
    times = collect(0.0:nt-1)
    rng = Xoshiro(1)
    reads = toy_reads(mesh, times, stations_n, rows_n, rng)
    stations = [findfirst(!iszero, reads.Z[r, :]) for r in axes(reads.Z, 1)]
    modes = spatial_modes(mesh)
    roots = Dict(
        :whitened => (θ -> kron(temporal_root(times, θ.ℓt), spatial_root(mesh, θ.ℓs))),
        :space_spectral => (θ -> kron(temporal_root(times, θ.ℓt), spatial_spectral_root(modes, θ.ℓs))),
        :space_hier => (θ -> kron(temporal_hier_root(times, θ.ℓt), spatial_spectral_root(modes, θ.ℓs))),
        :spectral => (θ -> kron(temporal_spectral_root(times, θ.ℓt), spatial_spectral_root(modes, θ.ℓs))))
    z = roots[:whitened](θ) * randn(rng, size(reads.A, 2))
    y = θ.σ .* (reads.A * z) .+ θ.τ .* randn(rng, size(reads.A, 1))
    marks = [100, 300, 1000, 3000, 10000, 30000]
    names = isempty(ARGS) ? (:diagonal, :whitened, :space_spectral, :spectral) : Symbol.(Tuple(ARGS))
    for name in names, lr in (0.01, 0.03, 0.1)
        tg = steps_target(name, mesh, times, reads, θ, y, roots)
        t = @elapsed trace = run_steps(name, tg, stations, stations_n, y; lr, iterations=last(marks), batch=4, marks)
        @printf("%-15s lr=%-5g KLopt=%8.1f (%.0f s)\n", name, lr, tg.klopt, t)
        for p in trace
            @printf("   it=%-6d excess KL=%12.1f  mean error (posterior sds) median=%7.3f p95=%7.3f\n",
                p.it, p.excess, p.median_mean_error, p.p95_mean_error)
        end
        flush(stdout)
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && main_steps()
