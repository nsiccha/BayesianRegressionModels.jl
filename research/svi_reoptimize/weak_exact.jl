# Exact references for the weak random intercept of
# test/nlme_svi_default_centering.jl (32 groups x one row, data seed 1; todo 12maq21):
#
#   a ~ N(0, 5), log τ ~ N(0, 1) (a lone intercept's scale), σ ~ Exponential(1),
#   u_g ~ N(0, τ), y_g ~ N(a + u_g, σ), so y_g ~ N(a, √(σ² + τ²)).
#
# 1. The posterior of the globals (packed: a, log σ, log τ), on a grid over
#    (log σ, log τ) with a integrated analytically: means and sds.
# 2. The best mean-field Gaussian q over those three coordinates to that marginal
#    posterior (units integrated out): the optimum of E_q[log p(y, g)] + H(q),
#    which SVI with every unit at its conditional optimum given each draw of the
#    globals (exact `unit_reoptimize`) targets. Gauss-Hermite in each coordinate,
#    coordinate golden sections.
#
#     julia --project=test research/svi_reoptimize/weak_exact.jl
using Random, LinearAlgebra, Statistics, Printf

rng = Xoshiro(1)
u = randn(rng, 32)
y = 0.5 .+ u .+ randn(rng, 32)
n, Sy, Syy = length(y), sum(y), sum(abs2, y)

# log p(y, a, ℓσ, ℓτ) in the packed coordinates (log-Jacobians included).
function logjoint(a, ls, lt)
    V = exp(2ls) + exp(2lt)
    -a^2 / 50 - lt^2 / 2 + ls - exp(ls) -
        n / 2 * log(V) - (Syy - 2a * Sy + n * a^2) / (2V)
end
# With a integrated out: a | V ~ N(m, 1/P).
function logmarginal_scales(ls, lt)
    V = exp(2ls) + exp(2lt)
    P = n / V + 1 / 25
    m = Sy / V / P
    -lt^2 / 2 + ls - exp(ls) - n / 2 * log(V) - Syy / (2V) + P * m^2 / 2 - log(P) / 2, m, P
end

grid = range(-12, 4; length=1601)
lw = [logmarginal_scales(ls, lt)[1] for ls in grid, lt in grid]
w = exp.(lw .- maximum(lw)); w ./= sum(w)
ms = [logmarginal_scales(ls, lt)[2] for ls in grid, lt in grid]
Ps = [logmarginal_scales(ls, lt)[3] for ls in grid, lt in grid]
Ea = sum(w .* ms)
Va = sum(w .* (1 ./ Ps .+ ms .^ 2)) - Ea^2
Els = sum(w .* grid); Vls = sum(w .* grid .^ 2) - Els^2
Elt = sum(w .* grid'); Vlt = sum(w .* (grid .^ 2)') - Elt^2
println("EXACT posterior (grid, a integrated):")
@printf("  mu_Intercept mean %.4f sd %.4f\n", Ea, sqrt(Va))
@printf("  log sigma    mean %.4f sd %.4f\n", Els, sqrt(Vls))
@printf("  log tau      mean %.4f sd %.4f\n", Elt, sqrt(Vlt))
@printf("  mass with log sigma < -3: %.4f\n", sum(w[grid .< -3, :]))

# Mean-field Gaussian optimum to the marginal posterior of (a, ℓσ, ℓτ).
J = SymTridiagonal(zeros(24), sqrt.(1.0:23.0))
F = eigen(J)
nodes, weights = F.values, F.vectors[1, :] .^ 2
function elbo(μ, s)
    total = 0.0
    for (i, x) in enumerate(nodes), (j, z) in enumerate(nodes), (k, v) in enumerate(nodes)
        total += weights[i] * weights[j] * weights[k] *
            logjoint(μ[1] + s[1] * x, μ[2] + s[2] * z, μ[3] + s[3] * v)
    end
    total + sum(log, s)
end
function golden(f, a, b)
    φ = (sqrt(5) - 1) / 2
    for _ in 1:60
        c, e = b - φ * (b - a), a + φ * (b - a)
        f(c) > f(e) ? (b = e) : (a = c)
    end
    (a + b) / 2
end
μ, s = [Ea, Els, Elt], [sqrt(Va), sqrt(Vls), sqrt(Vlt)]
for sweep in 1:12
    for k in 1:3
        μ[k] = golden(x -> elbo(setindex!(copy(μ), x, k), s), μ[k] - 3, μ[k] + 3)
        s[k] = exp(golden(l -> elbo(μ, setindex!(copy(s), exp(l), k)), log(s[k]) - 3,
            log(s[k]) + 3))
    end
end
println("MEAN-FIELD optimum to the marginal posterior (units integrated):")
@printf("  mu_Intercept mean %.4f sd %.4f\n", μ[1], s[1])
@printf("  log sigma    mean %.4f sd %.4f\n", μ[2], s[2])
@printf("  log tau      mean %.4f sd %.4f\n", μ[3], s[3])
@printf("  ELBO %.4f\n", elbo(μ, s))
