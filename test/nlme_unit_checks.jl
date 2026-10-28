# test/nlme_unit_checks.jl — checks of an NLME model (`brm_nlme_model`)
# against the full RK density, shared by the tests of terms whose per-level
# values are unit draws (test/nlme_grouped_hsgp.jl, test/rk_cdar.jl). Include
# it after nlme_fixtures.jl, with LogDensityProblems, LinearAlgebra and Random
# loaded.

# Random θ, σ and subject blocks; the packed point the NLME model evaluates.
function random_point(m, rng)
    θ = 0.2 .* randn(rng, length(m.theta))
    σ = 0.2 .* randn(rng, length(m.sigma))
    H = BRM._brm_nlme_ragged(m.view) ?
        [0.5 .* randn(rng, length(b)) for b in m.view.subject_coordinates] :
        0.5 .* randn(rng, length(m.view.block_margins), length(m.view.levels))
    θ, σ, H
end
column(H::AbstractMatrix, i) = H[:, i]
column(H::AbstractVector, i) = H[i]

# The full RK density's dependence on one unit's block equals that unit's
# problem's: density differences and the unit gradient (as in
# test/nlme_units_panel.jl).
function check_unit_problems(m, rng)
    v = m.view
    θ, σ, H = random_point(m, rng)
    Ω = BRM._brm_nlme_ragged(v) ? [Matrix{Float64}(I, K, K) for K in v.family_sizes] :
        Matrix{Float64}(I, length(v.block_margins), length(v.block_margins))
    full = rk_logdensity_problem(m.backend; ad_backend=NLME_AD)
    function point(i, z)
        u = copy(m.unit)
        u[m.theta] = θ
        u[m.sigma] = σ
        for (j, block) in enumerate(v.subject_coordinates)
            u[block] = j == i ? z : column(H, j)
        end
        u
    end
    for i in eachindex(v.levels)
        p = brm_nlme_unit_problem(m, i, θ, σ, Ω)
        d = LogDensityProblems.dimension(p)
        @test d == length(v.subject_coordinates[i])
        z1, z2 = 0.4 .* randn(rng, d), 0.4 .* randn(rng, d)
        @test LogDensityProblems.logdensity(p, z1) - LogDensityProblems.logdensity(p, z2) ≈
              LogDensityProblems.logdensity(full, point(i, z1)) -
              LogDensityProblems.logdensity(full, point(i, z2)) atol=1e-8
        _, gradient = LogDensityProblems.logdensity_and_gradient(p, z1)
        _, full_gradient = LogDensityProblems.logdensity_and_gradient(full, point(i, z1))
        @test gradient ≈ full_gradient[v.subject_coordinates[i]] atol=1e-8
    end
end

# Lockstep likelihoods sum to the RK likelihood at the packed point, each
# subject's η gradient is the full RK density's gradient on its block without
# the block's standard-normal prior (every scale is 1 at the unit point), and
# each subject's term does not move with another subject's block.
function check_lockstep(m, rng)
    θ, σ, H = random_point(m, rng)
    values = brm_nlme_loglikelihoods(m, θ, σ, H)
    u = BRM._brm_nlme_point(m, θ, σ, H)
    @test sum(values) ≈ Base.invokelatest(likelihood_query(m.backend), u) rtol=1e-10
    _, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    full = rk_logdensity_problem(m.backend; ad_backend=NLME_AD)
    _, full_gradient = LogDensityProblems.logdensity_and_gradient(full, u)
    for (i, block) in enumerate(m.view.subject_coordinates)
        @test column(G, i) ≈ full_gradient[block] .+ u[block] atol=1e-8
    end
    other = deepcopy(H)
    if H isa AbstractMatrix
        other[:, 2] .+= 0.7
    else
        other[2] .+= 0.7
    end
    moved = brm_nlme_loglikelihoods(m, θ, σ, other)
    @test moved[2] != values[2]
    @test moved[[1; 3:end]] == values[[1; 3:end]]
    θ, σ, H, values
end
