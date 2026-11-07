# Run: julia --project=test test/nlme_laplace_coloring.jl [testset-filter...]
#
# The whole-model Laplace (`BRMLaplaceProblem`) on connected units: a structural
# coloring of the units' η Hessian (src/nlme_laplace_coloring.jl) replaces one
# Hessian-vector product per within-unit position. Each problem is compared with
# itself under the position coloring, which is exact for any unit; both are
# exact, so they agree to rounding.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random

# Each subject has an intercept and a slope effect; observation i reads subjects
# left[i] and right[i], so the connected units join the subjects observations
# couple, and their Hessian couples only the effects one observation reads.
coloring_normal(data) = @brm data begin
    mu ~ 1 + x + (1 + x | effect | subject)
    @plate for i in eachindex(y)
        y[i] ~ Normal(mu[left[i]] + weight[i] * mu[right[i]], 0.5)
    end
end

# A non-Gaussian likelihood: its Hessian moves with the effects, so the
# gradient's implicit term is not zero.
coloring_poisson(data) = @brm data begin
    mu ~ 1 + (1 + x | effect | subject)
    @plate for i in eachindex(y)
        y[i] ~ Poisson(exp(mu[left[i]] + weight[i] * mu[right[i]]))
    end
end

function coloring_data(n, left, right; poisson=false)
    rows = length(left)
    y = poisson ? [mod(3i, 5) for i in 1:rows] :
        collect(range(-0.3, 0.7; length=rows))
    (; subject=["subject $i" for i in 1:n], x=collect(range(-1.0, 1.0; length=n)),
       left, right, weight=collect(range(0.4, 0.9; length=rows)), y)
end

# Subject 1 is read with every other subject, as one site effect is read with
# each path through the site: one unit, its Hessian an arrowhead.
function hub_data(n; poisson=false)
    left = repeat(2:n; inner=3)
    coloring_data(n, vcat(left, [1]), vcat(ones(Int, length(left)), [1]); poisson)
end

# Subject s is read with subject s + 1: one unit, its Hessian block-tridiagonal.
chain_data(n; poisson=false) =
    coloring_data(n, repeat(1:n-1; inner=2), repeat(2:n; inner=2); poisson)

connected_problem(model) = brm_laplace_problem(RKBRMI(model); ad_backend=NLME_AD,
    unit_partition=:connected, structured=false)

function by_position(p)
    q = deepcopy(p)
    q.coloring = nothing
    q
end

# The problem against itself by position, at the start and away from it: unit
# Hessians at a point off the modes, then the Laplace value and gradient (each
# evaluation runs the whole continuation mode search on the Hessians).
function check_against_position(p; points=2)
    q = by_position(p)
    rng = Xoshiro(20261107)
    u = copy(p.model.unit)
    u .+= 0.3 .* randn(rng, length(u))
    colored = BRM._brm_laplace_hessian_blocks(p, u)
    positional = BRM._brm_nlme_sampler_hessian_blocks(p.model, u)
    @test length(colored) == length(positional)
    for (a, b) in zip(colored, positional)
        @test a ≈ b rtol=1e-12 atol=1e-12
    end
    for k in 1:points
        φ = p.u[p.population] .+ (k - 1) .* 0.2 .* randn(rng, length(p.population))
        v, g = LogDensityProblems.logdensity_and_gradient(p, φ)
        w, h = LogDensityProblems.logdensity_and_gradient(q, φ)
        @test isfinite(v)
        @test v ≈ w rtol=1e-10
        @test g ≈ h rtol=1e-8 atol=1e-10
    end
    # The first assembly checked the pattern and kept it.
    @test p.coloring !== nothing && p.coloring.checked[]
end

@stestset "a hub unit's colors do not grow with its subjects" begin
    counts = map((5, 13)) do n
        p = connected_problem(coloring_normal(hub_data(n)))
        @test length(p.model.view.subject_coordinates) == 1
        @test only(length.(p.model.view.subject_coordinates)) == 2n
        @test p.coloring !== nothing && p.coloring.third
        length(p.coloring.directions)
    end
    @test counts[1] == counts[2]
    @test counts[2] <= 4
end

@stestset "Gaussian hub unit: colored and positional Laplace agree" begin
    check_against_position(connected_problem(coloring_normal(hub_data(6))))
end

@stestset "Poisson hub unit: colored and positional Laplace agree" begin
    check_against_position(connected_problem(coloring_poisson(hub_data(6; poisson=true))))
end

@stestset "Poisson chain unit: colored and positional Laplace agree" begin
    p = connected_problem(coloring_poisson(chain_data(7; poisson=true)))
    @test p.coloring !== nothing && p.coloring.third
    @test length(p.coloring.directions) < 14
    check_against_position(p)
end

# A pattern that misses curvature: the hub unit's cell supports cut to the
# subject each cell reads first, as if a support inventory had missed a read.
# The first assembly's check finds the hub couplings outside it, warns, and the
# problem takes the position coloring from then on.
@stestset "a pattern missing curvature falls back to the position coloring" begin
    p = connected_problem(coloring_normal(hub_data(6)))
    v = p.model.view
    cut(support) = support === :whole ? support : let owners = filter(!=(0),
            [v.coordinate_subject[c] for c in support])
        isempty(owners) ? support :
            filter(c -> v.coordinate_subject[c] in (0, minimum(owners)), support)
    end
    supports = Dict{Symbol,Vector{Union{Symbol,Vector{Int}}}}(
        name => Union{Symbol,Vector{Int}}[cut(s) for s in cells]
        for (name, cells) in v.factor_supports)
    fields = [f === :factor_supports ? supports : getfield(v, f)
        for f in fieldnames(typeof(v))]
    p.coloring = BRM._brm_laplace_coloring(typeof(v)(fields...))
    @test p.coloring !== nothing
    u = p.model.unit .+ 0.3 .* randn(Xoshiro(3), length(p.model.unit))
    blocks = @test_logs (:warn, r"curvature outside") BRM._brm_laplace_hessian_blocks(p, u)
    @test p.coloring === nothing
    for (a, b) in zip(blocks, BRM._brm_nlme_sampler_hessian_blocks(p.model, u))
        @test a ≈ b rtol=1e-12 atol=1e-12
    end
end

# Units the position coloring already serves keep it: subjects no observation
# couples (one unit each, two effects), and the strict partition, which records
# no cell supports.
@stestset "independent subjects keep the position coloring" begin
    data = coloring_data(4, [1, 2, 3, 4, 1, 3], [1, 2, 3, 4, 1, 3])
    @test connected_problem(coloring_normal(data)).coloring === nothing
    p = brm_laplace_problem(RKBRMI(coloring_normal(data)); ad_backend=NLME_AD,
        structured=false)
    @test p.coloring === nothing
end
