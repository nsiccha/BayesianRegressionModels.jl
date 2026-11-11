# The calibrated seal equation has joint support across survival and birth priors.
# Compare both authored helper sets with an independent density and coordinate
# Jacobian, including the implicit derivatives of the mortality support root.
using Test, StanBlocks, BridgeStan, Distributions, LinearAlgebra, Random

module SealSupportFormula
include(joinpath(@__DIR__, "..", "research", "seal", "grey_seal_brm.jl"))
probe(d) = @slic d begin
    phi_a_sc ~ restricted(uniform(0.0, 1.0); lower=(seal_adult_floor(n_age) - 0.8) / 0.2)
    shape ~ uniform(0.0, 1.0)
    adult = 0.8 + 0.2 * phi_a_sc
    pup_scale ~ restricted(uniform(0.0, 1.0); lower=seal_pup_floor(adult, shape, n_age))
    capacity = birth_rate_at_carrying_capacity(adult,
        mortality_rates(adult * pup_scale, adult, shape, n_age, 0.0, 0.0), n_age)
    intercept ~ normal(0.0, 1.0)
    maximum ~ uniform(0.0, 1.0)
    minimum_scale ~ restricted(uniform(0.0, 1.0);
        lower=seal_birth_fraction_lower(capacity, intercept))
    reference_fraction = minimum_scale + (1.0 - minimum_scale) * inv_logit(intercept)
    density_scaled ~ restricted(uniform(0.0, 1.0);
        upper=seal_density_scaled_upper(capacity, maximum, minimum_scale, intercept))
    margin = -log(maximum + (1.0 - maximum) * density_scaled) +
        log(maximum * reference_fraction / capacity)
    y ~ normal(margin, 1.0)
end
end

module SealSupportBundle
include(joinpath(@__DIR__, "..", "research", "seal", "grey_seal_ipm.jl"))
function probe(d)
    names = ("seal_adult_floor", "seal_mortality_support_residual", "seal_pup_floor",
        "seal_birth_fraction_lower", "seal_density_scaled_upper",
        "mortality_rates", "birth_rate_at_carrying_capacity")
    definitions = [p.slug => (; definition=_grey_seal_part_expr(p.source), markers=:stanonly)
        for p in GREY_SEAL_IPM_PARTS if p.slug in names]
    body = quote
        phi_a_sc ~ restricted(uniform(0.0, 1.0); lower=(seal_adult_floor(n_age) - 0.8) / 0.2)
        shape ~ uniform(0.0, 1.0)
        adult = 0.8 + 0.2 * phi_a_sc
        pup_scale ~ restricted(uniform(0.0, 1.0); lower=seal_pup_floor(adult, shape, n_age))
        capacity = birth_rate_at_carrying_capacity(adult,
            mortality_rates(adult * pup_scale, adult, shape, n_age, 0.0, 0.0), n_age)
        intercept ~ normal(0.0, 1.0)
        maximum ~ uniform(0.0, 1.0)
        minimum_scale ~ restricted(uniform(0.0, 1.0);
            lower=seal_birth_fraction_lower(capacity, intercept))
        reference_fraction = minimum_scale + (1.0 - minimum_scale) * inv_logit(intercept)
        density_scaled ~ restricted(uniform(0.0, 1.0);
            upper=seal_density_scaled_upper(capacity, maximum, minimum_scale, intercept))
        margin = -log(maximum + (1.0 - maximum) * density_scaled) +
            log(maximum * reference_fraction / capacity)
        y ~ normal(margin, 1.0)
    end
    compile_slic_bundle(d, Pair{String,Any}[], "body" => body;
        udf_definitions=definitions, name=:seal_joint_support_probe).model
end
end

function original_density(x, n_age, y)
    adult = 0.8 + 0.2 * x["phi_a_sc"]
    pup_mortality, adult_mortality = -log(adult * x["pup_scale"]), -log(adult)
    mortality = [exp((1 - (j / (n_age - 1)) ^ x["shape"]) * log(pup_mortality) +
                     (j / (n_age - 1)) ^ x["shape"] * log(adult_mortality))
                 for j in 0:(n_age - 2)]
    capacity = 2 * (1 - adult) / exp(-sum(mortality))
    fraction = x["minimum_scale"] + (1 - x["minimum_scale"]) / (1 + exp(-x["intercept"]))
    q = x["maximum"] + (1 - x["maximum"]) * x["density_scaled"]
    margin = -log(q) + log(x["maximum"] * fraction / capacity)
    @test 0 < capacity < fraction < 1
    @test margin > 0
    logpdf(Normal(), x["intercept"]) + logpdf(Normal(margin, 1.0), y)
end

rng = MersenneTwister(20261111)
@testset "actual mortality support: original joint law, coordinate Jacobian and native gradients" begin
    for build in (SealSupportFormula.probe, SealSupportBundle.probe), n_age in (2, 3, 6, 12)
        problem = StanBlocks.stan_instantiate(build((; n_age, y=0.2)))
        names = BridgeStan.param_names(problem.model)
        dimension = BridgeStan.param_unc_num(problem.model)
        @test dimension == 7
        points = [zeros(dimension), fill(-2.0, dimension), fill(2.0, dimension)]
        append!(points, [2 .* randn(rng, dimension) for _ in 1:5])
        for q in points
            constrained(v) = BridgeStan.param_constrain(problem.model, v)
            x = Dict(zip(names, constrained(q)))
            oracle = original_density(x, n_age, 0.2)
            plain = BridgeStan.log_density(problem.model, q; propto=false, jacobian=false)
            value, gradient = BridgeStan.log_density_gradient(problem.model, q; propto=false, jacobian=true)
            @test plain ≈ oracle atol=1e-10 rtol=1e-10
            @test isfinite(value) && all(isfinite, gradient)
            h = 1e-5
            jac = zeros(dimension, dimension)
            finite_gradient = zeros(dimension)
            for j in eachindex(q)
                plus, minus = copy(q), copy(q)
                plus[j] += h
                minus[j] -= h
                jac[:, j] = (constrained(plus) - constrained(minus)) / (2h)
                finite_gradient[j] = (BridgeStan.log_density(problem.model, plus; propto=false, jacobian=true) -
                    BridgeStan.log_density(problem.model, minus; propto=false, jacobian=true)) / (2h)
            end
            @test value - plain ≈ logabsdet(jac)[1] atol=2e-6 rtol=2e-7
            @test gradient ≈ finite_gradient atol=2e-5 rtol=2e-6
        end
    end
end
