# Vanko et al. (2026), Appendix C, Eqs. 42, 48, 51, 53.
# Test the actual helpers from both authored model forms against boundary
# identities and an independently integrated pregnancy-loss hazard.
using Test, StanBlocks, BridgeStan, Distributions, LogDensityProblems

module SealDensityFormula
include(joinpath(@__DIR__, "..", "research", "seal", "grey_seal_brm.jl"))
probe(d) = @slic d begin
    log_K ~ normal(log(60000.0), 0.3)
    theta0 = compute_density_dependence_intercept(b_max, loss_scaled)
    theta1 = compute_density_dependence_slope(exp(log_K), b_capacity, b_reference, theta0)
    y_birth ~ normal(update_birth_rate(b_reference, theta0, theta1, abundance), 0.05)
    y_pregnancy ~ normal(update_pregnancy_rate(b_reference, theta0, theta1, abundance, time), 0.05)
end
end

module SealDensityBundle
include(joinpath(@__DIR__, "..", "research", "seal", "grey_seal_ipm.jl"))
function probe(d)
    names = ("compute_density_dependence_intercept", "compute_density_dependence_slope",
        "update_birth_rate", "update_pregnancy_rate")
    definitions = [p.slug => (; definition=_grey_seal_part_expr(p.source), markers=:stanonly)
        for p in GREY_SEAL_IPM_PARTS if p.slug in names]
    body = quote
        log_K ~ normal(log(60000.0), 0.3)
        theta0 = compute_density_dependence_intercept(b_max, loss_scaled)
        theta1 = compute_density_dependence_slope(exp(log_K), b_capacity, b_reference, theta0)
        y_birth ~ normal(update_birth_rate(b_reference, theta0, theta1, abundance), 0.05)
        y_pregnancy ~ normal(update_pregnancy_rate(b_reference, theta0, theta1, abundance, time), 0.05)
    end
    compile_slic_bundle(d, Pair{String,Any}[], "body" => body;
        udf_definitions=definitions, name=:seal_density_probe).model
end
end

# Integrate theta_start * exp(theta1*N) * t over pregnancy, normalized so
# survival to birth (t=3/4) at abundance K gives the replacement birth rate.
# This uses the equilibrium identity rather than the implementation's nested exp.
function seal_reference(log_K, d)
    survival_zero = d.b_max + (1 - d.b_max) * d.loss_scaled
    loss_zero = -log(survival_zero)
    loss_at_K = loss_zero + log(d.b_reference / d.b_capacity)
    integrated_loss = loss_zero * (loss_at_K / loss_zero)^(d.abundance / exp(log_K))
    implantation = d.b_reference / survival_zero
    birth = implantation * exp(-integrated_loss)
    pregnancy = implantation * exp(-integrated_loss * (d.time / 0.75)^2)
    (; birth, pregnancy)
end

function seal_probe_density(log_K, d)
    r = seal_reference(log_K, d)
    logpdf(Normal(log(60000.0), 0.3), log_K) +
        logpdf(Normal(r.birth, 0.05), d.y_birth) +
        logpdf(Normal(r.pregnancy, 0.05), d.y_pregnancy)
end

@testset "seal density dependence: paper boundaries, both compiled sources, gradients" begin
    for build in (SealDensityFormula.probe, SealDensityBundle.probe)
        for abundance in (0.0, 3000.0, 60000.0, 120000.0), time in (0.0, 0.125, 0.75)
            d = (; b_max=0.9, loss_scaled=0.4, b_capacity=0.2, b_reference=0.7,
                abundance, time, y_birth=0.3, y_pregnancy=0.6)
            problem = StanBlocks.stan_instantiate(build(d))
            log_K = log(60000.0)
            lp, gradient = BridgeStan.log_density_gradient(problem.model, [log_K];
                propto=false, jacobian=true)
            @test lp ≈ seal_probe_density(log_K, d) atol=1e-9 rtol=1e-12
            h = 1e-5
            finite_difference = (seal_probe_density(log_K+h, d) - seal_probe_density(log_K-h, d))/(2h)
            @test gradient[1] ≈ finite_difference atol=2e-6 rtol=2e-7
            @test isfinite(lp) && all(isfinite, gradient)
            r = seal_reference(log_K, d)
            @test 0 <= r.birth <= r.pregnancy <= 1
            if abundance == 0
                @test r.birth ≈ d.b_reference
                @test abs(gradient[1]) < 1e-10
            elseif abundance == 60000
                @test r.birth ≈ d.b_capacity
                @test abs(gradient[1]) > 1e-4
            end
            time == 0.75 && @test r.pregnancy ≈ r.birth
        end
    end
end

@testset "seal full formula model: finite density and gradient at demographic start" begin
    sb = SealDensityFormula.SBBRMI(SealDensityFormula.grey_seal_brm_model())
    problem = StanBlocks.stan_instantiate(sb)
    start = SealDensityFormula.brm_start(sb, problem.model, (;
        phi_a_sc=0.88, phi_sc=0.85, survival_shape=0.5,
        male_pup_survival_offset=0.0, male_adult_survival_offset=0.0,
        carrying_capacity=60000.0, density_dependence_scaled=0.4,
        max_baseline_birth_rate=0.9, min_baseline_birth_rate_sc=0.7,
        population_init_size=3000.0,
        bbr_reference_logit=0.0,
        bbr_logit=(; herring_index_1=0.0, herring_index_2=0.0),
    ))
    # Call BridgeStan directly so a domain/dimension failure retains its cause.
    lp, gradient = BridgeStan.log_density_gradient(problem.model, start)
    @test isfinite(lp)
    @test all(isfinite, gradient)
end
