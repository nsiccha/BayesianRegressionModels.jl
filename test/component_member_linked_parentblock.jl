module MemberLinkedParentBlockTest

# A record member's linked predictor `logit(share) ~ ... + (1 | parentblock(:p) | g)`
# puts its margin in the parent's correlation block and binds `share` on the
# response scale, as the identity spelling `logit_share ~ ...` followed by
# `share = logistic.(logit_share)` does (snag member-linked-lp-39f85b56). The
# two spellings name the same coordinates, so their densities and gradients
# agree at every point, in compiled Stan and in RK, and RK agrees with Stan
# through the coordinate transport.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using ReactiveKernelsDistributionKernels

# The member's own terms come from the record: an intercept, or an intercept
# and a covariate, beside its margin in the parent's block.
@brm function linked_share(settings)
    logit(share) ~ rhs(settings.formula) + (1 | parentblock(:p) | settings.site)
    effect(share, Intercept) ~ Logistic(0.0, 1.0)
    optional(effect(share, :settings_dose)) ~ Normal(0.0, 1.0)
    return sum(share)
end

@brm function identity_share(settings)
    logit_share ~ rhs(settings.formula) + (1 | parentblock(:p) | settings.site)
    effect(logit_share, Intercept) ~ Logistic(0.0, 1.0)
    optional(effect(logit_share, :settings_dose)) ~ Normal(0.0, 1.0)
    share = logistic.(logit_share)
    return sum(share)
end

# No terms of its own: the margin is the whole predictor.
@brm function linked_margin(settings)
    log(rate) ~ 0 + (1 | parentblock(:p) | settings.site)
    return sum(rate)
end

@brm function identity_margin(settings)
    log_rate ~ 0 + (1 | parentblock(:p) | settings.site)
    rate = exp.(log_rate)
    return sum(rate)
end

panels(component) = brm(quote
    root ~ 0 + (1 | p | site)
    for entry in records
        readings[entry.id] ~ $component(entry)
    end
    primary ~ Normal(root, 1)
    y ~ Normal(readings, 1)
end; mod=@__MODULE__)

const site = ["x", "y"]
const records = [(id=:b, formula="1", site=["y", "y", "x"], dose=[0.2, 0.9, 0.4]),
                 (id=:a, formula="1 + dose", site=["x", "y"], dose=[0.7, 0.1])]
const input = (; records, site, primary=[0.1, -0.1], y=[0.4, 0.2])

points(n) = (zeros(n), collect(range(-0.32, 0.41; length=n)), [0.3sin(i) for i in 1:n])

function check_spellings(linked, identity_spelled, name)
    m, twin = panels(linked)(input), panels(identity_spelled)(input)
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    # The member margins join the parent's block: one block, three terms.
    @test only(ranef_blocks(sb)).n_terms == 3
    stan = consumer_stan(m, "member-linked-" * name; mod=@__MODULE__).model
    ref = consumer_stan(twin, "member-identity-" * name; mod=@__MODULE__).model
    names = BridgeStan.param_unc_names(stan)
    @test names == BridgeStan.param_unc_names(ref)
    for u in points(length(names))
        value, grad = BridgeStan.log_density_gradient(stan, u; propto=false)
        expected, ref_grad = BridgeStan.log_density_gradient(ref, u; propto=false)
        @test value ≈ expected
        @test grad ≈ ref_grad
    end
    rk, problem = consumer_problem(m)
    ref_rk, ref_problem = consumer_problem(twin)
    # RK names an intercept-only member's scalar coefficient after the name the
    # predictor binds, `share` rather than the twin's `logit_share`. The
    # coordinates are otherwise named and ordered alike.
    bound(n) = replace(string(n), r"\.share_Intercept\." => ".logit_share_Intercept.")
    @test bound.(coordinate_names(rk.model.layout)) == string.(coordinate_names(ref_rk.model.layout))
    for x in points(rk.model.layout.total)
        value, grad = LogDensityProblems.logdensity_and_gradient(problem, x)
        expected, ref_grad = LogDensityProblems.logdensity_and_gradient(ref_problem, x)
        @test value ≈ expected
        @test grad ≈ ref_grad
    end
    transport = brm_coordinate_transport(rk, sb, names)
    for u in points(length(names))
        value, grad = BridgeStan.log_density_gradient(stan, u; propto=false)
        x = brm_stan_point_to_rk(transport, u)
        rk_value, rk_grad = LogDensityProblems.logdensity_and_gradient(problem, x)
        @test rk_value ≈ brm_stan_logdensity_to_rk(transport, u, value)
        @test rk_grad ≈ brm_stan_gradient_to_rk(transport, u, grad)
    end
    (; m, sb, stan, names)
end

@stestset "A member's linked predictor puts its margin in the parent's block" begin
    (; sb, names) = check_spellings(linked_share, identity_share, "share")
    # Each member's coefficients keep their linked predictor's address.
    d = brm_descriptor(sb)
    for (key, coefficients) in ((:a, (:Intercept, :settings_dose)), (:b, (:Intercept,)))
        for coefficient in coefficients
            r = brm_population_effect_coordinates(d, :share, names; coefficient,
                submodel=(:readings, key))
            @test length(r.coordinates) == 1
            @test r.link === logit
        end
    end
end

@stestset "A linked member predictor that is only its parent-block margin" begin
    check_spellings(linked_margin, identity_margin, "margin")
end

end
