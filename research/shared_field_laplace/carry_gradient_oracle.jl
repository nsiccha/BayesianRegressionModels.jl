# Tiny mathematical counterexample for the implicit-mode correction after a
# correlated Gaussian carry replaces independent field priors. No fitting API.
# julia research/shared_field_laplace/carry_gradient_oracle.jl
using LinearAlgebra, Test

# Joint N(0,C) carry for (x,z), followed by y ~ Poisson(exp(x+z)).
# z is integrated by conditional Laplace; x remains explicit. The Poisson
# factorial is x-independent and omitted from every compared density.
function correlated_carry_laplace(x,y,rho)
    C = [1.0 rho;rho 1.0]
    P = inv(Symmetric(C)) # fixed two-coordinate oracle only
    z = 0.0
    for _ in 1:50
        rate = exp(x+z)
        score = -P[2,1]*x-P[2,2]*z+y-rate
        abs(score) < 1e-12 && break
        z += score/(P[2,2]+rate)
    end
    rate = exp(x+z)
    field_score = -P[2,1]*x-P[2,2]*z+y-rate
    abs(field_score) < 1e-10 || error("scalar oracle did not converge")
    u = [x,z]
    joint = -(2log(2pi)+logdet(Symmetric(C))+dot(u,P*u))/2+y*(x+z)-rate
    H = P[2,2]+rate
    value = joint+log(2pi)/2-log(H)/2
    direct = -P[1,1]*x-P[1,2]*z+y-rate
    trace_x = trace_z = -rate/H
    full_mixed = -P[1,2]-rate
    gradient = direct+(trace_x+full_mixed*trace_z/H)/2
    # This is the old independent-prior formula applied incorrectly to carry.
    missing_carry_mixed = direct+(trace_x-rate*trace_z/H)/2
    (;value,gradient,missing_carry_mixed,z,H,field_score,prior_precision=P[2,2])
end

@testset "correlated carry enters the implicit-mode mixed derivative" begin
    errors,wrong_errors = Float64[],Float64[]
    for x in (-.5,.4,1.1), rho in (-.4,.6), y in (2,5)
        c = correlated_carry_laplace(x,y,rho)
        eps = 1e-5
        finite_difference = (correlated_carry_laplace(x+eps,y,rho).value-
            correlated_carry_laplace(x-eps,y,rho).value)/(2eps)
        push!(errors,abs(c.gradient-finite_difference))
        push!(wrong_errors,abs(c.missing_carry_mixed-finite_difference))
        @test c.gradient ≈ finite_difference atol=1e-8
        @test abs(c.missing_carry_mixed-finite_difference) > 1e-3
        @test c.H >= c.prior_precision
    end
    independent = correlated_carry_laplace(.4,3,0.0)
    @test independent.gradient == independent.missing_carry_mixed
    println((;maximum_gradient_error=maximum(errors),
        missing_mixed_error_range=extrema(wrong_errors),
        scope="two-coordinate mathematical control; native carried-target AD unverified"))
end
