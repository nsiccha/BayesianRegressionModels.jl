include("iterate_stopping.jl")
using Random, Test

@testset "fixed-rate iterate diagnostics" begin
    x=randn(Xoshiro(1972),24,2000)
    d=iterate_precision(x,ones(24))
    @test d.passed
    @test d.minimum_ess>=50 && d.maximum_mcse<.1
    # Independent textbook two-half Rhat checks public API orientation.
    a,b=x[:,1:1000],x[:,1001:2000]
    W=(vec(var(a;dims=2))+vec(var(b;dims=2)))/2
    B=1000 .* (vec(mean(a;dims=2))-vec(mean(b;dims=2))).^2/2
    expected=sqrt.((999/1000).*ones(24)+B./(1000W))
    @test d.rh.rhat≈expected atol=1e-12
    factors=collect(2.:25.)
    scaled=iterate_precision(x.*factors,factors)
    @test scaled.normalized_mcse≈d.normalized_mcse atol=1e-12
    @test scaled.rh.rhat≈d.rh.rhat atol=1e-12
    drift=x.+reshape(vcat(fill(-4.,1000),fill(4.,1000)),1,:)
    @test !iterate_precision(drift,ones(24)).passed
    frozen=copy(x);frozen[1,:].=0
    @test iterate_precision(frozen,ones(24)).rh.reason==:constant_coordinate
    invalid=copy(x);invalid[1,1]=NaN
    @test iterate_precision(invalid,ones(24)).rh.reason==:nonfinite
    @test isempty(iterate_windows(200))
    @test iterate_windows(400)==[200,244,290,334,380]
    @test iterate_windows(5000)==[200,1336,2474,3612,4750]
    B=[.8 .1 -.2;.1 .5 .2;.2 -.1 .6];rho=log.([.3,.4,.5])
    raw=hcat(vcat(zeros(3),rho,vec(B)),vcat(zeros(3),rho,vec(-B)))
    gap=exact_iterate_covariance_gap(raw)
    @test gap.gap≈B*B' atol=1e-14
    @test minimum(eigvals(Symmetric(gap.gap)))>=-1e-14
    q=(;μ=[.1,-.2,.3],ρ=rho,Β=B)
    @test length(iterate_distribution(q))==9
    @test length(iterate_scales(q).raw)==15
    println((;maximum_rhat=d.rh.maximum,maximum_mcse=d.maximum_mcse,minimum_ess=d.minimum_ess))
end
