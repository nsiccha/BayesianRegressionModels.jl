# Shared numerical checks, also runnable with the three pure core source files
# included in a module named by BRM. No backend/compiler is needed here.
function multires_latent_covariance(g, theta, L)
    n = length(g.paths.state_time)
    K = zeros(n, n)
    for p in 1:(length(g.paths.state_ptr) - 1)
        rr = g.paths.state_ptr[p]:(g.paths.state_ptr[p + 1] - 1)
        for i in rr, j in rr
            K[i,j] = exp(-abs(g.times[g.paths.state_time[i]] - g.times[g.paths.state_time[j]]) / theta)
        end
    end
    kron(K, L * L')
end

@testset "Shared OU factors reproduce the full separable law on a small irregular support" begin
    rng = MersenneTwister(857)
    x, y = rand(rng, 7), rand(rng, 7)
    rows = [(p,t) for t in (0.,1.,13.,72.,105.) for p in 1:7 if rand(rng) > .3]
    east, north, time = [x[p] for (p,t) in rows], [y[p] for (p,t) in rows], last.(rows)
    g = BRM._brm_multiresolution_support(east, north, time, 2; rank=7)
    ell, theta = .7, 13.
    b = BRM._brm_multiresolution_basis(g.spatial, ell).weights
    L = [1. 0.; .4 sqrt(.84)]
    reads = [BRM._brm_multiresolution_read(g, e, n, t, c)
        for (e,n,t) in zip(east,north,time) for c in 1:2]
    Istate = Matrix{Float64}(I, g.ncoordinates, g.ncoordinates)
    H = [BRM._brm_multiresolution_value(view(Istate,:,j),b,g,r) for r in reads, j in 1:g.ncoordinates]
    actual = H * multires_latent_covariance(g, theta, L) * H'
    Omega = L * L'
    f = g.spatial.conditionals
    exact = [begin
        d = hypot(f.east[a.position]-f.east[z.position], f.north[a.position]-f.north[z.position])
        BRM._brm_vecchia_matern(sqrt(2.)*d/ell) * exp(-abs(g.times[a.time]-g.times[z.time])/theta) * Omega[a.channel,z.channel]
    end for a in reads, z in reads]
    @test actual ≈ exact atol=3e-12 rtol=3e-12
    @test maximum(abs.(diag(actual) .- 1)) < 2e-14
    @test g.ncoordinates == 2length(g.paths.state_time)
    @test_throws ArgumentError BRM._brm_multiresolution_read(g, east[1], north[1], 2.)
    @test_throws ArgumentError BRM._brm_multiresolution_read(g, east[1], north[1], time[1], 3)
end

@testset "Sparse spatial basis: variance, smooth derivative, stable identity and bounded width" begin
    rng = MersenneTwister(231)
    x, y = rand(rng, 137), rand(rng, 137)
    g = BRM._brm_multiresolution_spatial(x,y; rank=6)
    h = 1e-6
    b = BRM._brm_multiresolution_basis(g,.3; derivative=true)
    plus = BRM._brm_multiresolution_basis(g,.3+h).weights
    minus = BRM._brm_multiresolution_basis(g,.3-h).weights
    @test b.derivative ≈ (plus-minus)/(2h) atol=3e-7 rtol=3e-6
    for p in eachindex(g.conditionals.east)
        rr = g.basis_ptr[p]:(g.basis_ptr[p+1]-1)
        @test sum(abs2,b.weights[rr]) ≈ 1 atol=1e-14
        @test abs(dot(b.weights[rr],b.derivative[rr])) < 1e-12
        @test length(rr) <= 1+6*(1+floor(Int,log2(p)))
        @test g.basis_node[last(rr)] == p
    end
    perm = randperm(rng,length(x))
    shuffled = BRM._brm_multiresolution_spatial(x[perm],y[perm];rank=6)
    @test g.conditionals.east == shuffled.conditionals.east
    @test g.basis_ptr == shuffled.basis_ptr
    @test g.basis_node == shuffled.basis_node
    @test_throws DomainError BRM._brm_multiresolution_basis(g,0.)
    @test_throws ArgumentError BRM._brm_multiresolution_spatial([NaN],[0.])
end

@testset "Compressed time unions and exact OU covariance across sparse episodes" begin
    x = collect(1.:100.)
    # All global times differ: metadata must not allocate a position × time grid.
    g = BRM._brm_multiresolution_support(x,zero(x),x;rank=3)
    @test length(g.paths.state_time) <= 100*(1+3*(1+floor(Int,log2(100))))
    @test length(g.paths.bits) <= length(g.paths.state_time)
    @test_throws ArgumentError BRM._brm_multiresolution_read(g,1.,0.,2.)
    e,n,t = Float64[],Float64[],Float64[]
    for k in 0:199, p in 1:9
        (p == 1 || (k+p)%3 != 0) || continue
        push!(e,p);push!(n,sin(p));push!(t,k<=150 ? k : k+10000.)
    end
    g = BRM._brm_multiresolution_support(e,n,t;rank=3)
    b = BRM._brm_multiresolution_basis(g.spatial,2.).weights
    for lag in (1.,13.,72.,10199.)
        a = BRM._brm_multiresolution_read(g,1.,sin(1),0.)
        z = BRM._brm_multiresolution_read(g,1.,sin(1),lag)
        @test BRM._brm_multiresolution_covariance(g,b,a,z,13.) ≈ exp(-lag/13.) atol=2e-15
    end
    duplicate = BRM._brm_multiresolution_support(vcat(e,e),vcat(n,n),vcat(t,t);rank=3)
    @test duplicate.ncoordinates == g.ncoordinates
    @test duplicate.paths.state_time == g.paths.state_time
end

@testset "Normalized OU density and analytic derivatives against independent Gaussian algebra" begin
    rng = MersenneTwister(983)
    g = BRM._brm_multiresolution_support([0.,1.,0.,2.,1.,0.],zeros(6),[0.,1.,3.,3.,9.,72.],2;rank=2)
    L = [1.2 0.;-.3 .8]
    theta = 3.7
    field = .3randn(rng,g.ncoordinates)
    before = copy(field)
    r = BRM._brm_multiresolution_density(field,theta,L,g;derivative=true)
    K = multires_latent_covariance(g,theta,L)
    @test r.value ≈ -(dot(field,K\field)+logdet(Symmetric(K))+length(field)*log(2pi))/2 atol=2e-12
    @test r.field ≈ -(K\field) atol=2e-12
    h = 1e-5
    value(f,t,A) = BRM._brm_multiresolution_density(f,t,A,g).value
    @test r.temporal_length ≈ (value(field,theta+h,L)-value(field,theta-h,L))/(2h) atol=2e-9
    for c in 1:2, k in 1:c
        D = zeros(2,2);D[c,k]=h
        @test r.L[c,k] ≈ (value(field,theta,L+D)-value(field,theta,L-D))/(2h) atol=3e-9
    end
    @test field == before
    @test isfinite(BRM._brm_multiresolution_density(field,1e-300,L,g;derivative=true).temporal_length)
end
