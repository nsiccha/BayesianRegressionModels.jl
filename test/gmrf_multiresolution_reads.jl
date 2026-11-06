@testset "Compressed read-run maps and very short spatial lengths" begin
    rng=MersenneTwister(997)
    x,y=rand(rng,21),rand(rng,21)
    rows=[(p,t) for t in [0.,1.,13.,72.,5000.] for p in 1:21 if rand(rng)>.5]
    g=BRM._brm_multiresolution_support([x[p] for (p,t) in rows],[y[p] for (p,t) in rows],last.(rows),2;rank=4)
    plan=BRM._brm_multiresolution_read_plan(g)
    for p in eachindex(g.spatial.conditionals.east),j in g.spatial.basis_ptr[p]:(g.spatial.basis_ptr[p+1]-1)
        k=g.spatial.basis_node[j];m=plan.mapid[j]
        for r in plan.ptr[m]:(plan.ptr[m+1]-1),i in plan.starts[r]:plan.stops[r]
            @test g.reads.state_time[g.reads.state_ptr[p]+i-1] ==
                g.paths.state_time[g.paths.state_ptr[k]+i+plan.delta[r]-1]
        end
    end
    for ell in (exp(-365.),1e-300,nextfloat(0.))
        b=BRM._brm_multiresolution_basis(g.spatial,ell;derivative=true)
        @test all(isfinite,b.weights)
        @test all(iszero,b.derivative)
        @test count(==(1.),b.weights)==length(g.spatial.conditionals.east)
    end
end

@testset "The concrete intermittent-read regression retains OU covariance" begin
    rng = MersenneTwister(718)
    x,y = rand(rng,41),rand(rng,41)
    e,n,t = Float64[],Float64[],Float64[]
    for h in (0.,1.,2.,3.,6.,72.,73.,74.),p in eachindex(x)
        if rand(rng)>.35
            push!(e,x[p]);push!(n,y[p]);push!(t,h)
        end
    end
    @test length(t)==213
    g = BRM._brm_multiresolution_support(e,n,t;rank=30)
    weights = BRM._brm_multiresolution_basis(g.spatial,.7).weights
    reads,_ = BRM._brm_multiresolution_rows(g,[1],e,n,t,ones(Int,length(t)))
    H = zeros(length(t),g.ncoordinates)
    for (i,r) in enumerate(reads),j in BRM._brm_multiresolution_basis_range(g,r)
        H[i,BRM._brm_multiresolution_read_coordinate(g,r,j)] = weights[j]
    end
    K = H * multires_latent_covariance(g,13.,ones(1,1)) * H'
    @test maximum(abs,diag(K).-1) < 2e-14
    errors = [abs(K[i,j]/K[j,j]-exp(-abs(t[i]-t[j])/13.))
        for i in eachindex(t),j in eachindex(t) if e[i]==e[j] && n[i]==n[j]]
    @test maximum(errors) < 1e-6
end
