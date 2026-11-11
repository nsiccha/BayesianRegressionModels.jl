# No model evaluations or refitting. Recover the individual whitened scores
# from saved antithetic mean/covariance residuals and the exact recorded RNG
# sequence. Compare analytic-entropy and STL path-derivative estimators.
# STL: Roeder/Wu/Duvenaud2017, Eq8/Alg2. No extra entropy gradient is added.
using LinearAlgebra,Random,Statistics,Serialization,SHA,Test

upper_entries(A)=[A[i,j] for j in 1:3 for i in 1:j]
function white_contribution(w,xi;stl=false)
    v=stl ? w+xi : w
    A=(v*xi'+xi*v')/2
    stl || (A+=I)
    vcat(v,upper_entries(A))
end
function recover_white_pair(a,b,xi)
    S=zeros(3,3);k=0
    for j in 1:3,i in 1:j
        k+=1;S[i,j]=S[j,i]=b[k]
    end
    S-=I;r=dot(xi,xi)
    r>0 || error("zero Gaussian direction cannot identify the individual scores")
    d=(2S*xi-xi*(dot(xi,S*xi)/r))/r
    residual=maximum(abs.((d*xi'+xi*d')/2-S))
    (;plus=a+d,minus=a-d,residual)
end
function empirical_covariance(A)
    centered=A.-mean(A;dims=2)
    centered*centered'/size(A,2)
end
function directional_ratio(A,B)
    F=cholesky(Symmetric(B);check=false)
    issuccess(F) || return (;available=false,extrema=nothing)
    values=eigvals(Symmetric(F.L\A/F.L'))
    (;available=true,extrema=extrema(values))
end

@testset "saved native Gaussian gradient variance" begin
    base=joinpath(@__DIR__, "results")
    path=joinpath(base,"native-gaussian-block-endpoint-v1/audit.jls")
    audit=deserialize(path)
    native=deserialize(joinpath(base,"native-gaussian-block-fit-v1/comparison.jls"))
    audit.identity.fit_signature==native.identity.signature &&
        native.identity.julia==string(VERSION) || error("saved fit/RNG environment mismatch")
    N=audit.identity.npairs
    N==32768 && audit.identity.seed==1969 && size(audit.pairs)==(12,N) ||
        error("unexpected saved audit shape or RNG sequence")
    expected=vcat(["whitened_mean_$i" for i in 1:3],
        ["covariance_$(i)_$j" for j in 1:3 for i in 1:j])
    audit.labels[1:9]==expected || error("saved residual ordering differs")
    # Exact quadratic controls: path gradient vanishes at a matching Gaussian;
    # complete Gaussian cubature matches the known expectation off the optimum.
    G=[1.5 .2 -.1;.2 .8 .15;-.1 .15 1.2];m=[.2,-.4,.1]
    controls=zeros(9,2)
    for j in 1:3,sign in (-1,1)
        xi=sign*sqrt(3)*Matrix{Float64}(I,3,3)[:,j]
        @test white_contribution(-xi,xi;stl=true)==zeros(9)
        for (k,stl) in enumerate((false,true))
            controls[:,k].+=white_contribution(m-G*xi,xi;stl)/6
        end
    end
    @test controls[:,1]≈vcat(m,upper_entries(I-G)) atol=1e-14
    @test controls[:,2]≈controls[:,1] atol=1e-14
    raw_single=zeros(9,2N);stl_single=similar(raw_single)
    raw_pair=zeros(9,N);stl_pair=similar(raw_pair)
    cv=zeros(9,N);objective=zeros(2,N);rng=Xoshiro(audit.identity.seed)
    reconstruction_error=0.
    for j in 1:N
        # The original audit called randn(rng,3) once per pair. A bulk matrix
        # draw or randn! is not a guaranteed identical Xoshiro draw sequence.
        xi=randn(rng,3)
        r=recover_white_pair(audit.pairs[1:3,j],audit.pairs[4:9,j],xi)
        reconstruction_error=max(reconstruction_error,r.residual)
        raw_single[:,2j-1]=white_contribution(r.plus,xi)
        raw_single[:,2j]=white_contribution(r.minus,-xi)
        stl_single[:,2j-1]=white_contribution(r.plus,xi;stl=true)
        stl_single[:,2j]=white_contribution(r.minus,-xi;stl=true)
        raw_pair[:,j]=(raw_single[:,2j-1]+raw_single[:,2j])/2
        stl_pair[:,j]=(stl_single[:,2j-1]+stl_single[:,2j])/2
        cv[:,j]=vcat(zeros(3),upper_entries(xi*xi'-I))
        # Fully sampled log p-log q instead of log p+analytic entropy.
        # This is a value control variate at the SAME q, not an IWAE weight.
        objective[:,j]=[audit.elbos[1,j],audit.elbos[1,j]+(dot(xi,xi)-3)/2]
    end
    @test reconstruction_error<1e-10
    @test raw_pair≈audit.pairs[1:9,:] atol=1e-10
    @test stl_pair-raw_pair≈cv atol=1e-10
    @test stl_pair[1:3,:]≈raw_pair[1:3,:] atol=1e-12
    covariances=(;raw_single=empirical_covariance(raw_single),stl_single=empirical_covariance(stl_single),
        raw_pair=empirical_covariance(raw_pair),stl_pair=empirical_covariance(stl_pair))
    ratios=(;stl_single=diag(covariances.stl_single)./diag(covariances.raw_single),
        raw_pair_equal_cost=2diag(covariances.raw_pair)./diag(covariances.raw_single),
        stl_pair_equal_cost=2diag(covariances.stl_pair)./diag(covariances.stl_single),
        stl_vs_raw_pair=diag(covariances.stl_pair)./diag(covariances.raw_pair),
        stl_pair_vs_raw_single_equal_cost=2diag(covariances.stl_pair)./diag(covariances.raw_single))
    # Whole-pair blocks keep the dependence of the reconstructed +/- samples.
    # These are descriptive split checks, not independent confidence bounds.
    halves=map((1:N÷2,N÷2+1:N)) do indices
        first_sample,last_sample=2first(indices)-1,2last(indices)
        A=empirical_covariance(stl_pair[:,indices])
        B=empirical_covariance(raw_pair[:,indices])
        S=empirical_covariance(raw_single[:,first_sample:last_sample])
        (;stl_vs_raw_pair=diag(A)./diag(B),stl_pair_vs_raw_single_equal_cost=2diag(A)./diag(S))
    end
    directions=(;stl_single=directional_ratio(covariances.stl_single,covariances.raw_single),
        stl_vs_raw_pair=directional_ratio(covariances.stl_pair,covariances.raw_pair),
        stl_pair_vs_raw_single_equal_cost=directional_ratio(2covariances.stl_pair,covariances.raw_single))
    objective_covariance=empirical_covariance(objective)
    objective_summary=(;average=vec(mean(objective;dims=2)),covariance=objective_covariance,
        mcse=sqrt.(diag(objective_covariance)/(N-1)),
        variance_ratio=objective_covariance[2,2]/objective_covariance[1,1])
    directory=joinpath(base,"saved-gradient-variance-v2")
    ispath(directory) && error("preserve saved gradient variance diagnostic")
    mkpath(directory)
    identity=(;source_signature=bytes2hex(sha256(read(@__FILE__))),
        audit_signature=audit.identity.signature,audit_sha=bytes2hex(sha256(read(path))),
        fit_signature=native.identity.signature,julia=string(VERSION),npairs=N,seed=audit.identity.seed)
    result=(;identity,labels=expected,reconstruction_error,covariances,ratios,halves,directions,objective_summary,
        raw_average=vec(mean(raw_pair;dims=2)),stl_average=vec(mean(stl_pair;dims=2)),
        cv_average=vec(mean(cv;dims=2)),
        scope="saved exact-score oracle at one nonstationary q; no native trace noise, wall-time, refit or causal optimizer attribution")
    serialize(joinpath(directory,"comparison.jls"),result)
    println((;reconstruction_error,directions,objective_summary))
    for i in eachindex(expected)
        println((;label=expected[i],stl_single=ratios.stl_single[i],
            raw_pair_equal_cost=ratios.raw_pair_equal_cost[i],
            stl_vs_raw_pair=ratios.stl_vs_raw_pair[i],
            stl_pair_vs_raw_single_equal_cost=ratios.stl_pair_vs_raw_single_equal_cost[i]))
    end
end
