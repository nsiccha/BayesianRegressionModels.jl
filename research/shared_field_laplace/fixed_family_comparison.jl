# Fixed reference-derived Gaussian candidates on the ACTUAL collapsed target.
# No fitting, no ranking-based follow-up budget. Unit conditional Gaussians,
# including dense covariance, are integrated exactly at each global draw.
# These candidates use the accepted posterior reference and are diagnostics,
# not proposed inference output. A higher ELBO need not mean better moments.
# julia --project=test research/shared_field_laplace/fixed_family_comparison.jl 'fixed family actual target'
include("profile_optimization.jl")

const FAMILY_PAIRS = 8192
const FAMILY_SEED = 1959

function fixed_gaussian_state(mean,covariance)
    length(mean)==10 && size(covariance)==(10,10) || error("bounded Gaussian candidate")
    Cg = covariance[1:4,1:4]
    A = covariance[5:10,1:4]/Cg
    V = covariance[5:10,5:10]-A*covariance[1:4,5:10]
    (;mean,covariance,global_root=Matrix(cholesky(Symmetric(Cg)).L),A,V,
        unit_root=Matrix(cholesky(Symmetric(V)).L),
        entropy=5*(1+log(2pi))+logdet(Symmetric(covariance))/2)
end

function fixed_collapsed_value(state,r,g)
    M = state.mean[5:10]+state.A*(g-state.mean[1:4])
    r.base+dot(r.b,M)-dot(M,r.S*M)/2-dot(r.S,state.V)/2+state.entropy
end

function fixed_native_cubature_value(p,state,r,g)
    M = state.mean[5:10]+state.A*(g-state.mean[1:4])
    ne,nf = length(r.ei),length(r.fi)
    root = zeros(ne+nf,ne+nf)
    root[r.ei,1:ne] = state.unit_root
    root[r.fi,1:ne] = -r.coupling*state.unit_root
    root[r.fi,ne+1:end] = r.F.U\Matrix{Float64}(I,nf,nf)
    center = zeros(ne+nf)
    center[r.ei] = M
    center[r.fi] = r.z0-r.coupling*M
    u = copy(r.c.u)
    value = 0.
    for j in axes(root,2),sign in (-1,1)
        u[p.latent] = center+sign*sqrt(ne+nf)*root[:,j]
        value+=LogDensityProblems.logdensity(p.full,u)/(2(ne+nf))
    end
    value+nf/2*(1+log(2pi))-logdet(r.F)/2+state.entropy
end

@stestset "fixed family actual target" begin
    base = joinpath(@__DIR__, "results")
    surrogate = deserialize(joinpath(base,"dependence-surrogate-v1/controls.jls"))
    trained = deserialize(joinpath(base,"profile-optimization-v1/candidate.jls"))
    reference = deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    @test surrogate.reference_signature==trained.reference_signature==reference.signature
    @test reference.reference_ok
    names = (:full,:conditional_units,:forward_product,:product,:trained_endpoint)
    controls = surrogate.combined
    states = [fixed_gaussian_state(controls.mean,getproperty(controls.covariances,k))
        for k in names[1:4]]
    q = trained.q
    M,variance = audit_unit_state(q)
    C = zeros(10,10)
    C[1:4,1:4] = BRM._brm_svi_global_covariance(q)
    C[5:10,5:10] = Diagonal(variance)
    push!(states,fixed_gaussian_state(vcat(q.μ,M),C))
    # The first three candidates share the exact same global marginal. Reuse
    # its native quadratic evaluation, with different conditional unit moments.
    @test all(s->s.mean[1:4]==states[1].mean[1:4] &&
        s.global_root==states[1].global_root,states[1:3])
    groups = [1,1,1,2,3]
    representatives = [1,4,5]
    directory = get(ENV,"KB_LAPLACE_FAMILY_DIR",joinpath(base,"fixed-family-v1"))
    ispath(directory) && error("preserve existing fixed-family comparison")
    mkpath(directory)
    with_progress(:state;description="Fixed-family comparison on the actual target") do progress
        snapshot_progress(progress,joinpath(directory,"family.progress")) do
            fixture = with_progress(progress;description="Prepare native exact-latent oracle") do _
                (;rk,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,
                    spatial=2,prepare_model=false)
                exact_gaussian_reference(rk,data)
            end
            (;p,view,field,units) = fixture
            @test vcat(p.globals,units)==surrogate.coordinates
            with_progress(progress;description="Validate dense conditional unit expectation") do _
                for i in (1,5)
                    state = states[i]
                    g = state.mean[1:4]+state.global_root*[.3,-.5,.7,-.2]
                    r = profile_quadratic(p,field,units,g)
                    expected = fixed_collapsed_value(state,r,g)
                    cubature = fixed_native_cubature_value(p,state,r,g)
                    @test expected ≈ cubature atol=2e-10
                    if i==5
                        old = gaussian_unit_expectation(p,field,units,g,M,variance;score=false)
                        @test expected ≈ old.value+audit_entropy(state.global_root,variance) atol=2e-10
                    end
                end
            end
            pairs = zeros(length(names),FAMILY_PAIRS)
            rng = Xoshiro(FAMILY_SEED)
            with_progress(progress,FAMILY_PAIRS;description="Fixed paired candidate evaluations") do node
                for j in 1:FAMILY_PAIRS
                    xi = randn(rng,4)
                    for sign in (-1,1)
                        globals = [states[i].mean[1:4]+sign*states[i].global_root*xi
                            for i in representatives]
                        records = [profile_quadratic(p,field,units,g) for g in globals]
                        for i in eachindex(names)
                            k = groups[i]
                            pairs[i,j]+=fixed_collapsed_value(states[i],records[k],globals[k])/2
                        end
                    end
                    j%128==0 && update_progress!(node,j)
                end
            end
            summary = paired_summary(pairs)
            differences = pairs[1:4,:] .- pairs[5:5,:]
            contrast = paired_summary(differences)
            critical = quantile(TDist(FAMILY_PAIRS-1),1-.05/(2size(differences,1)))
            lower = contrast.average-critical*contrast.mcse
            upper = contrast.average+critical*contrast.mcse
            batches = [paired_summary(differences[:,(b-1)*512+1:b*512]) for b in 1:16]
            batch_mcse = vec(std(hcat(getproperty.(batches,:average)...);dims=2))/4
            squared = (differences .- contrast.average).^2
            total = vec(sum(squared;dims=2))
            variance_share = map(vec(maximum(squared;dims=2)),total) do largest,t
                iszero(t) ? 0. : largest/t
            end
            @test all(isfinite,pairs) && all(isfinite,variance_share)
            sources = [@__FILE__,joinpath(@__DIR__,"profile_optimization.jl"),
                joinpath(@__DIR__,"stationarity_audit.jl"),joinpath(@__DIR__,"exact_reference.jl"),
                joinpath(@__DIR__,"operator_probe.jl"),joinpath(@__DIR__,"../../test/Manifest.toml")]
            signature = bytes2hex(sha256(reduce(vcat,read.(sources))))
            result = (;names,states,pairs,summary,contrast,lower,upper,critical,
                batches,batch_mcse,variance_share,signature,seed=FAMILY_SEED,npairs=FAMILY_PAIRS,
                surrogate_signature=surrogate.signature,candidate_signature=trained.signature,
                reference_signature=reference.signature,
                scope="fixed reference-derived distributions; no fit, no family-optimality claim")
            serialize(joinpath(directory,"comparison.jls"),result)
            for i in eachindex(names)
                println((;family=names[i],elbo=summary.average[i],mcse=summary.mcse[i]))
            end
            for i in 1:4
                println((;family=names[i],gain_over_trained=contrast.average[i],
                    mcse=contrast.mcse[i],lower=lower[i],upper=upper[i],
                    batch_mcse=batch_mcse[i],variance_share=variance_share[i]))
            end
        end
    end
end
