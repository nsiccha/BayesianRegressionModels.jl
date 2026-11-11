# Diagnose the SAVED fit; no optimization, reference resampling or gate changes.
# The dense Gaussian oracle is hard-bounded in exact_reference.jl. It is never
# an implementation of the scalable target. Unit expectations are exact; only
# the four-global Gaussian expectation has Monte Carlo uncertainty.
# julia --project=test research/shared_field_laplace/stationarity_audit.jl 'saved SVI stationarity'
include("exact_reference.jl")
using Serialization, Statistics, SHA

const AUDIT_PAIRS = 64
const AUDIT_SEED = 1953
const AUDIT_PRACTICAL_TOLERANCE = .05
const AUDIT_FIT_SIGNATURE = "d19d528f00b5046188490bb779800e4980d182a86d6f3b78d54c9dee6fccbf3d"

# At fixed g integrate only the fields, then average over the held-fixed q_eta.
# The native global partial score is quadratic in the 22 Gaussian latents.
# Its symmetric 44-node cubature is therefore exact (Fisher identity), with
# no differentiation of the conditional nodes and no field/unit independence.
function gaussian_unit_expectation(p, field, units, g, M, variance; score=true)
    c = gaussian_reference_conditional(p,g)
    fi = [only(findall(==(i),p.latent)) for i in field]
    ei = [only(findall(==(i),p.latent)) for i in units]
    length(M) == length(ei) == length(variance) || error("unit shape mismatch")
    F = cholesky(Symmetric(c.H[fi,fi]))
    Hfe = c.H[fi,ei]
    coupling = F \ Hfe
    S = c.H[ei,ei] - Hfe' * coupling
    latent_mean = zeros(length(p.latent))
    latent_mean[ei] = M
    latent_mean[fi] = F \ (c.h[fi] - Hfe*M)
    u = copy(c.u); u[p.latent] = latent_mean
    value = LogDensityProblems.logdensity(p.full,u) + length(fi)/2*log(2pi) -
        logdet(F)/2 - dot(diag(S),variance)/2
    unit_score = c.h[ei] - Hfe'*(F\c.h[fi]) - S*M
    unit_logsd = 1 .- variance .* diag(S) # includes unit entropy
    score || return (;value,unit_score,unit_logsd)
    ne,nf = length(ei),length(fi)
    root = zeros(length(p.latent),ne+nf)
    Leta = Diagonal(sqrt.(variance))
    root[ei,1:ne] = Leta
    root[fi,1:ne] = -coupling*Leta
    root[fi,ne+1:end] = F.U \ Matrix{Float64}(I,nf,nf)
    w = zeros(length(g))
    for j in axes(root,2), sign in (-1,1)
        u[p.latent] = latent_mean + sign*sqrt(ne+nf)*root[:,j]
        _,full_score = LogDensityProblems.logdensity_and_gradient(p.full,u)
        w .+= full_score[p.globals] ./ (2(ne+nf))
    end
    (;value,unit_score,unit_logsd,w,S,root,latent_mean)
end

function audit_unit_state(q)
    all(length(m)==1 for m in q.M) || error("audit requires scalar unit blocks")
    M = only.(q.M)
    variance = [1/only(P) for P in q.Λ]
    all(>(0),variance) || error("invalid unit precision")
    M,variance
end

function audit_entropy(C,variance)
    (size(C,1)+length(variance))/2*(1+log(2pi)) +
        sum(log,diag(C)) + sum(log,variance)/2
end

# Complete expected ELBO for a held-out, common-random-number directional
# derivative. Matrix A acts in whitened global coordinates; T rescales a row
# in physical coordinates. Both transformations include their entropy change.
function directional_elbo(p,field,units,q,C,M,variance,epsilons,zetas,kind,t)
    D = length(q.μ)
    A = Matrix{Float64}(I,D,D)
    T = Matrix{Float64}(I,D,D)
    mean_shift = zeros(D)
    Mt,Vt = copy(M),copy(variance)
    if kind === :global_mean
        mean_shift = t*C[:,1]
    elseif kind === :global_row_scale
        T[1,1] = exp(t)
    elseif kind === :global_covariance
        A[1,2] = t/2; A[2,1] = t/2
    elseif kind === :unit_mean
        Mt[1] += t*sqrt(variance[1])
    elseif kind === :unit_logsd
        Vt[1] *= exp(2t)
    else
        error("unknown direction $kind")
    end
    value = 0.0
    for j in axes(epsilons,2), sign in (-1,1)
        offset = sign*(exp.(q.ρ).*epsilons[:,j]+q.Β*zetas[:,j])
        gt = q.μ + mean_shift + T*C*A*(C\offset)
        value += gaussian_unit_expectation(p,field,units,gt,Mt,Vt;score=false).value
    end
    value/(2size(epsilons,2)) + audit_entropy(C,Vt) + logdet(T) + logdet(A)
end

function audit_point(p,field,units,q,C,M,variance,epsilon,zeta)
    g = BRM._brm_svi_draw_global(q,epsilon,zeta)
    a = gaussian_unit_expectation(p,field,units,g,M,variance)
    xi = C\(g-q.μ)
    whitened_score = C'*a.w
    covariance = Matrix(Symmetric((whitened_score*xi' + xi*whitened_score')/2)) + I
    row_scale = 1 .+ a.w .* (g-q.μ)
    unit_mean = sqrt.(variance).*a.unit_score
    packed = BRM._brm_svi_global_gradient(q,a.w,epsilon,zeta)
    elbo = a.value + audit_entropy(C,variance)
    # Derivative of the symmetric off-diagonal transform has A12=A21=1/2.
    directions = [whitened_score[1],row_scale[1],covariance[1,2],
        unit_mean[1],a.unit_logsd[1]]
    residual = vcat(whitened_score,
        [covariance[i,j] for j in axes(covariance,2) for i in 1:j],
        row_scale,unit_mean,a.unit_logsd)
    (;elbo,residual,packed,directions,root=a.root,S=a.S)
end

function paired_summary(x)
    n = size(x,2)
    average = vec(mean(x;dims=2))
    mcse = vec(std(x;dims=2))/sqrt(n)
    centered = x .- average
    (;average,mcse,covariance_of_mean=centered*centered'/((n-1)*n))
end

@stestset "saved SVI stationarity" begin
    base = joinpath(@__DIR__, "results")
    fitdir = joinpath(base,"joint-fit-v1")
    reference = deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    saved = deserialize(joinpath(fitdir,"fit.jls"))
    comparison = deserialize(joinpath(fitdir,"comparison.jls"))
    @test reference.reference_ok
    @test saved.signature == comparison.signature == AUDIT_FIT_SIGNATURE
    @test !comparison.accuracy_ok
    directory = get(ENV,"KB_LAPLACE_AUDIT_DIR",joinpath(base,"stationarity-v1"))
    ispath(directory) && error("preserve existing audit directory: $directory")
    mkpath(directory)
    with_progress(:state;description="Audit saved collapsed SVI fit") do progress
        snapshot_progress(progress,joinpath(directory,"audit.progress")) do
            fixture = with_progress(progress;description="Prepare native Gaussian oracle") do _
                (;rk,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,
                    spatial=2,prepare_model=false)
                exact_gaussian_reference(rk,data)
            end
            (;p,view,field,units) = fixture
            @test view.coordinates == comparison.coordinates == reference.coordinates
            @test units == reduce(vcat,view.subject_coordinates)
            q = saved.fit.q
            M,variance = audit_unit_state(q)
            C = Matrix(cholesky(Symmetric(Matrix(BRM._brm_svi_global_covariance(q)))).L)
            D = length(q.μ)
            @test D == 4 && length(M)==6 && length(field)==16
            kinds = (:global_mean,:global_row_scale,:global_covariance,:unit_mean,:unit_logsd)
            heldout_rng = Xoshiro(1954)
            epsilons = randn(heldout_rng,D,2)
            zetas = randn(heldout_rng,size(q.Β,2),2)
            derivative_checks = with_progress(progress;description="Check held-out ELBO derivatives") do _
                points = [audit_point(p,field,units,q,C,M,variance,
                    s*epsilons[:,j],s*zetas[:,j]) for j in 1:2 for s in (-1,1)]
                analytic = sum(a.directions for a in points)/length(points)
                checks = map((1e-3,5e-4)) do h
                    fd = [(directional_elbo(p,field,units,q,C,M,variance,epsilons,zetas,k,h)-
                        directional_elbo(p,field,units,q,C,M,variance,epsilons,zetas,k,-h))/(2h)
                        for k in kinds]
                    @test fd ≈ analytic atol=2e-5 rtol=2e-5
                    (;h,fd,analytic,max_error=maximum(abs.(fd-analytic)))
                end
                println((;stage="directional_checks",checks))
                checks
            end
            labels = vcat(["global_mean:$i" for i in 1:D],
                ["global_covariance:$i,$j" for j in 1:D for i in 1:j],
                ["global_row_scale:$i" for i in 1:D],
                ["unit_mean:$i" for i in eachindex(M)],
                ["unit_logsd:$i" for i in eachindex(M)])
            pairs = Matrix{Float64}(undef,length(labels),AUDIT_PAIRS)
            packed_pairs = Matrix{Float64}(undef,length(BRM._brm_svi_global_vector(q)),AUDIT_PAIRS)
            elbos = zeros(1,AUDIT_PAIRS)
            rng = Xoshiro(AUDIT_SEED)
            with_progress(progress,AUDIT_PAIRS;description="Fixed independent antithetic pairs") do node
                for j in 1:AUDIT_PAIRS
                    epsilon,zeta = randn(rng,D),randn(rng,size(q.Β,2))
                    plus = audit_point(p,field,units,q,C,M,variance,epsilon,zeta)
                    minus = audit_point(p,field,units,q,C,M,variance,-epsilon,-zeta)
                    pairs[:,j] = (plus.residual+minus.residual)/2
                    packed_pairs[:,j] = (plus.packed+minus.packed)/2
                    elbos[1,j] = (plus.elbo+minus.elbo)/2
                    update_progress!(node,j)
                end
            end
            summary,packed,elbo = paired_summary(pairs),paired_summary(packed_pairs),paired_summary(elbos)
            # Bonferroni Student intervals are an approximate simultaneous MC
            # diagnostic, not a finite-sample coverage guarantee. Each pair is
            # ONE independent observation. Wide intervals mean inconclusive.
            critical = quantile(TDist(AUDIT_PAIRS-1),1-.05/(2length(labels)))
            lower = summary.average - critical*summary.mcse
            upper = summary.average + critical*summary.mcse
            material = (lower .> AUDIT_PRACTICAL_TOLERANCE) .|
                (upper .< -AUDIT_PRACTICAL_TOLERANCE)
            locally_small = all((lower .>= -AUDIT_PRACTICAL_TOLERANCE) .&
                (upper .<= AUDIT_PRACTICAL_TOLERANCE))
            verdict = any(material) ? "material nonstationarity" :
                locally_small ? "small local residuals" : "inconclusive"
            @test all(isfinite,pairs) && all(isfinite,packed_pairs) && all(isfinite,elbos)
            sources = [@__FILE__,joinpath(@__DIR__,"exact_reference.jl"),
                joinpath(@__DIR__,"operator_probe.jl"),joinpath(@__DIR__,"../../test/Manifest.toml")]
            signature = bytes2hex(sha256(reduce(vcat,read.(sources))))
            result = (;verdict,labels,summary,packed,elbo,lower,upper,material,critical,
                pairs,packed_pairs,elbos,derivative_checks,signature,
                fit_signature=saved.signature,reference_signature=reference.signature,
                seed=AUDIT_SEED,npairs=AUDIT_PAIRS,practical_tolerance=AUDIT_PRACTICAL_TOLERANCE,
                units,field,globals=p.globals,coordinates=view.coordinates,
                q=(;mu=q.μ,rho=q.ρ,B=q.Β,M,variance),
                scope="hard-bounded exact-unit-expectation audit; saved q unchanged")
            serialize(joinpath(directory,"audit.jls"),result)
            println((;stage="audit",verdict,elbo=elbo.average,elbo_mcse=elbo.mcse,
                critical,material_labels=labels[material]))
            for i in eachindex(labels)
                println((;label=labels[i],mean=summary.average[i],mcse=summary.mcse[i],
                    lower=lower[i],upper=upper[i]))
            end
        end
    end
end
