# Bounded diagnostic optimization in the UNCHANGED variational family.
# A four-global Gaussian is parametrized without redundant rho/B coordinates;
# independent scalar unit factors are profiled at every objective evaluation.
# This dense, exact-latent oracle never enters the production fit path.
# julia --project=test research/shared_field_laplace/profile_optimization.jl 'bounded profile optimization'
include("stationarity_audit.jl")

const PROFILE_PAIRS = 512
const PROFILE_SEED = 1956
const PROFILE_EVALUATIONS = 50 # includes initial evaluation and line-search trials

function profile_coordinates(x,mu0,C0)
    D = length(mu0)
    D == 4 && length(x)==14 || error("bounded four-global oracle only")
    L = zeros(D,D)
    for i in 1:D
        L[i,i] = exp(x[D+i])
    end
    k = 2D
    for j in 1:D, i in j+1:D
        k+=1; L[i,j]=x[k]
    end
    (;mu=mu0+C0*x[1:D],C=C0*L,L)
end

function profile_quadratic(p,field,units,g)
    c = gaussian_reference_conditional(p,g)
    fi = [only(findall(==(i),p.latent)) for i in field]
    ei = [only(findall(==(i),p.latent)) for i in units]
    F = cholesky(Symmetric(c.H[fi,fi]))
    Hfe = c.H[fi,ei]
    coupling = F\Hfe
    z0 = F\c.h[fi]
    S = c.H[ei,ei]-Hfe'*coupling
    b = c.h[ei]-Hfe'*z0
    base = c.f0+dot(c.h[fi],z0)/2+length(fi)/2*log(2pi)-logdet(F)/2
    (;c,fi,ei,F,coupling,z0,S,b,base)
end

function profile_score(p,record,M,variance)
    (;c,fi,ei,F,coupling,z0) = record
    nf,ne = length(fi),length(ei)
    latent_mean = zeros(nf+ne)
    latent_mean[ei] = M
    latent_mean[fi] = z0-coupling*M
    root = zeros(nf+ne,nf+ne)
    Leta = Diagonal(sqrt.(variance))
    root[ei,1:ne] = Leta
    root[fi,1:ne] = -coupling*Leta
    root[fi,ne+1:end] = F.U\Matrix{Float64}(I,nf,nf)
    u = copy(c.u)
    score = zeros(length(p.globals))
    for j in axes(root,2), sign in (-1,1)
        u[p.latent] = latent_mean+sign*sqrt(nf+ne)*root[:,j]
        _,native = LogDensityProblems.logdensity_and_gradient(p.full,u)
        score .+= native[p.globals]./(2(nf+ne))
    end
    score
end

function profile_objective(p,field,units,mu0,C0,points,x;gradient=true)
    (;mu,C,L) = profile_coordinates(x,mu0,C0)
    n = size(points,2)
    n == 2PROFILE_PAIRS || error("fixed sample budget changed")
    records = [profile_quadratic(p,field,units,mu+C*points[:,j]) for j in 1:n]
    S = sum(r.S for r in records)/n
    b = sum(r.b for r in records)/n
    M = Symmetric(S)\b
    variance = 1 ./ diag(S)
    base = sum(r.base for r in records)/n
    value = base+dot(b,M)-dot(M,S*M)/2-dot(diag(S),variance)/2+
        audit_entropy(C,variance)
    gradient || return (;value,M,variance,mu,C,S,b,base)
    # Envelope theorem: hold profiled units fixed in each native partial score.
    # g=mu0+C0*a+C0*L*xi. The entropy adds L^-T to the derivative in L.
    da,dL = zeros(4),zeros(4,4)
    for j in 1:n
        white_score = C0'*profile_score(p,records[j],M,variance)
        da .+= white_score/n
        dL .+= white_score*points[:,j]'/n
    end
    dL .+= inv(L)'
    grad = vcat(da,[dL[i,i]*L[i,i] for i in 1:4],
        [dL[i,j] for j in 1:4 for i in j+1:4])
    (;value,gradient=grad,M,variance,mu,C,S,b,base)
end

function profile_as_q(template,result)
    q = deepcopy(template)
    covariance = result.C*result.C'
    floor = .01eigmin(Symmetric(covariance))
    floor>0 || error("nonpositive oracle global covariance")
    size(q.Β)==(4,4) || error("unchanged full-rank four-global family required")
    q.μ .= result.mu
    q.ρ .= log(floor)/2
    q.Β .= cholesky(Symmetric(covariance-floor*I)).L
    for i in eachindex(q.M)
        q.M[i][1] = result.M[i]
        q.Λ[i][1,1] = 1/result.variance[i]
        q.U[i][1,1] = 1/sqrt(result.variance[i])
    end
    q
end

# Inverse BFGS, restricted to 14 oracle coordinates, with Armijo backtracking.
# The cap counts EVERY trial. A failed search or exhausted cap is an outcome,
# not a reason to extend this experiment or change its initialization.
function bounded_profile_bfgs(evaluate,x0,initial;node=nothing,checkpoint=nothing)
    x,current = copy(x0),initial
    B = Matrix{Float64}(I,length(x),length(x))
    evaluations = 1
    trace = [(;evaluations,value=current.value,gradient_norm=norm(current.gradient,Inf),step=0.)]
    reason = "evaluation budget"
    while evaluations<PROFILE_EVALUATIONS
        if norm(current.gradient,Inf)<=1e-5
            reason="surrogate gradient tolerance"; break
        end
        grad = -current.gradient
        direction = -B*grad
        if dot(direction,grad)>=0
            B = Matrix{Float64}(I,length(x),length(x))
            direction = -grad
        end
        direction .*= min(1.,1/norm(direction))
        slope = dot(direction,grad)
        accepted = false
        for backtrack in 0:11
            evaluations>=PROFILE_EVALUATIONS && break
            alpha = 2.0^-backtrack
            candidate_x = x+alpha*direction
            candidate = evaluate(candidate_x)
            evaluations+=1
            node===nothing || update_progress!(node,evaluations)
            if isfinite(candidate.value) && all(isfinite,candidate.gradient) &&
                    -candidate.value <= -current.value+1e-4*alpha*slope
                s = candidate_x-x
                y = -candidate.gradient-grad
                curvature = dot(s,y)
                if curvature>1e-10*norm(s)*norm(y)
                    V = I-s*y'/curvature
                    B = Matrix(Symmetric(V*B*V'+s*s'/curvature))
                else
                    B = Matrix{Float64}(I,length(x),length(x))
                end
                x,current = candidate_x,candidate
                push!(trace,(;evaluations,value=current.value,
                    gradient_norm=norm(current.gradient,Inf),step=alpha))
                checkpoint===nothing || checkpoint((;x,current,evaluations,trace))
                accepted = true
                break
            end
        end
        if !accepted
            reason = evaluations>=PROFILE_EVALUATIONS ? "evaluation budget" : "line search failed"
            break
        end
    end
    (;x,current,evaluations,trace,reason)
end

@stestset "bounded profile optimization" begin
    base = joinpath(@__DIR__, "results")
    saved = deserialize(joinpath(base,"joint-fit-v1/fit.jls"))
    reference = deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    @test saved.signature==AUDIT_FIT_SIGNATURE && reference.reference_ok
    directory = get(ENV,"KB_LAPLACE_PROFILE_DIR",joinpath(base,"profile-optimization-v1"))
    ispath(directory) && error("preserve existing profile optimization: $directory")
    mkpath(directory)
    with_progress(:state;description="Bounded same-family diagnostic optimization") do progress
        snapshot_progress(progress,joinpath(directory,"optimizer.progress")) do
            fixture = with_progress(progress;description="Prepare native exact-latent oracle") do _
                (;rk,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,
                    spatial=2,prepare_model=false)
                exact_gaussian_reference(rk,data)
            end
            (;p,view,field,units) = fixture
            @test view.coordinates==reference.coordinates
            q = saved.fit.q
            mu0 = copy(q.μ)
            C0 = Matrix(cholesky(Symmetric(Matrix(BRM._brm_svi_global_covariance(q)))).L)
            positive = randn(Xoshiro(PROFILE_SEED),4,PROFILE_PAIRS)
            points = hcat(positive,-positive)
            x0 = zeros(14)
            evaluate(x;gradient=true) = profile_objective(p,field,units,mu0,C0,points,x;gradient)
            initial = with_progress(progress;description="Profile units and check envelope derivatives") do _
                value = evaluate(x0)
                directions = [[i==1 ? .6 : i==5 ? -.3 : i==9 ? .4 : 0. for i in 1:14],
                    [i==4 ? .8 : i==14 ? -.6 : 0. for i in 1:14]]
                checks = map(Iterators.product(directions,(1e-4,5e-5))) do pair
                    d,h = pair
                    fd = (evaluate(x0+h*d;gradient=false).value-
                        evaluate(x0-h*d;gradient=false).value)/(2h)
                    analytic = dot(value.gradient,d)
                    @test isapprox(fd,analytic;atol=3e-5,rtol=3e-5)
                    (;h,fd,analytic,error=fd-analytic)
                end
                oldM,oldV = audit_unit_state(q)
                original = value.base+dot(value.b,oldM)-dot(oldM,value.S*oldM)/2-
                    dot(diag(value.S),oldV)/2+audit_entropy(C0,oldV)
                @test value.value>=original-1e-10
                println((;stage="profile_control",original,profiled=value.value,
                    gain=value.value-original,checks))
                serialize(joinpath(directory,"unit-only.jls"),(;q=profile_as_q(q,value),
                    result=value,original,checks,fit_signature=saved.signature,
                    seed=PROFILE_SEED,npairs=PROFILE_PAIRS))
                value
            end
            optimizer = with_progress(progress,PROFILE_EVALUATIONS;
                    description="Fixed evaluation budget including line searches") do node
                bounded_profile_bfgs(evaluate,x0,initial;node,
                    checkpoint=r->serialize(joinpath(directory,"accepted-iterate.jls"),r))
            end
            candidate = profile_as_q(q,optimizer.current)
            @test optimizer.evaluations<=PROFILE_EVALUATIONS
            @test optimizer.current.value>=initial.value-1e-10
            @test BRM._brm_svi_global_covariance(candidate) ≈
                optimizer.current.C*optimizer.current.C' atol=1e-12
            sources = [@__FILE__,joinpath(@__DIR__,"stationarity_audit.jl"),
                joinpath(@__DIR__,"exact_reference.jl"),joinpath(@__DIR__,"operator_probe.jl"),
                joinpath(@__DIR__,"../../src/nlme_svi.jl"),joinpath(@__DIR__,"../../test/Manifest.toml")]
            signature = bytes2hex(sha256(reduce(vcat,read.(sources))))
            serialize(joinpath(directory,"candidate.jls"),(;q=candidate,optimizer,signature,
                fit_signature=saved.signature,reference_signature=reference.signature,
                coordinates=view.coordinates,seed=PROFILE_SEED,npairs=PROFILE_PAIRS,
                budget=PROFILE_EVALUATIONS,scope="same-family bounded surrogate optimization; endpoint unchecked"))
            println((;stage="bounded_endpoint",optimizer.reason,optimizer.evaluations,
                surrogate_initial=initial.value,surrogate_final=optimizer.current.value,
                surrogate_gradient_norm=norm(optimizer.current.gradient,Inf),
                scope="not a convergence or posterior-accuracy claim; independent audit still required"))
        end
    end
end
