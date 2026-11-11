# A different private approximation family: globals remain variational;
# field AND Gaussian unit draws are integrated jointly. This is not a public
# block-selection change, a full-Laplace fit, or a passed posterior gate.
# Native matrix-free derivatives are checked against the hard-bounded exact
# Gaussian reference before ONE bounded four-global optimization experiment.
# julia --project=test research/shared_field_laplace/joint_latent_fit.jl 'bounded joint latent fit'
include("profile_optimization.jl")
include("whole_window_probe.jl")

const JOINT_FIT_SEED = 1965
# Keep the previous budget: 512 antithetic pairs, 50 objective/gradient
# evaluations including line searches, plus eight predeclared FD values.

function native_joint_latent_operator(m)
    field = whole_window_operator(m)
    latent = sort(vcat(field.field,m.view.subject_coordinates...))
    globals = setdiff(eachindex(m.unit),latent)
    t = (;globals,units=Vector{Int}[])
    WholeWindowOperator(m,t,latent,field.full,field.hvp,field.third)
end

function exact_probe_joint_state(op,u)
    d = length(op.field)
    d <= 32 || error("complete-probe joint control exceeds fixed dimension cap")
    state = shared_poisson_mode(op,u;tol=1e-9)
    basis = Matrix{Float64}(I,d,d)
    probes = [sqrt(d)*basis[:,j] for j in 1:d]
    trace = shared_trace(op,state.mode,probes).trace
    H(v) = shared_precision(op,state.mode,v)
    direction,info = KrylovKit.linsolve(H,trace[op.field];issymmetric=true,
        isposdef=true,rtol=1e-11,atol=1e-13,krylovdim=64,maxiter=1)
    info.converged == 1 || error("joint latent correction solve failed")
    gradient = state.gradient+(trace+shared_likelihood_hvp(op,state.mode,direction))/2
    quadratures = [shared_logquadratic(H,basis[:,j];spectral_lower=1.,
        maxdim=32,atol=1e-10,rtol=1e-10) for j in 1:d]
    all(q->q.converged,quadratures) || error("joint logdet control exhausted budget")
    logdeterminant = sum(q.value for q in quadratures)
    value = state.value+d*log(2pi)/2-logdeterminant/2
    (;value,gradient,mode=state.mode,logdeterminant)
end

function joint_global_objective(p,mu0,C0,points,x;gradient=true)
    (;mu,C,L) = profile_coordinates(x,mu0,C0)
    n = size(points,2)
    n == 2PROFILE_PAIRS || error("joint fit sample budget changed")
    value = 0.;da = zeros(4);dL = zeros(4,4)
    for j in 1:n
        g = mu+C*points[:,j]
        if gradient
            v,w = LogDensityProblems.logdensity_and_gradient(p,g)
            white = C0'*w
            da .+= white/n
            dL .+= white*points[:,j]'/n
            value += v/n
        else
            value += LogDensityProblems.logdensity(p,g)/n
        end
    end
    value += 2*(1+log(2pi))+sum(log,diag(C))
    gradient || return (;value,mu,C)
    dL .+= inv(L)'
    grad = vcat(da,[dL[i,i]*L[i,i] for i in 1:4],
        [dL[i,j] for j in 1:4 for i in j+1:4])
    (;value,gradient=grad,mu,C)
end

@stestset "bounded joint latent fit" begin
    base = joinpath(@__DIR__, "results")
    saved = deserialize(joinpath(base,"profile-optimization-v1/candidate.jls"))
    reference = deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    @test reference.reference_ok && saved.reference_signature == reference.signature
    # A warm-started family comparison, not independent initialization or an
    # assertion of a unique/global optimum. No reference moments initialize it.
    mu0 = copy(saved.q.μ)
    C0 = Matrix(cholesky(Symmetric(Matrix(BRM._brm_svi_global_covariance(saved.q)))).L)
    directory = joinpath(base,"joint-latent-fit-v1")
    ispath(directory) && error("preserve existing joint-latent fit")
    mkpath(directory)
    sources = [joinpath(@__DIR__,name) for name in
        ("joint_latent_fit.jl","profile_optimization.jl","stationarity_audit.jl",
         "exact_reference.jl","operator_probe.jl","trace_probe.jl",
         "nonlinear_probe.jl","whole_window_probe.jl","quadrature.jl","joint_svi_probe.jl")]
    append!(sources,[joinpath(@__DIR__,"../../src/nlme_svi.jl"),
        joinpath(@__DIR__,"../../test/Manifest.toml")])
    signature = bytes2hex(sha256(reduce(vcat,read.(sources))))
    identity = (;signature,seed=JOINT_FIT_SEED,npairs=PROFILE_PAIRS,
        evaluation_cap=PROFILE_EVALUATIONS,fd_values=8,
        warm_start_signature=saved.signature,reference_signature=reference.signature,
        julia=string(VERSION),enzyme=pkgdir(Enzyme),
        libenzyme=Enzyme.Compiler.Enzyme_jll.libEnzyme_path)
    serialize(joinpath(directory,"identity.jls"),identity)
    with_progress(:state;description="Native joint collapse and bounded global fit") do progress
        snapshot_progress(progress,joinpath(directory,"fit.progress")) do
            fixture = with_progress(progress;description="Prepare native joint and exact oracle targets") do _
                (;rk,m,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2)
                merge((;op=native_joint_latent_operator(m)),exact_gaussian_reference(rk,data))
            end
            (;op,p,view,field,units) = fixture
            @test op.field == p.latent
            @test op.t.globals == p.globals
            @test length(op.field)==22 && length(p.globals)==4
            @test view.coordinates == saved.coordinates == reference.coordinates
            native_checks = with_progress(progress,3;description="Native joint marginal derivative checks") do node
                map(enumerate((zeros(4),[.2,-.4,.3,.1],[-.3,.2,-.2,.4]))) do (i,shift)
                    g = p.start[p.globals]+shift
                    u = copy(p.start);u[p.globals] = g
                    native = exact_probe_joint_state(op,u)
                    exact = gaussian_reference_conditional(p,g)
                    value,score = LogDensityProblems.logdensity_and_gradient(p,g)
                    @test native.mode[p.latent] ≈ exact.mu atol=2e-8
                    @test native.value ≈ value atol=2e-7
                    @test native.gradient[p.globals] ≈ score atol=2e-7 rtol=2e-7
                    update_progress!(node,i)
                    (;value_error=native.value-value,
                        gradient_error=maximum(abs.(native.gradient[p.globals]-score)),
                        mode_error=maximum(abs.(native.mode[p.latent]-exact.mu)))
                end
            end
            serialize(joinpath(directory,"native-checks.jls"),(;identity,native_checks))
            all(c->abs(c.value_error)<2e-7 && c.gradient_error<2e-7 &&
                c.mode_error<2e-8,native_checks) || error("native joint control failed; no fit")
            positive = randn(Xoshiro(JOINT_FIT_SEED),4,PROFILE_PAIRS)
            points = hcat(positive,-positive)
            x0 = zeros(14)
            evaluate(x;gradient=true) = joint_global_objective(p,mu0,C0,points,x;gradient)
            initial = with_progress(progress;description="Check exact-global expectation derivatives") do _
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
                serialize(joinpath(directory,"initial.jls"),(;identity,value,checks,mu0,C0))
                println((;stage="joint fit initial",value=value.value,native_checks,checks));flush(stdout)
                all(c->abs(c.error)<=3e-5+3e-5abs(c.analytic),checks) ||
                    error("joint objective derivative validation failed; no fit")
                value
            end
            optimizer = with_progress(progress,PROFILE_EVALUATIONS;
                    description="Fixed global optimization evaluation budget") do node
                bounded_profile_bfgs(evaluate,x0,initial;node,
                    checkpoint=r->serialize(joinpath(directory,"accepted-iterate.jls"),(;identity,r)))
            end
            @test optimizer.evaluations <= PROFILE_EVALUATIONS
            @test optimizer.current.value >= initial.value-1e-10
            candidate = (;identity,optimizer,mu=optimizer.current.mu,C=optimizer.current.C,
                initial_value=initial.value,coordinates=view.coordinates,
                field,units,globals=p.globals,latent=p.latent,native_checks,
                scope="bounded global-only oracle optimization after native joint collapse control; independent endpoint audit still required")
            serialize(joinpath(directory,"candidate.jls"),candidate)
            println((;reason=optimizer.reason,evaluations=optimizer.evaluations,
                initial=initial.value,final=optimizer.current.value,
                surrogate_gradient=norm(optimizer.current.gradient,Inf),
                global_mean=candidate.mu,global_covariance=candidate.C*candidate.C'))
        end
    end
end
