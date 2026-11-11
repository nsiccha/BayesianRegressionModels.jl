# An implementation control, NOT a replacement for the failed nonlinear gate.
# All dense matrices, complete cubature sets and exact covariance injections
# are hard bounded here. The native field mode and draw actions stay iterative.
# The NLME reader explicitly refuses authored known random-effect scales
# (todo0840h34). This control conditions the supported sampled-hyperparameter
# model at fixed unconstrained hyperparameter coordinates instead; it does not
# claim known-value declaration support or integrate those hyperparameters.
# julia --project=test research/shared_field_laplace/gaussian_coupled_control.jl 'Gaussian coupled control'
include("whole_window_probe.jl")
using Serialization, Statistics

function known_field_fixture()
    groups, rows = 3, 4
    time = repeat(collect(range(.1,2.;length=rows)),groups)
    data = (;probe=repeat(1:groups;inner=rows),time,
        sx=cos.(1.3 .* (1:groups)),sy=sin.(.7 .* (1:groups)),
        y=.3 .+ .4sin.(time) .+ .7randn(Xoshiro(1960),groups*rows))
    brmi = @brm data begin
        mu ~ 1 + (1 | probe) + hsgp(time;k=2,by=probe,
            cor=multigroup((sx,sy),2.0;k=2))
        effect(mu,Intercept) ~ Normal(0,1.5)
        @plate for p in groups(probe)
            y[p] ~ Normal(mu[p],.7)
        end
    end
    rk = RKBRMI(brmi)
    m = brm_nlme_model(rk;ad_backend=NLME_AD)
    (;rk,m,data)
end

function known_gaussian_oracle(rk,m,data,field)
    d,n = length(m.unit),length(data.y)
    d <= 32 && n <= 64 || error("known-Gaussian oracle exceeds its hard size cap")
    globals = findall(==(:population),m.view.role)
    length(globals) == 1 || error("expected only the Gaussian intercept global")
    explicit = sort(vcat(globals,m.view.subject_coordinates...))
    active = sort(vcat(explicit,field))
    fixed = setdiff(1:d,active)
    length(fixed) == 3 || error("expected exactly three conditioned hyperparameters")
    observations = prepare_query(rk.model,RKEXT._rk_translated(rk),:observations)
    predict(u) = reduce(vcat,Base.invokelatest(observations,u).y.location)
    zero = zeros(d); offset = predict(zero)
    A = hcat(map(active) do j
        e = zeros(d); e[j] = 1
        predict(e)-offset
    end...)
    prior = ones(length(active)); prior[only(findall(in(globals),active))] = 1/1.5^2
    Ha = Diagonal(prior)+A'*A/.7^2
    ha = A'*(data.y-offset)/.7^2
    F = cholesky(Symmetric(Ha))
    H = zeros(d,d);H[active,active] = Ha
    h = zeros(d);h[active] = ha
    mean = zeros(d);mean[active] = F\ha
    covariance = zeros(d,d);covariance[active,active] = inv(F)
    Fz = cholesky(Symmetric(H[field,field]))
    K = H[explicit,explicit]-H[explicit,field]*(Fz\H[field,explicit])
    b = h[explicit]-H[explicit,field]*(Fz\h[field])
    conditional_A = -(Fz\H[field,explicit])
    conditional_b = Fz\h[field]
    (;explicit,globals,active,fixed,A,offset,prior,H,h,F,mean,covariance,Fz,K,b,
        conditional_A,conditional_b)
end

function known_collapsed(op,u,logdet_field)
    state = shared_poisson_mode(op,u;tol=1e-10)
    value = state.value+length(op.field)*log(2pi)/2-logdet_field/2
    (;value,gradient=state.gradient,mode=state.mode)
end

function inject_coupled_gaussian(m,op,explicit,mean,covariance)
    d = length(explicit)
    d <= 16 || error("exact factor injection is a bounded oracle only")
    t = JointWindowTarget(op,m,explicit,Vector{Int}[],2,2)
    floor = .1eigmin(Symmetric(covariance))
    q = BRM._brm_svi_params(t,mean,BRM.LowRankGlobal(d);
        global_scale=sqrt(floor),unit_scale=1.)
    q.Β .= cholesky(Symmetric(covariance-floor*I)).L
    (;t,q)
end

# Exact expectation for a quadratic target using complete symmetric cubature
# over the independent normals in the diagonal+rank sampling map.
function coupled_native_expectation(t,q,logdet_field)
    d,r = length(q.μ),size(q.Β,2)
    s = d+r; s <= 32 || error("cubature oracle exceeds its hard size cap")
    value = 0.; gradient = zeros(length(BRM._brm_svi_global_vector(q)))
    for j in 1:s, sign in (-1,1)
        noise = zeros(s); noise[j] = sign*sqrt(s)
        epsilon,eta = noise[1:d],noise[d+1:end]
        u = copy(t.m.unit)
        u[t.globals] = BRM._brm_svi_draw_global(q,epsilon,eta)
        state = known_collapsed(t.op,u,logdet_field)
        value += state.value/(2s)
        gradient .+= BRM._brm_svi_global_gradient(q,
            state.gradient[t.globals],epsilon,eta)/(2s)
    end
    value += BRM._brm_svi_entropy(q)+d*(1+log(2pi))/2
    (;value,gradient)
end

function known_reconstructed_moments(o,field,mean,covariance)
    d = length(o.mean); x = o.explicit
    fullmean = zeros(d); fullmean[x] = mean
    fullmean[field] = o.conditional_b+o.conditional_A*mean
    fullcov = zeros(d,d); fullcov[x,x] = covariance
    fullcov[x,field] = covariance*o.conditional_A'
    fullcov[field,x] = o.conditional_A*covariance
    fullcov[field,field] = inv(o.Fz)+o.conditional_A*covariance*o.conditional_A'
    (;mean=fullmean,covariance=fullcov)
end

@stestset "Gaussian coupled control" begin
    output = joinpath(@__DIR__, "results", "gaussian-coupled-v1")
    ispath(output) && error("refusing to overwrite Gaussian control artifacts: $output")
    mkpath(output)
    files = sort(filter(endswith(".jl"),readdir(@__DIR__;join=true)))
    identity = (;julia=string(VERSION),enzyme=pkgdir(Enzyme),
        libenzyme=Enzyme.Compiler.Enzyme_jll.libEnzyme_path,
        source_hashes=Dict(basename(f)=>bytes2hex(sha256(read(f))) for f in files),
        production_hash=bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/nlme_svi.jl")))))
    serialize(joinpath(output,"identity.jls"),identity)
    println((;stage="prepare native Gaussian slice at fixed hyperparameters"));flush(stdout)
    (;rk,m,data) = known_field_fixture()
    op = whole_window_operator(m)
    o = known_gaussian_oracle(rk,m,data,op.field)
    x,z = o.explicit,op.field
    d,k = length(m.unit),length(x)
    @test (d,k,length(z),length(o.fixed)) == (15,4,8,3)
    @test all(iszero,m.unit[o.fixed])
    @test length(o.globals) == 1
    @test all(length(b)==1 for b in m.view.subject_coordinates)
    @test o.mean[x] ≈ o.K\o.b atol=1e-11
    @test o.covariance[x,x] ≈ inv(Symmetric(o.K)) atol=1e-11
    # Independent observation-law and authored Gaussian-prior checks catch a
    # non-Gaussian intercept prior. Conditioned hyperpriors and Jacobians add
    # one constant, included in the native density but irrelevant to this slice.
    native0 = first(shared_value_gradient(op,zeros(d)))
    law0 = sum(logpdf.(Normal.(o.offset,.7),data.y))+
        sum(logpdf.(Normal.(0.,1 ./sqrt.(o.prior)),zeros(length(o.active))))
    constant = native0-law0
    rng = Xoshiro(1961)
    for j in 1:3
        u = zeros(d);u[o.active] = randn(rng,length(o.active))
        value,score = shared_value_gradient(op,u)
        expected = sum(logpdf.(Normal.(o.offset+o.A*u[o.active],.7),data.y))+
            sum(logpdf.(Normal.(0.,1 ./sqrt.(o.prior)),u[o.active]))+constant
        @test value ≈ expected atol=1e-10
        @test score[o.active] ≈ (o.h-o.H*u)[o.active] atol=1e-10
        @test norm(shared_third(op,u,randn(rng,length(z)))[o.active]) < 1e-11
    end
    # Full orthogonal probes remove trace sampling error ONLY in this small
    # control. A production fixed probe count still has trace sampling error.
    u0 = zeros(d); native_mode = shared_poisson_mode(op,u0).mode
    H(v) = shared_precision(op,native_mode,v)
    ld = sum(shared_logquadratic(H,Matrix{Float64}(I,length(z),length(z))[:,j];
        spectral_lower=1.,maxdim=16).value for j in eachindex(z))
    @test ld ≈ logdet(o.Fz) atol=1e-8
    f0 = known_collapsed(op,u0,ld).value
    for v in (zeros(k),fill(.2,k),collect(range(-.4,.3;length=k)))
        u = zeros(d);u[x] = v
        state = known_collapsed(op,u,ld)
        @test state.value ≈ f0+dot(o.b,v)-dot(v,o.K*v)/2 atol=1e-9
        @test state.gradient[x] ≈ o.b-o.K*v atol=1e-9
        @test state.mode[z] ≈ o.conditional_b+o.conditional_A*v atol=1e-9
    end
    println((;stage="native target checked; verify coupled and product optima"));flush(stdout)
    mean,C = o.mean[x],o.covariance[x,x]
    (;t,q) = inject_coupled_gaussian(m,op,x,mean,C)
    @test BRM._brm_svi_global_covariance(q) ≈ C atol=1e-11
    coupled = coupled_native_expectation(t,q,ld)
    # This is the log normalizer of the native slice, including the fixed
    # hyperprior/Jacobian constant above. Conditional log p(y | hypers) would
    # subtract `constant`. The posterior and KL comparisons cancel it.
    evidence = f0+dot(o.b,mean)/2+k*log(2pi)/2-logdet(Symmetric(o.K))/2
    @test coupled.value ≈ evidence atol=1e-8
    @test maximum(abs,coupled.gradient) < 1e-9

    product_cov = Matrix(Diagonal(1 ./diag(o.K)))
    product_target = JointWindowTarget(op,m,o.globals,m.view.subject_coordinates,2,2)
    qp = BRM._brm_svi_params(product_target,mean[1:1],BRM.MeanFieldGlobal();
        global_scale=sqrt(product_cov[1,1]),unit_scale=1.)
    @test product_target.globals == x[1:1]
    qp.ρ[1] = log(product_cov[1,1])/2
    for (j,block) in enumerate(product_target.units)
        i = only(findall(in(block),x))
        qp.M[j][1] = mean[i]
        qp.Λ[j][1,1] = 1/product_cov[i,i]
        qp.U[j][1,1] = sqrt(qp.Λ[j][1,1])
    end
    # Native exact expectation at the product optimum: coordinatewise mean
    # and log-SD residuals vanish even though cross-covariance is missing.
    scoremean,scalegrad = zeros(k),zeros(k)
    product_value = 0.
    for j in 1:k,sign in (-1,1)
        epsilon = zeros(k);epsilon[j] = sign*sqrt(k)
        u = zeros(d);u[x] = mean+sqrt.(diag(product_cov)).*epsilon
        state = known_collapsed(op,u,ld)
        scoremean .+= state.gradient[x]/(2k)
        scalegrad .+= (state.gradient[x].*(u[x]-mean).+1)/(2k)
        product_value += state.value/(2k)
    end
    product_value += BRM._brm_svi_entropy(qp)+k*(1+log(2pi))/2
    product_kl = (tr(o.K*product_cov)-k-logdet(Symmetric(o.K))-
        logdet(Symmetric(product_cov)))/2
    @test maximum(abs,scoremean) < 1e-9
    @test maximum(abs,scalegrad) < 1e-9
    @test evidence-product_value ≈ product_kl atol=1e-8
    @test product_kl > 1e-3
    exact_moments = known_reconstructed_moments(o,z,mean,C)
    product_moments = known_reconstructed_moments(o,z,mean,product_cov)
    @test exact_moments.mean ≈ o.mean atol=1e-10
    @test exact_moments.covariance ≈ o.covariance atol=1e-10
    @test norm(o.covariance[x,z]) > .01
    @test norm(product_moments.covariance-o.covariance) > .01

    # Fixed 2048 draws per injected family; no posterior fitting or until-green
    # resampling. Joint covariance checks include field/unit cross covariance.
    println((;stage="fixed conditional reconstruction draws",draws=2048));flush(stdout)
    diagnostics = map(((;name=:coupled,t,q,moments=exact_moments),
                      (;name=:product,t=product_target,q=qp,moments=product_moments))) do spec
        U = BRM._brm_svi_draws(Xoshiro(1962),spec.t,spec.q,2048)
        means = vec(Statistics.mean(U;dims=2))
        covar = cov(U;dims=2)
        sd = sqrt.(diag(spec.moments.covariance)[o.active])
        meanerror = maximum(abs.((means-spec.moments.mean)[o.active]./sd))
        coverror = maximum(abs.((covar-spec.moments.covariance)[o.active,o.active]./(sd*sd')))
        @test all(iszero,U[o.fixed,:])
        (;name=spec.name,meanerror,coverror,
            digest=bytes2hex(sha256(reinterpret(UInt8,vec(U)))))
    end
    result = (;identity,coordinates=m.view.coordinates,field=z,explicit=x,fixed=o.fixed,mean,C,
        evidence,product_value,product_kl,product_sd_ratio=sqrt.(diag(product_cov)./diag(C)),
        diagnostics,scope="Gaussian implementation control; nonlinear accuracy gate remains failed")
    serialize(joinpath(output,"control.jls"),result)
    println(result)
    for diagnostic in diagnostics
        @test diagnostic.meanerror < .12
        @test diagnostic.coverror < .16
    end
end
