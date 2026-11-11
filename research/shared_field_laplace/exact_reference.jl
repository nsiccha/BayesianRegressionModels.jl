# Hard-bounded reference only: exact integration of ALL conditionally Gaussian
# latents in the 26-coordinate Gaussian fixture, not a production Laplace path.
# julia --project=test research/shared_field_laplace/exact_reference.jl 'exact Gaussian reference'
include("operator_probe.jl")
using Treebars

struct ExactGaussianReference{P}
    full::P
    start::Vector{Float64}
    globals::Vector{Int}
    latent::Vector{Int}
end

function exact_gaussian_reference(rk, data)
    view = brm_nlme_view(rk)
    record = only(r for r in BRM._rk_coordinate_records(rk.plan) if r.kind === :hsgp)
    weights = Symbol(record.declaration, ".", record.locals.weights)
    field = findall(n -> BRM._brm_nlme_owned(n, weights), view.coordinates)
    units = reduce(vcat, view.subject_coordinates)
    latent = sort!(vcat(field, units))
    globals = setdiff(eachindex(view.coordinates), latent)
    # One-hot and dense reference operations may NEVER grow into the fit path.
    0 < length(latent) <= 32 && length(view.coordinates) <= 40 &&
        length(data.y) <= 128 || error("exact Gaussian oracle exceeds its hard size cap")
    length(unique(latent)) == length(latent) || error("overlapping latent blocks")
    start = first(BRM._brm_nlme_prior_median_point(rk, view, zeros(length(view.coordinates))))
    p = ExactGaussianReference(rk_logdensity_problem(rk;ad_backend=NLME_AD),
        start, globals, latent)
    (;p, view, field, units)
end

LogDensityProblems.dimension(p::ExactGaussianReference) = length(p.globals)
LogDensityProblems.capabilities(::Type{<:ExactGaussianReference}) =
    LogDensityProblems.LogDensityOrder{1}()

function gaussian_reference_conditional(p::ExactGaussianReference, g)
    u = copy(p.start); u[p.globals] = g; u[p.latent] .= 0
    f0, score0 = LogDensityProblems.logdensity_and_gradient(p.full, u)
    q = length(p.latent)
    h = score0[p.latent]
    H = Matrix{Float64}(undef, q, q)
    for j in 1:q
        u[p.latent[j]] = 1
        _, score = LogDensityProblems.logdensity_and_gradient(p.full, u)
        H[:,j] = h - score[p.latent]
        u[p.latent[j]] = 0
    end
    isapprox(H,H';rtol=1e-10,atol=1e-10) || error("non-symmetric Gaussian precision")
    F = cholesky(Symmetric(H))
    mu = F \ h
    u[p.latent] = mu
    fm = LogDensityProblems.logdensity(p.full,u)
    value = fm + q/2*log(2pi) - logdet(F)/2
    (;value,mu,F,H,h,f0,u)
end

LogDensityProblems.logdensity(p::ExactGaussianReference,g) =
    gaussian_reference_conditional(p,g).value

function LogDensityProblems.logdensity_and_gradient(p::ExactGaussianReference,g)
    c = gaussian_reference_conditional(p,g)
    q = length(p.latent)
    C = c.F.U \ Matrix{Float64}(I,q,q)
    gradient = zeros(length(g))
    u = copy(c.u)
    # Fisher identity: E[partial_g log p(g,u,y) | g,y]. The native partial
    # score is quadratic in u; these 2q Gaussian cubature nodes integrate it
    # exactly. Nodes/weights are NOT differentiated. Enzyme owns every score.
    for j in 1:q, sign in (-1,1)
        u[p.latent] = c.mu + sign*sqrt(q)*C[:,j]
        _, score = LogDensityProblems.logdensity_and_gradient(p.full,u)
        gradient .+= score[p.globals] ./ (2q)
    end
    c.value, gradient
end

# Independent observation-space Gaussian oracle, with native prior/Jacobian
# terms obtained by removing the likelihood and N(0,I) density at u=0.
function observation_gaussian_reference(p,g,mean_query,y;sigma=0.7)
    u = copy(p.start); u[p.globals] = g; u[p.latent] .= 0
    b = Base.invokelatest(mean_query,u)
    q = length(p.latent)
    A = hcat(map(1:q) do j
        u[p.latent[j]] = 1
        a = Base.invokelatest(mean_query,u) - b
        u[p.latent[j]] = 0
        a
    end...)
    r = y-b; n = length(y)
    R = cholesky(Symmetric(sigma^2*Matrix{Float64}(I,n,n)+A*A'))
    likelihood0 = -n/2*log(2pi*sigma^2)-sum(abs2,r)/(2sigma^2)
    marginal_likelihood = -n/2*log(2pi)-logdet(R)/2-dot(r,R\r)/2
    value = LogDensityProblems.logdensity(p.full,u)-likelihood0+
        q/2*log(2pi)+marginal_likelihood
    H = Matrix{Float64}(I,q,q)+A'*A/sigma^2
    (;value,H,mu=H\(A'*r/sigma^2))
end

@stestset "exact Gaussian reference" begin
    snapshot = get(ENV,"KB_PROGRESS_FILE",
        joinpath(@__DIR__, "results", "exact-reference.progress"))
    with_progress(:state;description="Validate exact Gaussian reference") do progress
        snapshot_progress(progress,snapshot) do
            fixture = with_progress(progress;description="Prepare native model and Enzyme") do _
                (;rk,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,
                    spatial=2,prepare_model=false)
                merge((;rk,data),exact_gaussian_reference(rk,data))
            end
            (;p,rk,data,field,units) = fixture
            @test length(p.globals) == 4
            @test length(field) == 16
            @test length(units) == 6
            @test length(p.latent) == 22
            observations = prepare_query(rk.model,RKEXT._rk_translated(rk),:observations)
            mean_query(u) = reduce(vcat,Base.invokelatest(observations,u).y.location)
            rng = Xoshiro(828)
            with_progress(progress,3;description="Check values, scores and joint draws") do checks
                for (k,shift) in enumerate((zeros(4),[.5,-.3,.2,.4],[-2.,-1.,1.,-.5]))
                    g = p.start[p.globals] + shift
                    c = gaussian_reference_conditional(p,g)
                    oracle = observation_gaussian_reference(p,g,mean_query,data.y)
                    @test c.H ≈ oracle.H rtol=1e-10 atol=1e-10
                    @test c.mu ≈ oracle.mu rtol=1e-10 atol=1e-10
                    @test c.value ≈ oracle.value atol=1e-10
                    # Verify conditional quadraticity away from its mode.
                    z = randn(rng,length(p.latent)); u = copy(c.u); u[p.latent] = z
                    fz,gz = LogDensityProblems.logdensity_and_gradient(p.full,u)
                    @test fz ≈ c.f0+dot(c.h,z)-dot(z,c.H*z)/2 atol=1e-10
                    @test gz[p.latent] ≈ c.h-c.H*z atol=1e-10
                    value,score = LogDensityProblems.logdensity_and_gradient(p,g)
                    eps = 1e-5
                    finite_difference = map(eachindex(g)) do j
                        plus,minus = copy(g),copy(g); plus[j]+=eps; minus[j]-=eps
                        (observation_gaussian_reference(p,plus,mean_query,data.y).value-
                         observation_gaussian_reference(p,minus,mean_query,data.y).value)/(2eps)
                    end
                    @test value == c.value
                    @test score ≈ finite_difference rtol=2e-6 atol=2e-7
                    # A complete symmetric cubature draw set recovers covariance
                    # and includes field/unit cross covariance, without MC error.
                    q = length(p.latent); C = c.F.U \ Matrix{Float64}(I,q,q)
                    centered = sqrt(q)*hcat(C,-C)
                    @test centered*centered'/(2q) ≈ inv(c.F) atol=1e-11
                    @test norm(inv(c.F)[1:6,7:end]) > 1e-3
                    println((;point=k,max_gradient_error=maximum(abs.(score-finite_difference)),
                        value_error=c.value-oracle.value,latent_coordinates=q))
                    update_progress!(checks,k)
                end
            end
        end
    end
end
