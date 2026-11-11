# Native current-window likelihood times a correlated Gaussian carry. Research
# only: no identity transport, fitted projection, or public streaming API.
# julia --project=test research/shared_field_laplace/native_carry_probe.jl 'native correlated carry'
include("whole_window_probe.jl")
include("block_projection_probe.jl")

struct CarriedWindowOperator{O,C,L,M,T}
    native::O
    carry::C
    likelihood::L
    m::M
    t::T
    field::Vector{Int}
    spectral_lower::Float64
end

function carried_window_operator(native,carry)
    m = native.m
    length(carry.mu) == length(m.unit) || throw(DimensionMismatch("native carry"))
    outer = setdiff(eachindex(m.unit),native.field)
    conditional = block_field_conditional(carry,native.field,outer)
    lower = block_precision_lower_bound(conditional.field_state)
    likelihood = prepare_query_ad(prepare_ad,m.backend.model,
        RKEXT._rk_translated(m.backend),:likelihood,NLME_AD,m.unit)
    CarriedWindowOperator(native,carry,likelihood,m,native.t,native.field,lower)
end

function shared_value_gradient(op::CarriedWindowOperator,u)
    value,gradient = ad_value_and_gradient(op.likelihood,u)
    prior = block_density_gradient(op.carry,u)
    value+prior.value,gradient+prior.gradient
end
shared_likelihood_hvp(op::CarriedWindowOperator,u,v) =
    shared_likelihood_hvp(op.native,u,v)
shared_third(op::CarriedWindowOperator,u,v) = shared_third(op.native,u,v)
function carried_joint_hvp(op,u,v)
    direction = zeros(length(u));direction[op.field] = v
    shared_likelihood_hvp(op,u,v)-block_precision(op.carry,direction)
end
shared_precision(op::CarriedWindowOperator,u,v) = -carried_joint_hvp(op,u,v)[op.field]

# The implicit-mode correction includes the carry's mixed field/outer block.
# A Gaussian carry has zero third derivative, but its mixed SECOND derivative
# is nonzero and cannot be removed from this correction.
function carried_collapsed_state(op,u,probes)
    state = shared_poisson_mode(op,u;tol=1e-10)
    trace = shared_trace(op,state.mode,probes).trace
    H(v) = shared_precision(op,state.mode,v)
    direction,info = KrylovKit.linsolve(H,trace[op.field];issymmetric=true,
        isposdef=true,rtol=1e-11,atol=1e-13,krylovdim=64,maxiter=1)
    info.converged == 1 || error("carried implicit solve failed")
    gradient = state.gradient+(trace+carried_joint_hvp(op,state.mode,direction))/2
    wrong_gradient = state.gradient+(trace+shared_likelihood_hvp(op,state.mode,direction))/2
    # Complete orthogonal probes are confined to this hard-bounded control.
    length(op.field) <= 16 || error("exact carried logdet control exceeds cap")
    logdeterminant = sum(shared_logquadratic(H,r;spectral_lower=op.spectral_lower,
        maxdim=32,atol=1e-10,rtol=1e-10).value for r in probes)/length(probes)
    value = state.value+length(op.field)*log(2pi)/2-logdeterminant/2
    (;value,gradient,wrong_gradient,mode=state.mode,logdeterminant)
end

function dense_carried_poisson_oracle(op,C,predictor,y,u)
    d,f = length(u),op.field
    d <= 32 && length(y) <= 64 || error("carried oracle exceeds hard size cap")
    P = inv(Symmetric(C))
    base = copy(u);base[f] .= 0
    b = predictor(base)
    A = hcat(map(f) do j
        v = copy(base);v[j] = 1
        predictor(v)-b
    end...)
    z = zeros(length(f))
    function evaluate(z)
        v = copy(base);v[f] = z
        delta = v-op.carry.mu
        rate = exp.(b+A*z)
        value = -(d*log(2pi)+logdet(Symmetric(C))+dot(delta,P*delta))/2+
            sum(logpdf.(Poisson.(rate),y))
        gradient = -(P*delta)[f]+A'*(y-rate)
        H = P[f,f]+A'*Diagonal(rate)*A
        (;value,gradient,H)
    end
    for i in 1:50
        state = evaluate(z)
        norm(state.gradient) < 1e-11 && break
        i == 50 && error("dense carried oracle failed to converge")
        step = state.H\state.gradient
        decrement = dot(state.gradient,step)
        accepted = false
        for half in 0:40
            scale = 2.0^-half;candidate = z+scale*step
            after = evaluate(candidate)
            if isfinite(after.value) &&
                    (decrement < 1e-12 || after.value >= state.value+1e-4scale*decrement)
                z = candidate;accepted = true;break
            end
        end
        accepted || error("dense carried oracle backtracking failed")
    end
    state = evaluate(z)
    norm(state.gradient) < 1e-9 || error("dense carried oracle residual")
    (;value=state.value+length(f)*log(2pi)/2-logdet(Symmetric(state.H))/2,
        z,H=state.H)
end

@stestset "native correlated carry" begin
    output = joinpath(@__DIR__, "results", "native-carry-v1")
    ispath(output) && error("refusing to overwrite native carry control")
    mkpath(output)
    files = sort(filter(endswith(".jl"),readdir(@__DIR__;join=true)))
    identity = (;julia=string(VERSION),enzyme=pkgdir(Enzyme),
        libenzyme=Enzyme.Compiler.Enzyme_jll.libEnzyme_path,
        source_hashes=Dict(basename(f)=>bytes2hex(sha256(read(f))) for f in files),
        production_hash=bytes2hex(sha256(read(joinpath(@__DIR__,"../../src/nlme_svi.jl")))))
    serialize(joinpath(output,"identity.jls"),identity)
    println((;stage="prepare native Poisson carried likelihood"));flush(stdout)
    (;rk,m,data) = shared_field_fixture(;poisson=true)
    native = whole_window_operator(m)
    d = length(m.unit)
    rng = Xoshiro(1963)
    B = .35randn(rng,d,3)
    blocks = [[j] for j in 1:d]
    bases = [reshape([.6+.1rand(rng)],1,1) for j in 1:d]
    c = block_carry(.1randn(rng,d),blocks,bases,B)
    op = carried_window_operator(native,c)
    # Dense carry construction is bounded and independent of the Woodbury
    # precision used by the native operator.
    @test d <= 32
    C = Diagonal([only(b) for b in bases])+B*B'
    f,outer = op.field,setdiff(1:d,op.field)
    @test op.spectral_lower > 0
    bound = rk_bound_data(rk)
    query = Base.invokelatest(ReactiveKernels.prepare,rk.model.spec;
        have=(:unconstrained,keys(bound)...),want=(:mu,),bound)
    function predictor(u)
        value = Base.invokelatest(query,u)
        value isa Tuple ? only(value) : value
    end
    probes = [sqrt(length(f))*Matrix{Float64}(I,length(f),length(f))[:,j]
        for j in eachindex(f)]
    results = map((zeros(d),.15randn(rng,d))) do u
        native = carried_collapsed_state(op,u,probes)
        exact = dense_carried_poisson_oracle(op,C,predictor,data.y,u)
        @test native.mode[f] ≈ exact.z atol=2e-9
        @test native.value ≈ exact.value atol=2e-8
        direction = randn(rng,length(f))
        @test shared_precision(op,native.mode,direction) ≈ exact.H*direction atol=2e-9
        @test op.spectral_lower <= eigmin(Symmetric(exact.H))*(1+1e-12)
        eps = 1e-5
        differences = map(outer) do j
            plus,minus = copy(u),copy(u);plus[j]+=eps;minus[j]-=eps
            (dense_carried_poisson_oracle(op,C,predictor,data.y,plus).value-
                dense_carried_poisson_oracle(op,C,predictor,data.y,minus).value)/(2eps)
        end
        error = maximum(abs.(native.gradient[outer]-differences))
        wrong_error = maximum(abs.(native.wrong_gradient[outer]-differences))
        @test error < 2e-6
        @test wrong_error > 1e-5
        (;point=u,value_error=native.value-exact.value,error,wrong_error,
            mode_error=maximum(abs.(native.mode[f]-exact.z)))
    end
    serialize(joinpath(output,"control.jls"),(;identity,results,field=f,outer,
        scope="native correlated carry derivative control; no streaming fit or identity transport"))
    println((;results,op.spectral_lower))
end
