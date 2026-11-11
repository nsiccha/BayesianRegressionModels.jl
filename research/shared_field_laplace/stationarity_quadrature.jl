# Fixed, bounded deterministic follow-up to the inconclusive global Monte
# Carlo audit. Orders are declared before the results: 7 and 11 in exactly
# four global coordinates. Successive agreement is evidence of numerical
# stability, not a certified integration-error bound or a production method.
# julia --project=test research/shared_field_laplace/stationarity_quadrature.jl 'fixed quadrature stationarity'
include("stationarity_audit.jl")

const STATIONARITY_ORDERS = (7,11)
const QUADRATURE_STABILITY_TOLERANCE = .01

function standard_normal_quadrature(n)
    2 <= n <= 11 || error("bounded quadrature order")
    e = eigen(SymTridiagonal(zeros(n),sqrt.(collect(1:n-1))))
    (;nodes=e.values,weights=e.vectors[1,:].^2)
end

function quadrature_stationarity(p,field,units,q,n;node=nothing)
    D = length(q.μ)
    D == 4 || error("tensor oracle is limited to exactly four globals")
    M,variance = audit_unit_state(q)
    C = Matrix(cholesky(Symmetric(Matrix(BRM._brm_svi_global_covariance(q)))).L)
    gh = standard_normal_quadrature(n)
    mean_score,covariance,row_scale = zeros(D),zeros(D,D),zeros(D)
    unit_mean,unit_logsd = zeros(length(M)),zeros(length(M))
    mean_S,mean_b = zeros(length(M),length(M)),zeros(length(M))
    expected_value,weight_sum = 0.,0.
    for (k,indices) in enumerate(Iterators.product(ntuple(_ -> 1:n,D)...))
        xi = [gh.nodes[i] for i in indices]
        weight = prod(gh.weights[i] for i in indices)
        g = q.μ + C*xi
        a = gaussian_unit_expectation(p,field,units,g,M,variance)
        ws = C'*a.w
        mean_score .+= weight .* ws
        covariance .+= weight .* ((ws*xi'+xi*ws')/2+I)
        row_scale .+= weight .* (1 .+ a.w.*(g-q.μ))
        unit_mean .+= weight .* sqrt.(variance).*a.unit_score
        unit_logsd .+= weight .* a.unit_logsd
        mean_S .+= weight .* a.S
        mean_b .+= weight .* (a.unit_score+a.S*M)
        expected_value += weight*a.value
        weight_sum += weight
        node !== nothing && k%128 == 0 && update_progress!(node,k)
    end
    node === nothing || update_progress!(node,n^D)
    # For FIXED q(g), this is the exact coordinate optimum of the product
    # scalar-unit Gaussian family under this quadrature approximation.
    # Mean coupling remains in S, although the variational covariance is diagonal.
    optimum_M = Symmetric(mean_S)\mean_b
    optimum_variance = 1 ./ diag(mean_S)
    shift = optimum_M-M
    conditional_gain = dot(shift,mean_S*shift)/2 +
        sum(log.(optimum_variance./variance) - diag(mean_S).*(optimum_variance-variance))/2
    residual = vcat(mean_score,
        [covariance[i,j] for j in 1:D for i in 1:j],row_scale,unit_mean,unit_logsd)
    (;order=n,nodes=n^D,weight_sum,residual,mean_score,covariance,row_scale,
        unit_mean,unit_logsd,mean_S,mean_b,optimum_M,optimum_variance,
        conditional_gain,elbo=expected_value+audit_entropy(C,variance))
end

@stestset "fixed quadrature stationarity" begin
    base = joinpath(@__DIR__, "results")
    saved = deserialize(joinpath(base,"joint-fit-v1/fit.jls"))
    audit = deserialize(joinpath(base,"stationarity-v1/audit.jls"))
    @test saved.signature == audit.fit_signature == AUDIT_FIT_SIGNATURE
    directory = get(ENV,"KB_LAPLACE_QUADRATURE_DIR",joinpath(base,"stationarity-quadrature-v1"))
    ispath(directory) && error("preserve existing quadrature output: $directory")
    mkpath(directory)
    for n in STATIONARITY_ORDERS
        gh = standard_normal_quadrature(n)
        @test sum(gh.weights) ≈ 1 atol=2e-14
        @test dot(gh.weights,gh.nodes) ≈ 0 atol=2e-14
        @test dot(gh.weights,gh.nodes.^2) ≈ 1 atol=2e-14
        @test dot(gh.weights,gh.nodes.^4) ≈ 3 atol=5e-13
        @test dot(gh.weights,gh.nodes.^6) ≈ 15 atol=2e-11
    end
    with_progress(:state;description="Fixed quadrature audit of saved SVI") do progress
        snapshot_progress(progress,joinpath(directory,"quadrature.progress")) do
            fixture = with_progress(progress;description="Prepare native exact-latent oracle") do _
                (;rk,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,
                    spatial=2,prepare_model=false)
                exact_gaussian_reference(rk,data)
            end
            (;p,view,field,units) = fixture
            @test view.coordinates == audit.coordinates
            q = saved.fit.q
            records = map(STATIONARITY_ORDERS) do order
                with_progress(progress,order^4;description="Order $order tensor quadrature") do node
                    seconds = @elapsed result = quadrature_stationarity(p,field,units,q,order;node)
                    serialize(joinpath(directory,"order-$order.jls"),(;result,seconds))
                    @test abs(result.weight_sum-1) < 1e-12
                    @test all(isfinite,result.residual)
                    println((;order,nodes=result.nodes,seconds,elbo=result.elbo,
                        global_mean=result.mean_score,global_row_scale=result.row_scale,
                        unit_logsd=result.unit_logsd,result.conditional_gain))
                    result
                end
            end
            coarse,fine = records
            residual_delta = abs.(fine.residual-coarse.residual)
            stable = maximum(residual_delta) <= QUADRATURE_STABILITY_TOLERANCE
            material = abs.(fine.residual) .> AUDIT_PRACTICAL_TOLERANCE
            # No assertion turns a scientific negative/inconclusive into a
            # test failure. Every result is retained before interpretation.
            sources = [@__FILE__,joinpath(@__DIR__,"stationarity_audit.jl"),
                joinpath(@__DIR__,"exact_reference.jl"),joinpath(@__DIR__,"operator_probe.jl"),
                joinpath(@__DIR__,"../../src/nlme_svi.jl"),
                joinpath(@__DIR__,"../../test/Manifest.toml")]
            signature = bytes2hex(sha256(reduce(vcat,read.(sources))))
            result = (;records,residual_delta,stable,material,labels=audit.labels,
                signature,fit_signature=saved.signature,orders=STATIONARITY_ORDERS,
                stability_tolerance=QUADRATURE_STABILITY_TOLERANCE,
                practical_tolerance=AUDIT_PRACTICAL_TOLERANCE,
                scope="fixed saved q; bounded four-global oracle; no certified error bound")
            serialize(joinpath(directory,"comparison.jls"),result)
            println((;stage="quadrature_stability",stable,
                max_residual_delta=maximum(residual_delta),elbo_delta=fine.elbo-coarse.elbo,
                material_labels=audit.labels[material],
                unit_mean_change=maximum(abs.(fine.optimum_M-coarse.optimum_M)),
                unit_variance_change=maximum(abs.(fine.optimum_variance-coarse.optimum_variance))))
        end
    end
end
