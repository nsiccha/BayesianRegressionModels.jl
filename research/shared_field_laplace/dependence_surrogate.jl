# Analytic information projections of a bounded Gaussian surrogate built from
# retained joint reference draws. This is NOT a Gaussian claim about the
# nonlinear authored posterior and is NOT an inference procedure for that target.
# julia --startup-file=no research/shared_field_laplace/dependence_surrogate.jl
using LinearAlgebra, Statistics, Serialization, SHA

function gaussian_kl(V,C)
    d = size(C,1)
    (tr(C\V)-d+logdet(Symmetric(C))-logdet(Symmetric(V)))/2
end

function dependence_controls(X)
    size(X,1)==10 && size(X,2)>=1000 || error("bounded ten-coordinate reference only")
    m = vec(mean(X;dims=2))
    centered = X .- m
    C = centered*centered'/(size(X,2)-1)
    F = cholesky(Symmetric(C))
    Q = inv(F)
    g,e = 1:4,5:10
    Vblock = zeros(10,10)
    Vblock[g,g] = inv(Symmetric(Q[g,g]))
    Vblock[e,e] = inv(Symmetric(Q[e,e]))
    Vproduct = copy(Vblock)
    Vproduct[e,e] = Diagonal(1 ./ diag(Q[e,e]))
    Vforward = zeros(10,10)
    Vforward[g,g] = C[g,g]
    Vforward[e,e] = Diagonal(diag(C[e,e]))
    # A dependence-capable conditional family: q(g) prod_i q(eta_i | g).
    # At the reverse-KL optimum of this Gaussian surrogate, q(g) has the exact
    # global marginal; unit regression uses the exact conditional means, but
    # its residual covariance is diagonal precision^-1, not marginal variance.
    A = C[e,g]/C[g,g]
    Vconditional = copy(C)
    Vconditional[e,e] = A*C[g,g]*A'+Diagonal(1 ./ diag(Q[e,e]))
    # Fixed rank=4, shrinkage=.02, in reference marginal-SD units. This is a
    # specified compression, NOT the reverse-KL optimum of a rank-four family.
    scales = sqrt.(diag(C))
    R = C./(scales*scales')
    eig = eigen(Symmetric(R))
    B = eig.vectors[:,end-3:end]*Diagonal(sqrt.(eig.values[end-3:end]))
    residual = diag(R)-vec(sum(abs2,B;dims=2))
    minimum(residual)>=-1e-12 || error("negative covariance residual")
    delta = .02
    D = (1-delta)*residual .+ delta
    all(>(0),D) || error("nonpositive projected base")
    Vrank = (scales*scales').*(Diagonal(D)+(1-delta)*B*B')
    covariances = (;full=C,independent_blocks=Vblock,product=Vproduct,
        conditional_units=Vconditional,forward_product=Vforward,rank4=Vrank)
    ratios = map(V->sqrt.(diag(V)./diag(C)),covariances)
    gaps = map(V->gaussian_kl(V,C),covariances)
    all(isposdef,Symmetric.(values(covariances))) || error("nonpositive control covariance")
    # Verify the compact reverse-KL expression independently against its full
    # trace expression for the three analytic optimum families.
    for V in (Vblock,Vproduct,Vconditional)
        gap = (logdet(Symmetric(C))-logdet(Symmetric(V)))/2
        isapprox(gap,gaussian_kl(V,C);atol=2e-11) || error("Gaussian projection identity")
    end
    norms = (;
        global_unit=norm(R[g,e]),unit_offdiagonal=norm(R[e,e]-I))
    functionals = zeros(10,4)
    functionals[3,1]=1;functionals[5,1]=1
    functionals[3,2]=1;functionals[5,2]=-1
    functionals[5:10,3] .= 1/sqrt(6)
    functionals[4,4]=1;functionals[5,4]=1
    # Functionals are in standardized parameter coordinates, not authored
    # response predictions. No interpretation as a new predictive gate.
    directions = functionals./scales
    functional_variances = map(V->diag(directions'*V*directions),covariances)
    (;mean=m,C,Q,covariances,ratios,gaps,norms,A,
        conditional_variance=1 ./ diag(Q[e,e]),functional_variances,
        rank=4,shrinkage=delta,reference_draws=size(X,2))
end

function run_dependence_surrogate()
    base = joinpath(@__DIR__, "results")
    directory = joinpath(base,"dependence-surrogate-v1")
    ispath(directory) && error("preserve existing surrogate results")
    reference = deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    reference.reference_ok || error("reference not accepted")
    records = [deserialize(joinpath(base,"exact-reference-v1/reference-$i.jls")) for i in 1:4]
    all(r->r.signature==reference.signature,records) || error("reference identity mismatch")
    coordinates = vcat(reference.globals,reference.units)
    length(coordinates)==10 || error("unexpected coordinate layout")
    arrays = [r.position[coordinates,:] for r in records]
    combined = dependence_controls(reduce(hcat,arrays))
    per_chain = dependence_controls.(arrays)
    mkpath(directory)
    signature = bytes2hex(sha256(vcat(read(@__FILE__),codeunits(reference.signature))))
    serialize(joinpath(directory,"controls.jls"),(;combined,per_chain,coordinates,
        names=reference.coordinates[coordinates],signature,reference_signature=reference.signature,
        scope="moment-matched Gaussian surrogate; per-chain sensitivity is not an iid error bar"))
    println((;stage="combined",draws=combined.reference_draws,gaps=combined.gaps,
        global_unit_correlation_norm=combined.norms.global_unit))
    for name in keys(combined.covariances)
        ratio = getproperty(combined.ratios,name)
        println((;family=name,global_sd_ratio=ratio[1:4],unit_sd_ratio=ratio[5:10],
            gap=getproperty(combined.gaps,name),
            functional_variance_ratio=getproperty(combined.functional_variances,name)./
                combined.functional_variances.full))
    end
    for (i,control) in enumerate(per_chain)
        println((;chain=i,product_global_sd_ratio=control.ratios.product[1:4],
            conditional_unit_sd_ratio=control.ratios.conditional_units[5:10],gaps=control.gaps))
    end
end

run_dependence_surrogate()
