# Bounded diagnostic of the first carry failure. Same retained 128 hyper draws,
# rank 8 and shrinkage .01; no fit, posterior resampling or production claim.
const CARRY_RESEARCH = isfile(joinpath(@__DIR__,"gaussian_block_comparison.jl")) ? (@__DIR__) :
    @__DIR__
include(joinpath(CARRY_RESEARCH,"gaussian_block_comparison.jl"))
include(joinpath(CARRY_RESEARCH,"block_projection_probe.jl"))

function bounded_moment_projection(mu,C,blocks,reference;rank,shrinkage)
    n=length(mu)
    n==26 && size(C)==(n,n) || error("26-coordinate covariance oracle only")
    # An exact moment set, not fresh posterior samples. This invokes the same
    # rank/shrinkage map on the supplied moments without sampling them again.
    L=Matrix(cholesky(Symmetric(C)).L)
    X=mu.+sqrt(n)*hcat(L,-L)
    project_block_carry(X,blocks,reference;rank,shrinkage).factor
end

@stestset "fixed carry conditional moments" begin
    base=joinpath(@__DIR__, "results")
    directory=joinpath(base,"carry-conditional-moments-v1")
    ispath(directory) && error("preserve carry conditional moment audit")
    draw_path=joinpath(base,"native-fixed-rate-fit-v1/draws.jls")
    draws=deserialize(draw_path)
    fit=deserialize(joinpath(base,"native-fixed-rate-fit-v1/comparison.jls"))
    reference=deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    initial=deserialize(joinpath(base,"native-fixed-core-initial-v1/comparison.jls"))
    fit.identity==draws.identity && reference.reference_ok &&
        reference.signature==fit.identity.reference_signature || error("fit/reference identity differs")
    n,M,rank,delta=26,128,8,.01
    size(draws.draws)==(n,1000) || error("retained draw shape differs")
    initial.identity.fit_signature==fit.identity.signature &&
        initial.identity.reference_signature==reference.signature &&
        initial.identity.draw_file_sha==bytes2hex(sha256(read(draw_path))) &&
        initial.identity.draw_selection==1:M && initial.identity.rank==rank &&
        initial.identity.delta==delta || error("initial carry provenance differs")
    indices=vcat([fit.globals,[fit.beta]],[[j] for j in vcat(fit.units,fit.field)])
    order=vcat(indices...)
    blocks=Vector{Int}[];offset=0
    for b in indices
        push!(blocks,collect(offset+1:offset+length(b)));offset+=length(b)
    end
    X=draws.draws[order,1:M]
    reference_blocks=[Matrix{Float64}(I,length(b),length(b)) for b in blocks]
    files=sort(filter(f->endswith(f,".jl"),readdir(CARRY_RESEARCH;join=true)))
    append!(files,[@__FILE__,joinpath(CARRY_RESEARCH,"../../test/Manifest.toml")])
    identity=(;fit_signature=fit.identity.signature,reference_signature=reference.signature,
        draw_file_sha=bytes2hex(sha256(read(draw_path))),
        source_sha=bytes2hex(sha256(reduce(vcat,read.(files)))),
        repository_commit=readchomp(`git -C $CARRY_RESEARCH rev-parse HEAD`),
        julia=string(VERSION),julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
        draw_selection=1:M,rank,delta,direction_seed=715)
    mkpath(directory);serialize(joinpath(directory,"identity.jls"),identity)
    with_progress(:state;description="Fixed carry conditional moment diagnostic") do progress
        snapshot_progress(progress,joinpath(directory,"moments.progress")) do
            fixture=with_progress(progress;description="Bounded Gaussian oracle") do _
                (;rk,data)=shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2,prepare_model=false)
                merge((;rk,data),exact_gaussian_reference(rk,data))
            end
            (;rk,data,p,view)=fixture
            view.coordinates==fit.coordinates==reference.coordinates || error("coordinate identity differs")
            block=affine_gaussian_block(rk,data,p,view)
            block.p3.globals==fit.globals || error("hyperparameter partition differs")
            means=zeros(n,M);conditional=zeros(n,n);max_residual=0.
            with_progress(progress,M;description="Integrate same retained hyperparameter points") do node
                for j in 1:M
                    c=gaussian_reference_conditional(block.p3,draws.draws[fit.globals,j])
                    means[:,j]=c.u[order]
                    C=zeros(n,n);C[block.p3.latent,block.p3.latent]=inv(c.F)
                    conditional.+=C[order,order]/M
                    max_residual=max(max_residual,norm(c.H*c.mu-c.h)/max(norm(c.h),1.))
                    j%16==0 && update_progress!(node,j)
                end
            end
            @test max_residual<1e-10
            mu=vec(mean(means;dims=2))
            A=(means.-mu)/sqrt(M)
            rb_covariance=conditional+A*A'
            empirical_mu=vec(mean(X;dims=2));centered=(X.-empirical_mu)/sqrt(M)
            empirical_covariance=centered*centered'
            residual=(X-means)/sqrt(M)
            decross_covariance=A*A'+residual*residual'
            @test all(iszero,residual[1:3,:])
            @test rb_covariance[1:3,1:3]≈empirical_covariance[1:3,1:3] atol=2e-13
            @test mu[1:3]==empirical_mu[1:3]
            rng=Xoshiro(715);directions=randn(rng,n,16)
            directions./=sqrt.(sum(abs2,directions;dims=1))
            results=NamedTuple[]
            for (name,m,C) in ((:joint_draws,empirical_mu,empirical_covariance),
                    (:mean_residual_decross,mu,decross_covariance),
                    (:conditional_moments,mu,rb_covariance))
                projected=bounded_moment_projection(m,C,blocks,reference_blocks;rank,shrinkage=delta)
                P=hcat([block_covariance_action(projected,Float64.((1:n).==j)) for j in 1:n]...)
                for (b,T) in zip(blocks,reference_blocks)
                    @test P[b,b]≈(1-delta)*C[b,b]+delta*T atol=3e-12
                end
                @test projected.mu≈m atol=2e-13
                moment_mean_error=abs.(m-reference.ref_mean[order])./reference.ref_sd[order]
                moment_sd_ratio=sqrt.(diag(C))./reference.ref_sd[order]
                projected_sd_ratio=sqrt.(diag(P))./reference.ref_sd[order]
                moment_ok=maximum(moment_mean_error)<=.3 && all(x->.8<=x<=1.2,moment_sd_ratio)
                projection_ok=maximum(moment_mean_error)<=.3 && all(x->.8<=x<=1.2,projected_sd_ratio)
                directional_error=[dot(v,(P-C)*v)/dot(v,C*v) for v in eachcol(directions)]
                if name===:joint_draws
                    @test projected_sd_ratio≈initial.sd_ratio atol=2e-12
                    @test moment_mean_error≈initial.mean_error atol=2e-12
                    @test directional_error≈initial.variance_relative_error atol=2e-12
                end
                result=(;name,moment_ok,projection_ok,moment_mean_error,moment_sd_ratio,
                    projected_sd_ratio,directional_error,mean=m,covariance=C,projected_covariance=P)
                push!(results,result)
                println((;name,moment_ok,projection_ok,max_mean_error=maximum(moment_mean_error),
                    sd_ratio=extrema(moment_sd_ratio),projected_sd_ratio=extrema(projected_sd_ratio),
                    directional_error=extrema(directional_error)))
            end
            serialize(joinpath(directory,"comparison.jls"),(;identity,results,order,
                coordinates=fit.coordinates[order],means,conditional,max_residual,
                scope="26-coordinate dense conditional-moment oracle at the same128 saved hyperparameter points; no refit/new posterior draws; hyperparameter sampling error remains; not a scalable covariance-diagonal algorithm"))
        end
    end
end
