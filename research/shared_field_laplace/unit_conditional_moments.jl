# Same retained128 joint draws: integrate only each small unit block given
# sampled hyperparameters/intercept/field. Dense algebra is a bounded oracle.
const UNIT_MOMENT_RESEARCH=isfile(joinpath(@__DIR__,"carry_conditional_moments.jl")) &&
    isfile(joinpath(@__DIR__,"gaussian_block_comparison.jl")) ? (@__DIR__) :
    @__DIR__
include(joinpath(UNIT_MOMENT_RESEARCH,"carry_conditional_moments.jl"))

# Reuse the validated fixed-M Gram projection. Increasing its numerical
# reference by alpha/delta times the conditional covariance adds precisely
# alpha*D_b to each local residual block, without changing the PCA of means.
function project_conditional_carry(means,blocks,conditional_blocks,reference;rank,shrinkage)
    0<shrinkage<=1 || throw(ArgumentError("conditional projection shrinkage must be in (0,1]"))
    length(blocks)==length(conditional_blocks)==length(reference) || error("conditional block counts differ")
    for (b,D) in zip(blocks,conditional_blocks)
        length(b)<=16 && size(D)==(length(b),length(b)) || error("conditional covariance block shape")
        issymmetric(D) && all(isfinite,D) && eigmin(Symmetric(D))>=0 || error("conditional covariance must be PSD")
    end
    adjusted=[T+((1-shrinkage)/shrinkage)*D for (T,D) in zip(reference,conditional_blocks)]
    result=project_block_carry(means,blocks,adjusted;rank,shrinkage)
    (;factor=result.factor,metadata=(;result.metadata...,
        method=:conditional_means_gram_plus_local_covariances,
        reference_blocks=deepcopy(reference),adjusted_reference_blocks=result.metadata.reference_blocks,
        conditional_blocks=deepcopy(conditional_blocks)))
end

# For a certified conditionally Gaussian independent-unit model ONLY.
# Existing colored full-density HVPs need max(block dimension) directions,
# not one global direction per stored unit. No growing dense Hessian here.
function native_unit_conditional(m,full,u)
    maximum(length,m.view.subject_coordinates)<=16 || error("unit block exceeds fixed cap")
    _,g=LogDensityProblems.logdensity_and_gradient(full,u)
    hessians=BRM._brm_nlme_sampler_hessian_blocks(m,u)
    means=copy(u);covariances=Matrix{Float64}[]
    for (indices,H) in zip(m.view.subject_coordinates,hessians)
        isapprox(H,H';atol=1e-10,rtol=1e-10) || error("unit Hessian is not symmetric")
        F=cholesky(Symmetric(-H))
        means[indices]=u[indices]+F\g[indices]
        push!(covariances,Matrix(inv(F)))
    end
    (;means,covariances)
end

@stestset "same draw unit conditional moments" begin
    base=joinpath(@__DIR__, "results")
    directory=joinpath(base,"unit-conditional-moments-v1")
    ispath(directory) && error("preserve unit moment diagnostic")
    previous_path=joinpath(base,"carry-conditional-moments-v1/comparison.jls")
    previous=deserialize(previous_path)
    draw_path=joinpath(base,"native-fixed-rate-fit-v1/draws.jls")
    draws=deserialize(draw_path)
    fit=deserialize(joinpath(base,"native-fixed-rate-fit-v1/comparison.jls"))
    reference=deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    fit.identity==draws.identity && reference.reference_ok &&
        previous.identity.fit_signature==fit.identity.signature &&
        previous.identity.reference_signature==reference.signature &&
        previous.identity.draw_file_sha==bytes2hex(sha256(read(draw_path))) || error("moment inputs differ")
    n,M,rank,delta=26,128,8,.01
    previous.identity.draw_selection==1:M && previous.identity.rank==rank &&
        previous.identity.delta==delta || error("moment controls differ")
    order=previous.order
    order==vcat(fit.globals,[fit.beta],fit.units,fit.field) || error("carried coordinate partition differs")
    blocks=Vector{Int}[];offset=0
    for b in vcat([fit.globals,[fit.beta]],[[j] for j in vcat(fit.units,fit.field)])
        push!(blocks,collect(offset+1:offset+length(b)));offset+=length(b)
    end
    X=draws.draws[:,1:M];means=copy(X);conditional=zeros(n,n)
    native_means=copy(X);native_conditional=zeros(n,n);native_costs=NamedTuple[]
    files=sort(filter(f->endswith(f,".jl"),readdir(UNIT_MOMENT_RESEARCH;join=true)))
    append!(files,[@__FILE__,joinpath(UNIT_MOMENT_RESEARCH,"../../test/Manifest.toml")])
    identity=(;previous.identity...,source_sha=bytes2hex(sha256(reduce(vcat,read.(files)))),
        previous_artifact_sha=bytes2hex(sha256(read(previous_path))),
        repository_commit=readchomp(`git -C $UNIT_MOMENT_RESEARCH rev-parse HEAD`),
        method=:unit_conditioning_on_sampled_physical_core)
    mkpath(directory);serialize(joinpath(directory,"identity.jls"),identity)
    with_progress(:state;description="Same-draw unit conditional moment diagnostic") do progress
        snapshot_progress(progress,joinpath(directory,"moments.progress")) do
            fixture=with_progress(progress;description="Bounded Gaussian oracle") do _
                (;rk,m,data)=shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2)
                merge((;rk,m,data),exact_gaussian_reference(rk,data))
            end
            (;rk,m,data,p,view)=fixture
            view.coordinates==fit.coordinates==reference.coordinates || error("coordinate order differs")
            sort(vcat(m.view.subject_coordinates...))==fit.units || error("unit packing differs")
            block=affine_gaussian_block(rk,data,p,view);latent=block.p3.latent
            block.p3.globals==fit.globals || error("hyperparameter partition differs")
            positions=[only(findall(==(i),latent)) for i in fit.units]
            kept=setdiff(eachindex(latent),positions)
            max_offdiagonal=0.;max_stationarity=0.;max_mean_difference=0.;max_sample_covariance_error=0.
            with_progress(progress,M;description="Condition small unit blocks at saved joint draws") do node
                for j in 1:M
                    c=gaussian_reference_conditional(block.p3,X[fit.globals,j])
                    max_mean_difference=max(max_mean_difference,maximum(abs.(c.u[order]-previous.means[:,j])))
                    H=c.H[positions,positions]
                    max_offdiagonal=max(max_offdiagonal,norm(H-Diagonal(diag(H)),Inf))
                    all(>(0),diag(H)) || error("unit block precision is not positive")
                    # Known fixture has one scalar per unit. Conditional units
                    # are independent given h, intercept and field weights.
                    conditional_mean=c.mu[positions]-(c.H[positions,kept]*(X[latent[kept],j]-c.mu[kept]))./diag(H)
                    means[fit.units,j]=conditional_mean
                    for (i,h) in zip(fit.units,diag(H))
                        conditional[i,i]+=inv(h)/M
                    end
                    restored=copy(X[latent,j]);restored[positions]=conditional_mean
                    max_stationarity=max(max_stationarity,norm((c.h-c.H*restored)[positions],Inf))
                    measurement=@timed native_unit_conditional(m,p.full,copy(X[:,j]))
                    native=measurement.value;native_means[:,j]=native.means
                    for (indices,C) in zip(m.view.subject_coordinates,native.covariances)
                        native_conditional[indices,indices].+=C/M
                        positions_in_latent=[only(findall(==(i),latent)) for i in indices]
                        exact=inv(Symmetric(c.H[positions_in_latent,positions_in_latent]))
                        max_sample_covariance_error=max(max_sample_covariance_error,maximum(abs.(C-exact)))
                    end
                    push!(native_costs,(;j,seconds=measurement.time,bytes=measurement.bytes,
                        inferred_full_gradient_calls=1,inferred_colored_hvp_directions=maximum(length,m.view.subject_coordinates)))
                    j%16==0 && update_progress!(node,j)
                end
            end
            @test max_mean_difference<1e-10
            @test max_offdiagonal<1e-10
            @test max_stationarity<1e-9
            native_mean_error=maximum(abs.(native_means-means))
            native_covariance_error=maximum(abs.(native_conditional-conditional))
            @test native_mean_error<1e-9
            @test native_covariance_error<1e-10
            @test max_sample_covariance_error<1e-10
            Y=native_means[order,:];mu=vec(mean(Y;dims=2));A=(Y.-mu)/sqrt(M)
            D=native_conditional[order,order];C=A*A'+D
            reference_blocks=[Matrix{Float64}(I,length(b),length(b)) for b in blocks]
            full_factor=bounded_moment_projection(mu,C,blocks,reference_blocks;rank,shrinkage=delta)
            projected=project_conditional_carry(Y,blocks,[D[b,b] for b in blocks],reference_blocks;rank,shrinkage=delta)
            mean_error=abs.(mu-reference.ref_mean[order])./reference.ref_sd[order]
            sd_ratio=sqrt.(diag(C))./reference.ref_sd[order]
            moment_ok=maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio)
            rng=Xoshiro(715);directions=randn(rng,n,16)
            directions./=sqrt.(sum(abs2,directions;dims=1))
            compressors=map(((:full_covariance_pca_oracle,full_factor),(:means_gram_local_covariance,projected.factor))) do (method,factor)
                P=hcat([block_covariance_action(factor,Float64.((1:n).==j)) for j in 1:n]...)
                for (b,T) in zip(blocks,reference_blocks)
                    @test P[b,b]≈(1-delta)*C[b,b]+delta*T atol=3e-12
                end
                @test factor.mu≈mu atol=2e-13
                projected_sd_ratio=sqrt.(diag(P))./reference.ref_sd[order]
                projection_ok=maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,projected_sd_ratio)
                directional_error=[dot(v,(P-C)*v)/dot(v,C*v) for v in eachcol(directions)]
                println((;method,moment_ok,projection_ok,max_mean_error=maximum(mean_error),sd_ratio=extrema(sd_ratio),
                    projected_sd_ratio=extrema(projected_sd_ratio),directional_error=extrema(directional_error)))
                (;method,projection_ok,projected_sd_ratio,directional_error,covariance=P)
            end
            result=(;identity,moment_ok,mean_error,sd_ratio,compressors,
                mean=mu,covariance=C,means=Y,projection=projected,
                conditional=D,A,max_offdiagonal,max_stationarity,max_mean_difference,coordinates=fit.coordinates[order],
                native_mean_error,native_covariance_error,max_sample_covariance_error,native_costs,
                scope="same128 existing joint draws; native colored unit Hessian blocks compared against bounded dense oracle; only unit conditional noise integrated, field/core sampling noise remains; initial independent-unit model only, not correlated-carry streaming acceptance")
            serialize(joinpath(directory,"comparison.jls"),result)
            println((;max_offdiagonal,max_stationarity,max_mean_difference,native_mean_error,native_covariance_error,max_sample_covariance_error,
                first_native_seconds=first(native_costs).seconds,
                median_later_native_seconds=median([c.seconds for c in native_costs[2:end]])))
        end
    end
end
