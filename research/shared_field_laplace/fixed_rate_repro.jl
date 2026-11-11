# Fresh-process reproduction of the actual capped algorithm and its draws.
# No new initialization, seed, budget, endpoint selection or reference sample.
# julia --project=test research/shared_field_laplace/fixed_rate_repro.jl 'native fixed rate reproducibility'
include("fixed_rate_fit.jl")

@stestset "native fixed rate reproducibility" begin
    base=joinpath(@__DIR__, "results")
    saved=deserialize(joinpath(base,"native-fixed-rate-fit-v1/fit.jls"))
    previous=deserialize(joinpath(base,"native-fixed-rate-fit-v1/comparison.jls"))
    previous_draws=deserialize(joinpath(base,"native-fixed-rate-fit-v1/draws.jls"))
    saved.identity==previous.identity==previous_draws.identity || error("retained fit identities differ")
    saved.identity.julia==string(VERSION) || error("reproduction Julia version differs")
    sources=[joinpath(@__DIR__,name) for name in
        ("fixed_rate_fit.jl","iterate_stopping.jl","path_derivative_fit.jl","gaussian_block_fit.jl",
         "gaussian_block_comparison.jl","stationarity_audit.jl","exact_reference.jl","operator_probe.jl",
         "whole_window_probe.jl","joint_svi_probe.jl","nonlinear_probe.jl","trace_probe.jl","quadrature.jl")]
    append!(sources,[joinpath(@__DIR__,"../../src",name) for name in
        ("nlme_view.jl","nlme_svi.jl","rk_backend.jl")])
    push!(sources,joinpath(@__DIR__,"../../test/Manifest.toml"))
    bytes2hex(sha256(reduce(vcat,read.(sources))))==saved.identity.signature ||
        error("fixed-rate algorithm/environment changed")
    directory=joinpath(base,"native-fixed-rate-repro-v1")
    ispath(directory) && error("preserve prior reproduction artifacts")
    mkpath(directory)
    identity=(;signature=bytes2hex(sha256(reduce(vcat,read.(vcat([@__FILE__],sources))))),
        fit_signature=saved.identity.signature,julia=string(VERSION),fit_seed=700,draw_seed=701,
        repository_commit=readchomp(`git -C $(@__DIR__) rev-parse HEAD`),
        julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
        enzyme=pkgdir(Enzyme),libenzyme=Enzyme.Compiler.Enzyme_jll.libEnzyme_path)
    serialize(joinpath(directory,"identity.jls"),identity)
    with_progress(:state;description="Fresh-process capped SVI reproduction") do progress
        snapshot_progress(progress,joinpath(directory,"repro.progress")) do
            fixture=with_progress(progress;description="Rebuild unchanged native target") do _
                (;rk,m,data)=shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2)
                merge((;rk,m,data),exact_gaussian_reference(rk,data))
            end
            (;rk,m,data,p,view)=fixture
            view.coordinates==saved.coordinates || error("reproduction coordinates differ")
            block=affine_gaussian_block(rk,data,p,view)
            joint=native_gaussian_block_target(m,block)
            joint.globals==saved.globals && joint.op.field==saved.latent || error("reproduction partition differs")
            t=PathDerivativeTarget(joint)
            start=first(BRM._brm_nlme_prior_median_point(m))
            q=BRM._brm_svi_params(t,start[t.globals],BRM.LowRankGlobal(3);global_scale=.2,unit_scale=.5)
            seconds=@elapsed fit=with_progress(progress;description="Repeat exact capped fit") do node
                fixed_rate_fit!(Xoshiro(700),t,q;node)
            end
            serialize(joinpath(directory,"fit.jls"),(;identity,fit,last_q=deepcopy(q),seconds))
            draw_seconds=@elapsed draws=with_progress(progress,1000;description="Repeat conditional draws") do node
                BRM._brm_svi_draws(Xoshiro(701),t,fit.q,1000;node)
            end
            serialize(joinpath(directory,"draws.jls"),(;identity,draws,seconds=draw_seconds))
            digest=bytes2hex(sha256(reinterpret(UInt8,vcat(fit.q.μ,fit.q.ρ,vec(fit.q.Β),fit.elbo,vec(draws)))))
            checks=(;digest=digest==previous.digest,draws=draws==previous_draws.draws,
                trajectory=fit.raw==saved.fit.raw && fit.distributions==saved.fit.distributions,
                last_q=BRM._brm_svi_global_vector(q)==BRM._brm_svi_global_vector(saved.last_q),
                iterate_checks=isequal(fit.checks,saved.fit.checks),
                status=(fit.iterations,fit.iterate_diagnostics_passed,fit.budget_exhausted,fit.averaging_window)==
                    (saved.fit.iterations,saved.fit.iterate_diagnostics_passed,saved.fit.budget_exhausted,saved.fit.averaging_window))
            serialize(joinpath(directory,"comparison.jls"),(;identity,checks,digest,seconds,draw_seconds,
                scope="same-host fresh-process exact fit/draw/status reproduction; no new accuracy or convergence evidence"))
            println((;checks,digest,seconds,draw_seconds))
            @test all(values(checks))
        end
    end
end
