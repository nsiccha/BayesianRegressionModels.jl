# Fit only after a saved exact-reference summary passes all diagnostic gates.
# julia --project=test research/shared_field_laplace/posterior_fit.jl 'SVI versus exact reference'
include("joint_svi_probe.jl")
using Treebars, Serialization, Statistics

@stestset "SVI versus exact reference" begin
    reference_dir = get(ENV,"KB_LAPLACE_REFERENCE_DIR",
        joinpath(@__DIR__, "results", "exact-reference-v1"))
    reference = deserialize(joinpath(reference_dir,"reference-summary.jls"))
    reference.reference_ok || error("saved reference failed its diagnostic gate")
    directory = get(ENV,"KB_LAPLACE_FIT_DIR",
        joinpath(@__DIR__, "results", "joint-fit-v1"))
    mkpath(directory)
    files = [@__FILE__,joinpath(@__DIR__,"joint_svi_probe.jl"),
        joinpath(@__DIR__,"operator_probe.jl"),joinpath(@__DIR__,"trace_probe.jl"),
        joinpath(@__DIR__,"nonlinear_probe.jl"),joinpath(@__DIR__,"quadrature.jl"),
        joinpath(@__DIR__,"../../test/Manifest.toml")]
    signature = bytes2hex(sha256(vcat(codeunits(reference.signature),reduce(vcat,read.(files)))))
    with_progress(:state;description="Joint-window SVI accuracy") do progress
        snapshot_progress(progress,joinpath(directory,"fit.progress")) do
            (;m) = with_progress(progress;description="Prepare native joint-field target") do node
                (;rk) = shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2,
                    prepare_model=false)
                (;m=brm_nlme_model(rk;ad_backend=NLME_AD,progress=node))
            end
            @test m.view.coordinates == reference.coordinates
            t = joint_window_target(m;trace_probes=8,logdet_probes=4)
            start = first(BRM._brm_nlme_prior_median_point(m))
            fitfile = joinpath(directory,"fit.jls")
            record = if isfile(fitfile)
                saved = deserialize(fitfile)
                saved.signature == signature || error("fit code/environment changed")
                saved
            else
                q = BRM._brm_svi_params(t,start[t.globals],BRM.LowRankGlobal(4);
                    global_scale=.2,unit_scale=.5)
                with_progress(progress;description="Optimize collapsed target") do node
                    seconds = @elapsed fit = BRM._brm_svi_fit!(Xoshiro(700),t,q;
                        batch_size=3,max_iterations=5000,lr_global=.01,unit_step=.1,
                        polish_lr=.02,window=50,patience=4,min_windows=8,
                        polish_windows=10,node)
                    saved = (;fit,seconds,signature)
                    serialize(fitfile,saved)
                    saved
                end
            end
            fit = record.fit
            drawfile = joinpath(directory,"draws.jls")
            draw_record = if isfile(drawfile)
                saved = deserialize(drawfile)
                saved.signature == signature || error("draw code/environment changed")
                saved
            else
                with_progress(progress,1000;description="Joint conditional draws") do node
                    seconds = @elapsed draws = BRM._brm_svi_draws(Xoshiro(701),t,fit.q,1000;node)
                    saved = (;draws,seconds,signature)
                    serialize(drawfile,saved)
                    saved
                end
            end
            mean_error = abs.(vec(mean(draw_record.draws;dims=2))-reference.ref_mean)./reference.ref_sd
            sd_ratio = vec(std(draw_record.draws;dims=2))./reference.ref_sd
            # Original, predeclared tolerances. No relaxation after observation.
            accuracy_ok = fit.converged && maximum(mean_error) <= .3 &&
                all(x -> .8 <= x <= 1.2,sd_ratio)
            summary = (;accuracy_ok,fit.converged,fit.iterations,seconds=record.seconds,
                draw_seconds=draw_record.seconds,mean_error,sd_ratio,
                coordinates=m.view.coordinates,field=t.op.field,globals=t.globals,
                reference_signature=reference.signature,signature,
                scope="full-current-window accuracy; no streaming/locality acceptance")
            serialize(joinpath(directory,"comparison.jls"),summary)
            println((;stage="accuracy",summary...))
            @test accuracy_ok
        end
    end
end
