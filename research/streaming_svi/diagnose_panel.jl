# Synthetic control for a misleading short-run batch reference. From the repo:
# julia --project=test research/streaming_svi/diagnose_panel.jl OUTPUT_DIR
# python3 research/streaming_svi/panel_oracle.py OUTPUT_DIR
# Both commands need the usual compute-token wrapper. The checked small-run
# results are retained in results/; the Python control requires NumPy only.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
include(joinpath(@__DIR__, "..", "..", "test", "nlme_svi_streaming_fixture.jl"))
output_dir = abspath(only(ARGS))
mkpath(output_dir)
labels = ["s$i" for i in 1:8]
windows = [svi_stream_data(Xoshiro(70+i), labels, 12) for i in 1:6]
open(joinpath(output_dir, "panel-stats.tsv"), "w") do io
    for (i,w) in enumerate(windows), label in labels
        y = w.y[w.subject .== label]
        println(io, join((i,label,length(y),sum(y),sum(abs2,y)), '\t'))
    end
end
function record(io, mode, i, fit)
    s = brm_svi_checkpoint(fit)
    for (j,name) in enumerate(s.globals)
        println(io, join((mode,i,name,s.q.μ[j],exp(2s.q.ρ[j])), '\t'))
    end
    for label in labels
        f=s.units[label]
        println(io, join((mode,i,label,only(f.M),inv(only(f.Λ))), '\t'))
    end
    println("PANEL_DIAG mode=$mode window=$i globals=$(s.globals) means=$(s.q.μ) " *
        "sd=$(exp.(s.q.ρ)) converged=$(fit.diagnostics.converged) iterations=$(fit.diagnostics.iterations)")
    flush(stdout)
    s
end
all_data=(;subject=reduce(vcat,[w.subject for w in windows]),
    x=reduce(vcat,[w.x for w in windows]), y=reduce(vcat,[w.y for w in windows]))
open(joinpath(output_dir, "panel-factors.tsv"), "w") do io
    for (mode,controls) in (
        ("short",(;max_iterations=4000,window=50,min_windows=20,patience=8,polish_windows=15)),
        ("long",(;max_iterations=30000,window=100,min_windows=200,patience=20,polish_windows=50)))
        first_fit=brm_svi_fit(svi_stream_model(first(windows));rng=Xoshiro(80),batch_size=8,ndraws=0,controls...)
        s=record(io,mode,1,first_fit)
        for i in 2:6
            fit=brm_svi_fit(s,svi_stream_model(windows[i]);rng=Xoshiro(80+i),batch_size=8,ndraws=0,controls...)
            s=record(io,mode,i,fit)
        end
        batch=brm_svi_fit(svi_stream_model(all_data);rng=Xoshiro(90),batch_size=8,ndraws=0,controls...)
        record(io,mode,0,batch)
        flush(io)
    end
end
