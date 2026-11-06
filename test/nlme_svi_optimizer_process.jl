# A fresh process of test/nlme_svi_optimizer.jl: rebuild the model from its data,
# continue an SVI optimizer checkpoint and report what it reached.
#
#     julia --project=test test/nlme_svi_optimizer_process.jl input.jls output.jls
#
# `input.mode`:
# - `:until_killed` resumes `input.state` with `save` writing every
#   `input.every` iterations to `input.path` by the atomic recipe of
#   `BRMSVIOptimizerState`. At iteration `input.kill_at` it writes part of a
#   checkpoint to the temporary file and terminates the process at once
#   (`_exit`, no cleanup), as an abrupt termination mid-write would.
# - `:finish` resumes the checkpoint stored at `input.path` to the end and
#   serializes the result to the output file.
# - `:callable` resumes `input.state` of `svi_callable_model` for `input.stop_after`
#   iterations and serializes the result with what names its callable here.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, Serialization, Random
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))

# `svi_shift` of test/nlme_svi_optimizer.jl from another driver: another position
# in the file, after other anonymous functions, one of them already serialized.
# Its generated name, line and Serialization's object numbers all differ there.
const svi_unrelated = (x -> x + 1, x -> x + 2)
serialize(IOBuffer(), first(svi_unrelated))
const svi_shift = eta -> eta .+ 0.1

function save_checkpoint(path, state)
    temporary = path * ".partial"
    open(io -> serialize(io, state), temporary, "w")
    mv(temporary, path; force=true)
end

input = deserialize(ARGS[1])
model = input.mode === :callable ? svi_callable_model(input.data) : svi_gauss_model(input.data)
if input.mode === :callable
    fit = brm_svi_resume(model, input.state; stop_after=input.stop_after, ndraws=0)
    backend = fit.diagnostics.model.backend
    serialize(ARGS[2], (; state=brm_svi_optimizer_checkpoint(fit),
        callable=nameof(typeof(svi_shift)),
        old_binding=BayesianRegressionModels._brm_rk_binding_identity(backend, "brm-svi-v1"; stable=false)))
elseif input.mode === :until_killed
    brm_svi_resume(model, input.state; checkpoint_every=input.every, ndraws=0) do state
        if state.run.iteration == input.kill_at
            io = IOBuffer()
            serialize(io, state)
            bytes = take!(io)
            write(input.path * ".partial", bytes[1:end÷2])
            ccall(:_exit, Cvoid, (Cint,), 9)
        end
        save_checkpoint(input.path, state)
    end
    error("the run ended before iteration $(input.kill_at)")
else
    fit = brm_svi_resume(model, deserialize(input.path); ndraws=input.ndraws)
    d = fit.diagnostics
    serialize(ARGS[2], (; fit.draws, d.q, d.elbo, d.elbo_windows, d.iterations,
        d.converged, d.unit_gradient_calls, state=brm_svi_optimizer_checkpoint(fit)))
end
