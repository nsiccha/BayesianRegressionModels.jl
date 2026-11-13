# The native density graph of an in-cell `HiddenMarkov(...)` observation
# (hidden_markov.jl): the steps × states log-emission matrix, then RK's
# forward algorithm (`hmm_logdensity`, or `hmm_log_weights_logdensity` for log
# weights).
#
# A state's law whose arguments are each read whole per state (`eachcol(Phi)`,
# `eachrow(Phi)`, `Ref(x)`, a listed law's own arguments) reads the state and
# not the step. Its column is then a plate over the steps whose cell evaluates
# the law at the step's value: the law is a plate-invariant subexpression of
# that plate, which RK evaluates once per state instead of once per
# (step, state) cell. For a categorical law that is one column slice and one
# simplex check per state rather than per step (snag
# `rkbrmi-hiddenmar-7ebba16b`: 2–3× the categorical notebook models' value
# and gradient). A broadcast emission (`F.(args...)`, K from the data) is a
# plate over the states around that step plate; listed laws (`[F1(...),
# F2(...)]`, K written out) give one step plate per law, with no branch on the
# state. A broadcast argument read per entry (`Normal.(mu, sigma)`: `mu[k]`,
# or `mu[k, t]` with a step axis) reads the step through `_brm_hmm_at`, so the
# law is built per cell whatever the plates; such a broadcast keeps one plate
# cell per (step, state), as RK's own `normal_hmm_logdensity` does, which
# costs less than nesting it.
#
# A state's law whose family RKPPL maps to a graph (`sampling_distribution_graph`)
# contributes that graph's `logpdf`; any other family its Distributions
# `logpdf`, the numerical route RKPPL itself takes for such a family.

# The Distributions log density of `F(arguments...)` at `value`.
_brm_hmm_numerical_logpdf(::Type{F}, value, arguments...) where {F} =
    BRM.Distributions.logpdf(F(arguments...), value)

function _rk_hmm_state_score(F, modes, captures, k, t, value)
    arguments = Tuple(
        mode === :row ? :($capture[$k, :]) :
        mode === :col ? :($capture[:, $k]) :
        mode === :each ? Expr(:call, GlobalRef(BRM, :_brm_hmm_at), capture, k, t) : capture
        for (capture, mode) in zip(captures, modes))
    object = ReactiveKernelsPPL.sampling_distribution_graph(F, arguments)
    object isa Expr && return :($object.logpdf($value))
    Expr(:call, GlobalRef(@__MODULE__, :_brm_hmm_numerical_logpdf), F, value, arguments...)
end

function ReactiveKernelsPPL.sampling_distribution_graph(
        ::Type{<:BRM._BRMHiddenMarkovLaw{F,M,B,L}}, args::Tuple) where {F,M,B,L}
    inputs, body = _rk_hmm_density_body(fieldtypes(F), M, B)
    _rk_hmm_density_spec(inputs, body,
        L ? DistributionKernelSources.hmm_log_weights_logdensity :
            DistributionKernelSources.hmm_logdensity)
end

# The density's inputs and body: the log-emission matrix, then the forward
# algorithm, bound as `brm_hmm_forward`.
function _rk_hmm_density_body(families, M, B)
    counts = map(length, M)
    starts = cumsum((0, counts[1:end-1]...))
    captures = Tuple(Symbol(:argument_, j) for j in 1:sum(counts))
    score(s, state) = _rk_hmm_state_score(families[s], M[s],
        captures[starts[s] .+ (1:counts[s])], state, :hmm_t, :(value[hmm_t]))
    # One state's log emissions at every step.
    column(state_score) = :(ReactiveKernels.plate(1:hmm_steps) do hmm_t
        $state_score
    end)
    emissions = if B && !(:each in M[1])
        (:(hmm_states = length(initial)),
         :(hmm_columns = ReactiveKernels.plate(1:hmm_states) do hmm_k
             $(column(score(1, :hmm_k)))
         end),
         :(log_emissions = reshape(reduce(vcat, hmm_columns), hmm_steps, hmm_states)))
    elseif B
        (:(hmm_states = length(initial)),
         :(hmm_cells = ReactiveKernels.plate(1:(hmm_steps * hmm_states)) do hmm_cell
             hmm_t = (hmm_cell - 1) % hmm_steps + 1
             hmm_k = (hmm_cell - 1) ÷ hmm_steps + 1
             $(score(1, :hmm_k))
         end),
         :(log_emissions = reshape(hmm_cells, hmm_steps, hmm_states)))
    else
        (:(log_emissions = $(Expr(:call, :hcat,
            (column(score(s, s)) for s in eachindex(families))...))),)
    end
    body = Expr(:block, :(hmm_steps = length(value)), emissions...,
        :(brm_hmm_forward(log_emissions, transition, initial)))
    (:value, :initial, :transition, captures...), body
end

# As `_rk_density_spec`, with the forward algorithm bound by name in the
# kernel's module, so the density composes it as a child graph.
function _rk_hmm_density_spec(inputs, body, forward)
    mod = _rk_new_module()
    Core.eval(mod, :(const ReactiveKernels = $ReactiveKernels))
    Core.eval(mod, :(using .ReactiveKernels: @kernel))
    Core.eval(mod, :(const brm_hmm_forward = $forward))
    Core.eval(mod, _rk_density_definition(:brm_owned_density, inputs, body))
end
