# The native density graph of an in-cell `HiddenMarkov(...)` observation
# (hidden_markov.jl): the steps × states log-emission matrix, one plate cell
# per (step, state), from each state's law, then RK's forward algorithm
# (`hmm_logdensity`, or `hmm_log_weights_logdensity` for log weights).
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
    families = fieldtypes(F)
    counts = map(length, M)
    starts = cumsum((0, counts[1:end-1]...))
    captures = Tuple(Symbol(:argument_, j) for j in 1:sum(counts))
    k, t, value = :hmm_k, :hmm_t, :(value[hmm_t])
    score(s, state) = _rk_hmm_state_score(families[s], M[s],
        captures[starts[s] .+ (1:counts[s])], state, t, value)
    cell = if B
        score(1, k)
    else
        branch = score(length(families), length(families))
        for s in (length(families) - 1):-1:1
            branch = :($k == $s ? $(score(s, s)) : $branch)
        end
        branch
    end
    body = quote
        hmm_steps = length(value)
        hmm_states = length(initial)
        hmm_cells = ReactiveKernels.plate(1:(hmm_steps * hmm_states)) do hmm_cell
            hmm_t = (hmm_cell - 1) % hmm_steps + 1
            hmm_k = (hmm_cell - 1) ÷ hmm_steps + 1
            $cell
        end
        log_emissions = reshape(hmm_cells, hmm_steps, hmm_states)
        brm_hmm_forward(log_emissions, transition, initial)
    end
    _rk_hmm_density_spec((:value, :initial, :transition, captures...), body,
        L ? DistributionKernelSources.hmm_log_weights_logdensity :
            DistributionKernelSources.hmm_logdensity)
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
