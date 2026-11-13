# ---- `HiddenMarkov(...)` in a cell on StanBlocks -------------------------------
#
# `y ~ HiddenMarkov(initial, transition, emissions)` in a `@plate` cell
# observes the cell's sequence `y` through BRM's forward-algorithm family:
#
#     y ~ brm_hidden_markov(log_initial, log_transition, emit, emit_rng)
#
# `emit(y_t, k, t)` is state k's emission log density at step t and
# `emit_rng(k, t)` draws from it; both are closures the cell builds from the
# authored emission argument (`_BRMHMMEmission`, hidden_markov.jl), so
# StanBlocks inlines each state's law into the generated function. Weights
# enter on the log scale: a probability `initial`/`transition` is passed as
# its `log`, a `log_weights = true` one as written. `initial` is the first
# step's state law, as in Stan's `hmm_marginal`, HiddenMarkovModels.jl and RK's
# `hmm_logdensity` (StanBlocks' `hmm_forward` instead moves its `rho` once
# before the first emission). The pointwise value is the sequence's log
# density, one per cell, as on RKBRMI.
#
# Discrete emissions observe an integer sequence (`_lpmf`), continuous ones a
# real sequence (`_lpdf`); Stan needs the suffix that matches the variate.

StanBlocks.@deffun begin
    "The log density of one sequence under a hidden Markov chain (forward algorithm, log weights)."
    brm_hidden_markov_lpdfs(y::vector[hmm_T], log_initial::vector[hmm_K],
            log_transition::matrix[hmm_K, hmm_K], emit, emit_rng)::real = begin
        hmm_alpha::vector[hmm_K]
        hmm_next::vector[hmm_K]
        hmm_total::real = 0.0
        for hmm_t in 1:hmm_T
            for hmm_j in 1:hmm_K
                if hmm_t == 1
                    hmm_next[hmm_j] = log_initial[hmm_j] + emit(y[hmm_t], hmm_j, hmm_t)
                else
                    hmm_next[hmm_j] = log_sum_exp(hmm_alpha + log_transition[:, hmm_j]) +
                        emit(y[hmm_t], hmm_j, hmm_t)
                end
            end
            hmm_alpha = hmm_next
        end
        if hmm_T > 0
            hmm_total = log_sum_exp(hmm_alpha)
        end
        return hmm_total
    end
    @lpxf brm_hidden_markov_lpdf(y::vector[hmm_T], log_initial::vector[hmm_K],
            log_transition::matrix[hmm_K, hmm_K], emit, emit_rng)::real =
        brm_hidden_markov_lpdfs(y, log_initial, log_transition, emit, emit_rng)
    "One forward simulation of a sequence: the state from its log weights, then the state's emission."
    brm_hidden_markov_rng(vector[hmm_T], log_initial::vector[hmm_K],
            log_transition::matrix[hmm_K, hmm_K], emit, emit_rng)::vector[hmm_T] = begin
        hmm_y::vector[hmm_T]
        hmm_z = 1
        for hmm_t in 1:hmm_T
            if hmm_t == 1
                hmm_z = categorical_logit_rng(log_initial)
            else
                hmm_z = categorical_logit_rng(log_transition[hmm_z, :]')
            end
            hmm_y[hmm_t] = emit_rng(hmm_z, hmm_t)
        end
        return hmm_y
    end
    "The log probability of one integer sequence under a hidden Markov chain (forward algorithm, log weights)."
    brm_hidden_markov_discrete_lpmfs(y::int[hmm_T], log_initial::vector[hmm_K],
            log_transition::matrix[hmm_K, hmm_K], emit, emit_rng)::real = begin
        hmm_alpha::vector[hmm_K]
        hmm_next::vector[hmm_K]
        hmm_total::real = 0.0
        for hmm_t in 1:hmm_T
            for hmm_j in 1:hmm_K
                if hmm_t == 1
                    hmm_next[hmm_j] = log_initial[hmm_j] + emit(y[hmm_t], hmm_j, hmm_t)
                else
                    hmm_next[hmm_j] = log_sum_exp(hmm_alpha + log_transition[:, hmm_j]) +
                        emit(y[hmm_t], hmm_j, hmm_t)
                end
            end
            hmm_alpha = hmm_next
        end
        if hmm_T > 0
            hmm_total = log_sum_exp(hmm_alpha)
        end
        return hmm_total
    end
    @lpxf brm_hidden_markov_discrete_lpmf(y::int[hmm_T], log_initial::vector[hmm_K],
            log_transition::matrix[hmm_K, hmm_K], emit, emit_rng)::real =
        brm_hidden_markov_discrete_lpmfs(y, log_initial, log_transition, emit, emit_rng)
    "One forward simulation of an integer sequence."
    brm_hidden_markov_discrete_rng(int[hmm_T], log_initial::vector[hmm_K],
            log_transition::matrix[hmm_K, hmm_K], emit, emit_rng)::int[hmm_T] = begin
        hmm_y::int[hmm_T]
        hmm_z = 1
        for hmm_t in 1:hmm_T
            if hmm_t == 1
                hmm_z = categorical_logit_rng(log_initial)
            else
                hmm_z = categorical_logit_rng(log_transition[hmm_z, :]')
            end
            hmm_y[hmm_t] = emit_rng(hmm_z, hmm_t)
        end
        return hmm_y
    end
    # State k's (and step t's) entry of a broadcast emission argument, by
    # Julia's broadcasting rules: a number is shared, a vector is read at its
    # state, a states × steps matrix at its state and step; an axis of length
    # one is shared.
    brm_hmm_at(x::real, hmm_k::int, hmm_t::int)::real = x
    brm_hmm_at(x::int, hmm_k::int, hmm_t::int)::int = x
    brm_hmm_at(x::vector[hmm_n], hmm_k::int, hmm_t::int)::real =
        x[hmm_n == 1 ? 1 : hmm_k]
    brm_hmm_at(x::int[hmm_n], hmm_k::int, hmm_t::int)::int =
        x[hmm_n == 1 ? 1 : hmm_k]
    brm_hmm_at(x::matrix[hmm_m, hmm_n], hmm_k::int, hmm_t::int)::real =
        x[hmm_m == 1 ? 1 : hmm_k, hmm_n == 1 ? 1 : hmm_t]
end

# The Stan spelling of state k's (step t's) emission law: its StanBlocks
# family name and arguments, each read per state by its mode.
function _sb_hmm_state_law(emission::_BRMHMMEmission, state, k, t, captures)
    F = emission.families[state]
    arguments = Any[]
    for (capture, mode) in zip(captures[state], emission.modes[state])
        push!(arguments,
            mode === :row ? Expr(Symbol("'"), Expr(:ref, capture, k, :(:))) :
            mode === :col ? Expr(:ref, capture, :(:), k) :
            mode === :each ? Expr(:call, brm_hmm_at, capture, k, t) : capture)
    end
    call = _sb_stan_distribution_call(F, Tuple(arguments), NamedTuple())
    discrete = F <: Distributions.DiscreteUnivariateDistribution
    (; name=first(call.args), arguments=call.args[2:end], discrete)
end

# Each state's law in turn: `law(state)` for state 1, else a branch on `k`.
function _sb_hmm_by_state(emission::_BRMHMMEmission, k, law)
    emission.broadcast && return law(1)
    K = length(emission.families)
    branch = law(K)
    for state in (K - 1):-1:1
        branch = Expr(:if, Expr(:call, :(==), k, state), law(state), branch)
    end
    branch
end

# `y ~ HiddenMarkov(...)` as cell statements: one local per non-name
# emission argument (computed once per cell, not once per state and step),
# then the observation through BRM's forward-algorithm family.
function _sb_cell_hidden_markov(x::Expr, scope)
    response, law = x.args[2], x.args[3]
    spec = _brm_hmm_call(law, head -> _sb_cell_family_binding(scope, head); context="sbimpl")
    emission = spec.emission
    statements = Any[]
    captures = map(enumerate(emission.arguments)) do (state, arguments)
        map(enumerate(arguments)) do (j, argument)
            value = _sb_cell_value(argument, scope)
            value isa Union{Symbol,Real} && return value
            name = Symbol(:brm_hmm_emission_, state, :_, j)
            push!(statements, Expr(:(=), name, value))
            name
        end
    end
    y, k, t = :brm_hmm_y, :brm_hmm_k, :brm_hmm_t
    discrete = _brm_hmm_discrete(emission)
    density = _sb_hmm_by_state(emission, k, state -> begin
        law = _sb_hmm_state_law(emission, state, k, t, captures)
        Expr(:call, Symbol(law.name, law.discrete ? :_lpmf : :_lpdf), y, law.arguments...)
    end)
    draw = _sb_hmm_by_state(emission, k, state -> begin
        law = _sb_hmm_state_law(emission, state, k, t, captures)
        Expr(:call, Symbol(law.name, :_rng), law.arguments...)
    end)
    initial = _sb_cell_value(spec.initial, scope)
    transition = _sb_cell_value(spec.transition, scope)
    spec.log_weights || (initial = Expr(:call, :log, initial);
        transition = Expr(:call, :log, transition))
    family = discrete ? brm_hidden_markov_discrete : brm_hidden_markov
    push!(statements, Expr(:call, :~, response, Expr(:call, family, initial, transition,
        Expr(:->, Expr(:tuple, y, k, t), density), Expr(:->, Expr(:tuple, k, t), draw))))
    statements
end

_sb_is_cell_hidden_markov(x, scope) = Meta.isexpr(x, :call, 3) && x.args[1] === :~ &&
    Meta.isexpr(x.args[3], :call) && _sb_cell_binding(scope, first(x.args[3].args)) === HiddenMarkov
