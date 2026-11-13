# ---- `HiddenMarkov(initial, transition, emissions)` --------------------------
#
# A sequence observed through a hidden discrete state that moves as a Markov
# chain (snag `hidden-markov-ob-6b35403a`, design decision `0sl1z6o`). The
# argument order is HiddenMarkovModels.jl's `HMM(init, trans, dists)`. In a
# `@plate` cell, `y[s] ~ HiddenMarkov(...)` observes the cell's whole sequence;
# each backend marginalizes the state path with the forward algorithm
# (`sb_hidden_markov.jl` on SBBRMI, `rk_hidden_markov.jl` on RKBRMI). This file
# holds the Julia law both backends are checked against and the parsing of the
# emission argument they share.

"""
    HiddenMarkov(initial, transition, emissions; log_weights = false)

The law of a sequence `y[1], …, y[T]` observed through a hidden state
`z[1], …, z[T]` in `1:K` that moves as a Markov chain:

- `initial[k]` is the probability that `z[1] = k`;
- `transition[i, j]` is the probability of a move from state `i` to state `j`
  (each row sums to one, as in Stan, StanBlocks, ReactiveKernels and
  HiddenMarkovModels.jl). A `K × K × (T - 1)` array gives the move into step
  `t` at slice `t - 1`; singleton axes broadcast;
- `emissions` holds one distribution per state: `y[t] | z[t] = k` follows
  `emissions[k]`. A `K × T` matrix of distributions gives the law of every
  state at every step.

`logpdf(d, y)` sums the density over every state path with the forward
algorithm, in log space. With `log_weights = true`, `initial` and `transition`
hold the logarithms of those weights (a log-softmax, say); no probability is
formed, and `-Inf` is a structural zero.

Inside a `@brm` `@plate` cell, `y[s] ~ HiddenMarkov(init, Gamma, Categorical.(eachrow(Phi)))`
observes cell `s`'s sequence `y[s]`. The emission argument is a broadcast of a
Distributions family over per-state arguments (`Normal.(mu, sigma)`,
`Categorical.(eachrow(Phi))`, `Poisson.(lambda)`), a vector of laws with one per
state (`[Normal(m, s), Poisson(l)]`, families may differ), or a comprehension
`[Normal(mu[k], s) for k in 1:K]`; a broadcast argument with a second, step
axis gives laws that change with the step. A column-stochastic matrix `Θ`
(`Θ[j, i]` the move from `i` to `j`) is passed as `permutedims(Θ)`, or declared
with simplex columns, `eachcol(Θ[1:K, 1:K]) .~ Dirichlet(K, 1)`, and passed
transposed.
"""
struct HiddenMarkov{S<:Distributions.ValueSupport,I,G,E} <:
       Distributions.Distribution{Distributions.Multivariate,S}
    initial::I
    transition::G
    emissions::E
    log_weights::Bool
end

_brm_hmm_support(::Type{<:Distribution{<:Any,Distributions.Discrete}}) = Distributions.Discrete
_brm_hmm_support(::Type) = Distributions.Continuous
function HiddenMarkov(initial::AbstractVector, transition::AbstractArray,
                      emissions::AbstractArray; log_weights::Bool=false)
    K = length(initial)
    ndims(emissions) in (1, 2) && size(emissions, 1) == K || throw(DimensionMismatch(
        "HiddenMarkov: `emissions` needs one law per state (K = $K rows), got size " *
        "$(size(emissions))"))
    ndims(transition) in (2, 3) && size(transition, 1) in (1, K) &&
        size(transition, 2) in (1, K) || throw(DimensionMismatch(
        "HiddenMarkov: `transition` must be K × K or K × K × (T - 1) (K = $K), got size " *
        "$(size(transition))"))
    discrete = all(e -> _brm_hmm_support(typeof(e)) === Distributions.Discrete, emissions)
    S = discrete ? Distributions.Discrete : Distributions.Continuous
    HiddenMarkov{S,typeof(initial),typeof(transition),typeof(emissions)}(
        initial, transition, emissions, log_weights)
end

Base.show(io::IO, d::HiddenMarkov) = print(io, "HiddenMarkov(K = ",
    length(d.initial), d.log_weights ? ", log_weights" : "", ")")

_brm_hmm_index(x, axis, i) = size(x, axis) == 1 ? 1 : i
_brm_hmm_emission(e::AbstractVector, k, t) = e[k]
_brm_hmm_emission(e::AbstractMatrix, k, t) = e[k, _brm_hmm_index(e, 2, t)]
_brm_hmm_log_transition(d, i, j, t) = (G = d.transition;
    w = ndims(G) == 2 ? G[_brm_hmm_index(G, 1, i), _brm_hmm_index(G, 2, j)] :
        G[_brm_hmm_index(G, 1, i), _brm_hmm_index(G, 2, j), _brm_hmm_index(G, 3, t)];
    d.log_weights ? w : log(w))
_brm_hmm_log_initial(d, k) = d.log_weights ? d.initial[k] : log(d.initial[k])

# The forward algorithm: `alpha[k]` is the log weight of every path in state k
# at step t, given the first t observations.
function Distributions.logpdf(d::HiddenMarkov, y::AbstractVector)
    K, T = length(d.initial), length(y)
    T == 0 && return 0.0
    ndims(d.emissions) == 2 && !(size(d.emissions, 2) in (1, T)) && throw(DimensionMismatch(
        "HiddenMarkov: the emission matrix has $(size(d.emissions, 2)) steps, the sequence $T"))
    alpha = [_brm_hmm_log_initial(d, k) + logpdf(_brm_hmm_emission(d.emissions, k, 1), y[1])
             for k in 1:K]
    for t in 2:T
        alpha = [logsumexp(alpha[i] + _brm_hmm_log_transition(d, i, j, t - 1) for i in 1:K) +
                 logpdf(_brm_hmm_emission(d.emissions, j, t), y[t]) for j in 1:K]
    end
    logsumexp(alpha)
end
Distributions.loglikelihood(d::HiddenMarkov, y::AbstractVector{<:Real}) = logpdf(d, y)

# One forward simulation of `T` steps: the hidden state path `z` and the
# observations `y`. (Distributions' `rand(rng, d, n)` means `n` draws of a
# multivariate law; a sequence length is not that, so this is its own name.)
function _brm_hidden_markov_draw(rng::AbstractRNG, d::HiddenMarkov, T::Integer)
    K = length(d.initial)
    weights(f) = (w = [f(k) for k in 1:K]; d.log_weights ? exp.(w .- logsumexp(w)) : w)
    z = Vector{Int}(undef, T)
    y = Vector{Any}(undef, T)
    for t in 1:T
        p = t == 1 ? weights(k -> d.initial[k]) :
            weights(j -> (G = d.transition; i = z[t - 1];
                ndims(G) == 2 ? G[_brm_hmm_index(G, 1, i), _brm_hmm_index(G, 2, j)] :
                    G[_brm_hmm_index(G, 1, i), _brm_hmm_index(G, 2, j),
                      _brm_hmm_index(G, 3, t - 1)]))
        z[t] = rand(rng, Categorical(p ./ sum(p)))
        y[t] = rand(rng, _brm_hmm_emission(d.emissions, z[t], t))
    end
    (; y=[y...], z)
end

# ---- The emission argument of an in-cell `HiddenMarkov(...)` -----------------
#
# Both backends read the authored emission argument the same way, as one law
# per state (and step):
#
# - a broadcast `F.(a1, …, an)`: state k's law is `F` at each argument's entry
#   k, by Julia's broadcasting rules (a number is shared, a vector is read at
#   `[k]`, a matrix at `[k, t]`, a singleton axis broadcasts). `eachrow(M)`
#   gives state k the row `M[k, :]`, `eachcol(M)` the column `M[:, k]`, and
#   `Ref(x)` shares `x` whole;
# - a vector of laws `[F1(…), …, FK(…)]`: state k's law is entry k, written out
#   (the families may differ), its arguments read whole;
# - a comprehension `[F(args…) for k in 1:K]` whose arguments read the state as
#   `x[k]`, `M[k, :]` or `M[:, k]`, or not at all: the broadcast it spells.
#
# `families` holds each law's Distributions constructor (one for a broadcast),
# `arguments` the authored argument expressions in order, and `modes` how each
# is read per state: `:each`, `:row`, `:col` or `:whole`.
struct _BRMHMMEmission
    broadcast::Bool
    families::Vector{Any}
    arguments::Vector{Vector{Any}}
    modes::Vector{Vector{Symbol}}
end

_brm_hmm_shown(x) = sprint(Base.show_unquoted, x)

# The keyword-free call `F(args...)` of one law, `F` resolved by `binding`.
function _brm_hmm_law_call(law, binding, context)
    Meta.isexpr(law, :call) && !isempty(law.args) || error(
        "$context: a state's emission law must be a distribution call such as " *
        "`Normal(mu, sigma)`; got `$(_brm_hmm_shown(law))`")
    any(arg -> Meta.isexpr(arg, (:parameters, :kw)), law.args[2:end]) && error(
        "$context: the emission law `$(_brm_hmm_shown(law))` takes positional " *
        "arguments only")
    family = binding(first(law.args))
    family isa Type && family <: Distributions.UnivariateDistribution || error(
        "$context: a state's emission law must be a univariate Distributions family; " *
        "`$(_brm_hmm_shown(first(law.args)))` is not one")
    family, Any[law.args[2:end]...]
end

# One broadcast argument and how it is read per state.
function _brm_hmm_broadcast_argument(arg)
    if Meta.isexpr(arg, :call, 2) && arg.args[1] in (:eachrow, :eachcol)
        return (arg.args[2], arg.args[1] === :eachrow ? :row : :col)
    elseif Meta.isexpr(arg, :call, 2) && arg.args[1] === :Ref
        return (arg.args[2], :whole)
    end
    (arg, :each)
end

# One comprehension argument, read at the comprehension's state `k`.
function _brm_hmm_comprehension_argument(arg, k, context)
    reads(x) = x === k || (x isa Expr && any(reads, x.args))
    reads(arg) || return (arg, :whole)
    if Meta.isexpr(arg, :ref, 2) && arg.args[2] === k && !reads(arg.args[1])
        return (arg.args[1], :each)
    elseif Meta.isexpr(arg, :ref, 3) && !reads(arg.args[1])
        row, col = arg.args[2], arg.args[3]
        row === k && col === :(:) && return (arg.args[1], :row)
        row === :(:) && col === k && return (arg.args[1], :col)
    end
    error("$context: a comprehension's emission argument reads its state as `x[$k]`, " *
        "`M[$k, :]` or `M[:, $k]`; compute `$(_brm_hmm_shown(arg))` per state in the " *
        "cell first, or write the broadcast `F.(...)`")
end

function _brm_hmm_emission(x, binding; context="@brm: HiddenMarkov")
    if Meta.isexpr(x, :., 2) && Meta.isexpr(x.args[2], :tuple)
        family, arguments = _brm_hmm_law_call(Expr(:call, x.args[1], x.args[2].args...),
            binding, context)
        read = map(_brm_hmm_broadcast_argument, arguments)
        # A literal vector argument, `Categorical.([p1, p2])`, lists its states'
        # entries: one law per entry, the other arguments read as broadcast.
        listed = findall(arg -> Meta.isexpr(arg, :vect), arguments)
        isempty(listed) && return _BRMHMMEmission(true, Any[family],
            [Any[first.(read)...]], [Symbol[last.(read)...]])
        K = length(arguments[first(listed)].args)
        all(i -> length(arguments[i].args) == K, listed) || error(
            "$context: the listed emission arguments of `$(_brm_hmm_shown(x))` have " *
            "different lengths")
        laws = [Any[i in listed ? arguments[i].args[state] : first(read[i])
                    for i in eachindex(arguments)] for state in 1:K]
        modes = [Symbol[i in listed ? :whole : last(read[i]) for i in eachindex(arguments)]
                 for _ in 1:K]
        return _BRMHMMEmission(false, Any[family for _ in 1:K], laws, modes)
    elseif Meta.isexpr(x, :vect)
        isempty(x.args) && error("$context: the emission vector holds no law")
        laws = [_brm_hmm_law_call(law, binding, context) for law in x.args]
        return _BRMHMMEmission(false, Any[first.(laws)...], [last(law) for law in laws],
            [fill(:whole, length(last(law))) for law in laws])
    elseif Meta.isexpr(x, :comprehension, 1) && Meta.isexpr(only(x.args), :generator, 2)
        generator = only(x.args)
        body, iteration = generator.args
        Meta.isexpr(iteration, :(=), 2) && iteration.args[1] isa Symbol || error(
            "$context: write the emission comprehension `[F(...) for k in 1:K]`")
        k = iteration.args[1]
        family, arguments = _brm_hmm_law_call(body, binding, context)
        read = [_brm_hmm_comprehension_argument(arg, k, context) for arg in arguments]
        return _BRMHMMEmission(true, Any[family], [Any[first.(read)...]],
            [Symbol[last.(read)...]])
    end
    error("$context: the emission argument is one law per state: a broadcast " *
        "`Normal.(mu, sigma)`, a vector `[Normal(m, s), Poisson(l)]` or a " *
        "comprehension `[Normal(mu[k], s) for k in 1:K]`; got `$(_brm_hmm_shown(x))`")
end

_brm_hmm_discrete(emission::_BRMHMMEmission) = all(emission.families) do F
    F <: Distributions.DiscreteUnivariateDistribution
end

# The parts of an in-cell `HiddenMarkov(initial, transition, emissions; log_weights)`.
function _brm_hmm_call(law, binding; context="@brm: HiddenMarkov")
    positional, keywords = Any[], Pair{Symbol,Any}[]
    for arg in law.args[2:end]
        for a in (Meta.isexpr(arg, :parameters) ? arg.args : Any[arg])
            if Meta.isexpr(a, :kw, 2)
                push!(keywords, a.args[1] => a.args[2])
            elseif a isa Symbol && Meta.isexpr(arg, :parameters)
                push!(keywords, a => a)
            else
                push!(positional, a)
            end
        end
    end
    length(positional) == 3 || error(
        "$context: write `HiddenMarkov(initial, transition, emissions)`; got " *
        "$(length(positional)) positional arguments")
    log_weights = false
    for (key, value) in keywords
        key === :log_weights || error("$context: unknown keyword `$key`; the only " *
            "keyword is `log_weights = true`")
        value isa Bool || error("$context: `log_weights` must be written `true` or " *
            "`false`; got `$(_brm_hmm_shown(value))`")
        log_weights = value
    end
    (; initial=positional[1], transition=positional[2],
       emission=_brm_hmm_emission(positional[3], binding; context), log_weights)
end

# ---- The RK law of an in-cell `HiddenMarkov(...)` -----------------------------
#
# RKBRMI observes `y ~ HiddenMarkov(initial, transition, emissions)` as
# `y ~ _BRMHiddenMarkovLaw{…}(initial, transition, arguments...)`: the emission
# arguments in order, and their structure (each law's family, how each
# argument is read per state, broadcast or listed laws, log weights) as type
# parameters, so the printed program replays with nothing registered. The
# ReactiveKernels extension maps the law to a native density graph
# (`ext/rk_hidden_markov.jl`): the steps × states log-emission matrix from each
# state's family, then `hmm_logdensity` (or its log-weights twin) from
# ReactiveKernelsDistributionKernels.
"""
    _BRMHiddenMarkovLaw{Families,Modes,Broadcast,LogWeights,Support}

Internal RK law of an in-cell [`HiddenMarkov`](@ref) observation. Never built
by hand.
"""
struct _BRMHiddenMarkovLaw{F<:Tuple,M,B,L,V<:Distributions.ValueSupport} <:
       Distributions.Distribution{Distributions.Multivariate,V} end

function _brm_hmm_law(emission::_BRMHMMEmission, log_weights::Bool)
    support = _brm_hmm_discrete(emission) ? Distributions.Discrete : Distributions.Continuous
    _BRMHiddenMarkovLaw{Tuple{emission.families...},
        Tuple(Tuple(modes) for modes in emission.modes), emission.broadcast,
        log_weights, support}
end

# State k's (step t's) entry of a broadcast emission argument, by Julia's
# broadcasting rules: a number is shared, a vector is read at its state, a
# states × steps matrix at its state and step; an axis of length one is shared.
_brm_hmm_at(x::Real, k, t) = x
_brm_hmm_at(x::AbstractVector, k, t) = x[size(x, 1) == 1 ? 1 : k]
_brm_hmm_at(x::AbstractMatrix, k, t) = x[size(x, 1) == 1 ? 1 : k, size(x, 2) == 1 ? 1 : t]
