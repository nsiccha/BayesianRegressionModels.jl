"""
    ode_rk45(rhs, y0, t0, ts, args...; reltol=1e-6, abstol=1e-6, maxsteps=10^6)
    ode_bdf(rhs, y0, t0, ts, args...; reltol=1e-6, abstol=1e-6, maxsteps=10^6)

Integrate `rhs(t, y, args...)` from the initial state vector `y0` at `t0`.
Row `i` of the result contains the state at `ts[i]`. Use explicit RK45 for
nonstiff problems or implicit BDF for stiff problems.

In a BRM plate, a named RHS with Stan and Julia implementations works on
both backends. Stan requires data-only tolerances and step limits. For Julia
evaluation load ReactiveKernels, SciMLBase, SciMLSensitivity and the solver
package (OrdinaryDiffEqLowOrderRK for RK45 or OrdinaryDiffEqBDF for BDF).
"""
function ode_rk45 end

@doc (@doc ode_rk45) function ode_bdf end

const _BRMCellODE = Union{typeof(ode_rk45),typeof(ode_bdf)}
const _BRM_ODE_DEFAULTS = (reltol=1e-6, abstol=1e-6, maxsteps=10^6)

for solve in (:ode_rk45, :ode_bdf)
    @eval function $solve(f::F, y0, t0, ts, args::Vararg{Any,N};
            reltol=_BRM_ODE_DEFAULTS.reltol, abstol=_BRM_ODE_DEFAULTS.abstol,
            maxsteps=_BRM_ODE_DEFAULTS.maxsteps) where {F,N}
        return _brm_ode_bridge($solve)(f, y0, t0, ts, reltol, abstol, maxsteps, args...)
    end
end

# Preserve the original Stan result type when a cell explicitly converts its
# times to a Stan array. An explicit Stan builtin also retains its own type.
_brm_ode_stan_times(args) = length(args) >= 4 && isxcall(args[4], :to_array_1d) &&
    !any(_sb_cell_keyword_argument, args)

function _brm_warn_legacy_ode(ex::Expr, scope, location=LineNumberNode(0, :none))
    if Meta.isexpr(ex, :call)
        head = first(ex.args)
        name = head isa GlobalRef ? head.name :
            Meta.isexpr(head, :., 2) && head.args[2] isa QuoteNode ? head.args[2].value : head
        if name in (:ode_rk45, :ode_rk45_tol, :ode_bdf, :ode_bdf_tol)
            callable = _sb_cell_binding(scope, head)
            builtin = getfield(StanBlocks.stan.builtin, name)
            legacy = callable === builtin || (callable === nothing && head === name) ||
                (callable isa _BRMCellODE && _brm_ode_stan_times(ex.args[2:end]))
            if legacy
                solver = name in (:ode_rk45, :ode_rk45_tol) ? :ode_rk45 : :ode_bdf
                message = "Stan's `$name` cell spelling is deprecated. Use " *
                    "`$solver(rhs, y0, t0, ts, args...; reltol, abstol, maxsteps)` " *
                    "with Julia vectors. Read the solution with `sol[:, k]`."
                @warn message _file=string(location.file) _line=location.line _id=(scope, name, location) maxlog=1
            end
        end
    end
    for arg in ex.args
        if arg isa LineNumberNode
            location = arg
        elseif arg isa Expr && arg.head !== :quote
            _brm_warn_legacy_ode(arg, scope, location)
        end
    end
end

function _brm_ode_bridge(solve)
    package = solve === ode_rk45 ? "OrdinaryDiffEqLowOrderRK" : "OrdinaryDiffEqBDF"
    throw(ArgumentError("load ReactiveKernels, $package, SciMLBase and SciMLSensitivity to call $solve in Julia"))
end

# A model call has the same keyword contract as a Julia call. Its operands
# are syntax: leave their evaluation to the backend, including data tolerances.
function _brm_ode_operands(solve, args)
    positional = filter(!_sb_cell_keyword_argument, args)
    length(positional) >= 4 || throw(ArgumentError("$solve needs rhs, y0, t0 and ts"))
    keywords = filter(_sb_cell_keyword_argument, args)
    all(_sb_statistic_keyword, keywords) ||
        throw(ArgumentError("$solve in a cell needs named solver controls"))
    controls = foldl(_sb_cell_keywords!, keywords; init=Pair{Symbol,Any}[])
    names = first.(controls)
    allunique(names) || throw(ArgumentError("repeated keyword in $solve"))
    for name in names
        haskey(_BRM_ODE_DEFAULTS, name) || throw(ArgumentError("unknown $solve keyword `$name`"))
    end
    options = merge(_BRM_ODE_DEFAULTS, NamedTuple(controls))
    return Any[positional[1:4]..., options..., positional[5:end]...]
end
