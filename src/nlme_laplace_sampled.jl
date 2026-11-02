# Zero admitted terms is an ordinary full-model target. Preserve the Laplace
# workflow's coordinate/draw interface without preparing any conditional AD.
mutable struct _BRMSampledLaplaceProblem{B,Q}
    backend::B
    target::Q
    population::Vector{Int}
    names::Vector{Symbol}
    integrated::Vector{Int}
    integrate::Vector{Symbol}
    blocks::Vector{_BRMSlapBlock}
    components::Vector{Vector{Int}}
    u::Vector{Float64}
    plan::BRMLaplacePlan
    tol::Float64
    maxiter::Int
    nonfinite::Int
    unconverged::Int
end

function _brm_sampled_laplace_problem(backend, plan; ad_backend, tol, maxiter)
    names = _rk_layout_coordinate_names(backend)
    u = zeros(length(names))
    target = rk_logdensity_problem(backend; ad_backend, u0=copy(u))
    _BRMSampledLaplaceProblem(backend, target, collect(eachindex(names)), names,
        Int[], Symbol[], _BRMSlapBlock[], Vector{Int}[], u, plan, Float64(tol), Int(maxiter), 0, 0)
end
LogDensityProblems.capabilities(::Type{<:_BRMSampledLaplaceProblem}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::_BRMSampledLaplaceProblem) = length(p.population)
function LogDensityProblems.logdensity(p::_BRMSampledLaplaceProblem, u::AbstractVector)
    value = LogDensityProblems.logdensity(p.target, u)
    isfinite(value) || (p.nonfinite += 1)
    p.u .= u
    value
end
function LogDensityProblems.logdensity_and_gradient(p::_BRMSampledLaplaceProblem,
        u::AbstractVector)
    value, gradient = LogDensityProblems.logdensity_and_gradient(p.target, u)
    isfinite(value) || (p.nonfinite += 1)
    p.u .= u
    value, gradient
end
function _brm_spl_mode!(p::_BRMSampledLaplaceProblem, u::AbstractVector)
    value = LogDensityProblems.logdensity(p, u)
    (; finite=isfinite(value), value, u=copy(p.u), iterations=0)
end
function brm_laplace_draws(p::_BRMSampledLaplaceProblem, P::AbstractMatrix;
        rng=Random.default_rng(), conditional::Integer=0)
    size(P, 1) == length(p.population) || throw(DimensionMismatch(
        "draw rows must match the full model's packed dimension"))
    conditional >= 0 || throw(ArgumentError("conditional must be nonnegative"))
    for k in axes(P, 2)
        isfinite(LogDensityProblems.logdensity(p, view(P, :, k))) ||
            _brm_nlme_error("brm_laplace_draws: draw $k has a nonfinite density")
    end
    newton_steps = zeros(Int, size(P, 2))
    diagnostics = conditional > 0 ?
        (; newton_steps, importance_ess=zeros(0, size(P, 2))) : (; newton_steps)
    (; draws=Matrix{Float64}(P), diagnostics)
end
Base.show(io::IO, p::_BRMSampledLaplaceProblem) = print(io,
    "Full sampled Laplace plan (", length(p.population), " sampled coordinates, 0 integrated)")
