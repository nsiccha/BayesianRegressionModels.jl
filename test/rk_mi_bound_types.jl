# test/rk_mi_bound_types.jl — RKBRMI binds an `mi()` column as two concretely
# typed inputs: its values (`<x>_raw`, `one(T)` where an entry is missing) and
# its presence mask (`<x>_present`, a `Vector{Bool}`), never as a
# `Vector{Union{Missing,T}}`. RK's type-stability check types every bound value,
# and a column of `Missing` unions is not concretely typed, so the data-only
# binding kernel that reads it could not be audited (snag
# mi-raw-columns-r-2b85d8f9, RK todo 03iuru0). The observed rows, the missing
# rows and a group-totals layout stay graph values over the pair.
#
# Oracle: every bound input of the fit, of a frozen `reprocess` and of each
# design binder design has a concrete element type; the bound values equal the
# data at the observed rows and the mask equals the data's presence. The
# densities themselves are pinned by test/rk_parity.jl (`mi()` responses),
# test/rk_missing_value_predictors.jl and test/mi_group_totals.jl.
#
# Run: julia --startup-file=no --project=test test/rk_mi_bound_types.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

untyped_columns(bound) = [k for (k, v) in pairs(bound)
    if v isa AbstractArray && !isconcretetype(eltype(v))]

# The bound pair of `mi()` column `x`, against the data column it binds.
function check_pair(bound, x, column)
    values, present = bound[Symbol(x, :_raw)], bound[Symbol(x, :_present)]
    @test values isa Vector{nonmissingtype(eltype(column))}
    @test present isa Vector{Bool}
    @test present == .!ismissing.(column)
    @test values[present] == collect(skipmissing(column))
end

function check_bound(rk, pairs, designs)
    bound = rk_bound_data(rk)
    @test isempty(untyped_columns(bound))
    for (x, column) in pairs
        check_pair(bound, x, column)
    end
    binder = rk_design_binder(rk)
    for design in designs
        @test isempty(untyped_columns(binder(design)))
        replayed = rk_bound_data(reprocess(rk, design))
        @test isempty(untyped_columns(replayed))
        for (x, _) in pairs
            check_pair(replayed, x, design[x])
        end
    end
end

# A frozen replay keeps the fit's missing mask, since its missing entries are
# fitted coordinates (`reprocess` refuses another mask); it may refresh the
# observed values.
scaled(v, s) = map(e -> ismissing(e) ? missing : s * e, v)

const N = 12
const X = collect(range(0.3, 4.1; length=N))
const XM = Union{Missing,Float64}[i % 5 == 0 ? missing : exp(0.5 + 0.3 * v)
    for (i, v) in enumerate(X)]
const Y = 0.3 .* X .+ 0.1 .* sin.(1:N)

@stestset "a completed covariate binds its values and presence" begin
    data = (; x=X, xm=XM, y=Y)
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mi(xm) ~ LogNormal(0.5, 0.6)
        mu ~ 1 + standardize(xm)
        y ~ Normal(mu, sigma)
    end)
    check_bound(rk, (:xm => XM,), [merge(data, (; xm=scaled(XM, 1.1))),
        merge(data, (; x=X .+ 0.2, xm=scaled(XM, 0.9), y=Y .- 0.1))])
end

@stestset "an unread mi() column binds its values and presence" begin
    data = (; xm=XM, y=Y)
    rk = RKBRMI(@brm data begin
        mi(xm) ~ LogNormal(0.5, 0.6)
        mu ~ Normal(0, 1)
        y ~ Normal(mu, 0.8)
    end)
    check_bound(rk, (:xm => XM,), [merge(data, (; xm=scaled(XM, 1.2)))])
end

@stestset "an mi() response binds its values and presence" begin
    y = Union{Missing,Float64}[i in (2, 7) ? missing : v for (i, v) in enumerate(Y)]
    data = (; x=X, y)
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        mi(y) ~ Normal(mu, sigma)
    end)
    check_bound(rk, (:y => y,), [merge(data, (; y=scaled(y, 1.1))),
        merge(data, (; x=X .+ 0.2, y=scaled(y, 0.9)))])
end

@stestset "a completion with group totals binds its values and presence" begin
    g = repeat(1:3, inner=4)
    cells = [1, 2]
    truth = exp.(1.5 .+ 0.4 .* X)
    gap = falses(N); gap[[1, 2, 6, 11]] .= true
    v = Union{Missing,Float64}[m ? missing : t for (m, t) in zip(gap, truth)]
    tot = [sum(truth[g .== k]) for k in cells]
    data = (; x=X, g, cells, tot, v, y=Y)
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        rows = groups(g => cells)
        mi(v)::shares(tot, rows) ~ LogNormal(mu, sigma)
        lv = log(v)
        m ~ 1 + lv
        y ~ Normal(m, 0.5)
    end)
    check_bound(rk, (:v => v,), [data])
end

@stestset "a complete mi() column binds an all-true presence" begin
    complete = exp.(0.5 .+ 0.3 .* X)
    data = (; x=X, xm=complete, y=Y)
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        mi(xm) ~ LogNormal(0.5, 0.6)
        mu ~ 1 + standardize(xm)
        y ~ Normal(mu, sigma)
    end)
    check_bound(rk, (:xm => complete,), [merge(data, (; xm=1.1 .* complete))])
end
