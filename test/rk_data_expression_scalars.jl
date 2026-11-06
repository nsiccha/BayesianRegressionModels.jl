# test/rk_data_expression_scalars.jl — a numeric scalar data field read by a data
# expression on RKBRMI: an offset, a coefficient's column, a fitted transform, an
# interaction, a random slope, a data-only assignment and a model-value offset.
#
# Such a value is a runtime input of the RK program (user principle `0ikeidi`:
# numeric values are never baked into model source). Two data sets that differ only
# in the value emit the same program, and the density follows the bound value.
# RK refused `offset(slope * x)` ("column `slope` must be a plain numeric vector for
# arithmetic") although SBBRMI accepted it, and folded the number of the assignment
# spelling `sx = slope * x` into the program (snag rkbrmi-refuses-a-ef38ae4a).
#
# Checks: an independent likelihood oracle where a model's rows are simple, the
# compiled Stan program of the same BRMI through the coordinate transport (value and
# native Reverse gradient), complete printed-source replay, program identity across
# values, frozen and data-free replay, a design binder and hold-outs.
#
# Run: julia --startup-file=no --project=test test/rk_data_expression_scalars.jl [testset substring ...]

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

scalar_scaled(v; s) = v * s

@brm function offset_component(x, s)
    value ~ 1 + offset(s * x)
    effect(value, Intercept) ~ Normal(0, 2)
    return value
end

function scalar_data()
    rng = Xoshiro(3)
    n = 8
    x = collect(range(-1.0, 1.0; length=n))
    (; x, x2=exp.(0.3 .* randn(rng, n)), y=0.4 .- 0.4 .* x .+ 0.5 .* randn(rng, n), row=collect(1:n),
        y2=0.2 .* randn(rng, n), k=rand(rng, 0:4, n), g=repeat([1, 2]; inner=4), gv=[0.5, 0.25],
        lab=repeat(["a", "b"]; outer=4), slope=-0.4, pos=0.7, count=2, flag=true)
end
changed_scalars(d) = merge(d, (; slope=1.7 * d.slope, pos=1.3 * d.pos, count=3))

# A one-coordinate model: `mu ~ 1 + <one offset>` over Normal(mu, 0.8) rows.
offset_builder(lines...) = brm(Expr(:block, lines...,
    :(effect(mu, Intercept) ~ Normal(0, 2)), :(y ~ Normal(mu, 0.8))); mod=@__MODULE__)
const OFFSETS = (
    (label="scalar times column", offset=d -> d.slope .* d.x,
        builder=offset_builder(:(mu ~ 1 + offset(slope * x)))),
    (label="column times scalar", offset=d -> d.x .* d.slope,
        builder=offset_builder(:(mu ~ 1 + offset(x * slope)))),
    (label="sum with a literal", offset=d -> d.slope .* d.x .+ 1.0,
        builder=offset_builder(:(mu ~ 1 + offset(slope * x + 1.0)))),
    (label="unary minus", offset=d -> -d.slope .* d.x,
        builder=offset_builder(:(mu ~ 1 + offset(-slope * x)))),
    (label="math of a scalar", offset=d -> log(d.pos) .* d.x,
        builder=offset_builder(:(mu ~ 1 + offset(log(pos) * x)))),
    (label="math of a column", offset=d -> d.slope .* log.(d.x2),
        builder=offset_builder(:(mu ~ 1 + offset(slope * log(x2))))),
    (label="two scalars", offset=d -> d.slope .* d.pos .* d.x,
        builder=offset_builder(:(mu ~ 1 + offset(slope * pos * x)))),
    (label="an integer scalar", offset=d -> d.count .* d.x,
        builder=offset_builder(:(mu ~ 1 + offset(count * x)))),
    (label="a gather", offset=d -> d.gv[d.g] .* d.slope,
        builder=offset_builder(:(mu ~ 1 + offset(gv[g] * slope)))),
    (label="three columns", offset=d -> d.x .* d.x2 .* d.x, moves=false,
        builder=offset_builder(:(mu ~ 1 + offset(x * x2 * x)))),
    (label="a sum of three", offset=d -> d.slope .+ d.x .+ d.x2,
        builder=offset_builder(:(mu ~ 1 + offset(slope + x + x2)))),
    # A scalar alone has no rows: one value on every row of the predictor.
    (label="a scalar alone", offset=d -> fill(d.slope, length(d.y)),
        builder=offset_builder(:(mu ~ 1 + offset(slope)))),
    (label="a form of scalars alone", offset=d -> fill(2 * d.slope, length(d.y)),
        builder=offset_builder(:(mu ~ 1 + offset(2 * slope)))),
    (label="math of a scalar alone", offset=d -> fill(log(d.pos), length(d.y)),
        builder=offset_builder(:(mu ~ 1 + offset(log(pos))))),
    # StanBlocks cannot trace an arbitrary callable inside an offset, whatever its
    # arguments: an independent limit, so the oracle alone checks this one.
    (label="a data call keyword", offset=d -> d.x .* d.slope, stan=false,
        builder=offset_builder(:(mu ~ 1 + offset(scalar_scaled(x; s=slope))))),
    (label="a data-only assignment", offset=d -> d.slope .* d.x,
        builder=offset_builder(:(sx = slope * x), :(mu ~ 1 + offset(sx)))),
    (label="an alias", offset=d -> d.slope .* d.x,
        builder=offset_builder(:(s2 = slope), :(mu ~ 1 + offset(s2 * x)))),
)
const OFFSET_ASSIGNMENT = only(c for c in OFFSETS if c.label == "a data-only assignment")
const OFFSET_ALONE = only(c for c in OFFSETS if c.label == "a scalar alone")

# Models whose every coordinate the compiled Stan program is the oracle of.
const COMPOSED = (
    (label="a coefficient's column", builder=@brm begin
        mu ~ 1 + slope * x
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, 0.8)
    end),
    (label="center of a form", builder=@brm begin
        mu ~ 1 + center(slope * x)
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, 0.8)
    end),
    (label="standardize of a form", builder=@brm begin
        mu ~ 1 + standardize(slope * x)
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, 0.8)
    end),
    (label="standardize of an assignment", builder=@brm begin
        sx = slope * x
        mu ~ 1 + standardize(sx)
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, 0.8)
    end),
    (label="a data call's keyword", builder=@brm begin
        mu ~ 1 + scalar_scaled(x; s=slope)
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, 0.8)
    end),
    (label="a predictor of a scalar alone", builder=@brm begin
        sigma ~ Exponential(1)
        mu ~ 0 + offset(slope)
        y ~ Normal(mu, sigma)
    end),
    (label="a random slope", builder=@brm begin
        mu ~ 1 + (0 + slope * x | g)
        effect(mu, :) ~ Normal(0, 2)
        sd(:, g) ~ Exponential(1)
        y ~ Normal(mu, 0.8)
    end),
    (label="a random intercept beside it", builder=@brm begin
        mu ~ 1 + offset(slope * x) + (1 | g)
        effect(mu, :) ~ Normal(0, 2)
        sd(:, g) ~ Exponential(1)
        y ~ Normal(mu, 0.8)
    end),
    (label="an offset-only predictor", builder=@brm begin
        sigma ~ Exponential(1)
        mu ~ 0 + offset(slope * x)
        y ~ Normal(mu, sigma)
    end),
    (label="a linked predictor of two offsets", builder=@brm begin
        log(rate) ~ 1 + offset(log(x2)) + offset(slope * x)
        effect(rate, :) ~ Normal(0, 2)
        k ~ Poisson(rate)
    end),
    # The reported composition: a linked predictor of offsets alone, with a sampled
    # value beside the scalar, read through a gather by a Normal law.
    (label="a linked sum of offsets read by a gather", builder=@brm begin
        sigma ~ Exponential(1)
        level ~ Normal(0, 1)
        log(q) ~ 0 + offset(level) + offset(log(x2)) + offset(slope * x)
        readout = q[row]
        y ~ Normal(readout, sigma)
    end),
    (label="a data law scale beside it", builder=@brm begin
        mu ~ 1 + offset(slope * x)
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, pos)
    end),
    (label="a plate reader", builder=@brm begin
        mu ~ 1 + offset(slope * x)
        effect(mu, :) ~ Normal(0, 2)
        @plate for i in eachindex(y)
            y[i] ~ Normal(mu[i], 0.8)
        end
    end),
    (label="a component's argument", builder=@brm begin
        field ~ offset_component(x, slope)
        y ~ Normal(field, 0.8)
    end),
    (label="a sampled value times the scalar", builder=@brm begin
        theta ~ Normal(0, 1)
        mu ~ 1 + offset(theta * slope * x)
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, 0.8)
    end),
    (label="a row-less sampled offset", builder=@brm begin
        theta ~ Normal(0, 1)
        mu ~ 1 + x + offset(theta * slope)
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, 0.8)
    end),
)

const REVERSE = AutoEnzyme(; mode=Enzyme.Reverse)
const SCALAR_FOLDER = joinpath(tempdir(), "brm-rk-data-expression-scalars")
function scalar_compiled(brmi, label)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    mkpath(SCALAR_FOLDER)
    code = BRM.stan_code(sb)
    stan = BRM.stan_instantiate(sb; path=joinpath(SCALAR_FOLDER,
        replace(label, r"[^a-z0-9]+" => "_") * "_" * string(hash(code); base=16) * ".stan"))
    (; sb, stan, names=BridgeStan.param_unc_names(stan.model))
end
scalar_probes(n) = (zeros(n), n == 1 ? [-0.3] : collect(range(-0.3, 0.4; length=n)), fill(0.2, n))
# The backend's own binding: a data-free replay's holds its frozen fitted values.
scalar_likelihood(backend) = prepare_query(backend.model, rk_bound_plan(backend), :likelihood)
offset_oracle(d, offset, u) = sum(logpdf.(Normal.(only(u) .+ offset, 0.8), d.y))

# The RK density and native Reverse gradient against the compiled Stan program.
function check_against_stan(backend, brmi, label)
    problem = rk_logdensity_problem(backend; ad_backend=REVERSE)
    c = scalar_compiled(brmi, label)
    t = brm_coordinate_transport(backend, c.sb, String.(c.names))
    for u in scalar_probes(LogDensityProblems.dimension(problem))
        before = copy(u)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(c.stan.model,
            brm_rk_to_stan(t, u); propto=false, jacobian=true)
        @test value ≈ stan_value atol=1e-10 rtol=1e-11
        @test gradient ≈ brm_stan_to_rk(t, stan_gradient) atol=1e-9 rtol=1e-10
        @test u == before
    end
end

# One program serves both values, and no emitted statement spells a value.
function check_one_program(builder, data, other)
    a, b = RKBRMI(builder(data)), RKBRMI(builder(other))
    ea, eb = BRM._rk_emit_ast(a.plan), BRM._rk_emit_ast(b.plan)
    @test ea.main == eb.main
    @test ea.defs == eb.defs
    @test string(a.model.program) == string(b.model.program)
    @test coordinate_names(a.model.layout) == coordinate_names(b.model.layout)
    source = sprint(Base.show_unquoted, ea.main)
    for value in (data.slope, data.pos)
        @test !occursin(string(value), source)
    end
    a, b
end

same_input(x, y) = isequal(x, y) || (x isa AbstractArray && y isa AbstractArray &&
    size(x) == size(y) && all(same_input.(x, y)))
same_inputs(x, y) = Set(keys(x)) == Set(keys(y)) && all(n -> same_input(x[n], y[n]), keys(y))

@testset "RKBRMI scalar data fields in data expressions" begin

@stestset "an offset binds the scalar at run time and agrees with SBBRMI" begin
    data = scalar_data()
    for case in OFFSETS
        @testset "$(case.label)" begin
            original = deepcopy(data)
            brmi = case.builder(data)
            backend = check_rk_source_roundtrip(RKBRMI(brmi); ad_backend=REVERSE)
            @test backend.model.layout.total == 1
            likelihood = scalar_likelihood(backend)
            for u in scalar_probes(1)
                @test Base.invokelatest(likelihood, u) ≈
                    offset_oracle(data, case.offset(data), u) atol=1e-10 rtol=1e-11
            end
            get(case, :stan, true) && check_against_stan(backend, brmi, case.label)
            @test data == original
        end
    end
end

@stestset "another value emits the same program and moves the density" begin
    data = scalar_data()
    other = changed_scalars(data)
    for case in OFFSETS
        @testset "$(case.label)" begin
            a, b = check_one_program(case.builder, data, other)
            # SBBRMI serves both values with one program too: the value is Stan data.
            get(case, :stan, true) &&
                @test BRM.stan_code(SBBRMI(case.builder(data); mod=@__MODULE__)) ==
                    BRM.stan_code(SBBRMI(case.builder(other); mod=@__MODULE__))
            la, lb = scalar_likelihood(a), scalar_likelihood(b)
            for u in scalar_probes(1)
                @test Base.invokelatest(lb, u) ≈
                    offset_oracle(other, case.offset(other), u) atol=1e-10 rtol=1e-11
                get(case, :moves, true) &&
                    @test !isapprox(Base.invokelatest(la, u), Base.invokelatest(lb, u); rtol=1e-6)
            end
        end
    end
end

@stestset "a composed model agrees with SBBRMI at both values, in one program" begin
    data = scalar_data()
    other = changed_scalars(data)
    for case in COMPOSED
        @testset "$(case.label)" begin
            for d in (data, other)
                brmi = case.builder(d)
                backend = check_rk_source_roundtrip(RKBRMI(brmi); ad_backend=REVERSE)
                check_against_stan(backend, brmi, case.label)
            end
            check_one_program(case.builder, data, other)
            @test BRM.stan_code(SBBRMI(case.builder(data); mod=@__MODULE__)) ==
                BRM.stan_code(SBBRMI(case.builder(other); mod=@__MODULE__))
        end
    end
end

# The coordinate transport has no name for an `&` coefficient over a staged form,
# scalar or not, so a precomputed column is this one's oracle. Neither model states
# a prior: on RKBRMI `effect(mu, :)` does not reach such a coefficient, with or
# without a scalar in the form (tracked on BRM's queue).
@stestset "an interaction operand equals a precomputed column" begin
    data = scalar_data()
    other = changed_scalars(data)
    scalar = @brm begin
        mu ~ 1 + (slope * x) & x2
        y ~ Normal(mu, 0.8)
    end
    column = @brm begin
        mu ~ 1 + slope_x & x2
        y ~ Normal(mu, 0.8)
    end
    for d in (data, other)
        rs = check_rk_source_roundtrip(RKBRMI(scalar(d)); ad_backend=REVERSE)
        rc = RKBRMI(column(merge(d, (; slope_x=d.slope .* d.x))))
        ps = rk_logdensity_problem(rs; ad_backend=REVERSE)
        pc = rk_logdensity_problem(rc; ad_backend=REVERSE)
        @test LogDensityProblems.dimension(ps) == LogDensityProblems.dimension(pc)
        for u in scalar_probes(LogDensityProblems.dimension(ps))
            vs, gs = LogDensityProblems.logdensity_and_gradient(ps, u)
            vc, gc = LogDensityProblems.logdensity_and_gradient(pc, u)
            @test vs ≈ vc atol=1e-12 rtol=1e-12
            @test gs ≈ gc atol=1e-12 rtol=1e-12
        end
    end
    check_one_program(scalar, data, other)
end

# The reporter's composition: a precomputed column is the same model at every value
# of the scalar, as a positive control that the two programs read one number each.
@stestset "the scalar spelling equals a precomputed column" begin
    data = scalar_data()
    scalar = @brm begin
        sigma ~ Exponential(1)
        level ~ Normal(0, 1)
        log(q) ~ 0 + offset(level) + offset(log(x2)) + offset(slope * x)
        readout = q[row]
        y ~ Normal(readout, sigma)
    end
    column = @brm begin
        sigma ~ Exponential(1)
        level ~ Normal(0, 1)
        log(q) ~ 0 + offset(level) + offset(log(x2)) + offset(slope_x)
        readout = q[row]
        y ~ Normal(readout, sigma)
    end
    for slope in (-0.4, 0.9)
        d = merge(data, (; slope))
        rs = RKBRMI(scalar(d))
        rc = RKBRMI(column(merge(d, (; slope_x=slope .* d.x))))
        ps = rk_logdensity_problem(rs; ad_backend=REVERSE)
        pc = rk_logdensity_problem(rc; ad_backend=REVERSE)
        @test coordinate_names(rs.model.layout) == coordinate_names(rc.model.layout)
        for u in scalar_probes(2)
            vs, gs = LogDensityProblems.logdensity_and_gradient(ps, u)
            vc, gc = LogDensityProblems.logdensity_and_gradient(pc, u)
            @test vs ≈ vc atol=1e-12 rtol=1e-12
            @test gs ≈ gc atol=1e-12 rtol=1e-12
        end
    end
end

# A scalar only a data expression reads is a design input: omitted, it keeps the fitted
# value, supplied, the design's (brm-use `reactivekernels.md`, frozen replay).
@stestset "a frozen replay reads the scalar as a design input" begin
    data = scalar_data()
    rows = (; x=data.x[1:3], x2=data.x2[1:3], y=data.y[1:3], g=data.g[1:3])
    for case in OFFSETS
        case.label == "a gather" && continue
        @testset "$(case.label)" begin
            fit = RKBRMI(case.builder(data))
            omitted = reprocess(fit, rows)
            offset(d) = case.offset(merge((; gv=data.gv, count=data.count, pos=data.pos,
                slope=data.slope), d))
            for u in scalar_probes(1)
                @test Base.invokelatest(scalar_likelihood(omitted), u) ≈
                    offset_oracle(rows, offset(rows), u) atol=1e-10 rtol=1e-11
            end
            changed = merge(rows, (; slope=0.9, pos=1.1, count=3))
            supplied = reprocess(fit, changed)
            for u in scalar_probes(1)
                @test Base.invokelatest(scalar_likelihood(supplied), u) ≈
                    offset_oracle(rows, offset(changed), u) atol=1e-10 rtol=1e-11
            end
        end
    end
end

@stestset "a fitted transform keeps the fit's scalar on a frozen replay" begin
    data = scalar_data()
    rows = (; x=data.x[1:3], y=data.y[1:3])
    forms = (
        (label="a staged form", builder=@brm begin
            mu ~ 0 + standardize(slope * x)
            effect(mu, :) ~ Normal(0, 2)
            y ~ Normal(mu, 0.8)
        end),
        (label="an assignment", builder=@brm begin
            sx = slope * x
            mu ~ 0 + standardize(sx)
            effect(mu, :) ~ Normal(0, 2)
            y ~ Normal(mu, 0.8)
        end),
    )
    training = data.slope .* data.x
    location, scale = sum(training) / length(training), sqrt(
        sum(abs2, training .- sum(training) / length(training)) / (length(training) - 1))
    for form in forms
        @testset "$(form.label)" begin
            fit = RKBRMI(form.builder(data))
            replay = reprocess(fit, merge(rows, (; slope=0.9)))
            column = (0.9 .* rows.x .- location) ./ scale
            for u in scalar_probes(1)
                @test Base.invokelatest(scalar_likelihood(replay), u) ≈
                    sum(logpdf.(Normal.(only(u) .* column, 0.8), rows.y)) atol=1e-10 rtol=1e-11
            end
        end
    end
end

# `Constant`: a `constant(...)` keeps its fitted value, whatever the design supplies.
@stestset "a constant in an offset is read by name and stays the fit's" begin
    data = scalar_data()
    builder = offset_builder(:(c = constant(2 * pos)), :(mu ~ 1 + offset(c * x)))
    fit = check_rk_source_roundtrip(RKBRMI(builder(data)); ad_backend=REVERSE)
    other = merge(data, (; pos=1.1))
    @test BRM._rk_emit_ast(fit.plan).main == BRM._rk_emit_ast(RKBRMI(builder(other)).plan).main
    rows = (; x=data.x[1:3], y=data.y[1:3], pos=1.1)
    frozen = reprocess(fit, rows)
    for u in scalar_probes(1)
        @test Base.invokelatest(scalar_likelihood(frozen), u) ≈
            offset_oracle(rows, 2 .* data.pos .* rows.x, u) atol=1e-10 rtol=1e-11
    end
end

# A number a prior reads cannot change on a replay; one name read by a prior and by an
# offset stays one input of the design and one number of the fit.
@stestset "a scalar a prior also reads is refused a new value on a replay" begin
    data = scalar_data()
    builder = @brm begin
        sigma ~ Exponential(pos)
        mu ~ 1 + offset(pos * x)
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, sigma)
    end
    brmi = builder(data)
    fit = check_rk_source_roundtrip(RKBRMI(brmi); ad_backend=REVERSE)
    check_against_stan(fit, brmi, "prior and offset")
    rows = (; x=data.x[1:3], y=data.y[1:3])
    @test reprocess(fit, rows) isa RKBRMI
    # refused: a design cannot change a fitted prior (brm-use reactivekernels.md).
    @test_throws ErrorException reprocess(fit, merge(rows, (; pos=1.1)))
end

@stestset "a design binder binds what a reprocess binds" begin
    data = scalar_data()
    rows = (; x=data.x[1:3], x2=data.x2[1:3], y=data.y[1:3])
    for (label, builder) in (("offset", OFFSETS[1].builder),
            ("assignment", OFFSET_ASSIGNMENT.builder),
            ("a scalar alone", OFFSET_ALONE.builder))
        @testset "$label" begin
            fit = RKBRMI(builder(data))
            binder = rk_design_binder(fit)
            for design in (rows, merge(rows, (; slope=0.9)),
                    (; x=data.x[2:5], x2=data.x2[2:5], y=data.y[2:5], slope=1.3))
                @test same_inputs(binder(design), rk_bound_data(reprocess(fit, design)))
            end
        end
    end
end

@stestset "a data-free replay reads the scalar like a frozen one" begin
    data = scalar_data()
    rows = (; x=data.x[1:3], y=data.y[1:3])
    for (label, builder) in (("offset", OFFSETS[1].builder),
            ("assignment", OFFSET_ASSIGNMENT.builder),
            ("a scalar alone", OFFSET_ALONE.builder))
        @testset "$label" begin
            fit = RKBRMI(builder(data))
            state = brm_fitted_state(fit)
            for design in (rows, merge(rows, (; slope=0.9)))
                free = RKBRMI(builder(design), state)
                frozen = reprocess(fit, design)
                for u in scalar_probes(1)
                    @test Base.invokelatest(scalar_likelihood(free), u) ≈
                        Base.invokelatest(scalar_likelihood(frozen), u) atol=1e-12 rtol=1e-12
                end
            end
        end
    end
    transform = @brm begin
        mu ~ 0 + standardize(slope * x)
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, 0.8)
    end
    fit = RKBRMI(transform(data))
    free = RKBRMI(transform(merge(rows, (; slope=0.9))), brm_fitted_state(fit))
    frozen = reprocess(fit, merge(rows, (; slope=0.9)))
    for u in scalar_probes(1)
        @test Base.invokelatest(scalar_likelihood(free), u) ≈
            Base.invokelatest(scalar_likelihood(frozen), u) atol=1e-12 rtol=1e-12
    end
end

@stestset "a held-out response keeps the scalar bound" begin
    data = scalar_data()
    builder = @brm begin
        mu ~ 1 + offset(slope * x)
        effect(mu, :) ~ Normal(0, 2)
        y ~ Normal(mu, 0.8)
        y2 ~ Normal(mu, 0.8)
    end
    backend = check_rk_source_roundtrip(RKBRMI(builder(data); held_out=(:y2,)))
    likelihood = scalar_likelihood(backend)
    for u in scalar_probes(1)
        @test Base.invokelatest(likelihood, u) ≈
            offset_oracle(data, data.slope .* data.x, u) atol=1e-10 rtol=1e-11
    end
end

@stestset "what stays refused" begin
    data = scalar_data()
    # RK's data expressions are numeric: a Bool column is refused, so a Bool field
    # is too (`_rk_scalar_value`, the rule its prior constants already follow).
    @test_throws "must be a plain numeric vector or a numeric scalar" BRM._brm_rk_plan(
        offset_builder(:(mu ~ 1 + offset(flag * x)))(data))
    @test_throws "a numeric scalar" BRM._brm_rk_plan((@brm begin
        theta ~ Normal(0, 1)
        mu ~ 1 + offset(theta * flag * x)
        y ~ Normal(mu, 0.8)
    end)(data))
end

end
