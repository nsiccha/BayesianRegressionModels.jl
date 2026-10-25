# test/dotted_operators.jl — dotted arithmetic in model-scope formulas.
#
# Run: julia --project=test test/dotted_operators.jl   (no sampling)
#
# Model-scope arithmetic applies per row: plain `*` already lowers to Stan's
# `.*`. Decision `0iefgns` (snag brm-broadcast-in-e3c18ba8, revisiting the
# parked `1jr1f6x`) admits the dotted spellings `.* ./ .+ .- .^` as exact
# aliases of the plain operators in `=` assignments and `~` arguments, so
# `a .* b` builds the `ExprColumn` of `a * b` and every backend emits the same
# program. Before it a dotted head failed when the builder was applied, with
# ``UndefVarError: `.*` not defined``. Any other dotted operator is refused at
# parse time with the admitted set.

using Test
using BayesianRegressionModels, StanBlocks
using Distributions: Normal, Exponential, Gamma
using Statistics: mean

const dotted_df = (x=collect(1.0:8), y=[2.1, 3.0, 3.9, 5.2, 5.8, 7.1, 7.9, 9.0])

dotted_stan(brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)
dotted_line(code, needle) =
    only(filter(line -> occursin(needle, line), split(code, '\n'))) |> strip
dotted_stanc_ok(code) = StanBlocks.stanc_check(code; warn_pedantic=false).ok
dotted_error(expr) = try
    Core.eval(@__MODULE__, expr)
    nothing
catch err
    err
end

@testset "dotted multiplicative composition (2PL shape)" begin
    dot = @brm dotted_df begin
        a ~ 1 + x
        theta ~ 1 + x
        b ~ 1 + x
        eta = a .* (theta .- b)
        y ~ Normal(eta, sigma)
        sigma ~ Exponential(1)
    end
    plain = @brm dotted_df begin
        a ~ 1 + x
        theta ~ 1 + x
        b ~ 1 + x
        eta = a * (theta - b)
        y ~ Normal(eta, sigma)
        sigma ~ Exponential(1)
    end
    # `.*` normalizes to `*` at parse time.
    @test getf(getargs(parent(dot.operations.eta))[2]) === (*)
    dot_code, plain_code = dotted_stan(dot), dotted_stan(plain)
    @test dotted_line(dot_code, " eta =") == "vector[x_n] eta = (a .* (theta - b));"
    # Exact alias: the whole emitted program matches the plain spelling.
    @test dot_code == plain_code
    @test dotted_stanc_ok(dot_code)
end

@testset "remaining dotted arithmetic operators" begin
    cases = (
        (:(a .+ b), :(a + b), "(a + b)"),
        (:(a .- b), :(a - b), "(a - b)"),
        (:(a ./ b), :(a / b), "(a ./ b)"),
        (:(a .^ 2), :(a ^ 2), "(a ^ 2)"),
        (:(.-a), :(-a), "(-a)"),
    )
    model(rhs) = Core.eval(@__MODULE__, :(@brm dotted_df begin
        a ~ 1 + x
        b ~ 1 + x
        eta = $rhs
        y ~ Normal(eta, sigma)
        sigma ~ Exponential(1)
    end))
    for (dotted, plain, stan) in cases
        code = dotted_stan(model(dotted))
        @test dotted_line(code, " eta =") == "vector[x_n] eta = $stan;"
        @test code == dotted_stan(model(plain))
        @test dotted_stanc_ok(code)
    end
end

@testset "a dotted likelihood argument (snag brm-broadcast-in-e3c18ba8)" begin
    # Cell means and sums of squares: a sum of squares is sigma^2 times a
    # chi-square with n - 1 degrees of freedom.
    cells = (g=repeat(1:4, inner=3), mittel=collect(range(-1.0, 1.0; length=12)),
             streuung=collect(range(0.2, 2.0; length=12)), halb_fg=fill(4.5, 12))
    dot = @brm cells begin
        mu ~ 1 + (1 | g)
        log(sigma) ~ 1 + (1 | g)
        mittel ~ Normal(mu, sigma)
        streuung ~ Gamma(halb_fg, 2 .* sigma .^ 2)
    end
    plain = @brm cells begin
        mu ~ 1 + (1 | g)
        log(sigma) ~ 1 + (1 | g)
        mittel ~ Normal(mu, sigma)
        streuung ~ Gamma(halb_fg, 2 * sigma ^ 2)
    end
    dot_code = dotted_stan(dot)
    @test dotted_line(dot_code, "streuung ~") ==
        "streuung ~ gamma(halb_fg, (1.0 ./ (2 .* (sigma ^ 2))));"
    @test dot_code == dotted_stan(plain)
    @test dotted_stanc_ok(dot_code)
end

@testset "other dotted operators fail at parse time" begin
    for (rhs, op) in ((:(a .% 2), ".%"), (:(a .== b), ".=="))
        err = dotted_error(:(@brm dotted_df begin
            a ~ 1 + x
            b ~ 1 + x
            eta = $rhs
            y ~ Normal(eta, sigma)
            sigma ~ Exponential(1)
        end))
        @test err isa LoadError && err.error isa ErrorException
        msg = sprint(showerror, err)
        @test occursin("the dotted operator `$op` is not admitted", msg)
        @test occursin(".* ./ .+ .- .^", msg)
    end
end

@testset "dotted spellings outside `_x` are unchanged" begin
    # A `.~` declaration reads its right-hand side as Julia broadcasting.
    vec = @brm dotted_df begin
        s ~ Exponential(1)
        w[1:3] .~ Normal.(0.0, 2 .* s; lower=0.0)
        y ~ Normal(w[1] * x, s)
    end
    @test dotted_stanc_ok(dotted_stan(vec))
    # A constant is evaluated once over whole columns, as Julia.
    con = @brm dotted_df begin
        m0 = constant(mean(log.(x) .- 1.0))
        y ~ Normal(m0 + 0.0 * x, 1.0)
    end
    @test occursin("m0", dotted_stan(con))
    # A model-scope dotted call keeps its own refusal, naming the plain call.
    err = dotted_error(:(@brm dotted_df begin
        y ~ Normal(abs2.(x), 1.0)
    end))
    @test err !== nothing &&
        occursin("A formula call already applies per row", sprint(showerror, err))
end
