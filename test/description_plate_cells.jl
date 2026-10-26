# Run: julia --startup-file=no --project=test test/description_plate_cells.jl
#
# A `@plate for` cell is described as written (snag brm-description-256b3b35,
# decision 011ibo0): its statements in the plate's own index, `x[i]` as x_i and
# `x[rows[i]]` as x_{rows_i} with the row set stated once, a sum of elementwise
# reads of one row set as a sum over those rows, an output `y[i] = …` as y_i,
# and a cell latent's law among the cell's statements and, in the cell's names,
# in the prior table. No input binding gets a symbol of its own. The page
# writes each part's sentences followed by its own equations, and an
# incomplete page opens with the number of uncovered parts and their callables.
using Test, BayesianRegressionModels, Distributions
using Random: Xoshiro

describe(builder; kwargs...) = brm_description(SBBRMI(builder; total_groups=()); kwargs...)
no_input_bindings(r) = !any(p -> startswith(p, "Cell input"), r.prose) &&
    !any(e -> startswith(e, "a_{") || occursin("\\mathcal K", e) || occursin("ragged", e) ||
        occursin("\\mathrm{nothing}", e), r.equations)

const TOTALS = let rng = Xoshiro(1)
    units, years = 24, 3
    district = string.("d", 1 .+ (0:units-1) .÷ 6)
    year = repeat(1:years; inner=units)
    cell(d, t) = string(d, "-", t)
    cells = [cell(d, t) for t in 1:2 for d in unique(district)]
    (; t=Float64.(year .- 2), x=randn(rng, units * years),
        unit_cell=[cell(district[(r-1)%units+1], year[r]) for r in 1:units*years],
        cell=cells, total_a=100 .+ 10 .* rand(rng, length(cells)),
        nl1=rand(rng, 5:20, length(cells)), elig=rand(rng, units * years))
end

@testset "a cell observing a sum over its rows reads as written" begin
    model = @brm TOTALS begin
        log_a ~ 1 + t + x
        cv_a ~ LogNormal(-2.5, 1)
        rows = groups(unit_cell => cell)
        @plate for kk in eachindex(total_a)
            total_a[kk] ~ Normal(sum(exp.(log_a[rows[kk]])), cv_a * total_a[kk])
        end
        log_nl1 ~ 1 + t
        cv_nl1 ~ LogNormal(-3, 1)
        rows_urs = groups(unit_cell => cell)
        @plate for kk in eachindex(nl1)
            nl1[kk] ~ NegativeBinomial2(sum(exp.(log_nl1[rows_urs[kk]]) .* elig[rows_urs[kk]]), 1.0 / cv_nl1^2)
        end
    end
    r = describe(model)
    @test r.complete
    @test no_input_bindings(r)
    @test "The statements of the `@plate` cell `total_a_cell` below hold for each entry \$\\mathit{kk}\$ of `total_a`." in r.prose
    @test "`rows = groups(unit_cell => cell)` gives entry \$\\mathit{kk}\$ the rows \$\\mathrm{rows}_{\\mathit{kk}}\$ whose `unit_cell` equals entry \$\\mathit{kk}\$ of `cell`." in r.prose
    @test "\\mathrm{rows}_{\\mathit{kk}}=\\{j:\\mathrm{unit\\_cell}_{j}=\\mathrm{cell}_{\\mathit{kk}}\\}" in r.equations
    observed = "{\\mathrm{total\\_a}}_{\\mathit{kk}}\\sim{}\\mathcal N\\left(\\sum_{j\\in \\mathrm{rows}_{\\mathit{kk}}}\\exp\\left({\\mathrm{log\\_a}}_{j}\\right),{\\left(\\mathrm{cv\\_a} \\cdot {\\mathrm{total\\_a}}_{\\mathit{kk}}\\right)}^{2}\\right)"
    @test observed in r.equations
    # A long relation names the sum as one term; its elementwise body stays whole.
    @test any(e -> endswith(e, "=\\sum_{j\\in \\mathrm{rows\\_urs}_{\\mathit{kk}}}\\left(\\exp\\left({\\mathrm{log\\_nl1}}_{j}\\right) \\cdot {\\mathrm{elig}}_{j}\\right)"), r.equations)
    # The cell predictor `total_a_cell` is no linear predictor of its own.
    @test !any(p -> startswith(p, "`total_a_cell` combines"), r.prose)

    # Each part's sentences, then its own equations, in model order.
    md = brm_description_markdown(r)
    sentence = findfirst("`total_a` contributes an observation likelihood.", md)
    equation = findfirst("\$\$\n" * observed * "\n\$\$", md)
    following = findfirst("`log_nl1` combines", md)
    @test first(sentence) < first(equation) < first(following)
    @test Tuple(p for s in r.sections for p in s.prose) == r.prose
    @test Tuple(e for s in r.sections for e in s.equations) == r.equations

    german = describe(model; texts=description_texts(:de))
    @test "Die folgenden Anweisungen der `@plate`-Zelle `total_a_cell` gelten für jeden Eintrag \$\\mathit{kk}\$ von `total_a`." in german.prose
    @test observed in german.equations
end

@testset "a cell latent is one of the cell's statements" begin
    model = @brm (; mu=[[10.0, 20.0, 5.0], [7.0, 3.0]], obs=[[1, 2], [1]],
                    y=[[11.0, 19.0], [6.5]], total=[36.0, 10.5]) begin
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(total)
            loc = log.(mu[i])
            log_T ~ Normal(loc, 0.5)
            T = exp.(log_T)
            y[i] ~ Normal(T[obs[i]], sigma)
            total[i] ~ Normal(sum(T), 0.01 * total[i])
        end
    end
    r = describe(model)
    @test r.complete
    @test no_input_bindings(r)
    @test "`log_T` is a latent quantity of the cell `y_cell`." in r.prose
    @test "\\mathrm{log\\_T}\\sim{}\\mathcal N\\left(\\mathrm{y\\_cell.loc},{0.5}^{2}\\right)" in r.equations
    @test "\\mathrm{y\\_cell.loc}=\\log\\left({\\mathrm{mu}}_{i}\\right)" in r.equations
    @test "{\\mathrm{y}}_{i}\\sim{}\\mathcal N\\left({\\mathrm{y\\_cell.T}}_{{\\mathrm{obs}}_{i}},{\\mathrm{sigma}}^{2}\\right)" in r.equations
    # The prior table names the latent's law in the cell's names too.
    @test occursin("Logical parameter: `kernel` / `y_cell` / `log_T`.\n\n\$\$\n\\mathcal N\\left(\\mathrm{y\\_cell.loc},{0.5}^{2}\\right)\n\$\$",
        brm_description_markdown(r))
    german = describe(model; texts=description_texts(:de))
    @test "`log_T` ist eine latente Größe der Zelle `y_cell`." in german.prose
end

@testset "a collected output and a grouped response read as written" begin
    model = @brm (; subject=[1, 2], obs_subject=[1, 2, 1], time=[0.2, 0.3, 0.8], y=[0.4, 1.2, 0.6]) begin
        sigma ~ Exponential(1)
        eta ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(subject)
            loc[i] = exp(eta[i]) .+ time[obs[i]]
            y[obs[i]] ~ Normal(loc[i], sigma)
        end
    end
    r = describe(model)
    @test r.complete
    @test no_input_bindings(r)
    @test "{\\mathrm{loc}}_{i}=\\left(\\exp\\left({\\mathrm{eta}}_{i}\\right) + {\\mathrm{time}}_{\\mathrm{obs}_{i}}\\right)" in r.equations
    @test "{\\mathrm{y}}_{\\mathrm{obs}_{i}}\\sim{}\\mathcal N\\left({\\mathrm{loc}}_{i},{\\mathrm{sigma}}^{2}\\right)" in r.equations
    @test !any(e -> occursin("loc.loc", e), r.equations)
    @test !any(p -> startswith(p, "`loc` combines"), r.prose)
end

@testset "positions read through indexin read as written" begin
    model = @brm (; subject=["a", "b"], dose_subject=["a", "a", "b"], log_dose=[0.1, 0.4, 0.2],
                    op_subject=["a", "a", "b", "a", "b"], op_dose=[1, 0, 3, 2, 0],
                    amount=[1.0, 0.0, 2.0, 1.5, 0.0], obs_subject=["a", "b"], y=[0.3, 0.5]) begin
        log_F ~ 0 + log_dose
        log_CL ~ 1 + (1 | p | subject)
        events = groups(op_subject => subject)
        doses = groups(dose_subject => subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(log_CL)
            F = vcat(0.0, log_F[doses[i]])
            slot = something.(indexin(op_dose[events[i]], doses[i]), 0) .+ 1
            dose = sum(amount[events[i]] .* exp.(F[slot]))
            y[obs[i]] ~ normal(dose .* exp(-exp(log_CL[i])), 0.3)
        end
    end
    r = describe(model)
    @test r.complete
    @test no_input_bindings(r)
    @test "\\mathrm{y\\_cell.slot}=\\operatorname{something}\\left(\\operatorname{indexin}\\left({\\mathrm{op\\_dose}}_{\\mathrm{events}_{i}},\\mathrm{doses}_{i}\\right),0\\right) + 1" in r.equations
    @test "\\mathrm{y\\_cell.F}=\\left[0.0;\\ {\\mathrm{log\\_F}}_{\\mathrm{doses}_{i}}\\right]" in r.equations
    @test "`indexin(a, b)` holds, for each entry of `a`, its position in `b`; `something(…, 0)` holds 0 where an entry has none." in r.prose
    @test count(p -> occursin(" = groups(", p), r.prose) == 3
end

@testset "a groups plate reads its group's rows; an incomplete page names its gaps" begin
    rng = Xoshiro(2)
    data = (; probe=repeat(["p1", "p2", "p3"], inner=4), day=Float64.(repeat([1, 2, 4, 5], 3)),
        x=randn(rng, 12), y=randn(rng, 12))
    model = @brm data begin
        phi ~ Uniform(-1, 1)
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | probe)
        @plate for p in groups(probe; order = day)
            y[p] ~ AR1Normal(mu[p], sigma, phi; time = day[p])
        end
    end
    r = describe(model)
    @test no_input_bindings(r)
    @test "The statements of the `@plate` cell `y_cell` below hold for each group \$p\$ of `probe`, whose rows are \$J_{p}\$, in increasing `day`." in r.prose
    @test "J_{p}=\\{j:\\mathrm{probe}_{j}=p\\}" in r.equations
    @test "{\\mathrm{y}}_{J_{p}}\\sim{}\\mathrm{AR1Normal}\\left({\\mathrm{mu}}_{J_{p}}, \\mathrm{sigma}, \\mathrm{phi}, time={\\mathrm{day}}_{J_{p}}\\right)" in r.equations
    # `AR1Normal` and `length` are known gaps (BRM todo 0m6bok4).
    @test !r.complete && length(r.diagnostics) == 3
    @test startswith(brm_description_markdown(r), "Incomplete model description: 3 parts have no scientific description (`AR1Normal`, `length`); they are listed under “Coverage gaps” below.\n")
    @test startswith(brm_description_markdown(describe(model; texts=description_texts(:de))),
        "Unvollständige Modellbeschreibung: 3 Teile haben keine wissenschaftliche Beschreibung (`AR1Normal`, `length`); sie sind unten unter „Abdeckungslücken“ aufgeführt.\n")
    one = describe(model; hooks=(AR1Normal => c -> BRMDescriptionFragment(covers=(c.id,)),))
    @test startswith(brm_description_markdown(one), "Incomplete model description: 1 part has no scientific description (`length`); it is listed under “Coverage gaps” below.\n")
end

@testset "a description built without sections keeps prose, then equations" begin
    d = BRMDescription("m", (), ("First.", "Second."), ("a=1", "b=2"), (), (), (), (), true)
    @test d.sections == ((; id=(), prose=("First.", "Second."), equations=("a=1", "b=2")),)
    @test brm_description_markdown(d) == "Complete model description.\n\nFirst.\n\nSecond.\n\n\$\$\na=1\n\$\$\n\n\$\$\nb=2\n\$\$\n"
end
