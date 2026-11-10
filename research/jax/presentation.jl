module JAXExampleResults

using Markdown, Printf

function table(name)
    lines = readlines(joinpath(@__DIR__, "results", name * ".tsv"))
    header = Tuple(Symbol.(split(first(lines), '\t')))
    cell(s) = something(tryparse(Float64, s), String(s))
    [NamedTuple{header}(Tuple(cell.(split(line, '\t')))) for line in lines[2:end]]
end

const POPULATION = [
    ("sigma", "Residual SD", "sigma"),
    ("pop_mu.beta_pop[1]", "Population intercept", "intercept"),
    ("pop_mu.beta_pop[2]", "Population slope", "slope"),
    ("b_mu__g_g.tau[1]", "Group intercept SD", "sd_intercept"),
    ("b_mu__g_g.tau[2]", "Group slope SD", "sd_slope"),
    ("b_mu__g_g.L[2]", "Intercept–slope correlation", "cor"),
]

mdtable(header, rows) = Markdown.MD([Markdown.Table([header, rows...], fill(:l, length(header)))])
interval(r) = @sprintf("%.2f [%.2f, %.2f]", r.q50, r.q025, r.q975)

function posterior_table()
    rows = table("posterior")
    rendered = [begin
        native = only(r for r in rows if r.quantity == key && r.method == "native")
        jax = only(r for r in rows if r.quantity == key && r.method == "jax")
        [label, interval(native), interval(jax), @sprintf("%.2f", native.z)]
    end for (key, label, _) in POPULATION]
    mdtable(["Quantity", "WarmupHMC: median [95% interval]", "NumPyro: median [95% interval]",
        "Mean difference / combined MCSE"], rendered)
end

function parity_table()
    rows = table("parity")
    mdtable(["Quantity", "Largest absolute difference"], [
        ["Log density", @sprintf("%.2g", maximum(r.log_density_abs_diff for r in rows))],
        ["Gradient component", @sprintf("%.2g", maximum(r.gradient_max_abs_diff for r in rows))],
    ])
end

function diagnostics_table()
    rows = table("posterior")
    run = Dict(r.key => r.value for r in table("run"))
    rendered = [begin
        sub = filter(r -> r.method == method, rows)
        completed = method == "native" ? run["native_completed_chains"] : run["chains"]
        div = run[method == "native" ? "native_divergences" : "python_divergences"]
        [label, string(Int(completed)), @sprintf("%.3f", maximum(r.rhat for r in sub)),
         string(round(Int, minimum(r.ess_bulk for r in sub))),
         string(round(Int, minimum(r.ess_tail for r in sub))), string(Int(div))]
    end for (method,label) in (("native", "WarmupHMC"), ("jax", "NumPyro"))]
    mdtable(["Sampler", "Completed chains", "Max. R̂", "Min. bulk ESS", "Min. tail ESS", "Divergences"], rendered)
end

function version_table()
    run = Dict(r.key => r.value for r in table("run"))
    keys = ["julia_version", "BayesianRegressionModels", "ReactiveKernels", "ReactiveKernelsPPL",
            "WarmupHMC", "python_version", "jax_version", "numpyro_version", "numpy_version"]
    mdtable(["Software", "Version"], [[replace(k, "_version" => ""), string(run[k])] for k in keys])
end

end
