# test/exported_link_functions.jl — `using BayesianRegressionModels` binds every
# function SBBRMI spells by identity and every link whose inverse is one of
# them, so a model module needs no `using LogExpFunctions` or
# `using StatsFuns` for a `logit(mu) ~ 1 + x` link or a `logistic(eta)`
# argument (snag `logit-link-needs-91302d71`). The distribution families are
# test/exported_families.jl (decision `1lixk35`).
#
# Run: julia --startup-file=no --project=test test/exported_link_functions.jl

using Test
using BayesianRegressionModels
import InverseFunctions, LogExpFunctions, StatsFuns
import StanBlocks

const BRM = BayesianRegressionModels

# The functions SBBRMI's Julia-to-Stan spelling table dispatches on through
# `::typeof(f)` (`_sb_julia_stan_name`, src/sb_cell_julia.jl). Base functions
# are bound in every module already.
function spelled_functions()
    fs = Set{Function}()
    for m in methods(BRM._sb_julia_stan_name)
        T = Base.unwrap_unionall(m.sig).parameters[2]
        T isa DataType && T <: Function && isdefined(T, :instance) || continue
        parentmodule(T.instance) === Base || push!(fs, T.instance)
    end
    fs
end

# A link `f(mu) ~ …` binds `mu = inverse(f)(eta)`, so a function whose inverse
# SBBRMI spells is a link it lowers.
function lowered_links(spelled)
    links = Set{Function}()
    for g in spelled
        f = InverseFunctions.inverse(g)
        f isa Function && parentmodule(f) !== Base && push!(links, f)
    end
    links
end

const SPELLED = spelled_functions()
const LOWERED = union(SPELLED, lowered_links(SPELLED))

@testset "every spelled function and lowered link is exported" begin
    # The reflection must see the seam; a renamed seam would read as empty.
    for f in (LogExpFunctions.logistic, LogExpFunctions.log1pexp,
              StatsFuns.normcdf, StatsFuns.norminvcdf)
        @test f in SPELLED
    end
    @test LogExpFunctions.logit in LOWERED
    exported = Set(names(BRM))
    missing_exports = filter(f -> !(nameof(f) in exported), LOWERED)
    @test isempty(missing_exports)
    # Bound to the package's own function, not a wrapper of BRM's.
    @test all(f -> getfield(BRM, nameof(f)) === f, LOWERED)
end

# The reporter's shape: a module that only does `using BayesianRegressionModels`
# and a data-free builder with a `logit(...)` link.
module OnlyBRMLinks
using BayesianRegressionModels
const data = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    n=[5, 5, 5, 6, 6, 6],
    c=[1, 2, 2, 4, 5, 5],
    k=[0, 1, 3, 3, 5, 6],
    z=[0.4, 0.9, 1.3, 0.2, 2.1, 0.7],
)
const builder = @brm begin
    logit(mu) ~ 1 + x
    phi ~ Gamma(2, 0.1)
    c ~ BetaBinomial2(n, mu, phi)
end
# The probit and softplus links, and a model-scope `logistic` call.
const others = @brm begin
    norminvcdf(q) ~ 1 + x
    logexpm1(s) ~ 1
    eta ~ 1 + x
    k ~ Binomial(n, q)
    c ~ Binomial(n, logistic(eta))
    z ~ Normal(0, s)
end
const source = quote
    logit(mu) ~ 1 + x
    phi ~ Gamma(2, 0.1)
    c ~ BetaBinomial2(n, mu, phi)
end
end

@testset "a link model needs only using BayesianRegressionModels" begin
    sb = SBBRMI(OnlyBRMLinks.builder(OnlyBRMLinks.data); mod=OnlyBRMLinks)
    code = stan_code(sb)
    @test occursin("inv_logit(logit_mu", code)
    @test StanBlocks.stan.compiles(sb.model)

    sb = SBBRMI(OnlyBRMLinks.others(OnlyBRMLinks.data); mod=OnlyBRMLinks)
    code = stan_code(sb)
    @test occursin("Phi(norminvcdf_q", code)
    @test occursin("log1p_exp(logexpm1_s", code)
    @test occursin("inv_logit(eta", code)
    @test StanBlocks.stan.compiles(sb.model)

    # `brm(source; mod)` resolves its names in the same module.
    built = brm(OnlyBRMLinks.source; mod=OnlyBRMLinks)(OnlyBRMLinks.data)
    @test occursin("inv_logit(logit_mu", stan_code(SBBRMI(built; mod=OnlyBRMLinks)))
end

# Importing LogExpFunctions and StatsFuns as well stays unambiguous: the
# bindings are identical.
module BRMAndLogExpFunctions
using BayesianRegressionModels, LogExpFunctions, StatsFuns
const resolved = (logit, logistic, log1pexp, normcdf, norminvcdf)
end

@testset "using BRM beside LogExpFunctions and StatsFuns" begin
    @test BRMAndLogExpFunctions.resolved === (LogExpFunctions.logit,
        LogExpFunctions.logistic, LogExpFunctions.log1pexp, StatsFuns.normcdf,
        StatsFuns.norminvcdf)
end
