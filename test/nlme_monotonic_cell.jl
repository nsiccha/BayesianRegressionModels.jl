# test/nlme_monotonic_cell.jl — a monotonic term read by a subject's @plate cell.
#
# RUN: julia --project=test test/nlme_monotonic_cell.jl [testset-filter...]
#
# A `mo()` term's level table, `cumsum(vcat(0.0, simplex_incr))`, has no row
# form. It holds one value per level and depends only on population coordinates,
# so a strict query that declares the subjects' effects as partitioned computes
# it once per population point, in the shared prefix, and each cell gathers from
# it at its own rows. These are toy data: a cell reads a per-event predictor
# through `groups`, as a dose stream does.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))

const MONOTONIC_CELL_DATA = (; subject=[1, 2, 3, 4], weight=[0.1, -0.2, 0.3, 0.0],
    op_subject=[1, 1, 2, 2, 3, 4, 4, 3],
    op_diet=[1, 2, 1, 3, 2, 4, 1, 2],
    op_x=[0.1, 0.2, -0.1, 0.3, 0.0, -0.2, 0.15, 0.05],
    op_amount=[100.0, 50.0, 80.0, 40.0, 120.0, 60.0, 90.0, 30.0],
    y=[0.8, 0.6, 0.5, 0.4, 0.7, 0.2, 0.1, 0.3])

function monotonic_cell_model(predictors, exponent)
    source = quote
        sigma ~ Exponential(1)
        $(predictors.args...)
        log_CL ~ 1 + weight + (1 | pk | subject)
        events = groups(op_subject => subject)
        @plate for i in eachindex(log_CL)
            loc[i] = 0.01 .* op_amount[events[i]] .* exp.($exponent .- exp(log_CL[i]))
            y[events[i]] ~ normal(loc[i], sigma)
        end
    end
    RKBRMI(brm(source; mod=@__MODULE__)(MONOTONIC_CELL_DATA))
end

const LOG_F_AT_EVENTS = :(log_F[events[i]])

# Each spelling emits its own level table: a coefficient on the contrast, a
# contrast beside a population block, the matrix of a joint block over two
# predictors, and a coefficient-free summand.
const MONOTONIC_CELL_SHAPES = (
    bare=(quote
        log_F ~ 0 + mo(op_diet)
    end, LOG_F_AT_EVENTS),
    population=(quote
        log_F ~ 1 + op_x + mo(op_diet)
    end, LOG_F_AT_EVENTS),
    joint=(quote
        log_F ~ 0 + mo(op_diet)
        log_K ~ 0 + mo(op_diet)
    end, :(log_F[events[i]] .+ log_K[events[i]])),
    summand=(quote
        log_F ~ 0 + op_x + mo1(op_diet)
    end, LOG_F_AT_EVENTS))

monotonic_cell_point(rk) = 0.1 .* sin.(1:rk.model.layout.total)

# One subject's log density, summed from the whole model's pointwise densities:
# the reference every strict evaluation below answers to.
function monotonic_cell_density(query, u, subject)
    pointwise = Base.invokelatest(query, u).y
    length(pointwise) == length(MONOTONIC_CELL_DATA.subject) ||
        error("expected one pointwise entry per subject's cell of y")
    sum(pointwise[subject])
end

function monotonic_cell_fd(f, u, coordinates; h=1e-5)
    map(coordinates) do j
        plus = copy(u); minus = copy(u)
        plus[j] += h; minus[j] -= h
        (f(plus) - f(minus)) / (2h)
    end
end

@stestset "a strict query computes a mo() level table once per population point" begin
    for (shape, (predictors, exponent)) in pairs(MONOTONIC_CELL_SHAPES)
        @testset "$shape" begin
            rk = monotonic_cell_model(predictors, exponent)
            bound = rk_bound_plan(rk)
            names = String.(coordinate_names(rk.model.layout))
            u = monotonic_cell_point(rk)
            owned = Symbol("b_pk_subject.z") => [1, 2, 3, 4]
            reference = pointwise_query(rk)

            query = prepare_cell_query(rk.model, bound, :y; locality=:error,
                partitioned=owned, u0=u)
            for subject in 1:4
                @test query(u, subject) ≈
                    monotonic_cell_density(reference, u, subject) rtol=1e-12
            end

            sampler = prepare_partition_sampler(rk.model, bound, :y, u; by=:subject,
                backend=NLME_AD, partitioned=owned, locality=:error)
            simplex = findall(n -> occursin(".simplex_incr.", n), names)
            @test !isempty(simplex)
            for subject in 1:4
                reads = partition_footprint(sampler, subject)
                # The term's population coordinates reach every subject, and no
                # subject reads another subject's effect.
                @test simplex ⊆ reads
                foreign = filter(j -> startswith(names[j], "b_pk_subject.z.") &&
                    !startswith(names[j], "b_pk_subject.z.$subject."), reads)
                @test isempty(foreign)

                gradient = zeros(length(reads))
                value, _ = partition_value_and_gradient!(sampler, gradient, u, subject)
                @test value ≈ monotonic_cell_density(reference, u, subject) rtol=1e-12
                @test gradient ≈ monotonic_cell_fd(
                    v -> monotonic_cell_density(reference, v, subject), u, reads) rtol=1e-6 atol=1e-7
            end
        end
    end
end

@stestset "brm_nlme_model evaluates a subject over a mo() term" begin
    for (shape, (predictors, exponent)) in pairs(MONOTONIC_CELL_SHAPES)
        @testset "$shape" begin
            rk = monotonic_cell_model(predictors, exponent)
            m = brm_nlme_model(rk; ad_backend=NLME_AD)
            @test any(n -> occursin(".simplex_incr.", String(n)), m.view.coordinates[m.theta])
            θ = 0.1 .* sin.(eachindex(m.theta))
            σ = [log(0.8)]
            H = [0.1 -0.2 0.3 0.05]
            reference = pointwise_query(rk)
            values, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
            for subject in 1:4
                u = BRM._brm_nlme_point(m, θ, σ, H)
                @test values[subject] ≈
                    monotonic_cell_density(reference, u, subject) rtol=1e-12
                @test brm_nlme_loglikelihood(m, subject, θ, σ, H[:, subject]) ≈
                    values[subject] rtol=1e-12
                value, gradient = brm_nlme_loglikelihood_and_gradient(
                    m, subject, θ, σ, H[:, subject])
                @test value ≈ values[subject] rtol=1e-12
                @test gradient ≈ G[:, subject] rtol=1e-10
                @test gradient ≈ monotonic_cell_fd(
                    η -> brm_nlme_loglikelihood(m, subject, θ, σ, η),
                    H[:, subject], eachindex(H[:, subject])) rtol=1e-6
            end
        end
    end
end
