module DescriptionRecordMembers
using Test, BayesianRegressionModels, Distributions

@brm function instrument(record)
    bias ~ Normal(0, record.width)
    if record.gaussian
        @plate for t in eachindex(record.readings)
            "Readings reported by this instrument."
            record.readings[t] ~ Normal(bias, 1)
        end
    else
        @plate for t in eachindex(record.readings)
            record.readings[t] ~ Cauchy(bias, 1)
        end
    end
    return bias
end

const instruments = @brm begin
    for item in sensors
        shifts[item.key] ~ instrument(item)
    end
end

@testset "Record members share rendered laws and retain keyed evidence" begin
    records = [(key=:south, gaussian=true, width=2., readings=[0.1]),
               (key=:north, gaussian=true, width=0.5, readings=[0.2,0.3]),
               (key=:west, gaussian=false, width=3., readings=[0.4])]
    fit = SBBRMI(instruments((;sensors=records)); mod=@__MODULE__, total_groups=())
    d = brm_descriptor(fit)
    prior(key) = (:member,:shifts,key,:parameter,:bias)
    anchors = Dict(prior(:south) => "#south-bias")
    result = brm_description(d; prior_anchors=anchors)
    @test result.model_id == d.id
    @test result.complete
    observations = filter(c -> c.kind === :observation, brm_description_components(result))
    @test Set(c.provenance.member for c in observations) == Set((:shifts,r.key) for r in records)
    members = filter(c -> c.callable === instrument, result.components)
    @test length(members) == 3
    @test length(result.priors) == 3
    for row in records
        member = only(c for c in members if last(c.provenance.member) === row.key)
        obs = only(c for c in observations if c.provenance.member == member.provenance.member)
        @test obs.provenance.observation_role === :conditioned
        @test prior(row.key) in brm_description_prior_references(obs)
        p = brm_description_prior(obs,prior(row.key))
        @test p.distribution.arguments[end] == row.width
        @test brm_description_prior_anchor(obs,p.id) == brm_description_prior_anchor(result,p.id)
        prediction = only(o for o in obs.outputs if o.role === :posterior_predictive)
        @test prediction.member == member.provenance.member
        @test prediction.name in (o.name for o in d.outputs)
        @test length(prediction.selection) == length(row.readings)
        @test reduce(vcat,fit.data[prediction.source])[prediction.selection] == row.readings
        @test brm_description_record(member,only(member.arguments)).width == row.width
        @test p.anchor == get(anchors,p.id,nothing)
    end
    @test any(p -> occursin("`north`, `south`",p),result.prose)
    equations = filter(eq -> occursin("readings",eq) && occursin("\\sim",eq),result.equations)
    @test length(equations) == 2
    normal_section = only(s for s in result.sections if any(eq -> occursin("readings",eq) &&
        occursin("\\sim{}\\mathcal N",eq),s.equations))
    @test normal_section.members == ((:shifts,:north),(:shifts,:south))
    @test any(eq -> occursin("\\mathcal N",eq),equations)
    @test any(eq -> occursin("cauchy",eq),equations)
    @test Tuple(e for section in result.sections for e in section.equations) == result.equations
    md = brm_description_markdown(result)
    @test occursin("Readings reported by this instrument.",md)
    @test Set(doc.path[2] for doc in result.documentation if doc.markdown == "Readings reported by this instrument.") == Set((:north,:south))
    @test all(doc -> last(doc.path) == Symbol("record.readings"),result.documentation)
    @test occursin("#south-bias",md)
    @test all(p -> occursin("](#"*brm_description_prior_anchor(p)*")",md),result.priors)

    seen = Set{Symbol}()
    hook = instrument => function(c)
        key = last(c.provenance.member)
        push!(seen,key)
        BRMDescriptionFragment(prose=("Calibration for $key.",),covers=(c.id,))
    end
    hooked = brm_description(d; hooks=(hook,), texts=description_texts(:de))
    @test seen == Set(r.key for r in records)
    @test hooked.complete
    @test all(r -> "Calibration for $(r.key)." in hooked.prose,records)
    @test any(p -> startswith(p,"Modell für"),hooked.prose)
    @test_throws ArgumentError brm_description(d; hooks=(instrument => c ->
        BRMDescriptionFragment(covers=(first(observations).id,)),))

    changed = [merge(row,(;width=row.width+1)) for row in reverse(records)]
    replay = reprocess(fit,(;sensors=changed))
    again = brm_description(replay)
    @test Set(p.id for p in again.priors) == Set(p.id for p in result.priors)
    @test again.equations == result.equations
    @test Dict(p.id=>last(p.distribution.arguments) for p in again.priors) ==
        Dict(prior(row.key)=>row.width for row in changed)
end
end

module DescriptionRecordHelpers
using Test, BayesianRegressionModels, Distributions
using StanBlocks: @deffun

@deffun instrument_curve(x::real)::real = x*x
@brm function transformed(record)
    bias ~ Normal(0,1)
    @plate for i in eachindex(record.y)
        mean_value[i] = instrument_curve(bias)
        record.y[i] ~ Normal(mean_value[i],1)
    end
    return bias
end
const builder = @brm begin
    for row in records
        values[row.key] ~ transformed(row)
    end
end

@testset "An uncovered helper has a gap for each use" begin
    data = (;records=[(;key=:a,y=[0.5]),(;key=:b,y=[0.9])])
    d = brm_descriptor(SBBRMI(builder(data); mod=@__MODULE__,total_groups=()))
    root = c -> BRMDescriptionFragment(covers=(c.id,))
    report = brm_description(d;hooks=(transformed=>root,))
    gaps = Set(x.id for x in report.coverage if x.status === :unsupported)
    unknown = filter(c -> c.callable === instrument_curve,brm_description_components(report))
    @test length(unknown) == length(gaps) == 2
    @test Set(c.id for c in unknown) == gaps
    @test !report.complete
    function describe_curve(c)
        key = last(c.provenance.member)
        BRMDescriptionFragment(prose=("Response calibration for $key.",),covers=(c.id,))
    end
    report = brm_description(d;hooks=(instrument_curve=>describe_curve,))
    @test report.complete
    @test all(r -> "Response calibration for $(r.key)." in report.prose,data.records)
    likelihoods = filter(s -> any(eq -> occursin("\\sim",eq) && occursin("record",eq),s.equations),report.sections)
    @test Set(s.members for s in likelihoods) == Set((((:values,:a),),((:values,:b),)))
end

@brm function offset(record)
    location ~ Normal(0,record.scale)
    return location
end
const offsets = @brm begin
    for row in devices
        location[row.key] ~ offset(row)
    end
    y ~ Normal(location,1)
end

@testset "Members without a fitted plate retain their own prior" begin
    data = (;devices=[(;key=4,scale=1.),(;key=2,scale=3.)],y=[0.,1.])
    d = brm_descriptor(SBBRMI(offsets(data); mod=@__MODULE__,total_groups=()))
    r = brm_description(d)
    @test r.complete
    @test Dict(last(p.provenance.member)=>last(p.distribution.arguments) for p in r.priors) ==
        Dict(4=>1.,2=>3.)
    @test only(c for c in brm_description_components(r) if c.kind === :observation).provenance.owner === :y
    empty_data = (;devices=data.devices[1:0],y=Float64[])
    empty_report = brm_description(SBBRMI(offsets(empty_data); mod=@__MODULE__,total_groups=()))
    @test empty_report.complete
    @test isempty(empty_report.priors)
    @test !any(c -> haskey(c.provenance,:member),brm_description_components(empty_report))
end
end
