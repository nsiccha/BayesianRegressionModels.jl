# Weber's public warfarin data (data/README.md) in the per-subject form the
# models take: one entry per subject for the dose, the weight and the PK
# medians the second stage conditions on, and one vector per subject for the
# sampling times and observations of each channel.

const WARFARIN_DATA_FILE = joinpath(@__DIR__, "data", "warfarin_data.txt")
const WARFARIN_PK_MEDIANS_FILE = joinpath(@__DIR__, "results", "pk_subject_medians.tsv")

"""
    warfarin_data([path])

The 32 subjects of `data/warfarin_data.txt`, in order of first appearance, as
the PK and joint models take them. Follows Weber's `munge.R`: a subject's dose
is its `amt` at time 0, PK observations are the `dvid = 1` rows at 1 h or
later (the five zero concentrations at 0.5 h are dropped, because the lag time
may be as long as 1 h), PD observations are the `dvid = 2` rows.
"""
function warfarin_data(path = WARFARIN_DATA_FILE)
    lines = readlines(path)
    header = Symbol.(split(lines[1], '\t'))
    header == [:id, :time, :amt, :dv, :dvid, :wt, :age, :sex] ||
        error("unexpected columns in $path: $header")
    number(s) = s == "." ? missing : parse(Float64, s)
    rows = [NamedTuple{Tuple(header)}(Tuple(number.(split(line, '\t')))) for line in lines[2:end]]

    ids = unique(Int(row.id) for row in rows)               # order of first appearance
    position = Dict(id => i for (i, id) in enumerate(ids))
    J = length(ids)
    dose = zeros(J); weight = zeros(J)
    pk_time = [Float64[] for _ in 1:J]; pk_dv = [Float64[] for _ in 1:J]
    pd_time = [Float64[] for _ in 1:J]; pd_dv = [Float64[] for _ in 1:J]
    for row in rows
        i = position[Int(row.id)]
        if !ismissing(row.amt) && row.time == 0
            dose[i] = row.amt
            weight[i] = row.wt
        elseif ismissing(row.amt) && row.dvid == 1 && row.time >= 1
            push!(pk_time[i], row.time); push!(pk_dv[i], row.dv)
        elseif ismissing(row.amt) && row.dvid == 2
            push!(pd_time[i], row.time); push!(pd_dv[i], row.dv)
        end
    end
    all(>(0), dose) || error("a subject has no dosing row")
    return (;
        subject = collect(1:J),
        source_subject = ids,
        dose,
        weight,
        log_weight_ratio = log.(weight ./ 70.0),
        pk_time, pk_dv, pd_time, pd_dv,
    )
end

"""
    warfarin_pd_data([data]; medians = WARFARIN_PK_MEDIANS_FILE)

The PD stage's data: `data` plus each subject's PK posterior medians of
`log(tlag)`, `log(ka)`, `log(CL)` and `log(V)` (the last two including the
weight effect), read from the table `fit.jl` writes after the PK fit.
"""
function warfarin_pd_data(data = warfarin_data(); medians = WARFARIN_PK_MEDIANS_FILE)
    isfile(medians) || error("$medians is missing: run research/warfarin/fit.jl first")
    lines = readlines(medians)
    header = Symbol.(split(lines[1], '\t'))
    table = Dict(name => Float64[] for name in header)
    for line in lines[2:end]
        for (name, value) in zip(header, split(line, '\t'))
            push!(table[name], parse(Float64, value))
        end
    end
    Int.(table[:source_subject]) == data.source_subject ||
        error("$medians lists other subjects than the data")
    return merge(data, (;
        pk_log_tlag = table[:pk_log_tlag],
        pk_log_ka = table[:pk_log_ka],
        pk_log_cl = table[:pk_log_cl],
        pk_log_v = table[:pk_log_v],
    ))
end
