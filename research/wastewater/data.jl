using Dates

"CDC's published simulated example, restricted to one catchment and 35 observed days."
function wastewater_public_data(; nt=35, uot=50, ht=7)
    directory = joinpath(@__DIR__, "data")
    rows(file) = [split(line, '\t') for line in readlines(joinpath(directory, file))[2:end]]
    first_date = Date("2023-09-01")
    day(s) = Dates.value(Date(s) - first_date) + 1
    ww = filter(r -> r[2] == "1" && 1 <= day(r[1]) <= nt, rows("ww_data.tsv"))
    hosp = filter(r -> 1 <= day(r[1]) <= nt, rows("hosp_data.tsv"))
    length(hosp) == nt || error("hospital series must cover each observed day")
    population = parse(Float64, hosp[1][3])
    catchment_population = parse(Float64, ww[1][6])
    weights = [1 - catchment_population / population, catchment_population / population]
    g = [parse(Float64, r[2]) for r in rows("default_covid_gi.tsv")]
    dl = [parse(Float64, r[2]) for r in rows("default_covid_inf_to_hosp.tsv")]
    base = cdc_ww_brm_fixture(; K=2, nt, uot, ht)
    forecast_days = (nt + 1):(nt + ht)
    merge(base, (;
        subpopulation=["uncovered", "site-1"], w=weights,
        ww_lab=["lab-1"], n_labs=1, lab_to_subpop=[2],
        ww_time=[uot + day(r[1]) for r in ww], ww_subpop=fill(2, length(ww)),
        ww_obs_lab=ones(Int, length(ww)),
        ww_lod=[parse(Float64, r[5]) for r in ww],
        ww_log_conc=[parse(Float64, r[4]) for r in ww],
        count_stream=["state admissions"], n_count_streams=1,
        count_week_grid=[collect(1.0:base.n_weeks)],
        count_subpop_weights=reshape(weights, 2, 1), count_delay=reshape(dl, :, 1),
        count_population=[population],
        count_start=uot .+ collect(1:nt), count_stop=uot .+ collect(1:nt),
        count_obs_stream=ones(Int, nt), count=[parse(Int, r[2]) for r in hosp],
        forecast_count_start=uot .+ collect(forecast_days),
        forecast_count_stop=uot .+ collect(forecast_days),
        forecast_count_obs_stream=ones(Int, ht),
        forecast_ww_time=[uot + nt + ht], forecast_ww_subpop=[2], forecast_ww_obs_lab=[1],
        g, dl, nsh=30, mwpd=757000.0, n_pop=population,
    ))
end

function wastewater_public_model(data=wastewater_public_data())
    cdc_ww_brm_model(data)
end
