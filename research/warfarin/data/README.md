# Warfarin data

`warfarin_data.txt` is a verbatim copy of the file of the same name in
Sebastian Weber's StanCon 2018 contribution, "Solving ODEs in the wild:
Scalable pharmacometrics with Stan":

<https://github.com/stan-dev/stancon_talks/tree/master/2018-helsinki/Contributed-Talks/weber/stancon18-master>

The stancon_talks repository distributes its code under the New BSD License
and its text under CC BY 4.0, copyright to the authors. The data are the
classic warfarin dataset of O'Reilly and Aggeler (1968), as distributed with
Holford's teaching material.

Columns, tab separated, `.` for a missing value:

| column | meaning |
| --- | --- |
| `id` | subject (32 subjects) |
| `time` | hours after the dose |
| `amt` | dose in mg, on the dosing row (`time = 0`) only; 1.5 mg per kg |
| `dv` | the observation: plasma warfarin concentration in mg/L (`dvid = 1`) or prothrombin complex activity in percent of normal (`dvid = 2`) |
| `dvid` | 1 for a PK observation, 2 for a PD observation |
| `wt` | body weight in kg |
| `age` | age in years |
| `sex` | 1 male, 0 female |

`data.jl` reads the file into the per-subject form the models take, following
Weber's `munge.R`: the five PK observations at 0.5 h, all zero, are dropped
because the lag time may be as long as 1 h.
