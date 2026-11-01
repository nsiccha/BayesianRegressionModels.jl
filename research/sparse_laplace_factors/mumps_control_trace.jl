# Diagnostic only: instruments BRM's own inertia wrapper in this process.
# Run with julia --project=test under one compute token. This checks the
# control state around MUMPS calls; logging changes invalidate timing claims.
using BayesianRegressionModels, Logging
const BRM_TRACE = BayesianRegressionModels
const MUMPS_TRACE_PHASE = Ref(:none)
const MUMPS_TRACE_WARNINGS = Ref(0)
const MUMPS_TRACE_FACTORS = Ref(0)
struct MumpsTraceLogger{L} <: AbstractLogger
    parent::L
end
Logging.min_enabled_level(l::MumpsTraceLogger) = Logging.min_enabled_level(l.parent)
Logging.shouldlog(l::MumpsTraceLogger,args...) = Logging.shouldlog(l.parent,args...)
Logging.catch_exceptions(l::MumpsTraceLogger) = false
function Logging.handle_message(l::MumpsTraceLogger,level,message,args...;kwargs...)
    Logging.handle_message(l.parent,level,message,args...;kwargs...)
    if occursin("out-of-core storage",string(message))
        MUMPS_TRACE_WARNINGS[] += 1
        println("MUMPS warning phase=",MUMPS_TRACE_PHASE[]," completed factors=",MUMPS_TRACE_FACTORS[])
        for frame in Iterators.take(stacktrace(), 20)
            println("  ", frame.func, " at ", frame.file, ":", frame.line)
        end
        flush(stdout)
    end
end
source=read(joinpath(@__DIR__, "../../src/nlme_laplace_factors.jl"),String)
start=findfirst("function _brm_spl_inertia(K)",source).start
stop=findfirst("function _brm_spl_factor_selection",source).start-1
body=source[start:stop]
body=replace(body,"m = MUMPS.Mumps"=>"Main.MUMPS_TRACE_PHASE[] = :constructor\n        before_warnings = Main.MUMPS_TRACE_WARNINGS[]\n        m = MUMPS.Mumps")
body=replace(body,"copy(MUMPS.default_cntl64); backend=MUMPS.Sequential())"=>"""copy(MUMPS.default_cntl64); backend=MUMPS.Sequential())
        @assert m.icntl[22]==0 && m.icntl[16]==1 && m.icntl[48]==0
        if Main.MUMPS_TRACE_WARNINGS[] != before_warnings
            println("MUMPS returned constructor controls22/16/48=",m.icntl[[22,16,48]])
        end
        Main.MUMPS_TRACE_PHASE[] = :controls""")
body=replace(body,"MUMPS.invoke_mumps!(m)"=>"""Main.MUMPS_TRACE_PHASE[] = :factorization
            MUMPS.invoke_mumps!(m)
            @assert m.icntl[22]==0 && m.icntl[16]==1 && m.icntl[48]==0
            Main.MUMPS_TRACE_FACTORS[] += 1""")
body=replace(body,"Base.finalize(m)"=>"""Main.MUMPS_TRACE_PHASE[] = :finalization
            Base.finalize(m)
            Main.MUMPS_TRACE_PHASE[] = :none""")
include_string(BRM_TRACE,body,"instrumented-own-inertia.jl")
with_logger(MumpsTraceLogger(current_logger())) do
    include(joinpath(@__DIR__, "fitting_scaling.jl"))
end
println("MUMPS_FITTING_TRACE_PASSED factors=",MUMPS_TRACE_FACTORS[]," warnings=",MUMPS_TRACE_WARNINGS[])
