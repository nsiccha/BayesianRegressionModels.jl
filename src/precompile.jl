# These signatures consume syntax, rather than specializing on each model's
# expression tree. Generate their native code in the package image so the first
# artifact request can reuse it. No data, user callable or numerical kernel is
# executed here; emitted numerical code keeps its existing specialization.
if ccall(:jl_generating_output, Cint, ()) == 1
    let info = typeof(_brm_parse_info(_brm_plate_groupings(Expr(:block))))
        precompile(_brm_parse!, (Expr, info))
        precompile(_brm_parselocals!, (Expr, info, Symbol))
    end
    precompile(_brm_expand, (Expr, Module))
    precompile(rewrite_ranef_ids, (Expr,))
    precompile(_brm_collect_additive_terms!, (Vector{Any}, Any))
    precompile(_brm_operation_references!, (Set{Symbol}, ExprColumn))
    precompile(_rk_collect_prior_reads!, (Set{Symbol}, Dict{Symbol,Tuple{Any,Symbol}},
        _BRMPreparedExpr{UnionAll}, Dict{Symbol,Any}))
    precompile(_rk_source_symbols!, (Set{Symbol}, Vector{Any}))
    precompile(_rk_plan_reads!, (Set{Symbol}, Symbol))
    precompile(_RKPredictorSpec, (Symbol, Symbol, Vector{_RKTermSpec}, Symbol, Symbol))
    precompile(_rk_eval_dotted, (Expr, Dict{Symbol,Any}, Vector{_RKDerivedSpec}, Dict{Symbol,Any}))
    precompile(_brm_backend_context, (BRMI,))
    precompile(_brm_prepare_program, (BRMI,))
    precompile(_brm_prepare_model, (BRMI,))
    precompile(_brm_rk_unselected_plan, (BRMI,))
    precompile(_brm_rk_unselected_plan, (BRMI, Ref{Any}))
    precompile(_brm_rk_plan, (BRMI,))
    precompile(_rk_emit_ast, (_RKStructuralPlan,))
    precompile(_rk_emit_ast, (_RKValuePlan,))
end
