# Attribute the allocation constant, independently of the scaling check.
empty!(ARGS); push!(ARGS, "selections stay compact")
include(joinpath(@__DIR__, "../../test/laplace_factors.jl"))

function profile_assembly(n)
    rk = spl_sum_fixture(n)
    packed = length(BRM._rk_layout_coordinate_names(rk))
    u = zeros(packed)
    f = BRM._brm_spl_prepare_factors(rk, :y, u, 1, 1:packed, Int[])
    f = BRM._brm_spl_factor_operators(f, u, NLME_AD)
    c = only(f.cells); t = first(only(c.terms))
    values() = BRM._brm_spl_factor_values(f,u,c)
    block() = BRM._rk_spl_factor_block(t.ad,u,t.sel,c.i;term=t.t)
    state() = BRM._brm_spl_factor_blocks(f,u,c)
    assembly() = BRM._brm_spl_factor_curvature(f,u,packed,0)
    values(); block(); state(); assembly()
    merge_term() = merge(t,(;g=zeros(length(t.sel)),h=zeros(length(t.sel),length(t.sel))))
    merge_term()
    println((;n,term_record_bytes=sizeof(t),operator_record_bytes=sizeof(t.ad),
        value_bytes=@allocated(values()), derivative_bytes=@allocated(block()),
        merge_bytes=@allocated(merge_term()), state_bytes=@allocated(state()),
        assembly_bytes=@allocated(assembly())))
end
profile_assembly(32)
profile_assembly(512)
