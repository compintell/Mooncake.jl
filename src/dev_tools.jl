"""
    primal_ir(sig; interp=get_interpreter())::IRCode

Get the `Core.Compiler.IRCode` associated to `sig` from which the a rule can be derived.
Roughly equivalent to `Base.code_ircode_by_type(sig; interp)`.
"""
function primal_ir(sig; interp=get_interpreter())::IRCode
    return generate_ir(interp, sig).primal_ir
end

"""
    fwd_ir(sig; interp=get_interpreter(), debug_mode=false, do_inline=true)::IRCode

Generate the `Core.Compiler.IRCode` used to construct the forwards-pass of AD. Take a look
at how `build_rrule` makes use of `generate_ir` to see exactly how this is used in practice.

# Arguments
- `sig::Type{<:Tuple}`: the signature of the call to be differentiated.

# Keyword Arguments
- `interp`: the interpreter to use to obtain the primal IR.
- `debug_mode::Bool`: whether the generated IR should make use of Mooncake's debug mode.
- `do_inline::Bool`: whether to apply an inlining pass prior to returning the ir generated
    by this function. This is `true` by default, but setting it to `false` can sometimes be
    helpful if you need to understand what function calls are generated in order to perform
    AD, before lots of it gets inlined away.
"""
function fwd_ir(
    sig; interp=get_interpreter(), debug_mode::Bool=false, do_inline::Bool=true
)::IRCode
    return generate_ir(interp, sig; debug_mode, do_inline).fwd_ir
end

"""
    rvs_ir(sig; interp=get_interpreter(), debug_mode=false, do_inline=true)::IRCode

Generate the `Core.Compiler.IRCode` used to construct the reverse-pass of AD. Take a look
at how `build_rrule` makes use of `generate_ir` to see exactly how this is used in practice.

# Arguments
- `sig::Type{<:Tuple}`: the signature of the call to be differentiated.

# Keyword Arguments
- `interp`: the interpreter to use to obtain the primal IR.
- `debug_mode::Bool`: whether the generated IR should make use of Mooncake's debug mode.
- `do_inline::Bool`: whether to apply an inlining pass prior to returning the ir generated
    by this function. This is `true` by default, but setting it to `false` can sometimes be
    helpful if you need to understand what function calls are generated in order to perform
    AD, before lots of it gets inlined away.
"""
function rvs_ir(
    sig; interp=get_interpreter(), debug_mode::Bool=false, do_inline::Bool=true
)::IRCode
    return generate_ir(interp, sig; debug_mode, do_inline).rvs_ir
end
