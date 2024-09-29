module MooncakeLuxLibExt

using LuxLib, Random, Mooncake
using Base: IEEEFloat

import LuxLib.Impl: matmul, matmuladd, fused_dense
import Mooncake: @from_rrule, DefaultCtx, MooncakeInterpreter

@from_rrule(DefaultCtx, Tuple{typeof(matmul), Array{P}, Array{P}} where {P<:IEEEFloat})
@from_rrule(
    DefaultCtx,
    Tuple{typeof(matmuladd), Array{P}, Array{P}, Vector{P}} where {P<:IEEEFloat},
)

# Unfused version of `fused_dense`, which `build_rrule` makes use of.
function unfused_dense(
    opmode,
    act::F,
    weight::AbstractMatrix,
    x::AbstractMatrix,
    b::LuxLib.Optional{<:AbstractVector},
) where {F}
    return bias_activation(act, matmul(opmode, weight, x), b)
end

# function Mooncake.build_rrule(interp::MooncakeInterpreter, sig_or_mi; kwargs...)
#     return Mooncake.build
# end

end
