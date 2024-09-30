module MooncakeLuxLibExt

using LuxLib, Random, Mooncake
using Base: IEEEFloat
using Base.Experimental: @overlay

import LuxLib.Impl: matmul, matmuladd
import Mooncake: @from_rrule, DefaultCtx, MooncakeInterpreter, mooncake_method_table

@from_rrule(DefaultCtx, Tuple{typeof(matmul), Array{P}, Array{P}} where {P<:IEEEFloat})
@from_rrule(
    DefaultCtx,
    Tuple{typeof(matmuladd), Array{P}, Array{P}, Vector{P}} where {P<:IEEEFloat},
)

# Re-implement a bunch of methods to ensure that Mooncake can differentiate them.
@overlay mooncake_method_table function LuxLib.Impl.fused_dense(
    opmode,
    act::F,
    weight::AbstractMatrix,
    x::AbstractMatrix,
    b::LuxLib.Optional{<:AbstractVector},
) where {F}
    return bias_activation(act, matmul(weight, x), b)
end

@overlay mooncake_method_table function LuxLib.Impl.bias_activation_loop!(
    y::AbstractArray{yT, 3}, σ::F, x::AbstractArray{xT, 3}, bias::AbstractVector
) where {F, xT, yT}
    return LuxLib.Impl.bias_activation_simd_loop!(y, σ, x, bias)
end

@overlay mooncake_method_table function LuxLib.Impl.activation_loop!(
    y::AbstractArray, σ::F, x::AbstractArray
) where {F}
    return LuxLib.Impl.activation_simd_loop!(y, σ, x)
end

@overlay mooncake_method_table function LuxLib.Impl.fused_conv(
    ::LuxLib.Impl.AbstractInternalArrayOpMode,
    act::F,
    weight::AbstractArray{wT, N},
    x::AbstractArray{xT, N},
    bias::LuxLib.Optional{<:AbstractVector},
    cdims::LuxLib.Impl.ConvDims,
) where {F, wT, xT, N}
    return LuxLib.Impl.bias_activation(act, LuxLib.Impl.conv(x, weight, cdims), bias)
end

# IMPORT SLEEFPirates RULES! Use a loop.

end
