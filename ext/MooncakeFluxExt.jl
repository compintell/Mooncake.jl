module MooncakeFluxExt

using Mooncake, Flux
using Base: IEEEFloat
import Mooncake: DefaultCtx, rrule!!, @is_primitive, CoDual, zero_fcodual, NoRData

@is_primitive DefaultCtx Tuple{
    typeof(Flux.Losses.mse),Array{P},Array{P}
} where {P<:IEEEFloat}

function rrule!!(
    ::CoDual{typeof(Flux.Losses.mse)}, X::CoDual{<:Array{P}}, Y::CoDual{<:Array{P}}
) where {P<:IEEEFloat}
    norm_factor = P(2) / P(length(X.x)) .* (X.x .- Y.x)

    function flux_mse_pullback(dloss::P)
        # adjoints got by VJP reverse pass equations.
        temp = norm_factor .* dloss
        @. X.dx += temp
        @. Y.dx -= temp
        return NoRData(), NoRData(), NoRData()
    end

    # in forward pass it returns codual float64, hence coduals dx is NoFData().
    return zero_fcodual(Flux.Losses.mse(X.x, Y.x)), flux_mse_pullback
end

end
