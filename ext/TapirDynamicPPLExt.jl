module TapirDynamicPPLExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL
    using Tapir: Tapir
else
    using ..DynamicPPL: DynamicPPL
    using ..Tapir: Tapir
end

using Tapir: DefaultCtx, CoDual, NoPullback, primal, zero_fcodual

# This is purely an optimisation.
Tapir.@is_primitive DefaultCtx Tuple{typeof(DynamicPPL.istrans), Vararg}
function Tapir.rrule!!(f::CoDual{typeof(DynamicPPL.istrans)}, x::Vararg{CoDual, N}) where {N}
    return simple_zero_adjoint(f, x...)
end

end # module
