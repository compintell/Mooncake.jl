module TapirDynamicPPLExt

using DynamicPPL: DynamicPPL, istrans
using Tapir: Tapir

using Tapir: DefaultCtx, CoDual, simple_zero_adjoint

# This is purely an optimisation.
Tapir.@is_primitive DefaultCtx Tuple{typeof(istrans), Vararg}
Tapir.rrule!!(f::CoDual{typeof(istrans)}, x::CoDual...) = simple_zero_adjoint(f, x...)

end # module
