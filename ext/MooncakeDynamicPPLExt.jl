module MooncakeDynamicPPLExt

using DynamicPPL: DynamicPPL, istrans
using Mooncake: Mooncake

# This is purely an optimisation.
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(istrans), Vararg}

end # module
