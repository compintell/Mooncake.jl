module MooncakeDynamicPPLExt

if isdefined(Base, :get_extension)
    using DynamicPPL: DynamicPPL, istrans
    using Mooncake: Mooncake
else
    using ..DynamicPPL: DynamicPPL, istrans
    using ..Mooncake: Mooncake
end

using Mooncake: DefaultCtx, CoDual, simple_zero_adjoint

# This is purely an optimisation.
Mooncake.@is_primitive DefaultCtx Tuple{typeof(istrans), Vararg}
Mooncake.rrule!!(f::CoDual{typeof(istrans)}, x::CoDual...) = simple_zero_adjoint(f, x...)

end # module
