module MooncakeDiffEqBaseExt

using DiffEqBase, Mooncake
using DiffEqBase: SciMLBase
using SciMLBase: ADOriginator, MooncakeOriginator
import Mooncake:
    rrule!!,
    CoDual,
    zero_fcodual,
    NoRData,
    @is_primitive,
    @from_rrule,
    @zero_adjoint,
    @mooncake_overlay,
    DefaultCtx,
    MinimalCtx,
    NoPullback

@is_primitive MinimalCtx Tuple{
    typeof(DiffEqBase.set_mooncakeoriginator_if_mooncake),SciMLBase.ChainRulesOriginator
}
function rrule!!(
    f::CoDual{typeof(DiffEqBase.set_mooncakeoriginator_if_mooncake)},
    X::CoDual{SciMLBase.ChainRulesOriginator},
)
    return zero_fcodual(SciMLBase.MooncakeOriginator()), NoPullback(f, X)
end
end
