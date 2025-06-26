module MooncakeDiffEqBaseExt

using DiffEqBase, Mooncake
using DiffEqBase: SciMLBase
using SciMLBase: ADOriginator, MooncakeOriginator
println("mooncakke diff eqbase ext loaded !")
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

# @from_rrule(
#     MinimalCtx,
#     Tuple{
#         typeof(DiffEqBase.solve_up),
#         DiffEqBase.AbstractDEProblem,
#         Union{Nothing,DiffEqBase.AbstractSensitivityAlgorithm},
#         Any,
#         Any,
#         Any,
#     },
#     true,
# )

# # Dispatch for auto-alg
# @from_rrule(
#     MinimalCtx,
#     Tuple{
#         typeof(DiffEqBase.solve_up),
#         DiffEqBase.AbstractDEProblem,
#         Union{Nothing,DiffEqBase.AbstractSensitivityAlgorithm},
#         Any,
#         Any,
#     },
#     true,
# )

# @zero_adjoint MinimalCtx Tuple{typeof(DiffEqBase.numargs),Any}
# @mooncake_overlay DiffEqBase.set_mooncakeoriginator_if_mooncake(x::ADOriginator) =
#     MooncakeOriginator

@is_primitive MinimalCtx Tuple{
    typeof(DiffEqBase.set_mooncakeoriginator_if_mooncake),SciMLBase.ChainRulesOriginator
}
function rrule!!(
    f::CoDual{typeof(DiffEqBase.set_mooncakeoriginator_if_mooncake)},
    X::CoDual{SciMLBase.ChainRulesOriginator},
)
    return zero_fcodual(SciMLBase.MooncakeOriginator()), NoPullback(f, X)
end
# o method matching rrule!!(::CoDual{typeof(DiffEqBase.set_mooncakeoriginator_if_mooncake), Mooncake.NoFData}, ::CoDual{SciMLBase.ChainRulesOriginator, Mooncake.NoFData})
# Mooncake.DerivedRule{
#     Tuple{
#         typeof(DiffEqBase.set_mooncakeoriginator_if_mooncake),SciMLBase.ChainRulesOriginator
#     },
#     Tuple{
#         Mooncake.CoDual{
#             typeof(DiffEqBase.set_mooncakeoriginator_if_mooncake),Mooncake.NoFData
#         },
#         Mooncake.CoDual{SciMLBase.ChainRulesOriginator,Mooncake.NoFData},
#     },
#     Mooncake.CoDual{Type{SciMLBase.MooncakeOriginator},Mooncake.NoFData},
#     Tuple{Mooncake.NoRData},
#     Tuple{Mooncake.NoRData,Mooncake.NoRData},
#     false,
#     Val{2},
# }
end