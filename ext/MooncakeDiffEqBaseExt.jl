module MooncakeDiffEqBaseExt

using DiffEqBase, Mooncake

Mooncake.@from_rrule(
    Mooncake.MinimalCtx,
    Tuple{
        typeof(DiffEqBase.solve_up),
        DiffEqBase.AbstractDEProblem,
        Union{Nothing, DiffEqBase.AbstractSensitivityAlgorithm},
        Any,
        Any,
        Any,
    },
    true,
)

Mooncake.@zero_adjoint Mooncake.MinimalCtx Tuple{typeof(DiffEqBase.numargs), Any}

end
