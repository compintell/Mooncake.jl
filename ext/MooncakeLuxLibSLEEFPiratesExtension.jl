module MooncakeLuxLibSLEEFPiratesExtension

using LuxLib, Mooncake, SLEEFPirates
using Base: IEEEFloat
using Mooncake: @from_rrule, DefaultCtx

@static if VERSION >= v"1.11"

    # Workaround for package load order problems. See
    # https://github.com/JuliaLang/julia/issues/56204#issuecomment-2419553167 for more context.
    function __init__()
        Base.generating_output() && return nothing

        for f in Any[
            LuxLib.NNlib.sigmoid_fast,
            LuxLib.NNlib.softplus,
            LuxLib.NNlib.logsigmoid,
            LuxLib.NNlib.swish,
            LuxLib.NNlib.lisht,
            Base.tanh,
            LuxLib.NNlib.tanh_fast,
        ]
            f_fast = LuxLib.Impl.sleefpirates_fast_act(f)
            @eval @from_rrule DefaultCtx Tuple{typeof($f_fast),IEEEFloat}
            @eval @from_rrule(
                DefaultCtx,
                Tuple{
                    typeof(Broadcast.broadcasted),
                    typeof($f_fast),
                    Union{IEEEFloat,Array{<:IEEEFloat}},
                },
            )
        end
    end

else
    for f in Any[
        LuxLib.NNlib.sigmoid_fast,
        LuxLib.NNlib.softplus,
        LuxLib.NNlib.logsigmoid,
        LuxLib.NNlib.swish,
        LuxLib.NNlib.lisht,
        Base.tanh,
        LuxLib.NNlib.tanh_fast,
    ]
        f_fast = LuxLib.Impl.sleefpirates_fast_act(f)
        @eval @from_rrule DefaultCtx Tuple{typeof($f_fast),IEEEFloat}
        @eval @from_rrule(
            DefaultCtx,
            Tuple{
                typeof(Broadcast.broadcasted),
                typeof($f_fast),
                Union{IEEEFloat,Array{<:IEEEFloat}},
            },
        )
    end
end

end
