module MooncakeLuxLibSLEEFPiratesExtension

using LuxLib, Mooncake, SLEEFPirates
using Base: IEEEFloat
using Mooncake: @from_rrule, DefaultCtx

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
