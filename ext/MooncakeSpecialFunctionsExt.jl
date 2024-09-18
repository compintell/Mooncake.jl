module MooncakeSpecialFunctionsExt

    using SpecialFunctions, Mooncake

    import Mooncake: @from_rrule, DefaultCtx

    @from_rrule DefaultCtx Tuple{typeof(airyai), Float64}
    @from_rrule DefaultCtx Tuple{typeof(airyaix), Float64}
    @from_rrule DefaultCtx Tuple{typeof(erfc), Float64}
    @from_rrule DefaultCtx Tuple{typeof(erfcx), Float64}
end
