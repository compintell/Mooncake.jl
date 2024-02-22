module TapedSpecialFunctionsExt

    using SpecialFunctions, Taped

    import Taped: @from_rrule, DefaultCtx

    @from_rrule DefaultCtx Tuple{typeof(airyai), Float64}
    @from_rrule DefaultCtx Tuple{typeof(airyaix), Float64}
    @from_rrule DefaultCtx Tuple{typeof(erfc), Float64}
end
