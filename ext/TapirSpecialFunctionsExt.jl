module TapirSpecialFunctionsExt

    using SpecialFunctions, Tapir

    import Tapir: @from_rrule, DefaultCtx

    @from_rrule DefaultCtx Tuple{typeof(airyai), Float64}
    @from_rrule DefaultCtx Tuple{typeof(airyaix), Float64}
    @from_rrule DefaultCtx Tuple{typeof(erfc), Float64}
end
