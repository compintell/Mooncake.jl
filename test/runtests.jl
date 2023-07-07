using
    ChainRulesCore,
    FiniteDifferences,
    LinearAlgebra,
    Random,
    Taped,
    Test,
    Umlaut

using Taped: CoDual, to_reverse_mode_ad, Tangent, _wrap_field

include("test_resources.jl")

@testset "Taped.jl" begin
    include("tracing.jl")
    include("tangents.jl")
    include("reverse_mode_ad.jl")
end
