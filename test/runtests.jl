using
    ReverseDiff,
    Taped,
    Test,
    Umlaut

include("test_resources.jl")

@testset "Taped.jl" begin
    include("tracing.jl")
    include("algorithmic_differentiation.jl")
end
