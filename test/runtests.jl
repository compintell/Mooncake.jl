using
    ReverseDiff,
    Taped,
    Test,
    Umlaut

using Taped: Dual, to_forwards_mode_ad

include("test_resources.jl")

@testset "Taped.jl" begin
    include("tracing.jl")
    include("is_pure.jl")
    include("vmap.jl")
    include("forwards_mode_ad.jl")
    # include("algorithmic_differentiation.jl")
end
