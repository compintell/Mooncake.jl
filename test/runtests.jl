using Distributions, ReverseDiff, Taped, Test, Umlaut

using Taped: Dual, Shadow, to_forwards_mode_ad, to_reverse_mode_ad, assume

include("test_resources.jl")

@testset "Taped.jl" begin
    include("tracing.jl")
    include("is_pure.jl")
    include("vmap.jl")
    include("forwards_mode_ad.jl")
    include("reverse_mode_ad.jl")
    include("logpdf.jl")
end
