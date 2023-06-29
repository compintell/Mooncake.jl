using
    BenchmarkTools,
    ChainRulesCore,
    Distributions,
    FiniteDifferences,
    FunctionWrappers,
    Random,
    ReverseDiff,
    Taped,
    Test,
    Umlaut

using FunctionWrappers: FunctionWrapper

using Taped:
    Dual,
    CoDual,
    to_forwards_mode_ad,
    to_reverse_mode_ad,
    assume,
    Instruction,
    accelerate,
    execute!,
    AcceleratedTape,
    Tangent

import Taped: InplaceData

include("test_resources.jl")

@testset "Taped.jl" begin
    # include("tracing.jl")
    # include("is_pure.jl")
    # include("vmap.jl")
    # include("forwards_mode_ad.jl")
    include("tangents.jl")
    include("reverse_mode_ad.jl")
    # include("logpdf.jl")
    # include("inplace.jl")
    # include("accelerate_tape.jl")
end
