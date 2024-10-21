using Pkg
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DifferentiationInterface, DifferentiationInterfaceTest
import Mooncake

test_differentiation(
    [AutoMooncake(; config=nothing), AutoMooncake(; config=Mooncake.Config())];
    excluded=SECOND_ORDER, logging=true,
)
