# Test case with isbits data.

bleh(x::Float64, y::Int) = x * y

function ChainRulesCore.rrule(::typeof(bleh), x::Float64, y::Int)
    return x * y, dz -> (ChainRulesCore.NoTangent(), dz * y, ChainRulesCore.NoTangent())
end

Tapir.@from_rrule DefaultCtx Tuple{typeof(bleh), Float64, Int}

# Test case with heap-allocated data.

test_sum(x) = sum(x)

function ChainRulesCore.rrule(::typeof(test_sum), x::AbstractArray{<:Real})
    test_sum_pb(dy::Real) = ChainRulesCore.NoTangent(), fill(dy, size(x))
    return test_sum(x), test_sum_pb
end

Tapir.@from_rrule DefaultCtx Tuple{typeof(test_sum), Array{<:Base.IEEEFloat}}

@testset "chain_rules_macro" begin
    @testset "to_cr_tangent and to_tapir_tangent" for (t, t_cr) in Any[
        (5.0, 5.0),
        (ones(5), ones(5)),
        (NoTangent(), ChainRulesCore.NoTangent()),
    ]
        @test Tapir.to_cr_tangent(t) == t_cr
        @test Tapir.to_tapir_tangent(t_cr) == t
        @test Tapir.to_tapir_tangent(Tapir.to_cr_tangent(t)) == t
        @test Tapir.to_cr_tangent(Tapir.to_tapir_tangent(t_cr)) == t_cr
    end
    @testset "rules" begin
        Tapir.TestUtils.test_rule(Xoshiro(1), bleh, 5.0, 4; perf_flag=:stability)
        Tapir.TestUtils.test_rule(Xoshiro(1), test_sum, ones(5); perf_flag=:stability)
    end
end
