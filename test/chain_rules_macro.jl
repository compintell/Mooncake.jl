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

Tapir.@is_primitive DefaultCtx Tuple{typeof(test_sum), Array{<:Base.IEEEFloat}}
function Tapir.rrule!!(f::CoDual{typeof(test_sum)}, x::CoDual{<:Array{<:Base.IEEEFloat}})
    return Tapir.rrule_wrapper_implementation(f, x)
end

@testset "chain_rules_macro" begin
    Tapir.TestUtils.test_rule(Xoshiro(1), bleh, 5.0, 4; perf_flag=:stability)
    Tapir.TestUtils.test_rule(Xoshiro(1), test_sum, ones(5); perf_flag=:stability)
end
