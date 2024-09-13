module ChainRulesInteropTestResources

using ChainRulesCore, LinearAlgebra, Tapir

using Base: IEEEFloat
using Tapir: DefaultCtx, @from_rrule

# Test case with isbits data.

bleh(x::Float64, y::Int) = x * y

function ChainRulesCore.rrule(::typeof(bleh), x::Float64, y::Int)
    return x * y, dz -> (ChainRulesCore.NoTangent(), dz * y, ChainRulesCore.NoTangent())
end

@from_rrule DefaultCtx Tuple{typeof(bleh), Float64, Int}

# Test case with heap-allocated input.

test_sum(x) = sum(x)

function ChainRulesCore.rrule(::typeof(test_sum), x::AbstractArray{<:Real})
    test_sum_pb(dy::Real) = ChainRulesCore.NoTangent(), fill(dy, size(x))
    return test_sum(x), test_sum_pb
end

@from_rrule DefaultCtx Tuple{typeof(test_sum), Array{<:Base.IEEEFloat}}

# Test case with heap-allocated output.

test_scale(x::Real, y::AbstractVector{<:Real}) = x * y

function ChainRulesCore.rrule(::typeof(test_scale), x::Real, y::AbstractVector{<:Real})
    function test_scale_pb(dout::AbstractVector{<:Real})
        return ChainRulesCore.NoTangent(), dot(dout, y), dout * x
    end
    return x * y, test_scale_pb
end

@from_rrule DefaultCtx Tuple{typeof(test_scale), Base.IEEEFloat, Vector{<:Base.IEEEFloat}}

# Test case with non-differentiable type as output.

test_nothing() = nothing

function ChainRulesCore.rrule(::typeof(test_nothing))
    test_nothing_pb(::ChainRulesCore.NoTangent) = (ChainRulesCore.NoTangent(),)
    return nothing, test_nothing_pb
end

@from_rrule DefaultCtx Tuple{typeof(test_nothing)}

# Test case in which ChainRulesCore returns a tangent which is of the "wrong" type from the
# perspective of Tapir.jl. In this instance, some kind of error should be thrown, rather
# than it being possible for the error to propagate.

test_bad_rdata(x::Real) = 5x

function ChainRulesCore.rrule(::typeof(test_bad_rdata), x::Float64)
    test_bad_rdata_pb(dy::Float64) = ChainRulesCore.NoTangent(), Float32(dy * 5)
    return 5x, test_bad_rdata_pb
end

@from_rrule DefaultCtx Tuple{typeof(test_bad_rdata), Float64}

end

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
    @testset "rules: $(typeof(fargs))" for fargs in Any[
        (ChainRulesInteropTestResources.bleh, 5.0, 4),
        (ChainRulesInteropTestResources.test_sum, ones(5)),
        (ChainRulesInteropTestResources.test_scale, 5.0, randn(3)),
        (ChainRulesInteropTestResources.test_nothing,),
    ]
        test_rule(sr(1), fargs...; perf_flag=:stability, is_primitive=true)
    end
    @testset "bad rdata" begin
        f = ChainRulesInteropTestResources.test_bad_rdata
        out, pb!! = Tapir.rrule!!(zero_fcodual(f), zero_fcodual(3.0))
        @test_throws TypeError pb!!(5.0)
    end
end
