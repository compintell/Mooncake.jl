module ToolsForRulesResources

# Note: do not `using Mooncake` in this module to ensure that all of the macros work
# correctly if `Mooncake` is not in scope.
using ChainRulesCore, LinearAlgebra
using Base: IEEEFloat
using Mooncake: @mooncake_overlay, @zero_adjoint, @from_rrule, MinimalCtx, DefaultCtx

local_function(x) = 5x
overlay_tester(x) = 2x
@mooncake_overlay overlay_tester(x) = local_function(x)

zero_tester(x) = 0
@zero_adjoint MinimalCtx Tuple{typeof(zero_tester),Float64}

vararg_zero_tester(x...) = 0
@zero_adjoint MinimalCtx Tuple{typeof(vararg_zero_tester),Vararg}

# Test case with isbits data.

bleh(x::Float64, y::Int) = x * y

function ChainRulesCore.rrule(::typeof(bleh), x::Float64, y::Int)
    return x * y, dz -> (ChainRulesCore.NoTangent(), dz * y, ChainRulesCore.NoTangent())
end

@from_rrule DefaultCtx Tuple{typeof(bleh),Float64,Int} false

# Test case with heap-allocated input.

test_sum(x) = sum(x)

function ChainRulesCore.rrule(::typeof(test_sum), x::AbstractArray{<:Real})
    test_sum_pb(dy::Real) = ChainRulesCore.NoTangent(), fill(dy, size(x))
    return test_sum(x), test_sum_pb
end

@from_rrule DefaultCtx Tuple{typeof(test_sum),Array{<:Base.IEEEFloat}} false

# Test case with heap-allocated output.

test_scale(x::Real, y::AbstractVector{<:Real}) = x * y

function ChainRulesCore.rrule(::typeof(test_scale), x::Real, y::AbstractVector{<:Real})
    function test_scale_pb(dout::AbstractVector{<:Real})
        return ChainRulesCore.NoTangent(), dot(dout, y), dout * x
    end
    return x * y, test_scale_pb
end

@from_rrule(
    DefaultCtx, Tuple{typeof(test_scale),Base.IEEEFloat,Vector{<:Base.IEEEFloat}}, false
)

# Test case with non-differentiable type as output.

test_nothing() = nothing

function ChainRulesCore.rrule(::typeof(test_nothing))
    test_nothing_pb(::ChainRulesCore.NoTangent) = (ChainRulesCore.NoTangent(),)
    return nothing, test_nothing_pb
end

@from_rrule DefaultCtx Tuple{typeof(test_nothing)} false

# Test case in which ChainRulesCore returns a tangent which is of the "wrong" type from the
# perspective of Mooncake.jl. In this instance, some kind of error should be thrown, rather
# than it being possible for the error to propagate.

test_bad_rdata(x::Real) = 5x

function ChainRulesCore.rrule(::typeof(test_bad_rdata), x::Float64)
    test_bad_rdata_pb(dy::Float64) = ChainRulesCore.NoTangent(), Float32(dy * 5)
    return 5x, test_bad_rdata_pb
end

@from_rrule DefaultCtx Tuple{typeof(test_bad_rdata),Float64} false

# Test case for rule with diagonal dispatch.
test_add(x, y) = x + y
function ChainRulesCore.rrule(::typeof(test_add), x, y)
    test_add_pb(dout) = ChainRulesCore.NoTangent(), dout, dout
    return x + y, test_add_pb
end
@from_rrule DefaultCtx Tuple{typeof(test_add),T,T} where {T<:IEEEFloat} false

# Test case for rule with non-differentiable kwargs.
test_kwargs(x; y::Bool=false) = y ? x : 2x

function ChainRulesCore.rrule(::typeof(test_kwargs), x::Float64; y::Bool=false)
    test_kwargs_pb(dz::Float64) = ChainRulesCore.NoTangent(), y ? dz : 2dz
    return y ? x : 2x, test_kwargs_pb
end

@from_rrule(DefaultCtx, Tuple{typeof(test_kwargs),Float64}, true)

# Test case for rule with differentiable types used in a non-differentiable way.
test_kwargs_conditional(x; y::Float64=1.0) = y > 0 ? x : 2x

function ChainRulesCore.rrule(::typeof(test_kwargs_conditional), x::Float64; y::Float64=1.0)
    test_kwargs_cond_pb(dz::Float64) = ChainRulesCore.NoTangent(), y > 0 ? dz : 2dz
    return y > 0 ? x : 2x, test_kwargs_cond_pb
end

@from_rrule(DefaultCtx, Tuple{typeof(test_kwargs_conditional),Float64}, true)

end

@testset "tools_for_rules" begin
    @testset "mooncake_overlay" begin
        f = ToolsForRulesResources.overlay_tester
        rule = Mooncake.build_rrule(Tuple{typeof(f),Float64})
        @test value_and_gradient!!(rule, f, 5.0) == (15.0, (NoTangent(), 3.0))
    end
    @testset "zero_adjoint" begin
        f_zero = ToolsForRulesResources
        test_rule(
            sr(123),
            ToolsForRulesResources.zero_tester,
            5.0;
            is_primitive=true,
            perf_flag=:stability_and_allocs,
        )
        test_rule(
            sr(123),
            ToolsForRulesResources.vararg_zero_tester,
            5.0,
            4.0;
            is_primitive=true,
            perf_flag=:stability_and_allocs,
        )
    end
    @testset "chain_rules_macro" begin
        @testset "to_cr_tangent" for (t, t_cr) in Any[
            (5.0, 5.0),
            (ones(5), ones(5)),
            (NoTangent(), ChainRulesCore.NoTangent()),
            ((5.0, 4.0), ChainRulesCore.Tangent{Any}(5.0, 4.0)),
            ([ones(5), NoTangent()], [ones(5), ChainRulesCore.NoTangent()]),
            (
                Tangent((a=5.0, b=NoTangent())),
                ChainRulesCore.Tangent{Any}(; a=5.0, b=ChainRulesCore.NoTangent()),
            ),
            (
                MutableTangent((a=5.0, b=ones(3))),
                ChainRulesCore.Tangent{Any}(; a=5.0, b=ones(3)),
            ),
        ]
            @test Mooncake.to_cr_tangent(t) == t_cr
        end
        @testset "rules: $(typeof(fargs))" for fargs in Any[
            (ToolsForRulesResources.bleh, 5.0, 4),
            (ToolsForRulesResources.test_sum, ones(5)),
            (ToolsForRulesResources.test_scale, 5.0, randn(3)),
            (ToolsForRulesResources.test_nothing,),
            (Core.kwcall, (y=true,), ToolsForRulesResources.test_kwargs, 5.0),
            (Core.kwcall, (y=false,), ToolsForRulesResources.test_kwargs, 5.0),
            (ToolsForRulesResources.test_kwargs, 5.0),
            (Core.kwcall, (y=-1.0,), ToolsForRulesResources.test_kwargs_conditional, 5.0),
            (Core.kwcall, (y=1.0,), ToolsForRulesResources.test_kwargs_conditional, 5.0),
            (ToolsForRulesResources.test_kwargs_conditional, 5.0),
        ]
            test_rule(sr(1), fargs...; perf_flag=:stability, is_primitive=true)
        end
        @testset "bad rdata" begin
            f = ToolsForRulesResources.test_bad_rdata
            out, pb!! = Mooncake.rrule!!(zero_fcodual(f), zero_fcodual(3.0))
            @test_throws MethodError pb!!(5.0)
        end
    end
end
