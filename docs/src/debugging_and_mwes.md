# Debugging and MWEs

There's a reasonable chance that you'll run into an issue with Tapir.jl at some point.
In order to debug what is going on when this happens, or to produce an MWE, it is helpful to have a convenient way to run Tapir.jl on whatever function and arguments you have which are causing problems.

We recommend making use of Tapir.jl's testing functionality to generate your test cases:

```@docs
Tapir.TestUtils.test_rule
```

This approach is convenient because it can
1. check whether AD runs at all,
1. check whether AD produces the correct answers,
1. check whether AD is performant, and
1. can be used without having to manually generate tangents.

## Example

```@meta
DocTestSetup = quote
    using Random, Tapir
end
```

For example:
```jldoctest
f(x) = Core.bitcast(Float64, x)
Tapir.TestUtils.test_rule(
    Random.Xoshiro(123), f, 3;
    is_primitive=false, perf_flag=:none, interp=Tapir.TapirInterpreter(),
)

# output
ERROR: MethodError: Cannot `convert` an object of type
  MistyClosures.MistyClosure{Core.OpaqueClosure{Tuple{Tapir.CoDual{typeof(f),NoFData},Tapir.CoDual{Int64,NoFData}},Union{}}} to an object of type
  MistyClosures.MistyClosure{Core.OpaqueClosure{Tuple{Tapir.CoDual{typeof(f),NoFData},Tapir.CoDual{Int64,NoFData}},Tapir.CoDual{Float64, NoFData}}}

Closest candidates are:
  convert(::Type{T}, !Matched::T) where T
   @ Base Base.jl:84
  (::Type{MistyClosures.MistyClosure{Toc}} where Toc<:Core.OpaqueClosure)(::Any, !Matched::Any)
   @ MistyClosures ~/.julia/packages/MistyClosures/UrOU5/src/MistyClosures.jl:7

Stacktrace:
 [1] Tapir.DerivedRule{MistyClosures.MistyClosure{Core.OpaqueClosure{Tuple{Tapir.CoDual{typeof(f), NoFData}, Tapir.CoDual{Int64, NoFData}}, Tapir.CoDual{Float64, NoFData}}}, MistyClosures.MistyClosure{Core.OpaqueClosure{Tuple{Float64}, Tuple{NoRData, NoRData}}}, Val{false}, Val{2}}(fwds_oc::MistyClosures.MistyClosure{Core.OpaqueClosure{Tuple{Tapir.CoDual{typeof(f), NoFData}, Tapir.CoDual{Int64, NoFData}}, Union{}}}, pb_oc::MistyClosures.MistyClosure{Core.OpaqueClosure{Tuple{Float64}, Tuple{NoRData, NoRData}}}, isva::Val{false}, nargs::Val{2})
   @ Tapir ~/repos/Tapir.jl/src/interpreter/s2s_reverse_mode_ad.jl:666
 [2] build_rrule(interp::Tapir.TapirInterpreter{Tapir.DefaultCtx}, sig::Type{Tuple{typeof(f), Int64}}; safety_on::Bool, silence_safety_messages::Bool)
   @ Tapir ~/repos/Tapir.jl/src/interpreter/s2s_reverse_mode_ad.jl:826
 [3] test_rule(::Any, ::Any, ::Vararg{Any}; interface_only::Bool, is_primitive::Bool, perf_flag::Symbol, interp::Tapir.TapirInterpreter{Tapir.DefaultCtx}, safety_on::Bool)
   @ Tapir.TestUtils ~/repos/Tapir.jl/src/test_utils.jl:451
 [4] top-level scope
   @ none:1
```
This particular error is caused by Tapir.jl preventing you from doing (potentially) unsafe casting.
In this particular instance, Tapir.jl just fails to compile, but in other instances other things can happen.

In any case, the point here is that `Tapir.TestUtils.test_rule` provides a convenient way to produce and report an error.



## Segfaults

These are everyone's least favourite kind of problem, and they should be _extremely_ rare in Tapir.jl.
However, if you are unfortunate enough to encounter one, please re-run your problem with the `safety_on` kwarg set to `true`.
See [Safe Mode](@ref) for more info.
In general, this will catch problems before they become segfaults, at which point the above strategy for debugging and error reporting should work well.
