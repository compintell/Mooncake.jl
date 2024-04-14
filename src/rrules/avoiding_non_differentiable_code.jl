# Avoid troublesome bitcast magic -- we can't handle converting from pointer to UInt,
# because we drop the gradient, because the tangent type of integers is NoTangent.
# https://github.com/JuliaLang/julia/blob/9f9e989f241fad1ae03c3920c20a93d8017a5b8f/base/pointer.jl#L282
@is_primitive MinimalCtx Tuple{typeof(Base.:(+)), Ptr, Integer}
function rrule!!(::CoDual{typeof(Base.:(+))}, x::CoDual{<:Ptr}, y::CoDual{<:Integer})
    pb!! = NoPullback((NoRvsData(), NoRvsData(), NoRvsData()))
    return CoDual(primal(x) + primal(y), tangent(x) + primal(y)), pb!!
end

@is_primitive MinimalCtx Tuple{typeof(randn), Xoshiro, Vararg}
function rrule!!(::CoDual{typeof(randn)}, rng::CoDual{Xoshiro}, args::CoDual...)
    x = randn(primal(rng), map(primal, args)...)
    return CoDual(x, zero(x)), NoPullback()
end

function generate_hand_written_rrule!!_test_cases(
    rng_ctor, ::Val{:avoiding_non_differentiable_code}
)
    _x = Ref(5.0)
    _dx = Ref(4.0)
    test_cases = Any[
        # Rules to avoid pointer type conversions.
        (
            true,
            :stability,
            nothing,
            +,
            CoDual(
                bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
            ),
            2,
        ),

        # Rule to avoid llvmcall
        (true, :stability, nothing, randn, Xoshiro(1)),
        (true, :stability, nothing, randn, Xoshiro(1), 2),
        (true, :stability, nothing, randn, Xoshiro(1), 3, 2),
    ]
    memory = Any[_x, _dx]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(
    rng_ctor, ::Val{:avoiding_non_differentiable_code},
)
    return Any[], Any[]
end
