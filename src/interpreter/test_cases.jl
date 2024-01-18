a_primitive(x) = sin(x)
non_primitive(x) = sin(x)

is_primitive(::DefaultCtx, ::Type{<:Tuple{typeof(a_primitive), Any}}) = true
is_primitive(::DefaultCtx, ::Type{<:Tuple{typeof(non_primitive), Any}}) = false

contains_primitive(x) = @inline a_primitive(x)
contains_non_primitive(x) = @inline non_primitive(x)
contains_primitive_behind_call(x) = @inline contains_primitive(x)

function to_benchmark(__rrule!!::R, df::F, dx::X) where {R, F, X}
    out, pb!! = __rrule!!(df, dx...)
    pb!!(tangent(out), tangent(df), map(tangent, dx)...)
end

function gradient(__rrule!!::R, df::F, dx::X) where {R, F, X}
    out, pb!! = __rrule!!(df, dx...)
    @assert out isa CoDual{Float64, Float64}
    return pb!!(1.0, tangent(df), map(tangent, dx)...)
end
