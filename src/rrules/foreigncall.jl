
#
# Rules to handle / avoid foreigncall nodes
#

isprimitive(::RMC, ::typeof(Base.allocatedinline), ::Type) = true
function rrule!!(::CoDual{typeof(Base.allocatedinline)}, T::CoDual{<:Type})
    return CoDual(Base.allocatedinline(primal(T)), NoTangent()), NoPullback()
end

isprimitive(::RMC, ::typeof(pointer_from_objref), x) = true
function rrule!!(::CoDual{typeof(pointer_from_objref)}, x)
    y = CoDual(
        pointer_from_objref(primal(x)),
        bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(shadow(x))),
    )
    return y, NoPullback()
end

isprimitive(::RMC, ::typeof(Base.unsafe_pointer_to_objref), x::Ptr) = true
function rrule!!(::CoDual{typeof(Base.unsafe_pointer_to_objref)}, x::CoDual{<:Ptr})
    dx_ref = unsafe_pointer_to_objref(shadow(x))
    _y = unsafe_pointer_to_objref(primal(x))
    dy = build_tangent(typeof(_y), dx_ref.x)
    return CoDual(_y, dy), NoPullback()
end

function isprimitive(
    ::RMC, ::Type{Array{T, N}}, ::typeof(undef), ::Vararg{Int, N}
) where {T,N}
    return true
end
function rrule!!(
    ::CoDual{Type{Array{T, N}}}, ::CoDual{typeof(undef)}, m::Vararg{CoDual{Int}, N}
) where {T, N}
    _m = map(primal, m)
    x = CoDual(Array{T, N}(undef, _m...), Array{tangent_type(T), N}(undef, _m...))
    return x, NoPullback()
end

function isprimitive(
    ::RMC, ::Type{Array{T, N}}, ::typeof(undef), ::NTuple{N, Int}
) where {T, N}
    return true
end
function rrule!!(
    ::CoDual{Type{Array{T, N}}}, ::CoDual{typeof(undef)}, m::CoDual{NTuple{N, Int}},
) where {T, N}
    _m = primal(m)
    x = CoDual(Array{T, N}(undef, _m), Array{tangent_type(T), N}(undef, _m))
    return x, NoPullback()
end

isprimitive(::RMC, ::typeof(Base._growend!), a::Vector, delta::Integer) = true
function rrule!!(
    ::CoDual{typeof(Base._growend!)}, a::CoDual{<:Vector}, delta::CoDual{<:Integer},
)
    _d = primal(delta)
    _a = primal(a)
    Base._growend!(_a, _d)
    Base._growend!(shadow(a), _d)
    function _growend!_pullback!!(dy, df, da, ddelta)
        Base._deleteend!(_a, _d)
        Base._deleteend!(da, _d)
        return df, da, ddelta
    end
    return CoDual(nothing, zero_tangent(nothing)), _growend!_pullback!!
end

isprimitive(::RMC, ::typeof(copy), ::Array) = true
function rrule!!(::CoDual{typeof(copy)}, a::CoDual{<:Array})
    y = CoDual(copy(primal(a)), copy(shadow(a)))
    copy_pullback!!(dy, df, dx) = df, increment!!(dx, dy)
    return y, copy_pullback!!
end

isprimitive(::RMC, ::typeof(typeintersect), a, b) = true
function rrule!!(::CoDual{typeof(typeintersect)}, @nospecialize(a), @nospecialize(b))
    y = typeintersect(primal(a), primal(b))
    return CoDual(y, zero_tangent(y)), NoPullback()
end

isprimitive(::RMC, ::typeof(fill!), ::Union{Array{UInt8}, Array{Int8}}, x::Integer) = true
function rrule!!(
    ::CoDual{typeof(fill!)},
    a::CoDual{<:Union{Array{UInt8}, Array{Int8}}, <:Array{NoTangent}},
    x::CoDual{<:Integer},
)
    old_value = copy(primal(a))
    fill!(primal(a), primal(x))
    function fill!_pullback!!(dy, df, da, dx)
        primal(a) .= old_value
        return df, da, dx
    end
    return a, fill!_pullback!!
end

isprimitive(::RMC, ::typeof(Core.Compiler.return_type), args...) = true
function rrule!!(::CoDual{typeof(Core.Compiler.return_type)}, args...)
    y = Core.Compiler.return_type(map(primal, args)...)
    return CoDual(y, zero_tangent(y)), NoPullback()
end



#
# general foreigncall nodes
#

function rrule!!(
    ::CoDual{typeof(__foreigncall__)},
    ::CoDual{Val{:jl_array_ptr}},
    ::CoDual{Val{Ptr{T}}},
    ::CoDual{Tuple{Val{Any}}},
    ::CoDual, # nreq
    ::CoDual, # calling convention
    a::CoDual{<:Array{T}, <:Array{V}}
) where {T, V}
    y = CoDual(
        ccall(:jl_array_ptr, Ptr{T}, (Any, ), primal(a)),
        ccall(:jl_array_ptr, Ptr{V}, (Any, ), shadow(a)),
    )
    return y, NoPullback()
end

function rrule!!(
    ::CoDual{typeof(__foreigncall__)},
    ::CoDual{Val{:memmov}},
    ::CoDual{Val{Ptr{Nothing}}},
    ::CoDual{Tuple{Val{Ptr{Nothing}}, Val{Ptr{Nothing}}, Val{UInt64}}},
    ::CoDual, # nreq
    ::CoDual, # calling convention
    dest::CoDual,
    src::CoDual,
    n::CoDual,
    args...
)
    T_in = (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t)
    y = CoDual(
        ccall(:memmov, Ptr{Cvoid}, T_in, primal(dest), primal(src), primal(n)),
        ccall(:memmov, Ptr{Cvoid}, T_in, shadow(dest), shadow(src), primal(n)),
    )
    function memmov_pb!!(_, d1, d2, d3, d4, d5, ddest, dsrc, dn, dargs...)
        
        return d1, d2, d3, d4, d5, ddest, dsrc, dn, dargs...
    end
    return y, memmov_pb!!
end

# :memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
#           dest, src, n * aligned_sizeof(T)
# ccall(:memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), dst, src, n)

for name in [
    :(:jl_alloc_array_1d), :(:jl_alloc_array_2d), :(:jl_alloc_array_3d), :(:jl_new_array),
    :(:jl_array_grow_end), :(:jl_array_del_end), :(:jl_array_copy),
    :(:jl_type_intersection), :(:memset), :(:jl_get_tls_world_age),
]
    @eval function rrule!!(::CoDual{typeof(__foreigncall__)}, ::CoDual{Val{$name}}, args...)
        nm = $name
        throw(error(
            "AD has hit a :($nm) ccall. This should not happen. " *
            "Please open an issue with a minimal working example in order to reproduce. ",
            "This is true unless you have intentionally written a ccall to :$(nm), ",
            "in which case you must write a :foreigncall rule."
        ))
    end
end
