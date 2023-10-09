
#
# Rules to handle / avoid foreigncall nodes
#

isprimitive(::RMC, ::typeof(objectid), @nospecialize(x)) = true
function rrule!!(::CoDual{typeof(objectid)}, @nospecialize(x))
    return CoDual(objectid(primal(x)), NoTangent()), NoPullback()
end

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

function _increment_pointer!(x::Ptr{T}, y::Ptr{T}, N::Integer) where {T}
    increment!!(unsafe_wrap(Vector{T}, x, N), unsafe_wrap(Vector{T}, y, N))
    return x
end

# unsafe_copyto! is the only function in Julia that appears to rely on a ccall to `memmove`.
# Since we can't differentiate `memmove` (due to a lack of type information), it is
# necessary to work with `unsafe_copyto!` instead.
isprimitive(::RMC, ::typeof(unsafe_copyto!), args...) = true
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)}, dest::CoDual{Ptr{T}}, src::CoDual{Ptr{T}}, n::CoDual
) where {T}
    _n = primal(n)

    # Record values that will be overwritten.
    dest_copy = Vector{T}(undef, _n)
    ddest_copy = Vector{T}(undef, _n)
    unsafe_copyto!(pointer(dest_copy), primal(dest), _n)
    unsafe_copyto!(pointer(ddest_copy), shadow(dest), _n)

    # Run primal computation.
    unsafe_copyto!(primal(dest), primal(src), _n)
    unsafe_copyto!(shadow(dest), shadow(src), _n)

    function unsafe_copyto!_pb!!(_, df, ddest, dsrc, dn)

        # Increment dsrc.
        dsrc = _increment_pointer!(dsrc, ddest, _n)

        # Restore initial state.
        unsafe_copyto!(primal(dest), pointer(dest_copy), _n)
        unsafe_copyto!(shadow(dest), pointer(ddest_copy), _n)

        return df, ddest, dsrc, dn
    end
    return dest, unsafe_copyto!_pb!!
end

# same structure as the previous method, just without the pointers.
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)},
    dest::CoDual{<:Array{T}},
    doffs::CoDual,
    src::CoDual{<:Array{T}},
    soffs::CoDual,
    n::CoDual,
) where {T}
    _n = primal(n)

    # Record values that will be overwritten.
    _doffs = primal(doffs)
    dest_idx = _doffs:_doffs + _n - 1
    _soffs = primal(soffs)
    dest_copy = primal(dest)[dest_idx]
    ddest_copy = shadow(dest)[dest_idx]

    # Run primal computation.
    unsafe_copyto!(primal(dest), _doffs, primal(src), _soffs, _n)
    unsafe_copyto!(shadow(dest), _doffs, shadow(src), _soffs, _n)

    function unsafe_copyto_pb!!(_, df, ddest, ddoffs, dsrc, dsoffs, dn)

        # Increment dsrc.
        src_idx = _soffs:_soffs + _n - 1
        dsrc[src_idx] .= increment!!.(view(dsrc, src_idx), view(ddest, dest_idx))

        # Restore initial state.
        primal(dest)[dest_idx] .= dest_copy
        shadow(dest)[dest_idx] .= ddest_copy

        return df, ddest, ddoffs, dsrc, dsoffs, dn
    end

    return dest, unsafe_copyto_pb!!
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
    ::CoDual{Val{:jl_reshape_array}},
    ::CoDual{Val{Array{P, M}}},
    ::CoDual{Tuple{Val{Any}, Val{Any}, Val{Any}}},
    ::CoDual, # nreq
    ::CoDual, # calling convention
    x::CoDual{Type{Array{P, M}}},
    a::CoDual{Array{P, N}, Array{T, N}},
    dims::CoDual,
) where {P, T, M, N}
    d = primal(dims)
    y = CoDual(
        ccall(:jl_reshape_array, Array{P, M}, (Any, Any, Any), Array{P, M}, primal(a), d),
        ccall(:jl_reshape_array, Array{T, M}, (Any, Any, Any), Array{T, M}, shadow(a), d),
    )
    return y, NoPullback()
end

for name in [
    :(:jl_alloc_array_1d), :(:jl_alloc_array_2d), :(:jl_alloc_array_3d), :(:jl_new_array),
    :(:jl_array_grow_end), :(:jl_array_del_end), :(:jl_array_copy), :(:jl_object_id),
    :(:jl_type_intersection), :(:memset), :(:jl_get_tls_world_age), :(:memmove),
]
    @eval function rrule!!(::CoDual{typeof(__foreigncall__)}, ::CoDual{Val{$name}}, args...)
        nm = $name
        throw(error(
            "AD has hit a :($nm) ccall. This should not happen. " *
            "Please open an issue with a minimal working example in order to reproduce. ",
            "This is true unless you have intentionally written a ccall to :$(nm), ",
            "in which case you must write a :foreigncall rule. It may not be possible ",
            "to implement a :foreigncall rule if too much type information has been lost ",
            "in which case your only recourse is to write a rule for whichever Julia ",
            "function calls this one (and retains enough type information).",
        ))
    end
end
