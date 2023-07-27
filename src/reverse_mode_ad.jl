struct ReverseModeADContext <: TapedContext end

const RMC = ReverseModeADContext

struct CoDual{Tx, Tdx}
    x::Tx
    dx::Tdx
end

# Always sharpen the first thing if it's a type, in order to preserve dispatch possibility.
CoDual(::Type{P}, dx::NoTangent) where {P} = CoDual{Type{P}, NoTangent}(P, dx)

primal(x::CoDual) = x.x
shadow(x::CoDual) = x.dx

function verify_codual_type(::CoDual{P, T}) where {P, T}
    Tt = tangent_type(P)
    if Tt !== T
        throw(error("for primal of type $P, expected tangent of type $Tt, but found $T"))
    end
end

struct NoPullback end

@inline (::NoPullback)(dy, dx...) = dx

function rrule!!(::CoDual{typeof(verify)}, args...)
    v = verify(map(primal, args)...)
    return CoDual(v, zero_tangent(v)), NoPullback()
end

function rrule!!(::CoDual{typeof(Umlaut.__new__)}, xs...)
    y = Umlaut.__new__(map(primal, xs)...)
    P = primal(xs[1])
    dy = build_tangent(P, map(shadow, xs[2:end])...)
    function __new__pullback(dy, d__new__, df, dxs...)
        new_dxs = map((x, y) -> increment!!(x, _value(y)), dxs, dy.fields)
        return d__new__, df, new_dxs...
    end
    function __new__pullback(dy::NamedTuple, d__new__, df, dxs...)
        new_dxs = map((x, y) -> increment!!(x, _value(y)), dxs, dy)
        return d__new__, df, new_dxs...
    end
    return CoDual(y, dy), __new__pullback
end

get_fields(x::Union{Tangent, MutableTangent}) = x.fields
get_fields(x) = x

#
# Core.Builtin -- these are "primitive" functions which must have rrules because no IR
# is available.
# There is a finite number of these functions.
# Any built-ins which don't have rules defined are left as comments with their names
# in this block of code. They will inevitably get implemented as we attempt to differentiate
# more and more code.
# The list of Core.IntrinsicFunction instances is not complete.
#

# This implementation for intrinsics is not at all performant.
# Unlike regular functions, intrinsics are _instances_ of `Core.IntrinsicFunction`,
# so they all have the same type, meaning that dispatch isn't possible.
# We'll need to produce re-write rules which replace any intrinsic function with a
# regular function that we can dispatch on. This is the same strategy as employed for
# IR nodes like `new`, `splatnew`, etc, so is clearly possible.
function rrule!!(f::CoDual{<:Core.IntrinsicFunction}, x)
    _f = primal(f)
    _x = primal(x)
    if _f === arraylen
        return CoDual(arraylen(_x), NoTangent()), NoPullback()
    elseif _f === not_int
        return CoDual(not_int(_x), NoTangent()), NoPullback()
    elseif _f === neg_int
        return CoDual(neg_int(_x), NoTangent()), NoPullback()
    elseif _f === sqrt_llvm
        llvm_sqrt_pullback!!(dy, df, dx) = df, dx + dy * inv(2 * sqrt(_x))
        return CoDual(sqrt_llvm(_x), zero(_x)), llvm_sqrt_pullback!!
    elseif _f === floor_llvm
        return CoDual(floor_llvm(_x), zero(_x)), NoPullback()
    elseif _f === cttz_int
        return CoDual(cttz_int(_x), NoTangent()), NoPullback()
    else
        throw(error("unknown unary Core.IntrinsicFunction $f with argument $x"))
    end
end

# See comment above.
function rrule!!(f::CoDual{<:Core.IntrinsicFunction}, a::CoDual, b::CoDual)
    _f = primal(f)
    _a = primal(a)
    _b = primal(b)
    if _f === sitofp
        x = sitofp(_a, _b)
        return CoDual(x, zero_tangent(x)), NoPullback()
    elseif _f === sle_int
        return CoDual(sle_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === slt_int
        return CoDual(slt_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === sub_int
        return CoDual(sub_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === add_int
        return CoDual(add_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === add_float
        add_float_pb!!(c̄, ::NoTangent, ā, b̄) = NoTangent(), c̄ + ā, c̄ + b̄
        c = add_float(_a, _b)
        return CoDual(c, zero_tangent(c)), add_float_pb!!
    elseif _f === mul_float
        function mul_float_pb!!(c̄, ::NoTangent, ā, b̄)
            return NoTangent(), ā + c̄ * _b, b̄ + _a * c̄
        end
        c = mul_float(_a, _b)
        return CoDual(c, zero_tangent(c)), mul_float_pb!!
    elseif _f === eq_float
        return CoDual(eq_float(_a, _b), NoTangent()), NoPullback()
    elseif _f === bitcast
        v = bitcast(_a, _b)
        if _a <: Ptr
            if _b isa Ptr
                dv = bitcast(Ptr{tangent_type(eltype(_a))}, shadow(b))
            else
                dv = bitcast(Ptr{tangent_type()})
            end
        else
            dv = zero_tangent(v)
        end
        return CoDual(v, dv), NoPullback() # NOT SURE THAT THIS IS QUITE RIGHT
    elseif _f === mul_int
        return CoDual(mul_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === and_int
        return CoDual(and_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === or_int
        return CoDual(or_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === sext_int
        return CoDual(sext_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === lshr_int
        return CoDual(lshr_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === shl_int
        return CoDual(shl_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === trunc_int
        return CoDual(trunc_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === div_float
        _y = div_float(_a, _b)
        function div_float_pullback!!(dy, df, da, db)
            da += div_float(dy, _b)
            db -= dy * _a / _b^2
            return df, da, db
        end
        y = CoDual(_y, zero_tangent(_y)), div_float_pullback!!
    elseif _f === lt_float
        return CoDual(lt_float(_a, _b), NoTangent()), NoPullback()
    elseif _f === le_float
        return CoDual(le_float(_a, _b), NoTangent()), NoPullback()
    elseif _f === fptosi
        return CoDual(fptosi(_a, _b), NoTangent()), NoPullback()
    elseif _f === zext_int
        return CoDual(zext_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === eq_int
        return CoDual(eq_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === ashr_int
        return CoDual(ashr_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === checked_srem_int
        return CoDual(checked_srem_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === flipsign_int
        return CoDual(flipsign_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === checked_sdiv_int
        return CoDual(checked_sdiv_int(_a, _b), NoTangent()), NoPullback()
    elseif _f === checked_smul_int
        return CoDual(checked_smul_int(_a, _b), (NoTangent(), NoTangent())), NoPullback()
    elseif _f === add_ptr
        throw(error("add_ptr intrinsic hit. This should never happen. Open an issue"))
    else
        throw(error("unknown binary Core.IntrinsicFunction $_f with args $((_a, _b))"))
    end
end

function rrule!!(f::CoDual{Core.IntrinsicFunction}, x::CoDual, y::CoDual, z::CoDual)
    _f = primal(f)
    _x = primal(x)
    _y = primal(y)
    _z = primal(z)
    if _f === pointerref
        x_s = shadow(x)
        a = CoDual(pointerref(_x, _y, _z), pointerref(x_s, _y, _z))
        function pointerref_pullback!!(da, ::NoTangent, dx, dy, dz)
            dx_v = pointerref(dx, _y, _z)
            new_dx_v = increment!!(dx_v, da)
            pointerset(dx, new_dx_v, _y, _z)
            return NoTangent(), dx, dy, dz
        end
        return a, pointerref_pullback!!
    else
        throw(error("unknown ternary Core.IntrinsicFunction $_f with arg $((_x, _y, _z)) "))
    end
end

function rrule!!(f::CoDual{Core.IntrinsicFunction}, a, b, c, d)
    _f = primal(f)
    _a = primal(a)
    _b = primal(b)
    _c = primal(c)
    _d = primal(d)
    throw(error("unknown quaternary Core.IntrinsicFunction $_f with arg $((_a, _b, _c, _d)) "))
end

function rrule!!(::CoDual{typeof(<:)}, T1, T2)
    return CoDual(<:(primal(T1), primal(T2)), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(===)}, args...)
    return CoDual(===(map(primal, args)...), NoTangent()), NoPullback()
end

# Core._abstracttype
# Core._apply_iterate
# Core._apply_pure
# Core._call_in_world
# Core._call_in_world_total
# Core._call_latest
# Core._compute_sparams
# Core._equiv_typedef
# Core._expr
# Core._primitivetype
# Core._setsuper!
# Core._structtype
# Core._svec_ref
# Core._typebody!
# Core._typevar

function rrule!!(::CoDual{typeof(Core.apply_type)}, args...)
    arg_primals = map(primal, args)
    T = Core.apply_type(arg_primals...)
    return CoDual(T, zero_tangent(T)), NoPullback()
end

# Core.arrayref
# Core.arrayset

function rrule!!(::CoDual{typeof(Core.arraysize)}, X, dim)
    return CoDual(Core.arraysize(primal(X), primal(dim)), NoTangent()), NoPullback()
end

# Core.compilerbarrier
# Core.const_arrayref
# Core.donotdelete
# Core.finalizer
# Core.get_binding_type

function rrule!!(::CoDual{typeof(Core.ifelse)}, cond, a, b)
    _cond = primal(cond)
    function ifelse_pullback!!(dc, df, ::NoTangent, da, db)
        da = _cond ? increment!!(da, dc) : da
        db = _cond ? db : increment!!(db, dc)
        return df, NoTangent(), da, db
    end
    return ifelse(_cond, a, b), ifelse_pullback!!
end

# Core.set_binding_type!

function rrule!!(::CoDual{typeof(Core.sizeof)}, x)
    return CoDual(Core.sizeof(primal(x)), NoTangent()), NoPullback()
end

# Core.svec

function rrule!!(::CoDual{typeof(applicable)}, f, args...)
    return CoDual(applicable(primal(f), map(primal, args)...), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(Core.fieldtype)}, args...)
    arg_primals = map(primal, args)
    return CoDual(Core.fieldtype(arg_primals...), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(getfield)}, value::CoDual, name::CoDual)
    _name = primal(name)
    function getfield_pullback(dy, ::NoTangent, dvalue, ::NoTangent)
        new_dvalue = increment_field!!(dvalue, dy, _name)
        return NoTangent(), new_dvalue, NoTangent()
    end
    y = CoDual(getfield(primal(value), _name), _getfield(shadow(value), _name))
    return y, getfield_pullback
end

function rrule!!(::CoDual{typeof(getfield)}, value::CoDual, name::CoDual, order::CoDual)
    _name = primal(name)
    _order = primal(order)
    function getfield_pullback(dy, df, dvalue, dname, dorder)
        new_dvalue = increment_field!!(dvalue, dy, _name)
        return df, new_dvalue, dname, dorder
    end
    _order = _order isa Expr ? true : _order
    y = CoDual(
        getfield(primal(value), _name, _order),
        _getfield(shadow(value), _name, _order),
    )
    return y, getfield_pullback
end

# getglobal

function rrule!!(::CoDual{typeof(getglobal)}, a, b)
    v = getglobal(primal(a), primal(b))
    return CoDual(v, zero_tangent(v)), NoPullback()
end

# invoke

function rrule!!(::CoDual{typeof(isa)}, x, T)
    return CoDual(isa(primal(x), primal(T)), NoTangent()), NoPullback()
end

function rrule!!(::CoDual{typeof(isdefined)}, args...)
    return CoDual(isdefined(map(primal, args)...), NoTangent()), NoPullback()
end

# modifyfield!

function rrule!!(::CoDual{typeof(nfields)}, x)
    return CoDual(nfields(primal(x)), NoTangent()), NoPullback()
end

# replacefield!

function _setfield!(value::MutableTangent, name, x)
    @set value.fields.$name = x
    return x
end

function rrule!!(::CoDual{typeof(setfield!)}, value, name, x)
    _name = primal(name)
    old_x = isdefined(primal(value), _name) ? getfield(primal(value), _name) : nothing
    function setfield!_pullback(dy, df, dvalue, ::NoTangent, dx)
        new_dx = increment!!(dx, getfield(dvalue.fields, _name).tangent)
        set_field_to_zero!!(dvalue, _name)
        new_dx = increment!!(new_dx, dy)
        old_x !== nothing && setfield!(primal(value), _name, old_x)
        return df, dvalue, NoTangent(), new_dx
    end
    y = CoDual(
        setfield!(primal(value), _name, primal(x)),
        _setfield!(shadow(value), _name, shadow(x)),
    )
    return y, setfield!_pullback
end

# swapfield!
# throw

function rrule!!(::CoDual{typeof(tuple)}, args...)
    y = CoDual(tuple(map(primal, args)...), tuple(map(shadow, args)...))
    tuple_pullback(dy, ::NoTangent, dargs...) = NoTangent(), map(increment!!, dargs, dy)...
    return y, tuple_pullback
end

function rrule!!(::CoDual{typeof(typeassert)}, x, type)
    function typeassert_pullback(dy, ::NoTangent, dx, ::NoTangent)
        return NoTangent(), increment!!(dx, dy), NoTangent()
    end
    return CoDual(typeassert(primal(x), primal(type)), shadow(x)), typeassert_pullback
end

rrule!!(::CoDual{typeof(typeof)}, x) = CoDual(typeof(primal(x)), NoTangent()), NoPullback()



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

for name in [
    :(:jl_alloc_array_1d), :(:jl_alloc_array_2d), :(:jl_alloc_array_3d), :(:jl_new_array),
    :(:jl_array_grow_end), :(:jl_array_del_end), :(:jl_array_copy),
    :(:jl_type_intersection), :(:memset),
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



#
# Rules to avoid hitting limitations of Umlaut
# These should be removed at a later date.
#

isprimitive(::RMC, ::typeof(eltype), x) = true
function rrule!!(::CoDual{typeof(eltype)}, x)
    return CoDual(eltype(primal(x)), NoTangent()), NoPullback()
end

isprimitive(::RMC, ::typeof(Base.promote_op), x, S::Type...) = true
function rrule!!(::CoDual{typeof(Base.promote_op)}, args...)
    return CoDual(Base.promote_op(map(primal, args)...), NoTangent()), NoPullback()
end

isprimitive(::RMC, ::Core.Typeof(String), args...) = true
function rrule!!(::CoDual{Core.Typeof(String)}, args::CoDual...)
    s = String(map(primal, args)...)
    return CoDual(s, zero_tangent(s)), NoPullback()
end


#
# Rules to avoid / cope with Umlaut internals.
# We _might_ remove these at a later date if they turn out not to be needed.
#

function rrule!!(::CoDual{typeof(Umlaut.check_variable_length)}, args...)
    v = Umlaut.check_variable_length(map(primal, args)...)
    return CoDual(v, zero_tangent(v)), NoPullback()
end

# Umlaut occassionally pushes `getindex` onto the tape.
# Easiest just to handle it like this.
# Might remove at a later date when `Umlaut.primitivize` works properly.
isprimitive(::RMC, ::typeof(getindex), ::Tuple, ::Int) = true
function rrule!!(::CoDual{typeof(getindex)}, x::CoDual{<:Tuple}, i::CoDual{Int})
    function getindex_pullback!!(dy, df, dx, ::NoTangent)
        dx = ntuple(n -> n == primal(i) ? increment!!(dx[n], dy) : dx[n], length(dx))
        return df, dx, NoTangent()
    end
    return CoDual(primal(x)[primal(i)], shadow(x)[primal(i)]), getindex_pullback!!
end



#
# (in-principle) non-essential rules.
# Some of these might be removed in the future in favour of lower-level rules.
#

isprimitive(::RMC, ::typeof(sin), ::Float64) = true
function rrule!!(::CoDual{typeof(sin)}, x::CoDual{Float64})
    x = primal(x)
    partial = cos(x)
    function sin_pullback!!(dy::Float64, dsin, dx::Float64)
        return dsin, increment!!(dx, dy * partial)
    end
    y = sin(x)
    return CoDual(y, zero(y)), sin_pullback!!
end

isprimitive(::RMC, ::typeof(cos), ::Float64) = true
function rrule!!(::CoDual{typeof(cos)}, x::CoDual{Float64})
    x = primal(x)
    partial = -sin(x)
    function cos_pullback!!(dy::Float64, dcos, dx::Float64)
        return dcos, increment!!(dx, dy * partial)
    end
    y = cos(x)
    return CoDual(y, zero(y)), cos_pullback!!
end

isprimitive(::RMC, ::typeof(Base.getindex), x::Array, inds::Int...) = true
function rrule!!(
    ::CoDual{typeof(Base.getindex)},
    x::CoDual{<:Array, Tdx},
    inds::CoDual{Int}...,
) where {V, Tdx <: Array{V}}
    ind_primals = map(primal, inds)
    dx = shadow(x)
    y_primal = getindex(primal(x), ind_primals...)
    y_shadow = zero_tangent(y_primal)
    y = CoDual(y_primal, y_shadow)
    function getindex_pullback!!(dy::V, df, dx::Tdx, dinds::NoTangent...)
        setindex!(dx, increment!!(getindex(dx, ind_primals...), dy), ind_primals...)
        return df, dx, dinds...
    end
    return y, getindex_pullback!!
end

isprimitive(::RMC, ::typeof(setindex!), x::Array, v, inds...) = true
function rrule!!(
    ::CoDual{typeof(Base.setindex!)},
    A::CoDual{<:Array, TdA},
    v::CoDual,
    inds::CoDual{Int}...,
) where {V, TdA <: Array{V}}
    ind_primals = map(primal, inds)
    old_A_v = getindex(primal(A), ind_primals...)
    old_A_v_t = getindex(shadow(A), ind_primals...)
    setindex!(primal(A), primal(v), ind_primals...)
    setindex!(shadow(A), shadow(v), ind_primals...)
    function setindex_pullback!!(dA::TdA, df, dA2::TdA, dv, dinds::NoTangent...)
        dv_new = increment!!(dv, getindex(dA, ind_primals...))
        setindex!(primal(A), old_A_v, ind_primals...)
        setindex!(dA, old_A_v_t, ind_primals...)
        return df, dA, dv_new, dinds...
    end
    return A, setindex_pullback!!
end



#
# Rules to work around a lack of activity analysis.
#

# isprimitive(::RMC, ::typeof(Base.elsize), x) = true
# function rrule!!(::CoDual{typeof(Base.elsize)}, x::CoDual)
#     y = Base.elsize(primal(x))
#     return CoDual(y, zero_tangent(y)), NoPullback()
# end

for name in [
    :(Base.elsize),
    :(Core.Compiler.sizeof_nothrow),
    :(Base.datatype_haspadding),
    :(Base.datatype_nfields),
    :(Base.datatype_pointerfree),
    :(Base.datatype_alignment),
    :(Base.datatype_pointerfree),
    :(Base.datatype_fielddesc_type),
]
    @eval isprimitive(::RMC, ::Core.Typeof($name), @nospecialize(args)...) = true
    @eval function rrule!!(::CoDual{Core.Typeof($name)}, args...)
        y = $(name)(map(primal, args)...)
        return CoDual(y, zero_tangent(y)), NoPullback()
    end
end

# isprimitive(::RMC, ::typeof(Core.Compiler.sizeof_nothrow), @nospecialize(x)) = true
# function rrule!!(::CoDual{typeof(Core.Compiler.sizeof_nothrow)}, @nospecialize(x))
#     y = Core.Compiler.sizeof_nothrow(primal(x))
#     return CoDual(y, zero_tangent(y)), NoPullback()
# end

# isprimitive(::RMC, ::typeof(Base.datatype_haspadding), dt::DataType) = true
# function rrule!!(::CoDual{typeof(Base.datatype_haspadding)}, dt)
#     y = Base.datatype_haspadding(primal(dt))
#     return CoDual(y, zero_tangent(y)), NoPullback()
# end



#
# Rules to avoid pointer magic
#

isprimitive(::RMC, ::typeof(Base.:(+)), x::Ptr, y::Integer) = true
function rrule!!(::CoDual{typeof(Base.:(+))}, x::CoDual{<:Ptr}, y::CoDual{<:Integer})
    return CoDual(primal(x) + primal(y), shadow(x) + primal(y)), NoPullback()
end


#
# LinearAlgebra
#

function rrule!!(::CoDual{typeof(LinearAlgebra.chkstride1)}, args...)
    return CoDual(LinearAlgebra.chkstride1(args...), NoTangent()), NoPullback()
end



#
# LinearAlgebra.BLAS
#

blas_name(name::Symbol) = (Symbol(name, "64_"), Symbol(BLAS.libblastrampoline))

for (fname, elty) in ((:dscal_, :Float64), (:sscal_, :Float32))
    @eval function Taped.rrule!!(
        ::CoDual{typeof(__foreigncall__)},
        ::CoDual{Val{$(blas_name(fname))}},
        ::CoDual, # return type
        ::CoDual, # argument types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        n::CoDual{Ptr{BLAS.BlasInt}},
        DA::CoDual{Ptr{$elty}},
        DX::CoDual{Ptr{$elty}},
        incx::CoDual{Ptr{BLAS.BlasInt}},
        args...,
    )
        # Load in values from pointers, and turn pointers to memory buffers into Vectors.
        _n = unsafe_load(primal(n))
        _incx = unsafe_load(primal(incx))
        _DA = unsafe_load(primal(DA))
        _DX = unsafe_wrap(Vector{$elty}, primal(DX), _n * _incx)
        _DX_s = unsafe_wrap(Vector{$elty}, shadow(DX), _n * _incx)

        inds = 1:_incx:(_incx * _n)
        DX_copy = _DX[inds]
        BLAS.scal!(_n, _DA, _DX, _incx)

        function dscal_pullback!!(_, a, b, c, d, e, f, dn, dDA, dDX, dincx, dargs...)

            # Set primal to previous state.
            _DX[inds] .= DX_copy

            # Compute cotangent w.r.t. scaling.
            unsafe_store!(dDA, BLAS.dot(_n, _DX, _incx, dDX, _incx) + unsafe_load(dDA))

            # Compute cotangent w.r.t. DX.
            BLAS.scal!(_n, _DA, _DX_s, _incx)

            return a, b, c, d, e, f, dn, dDA, dDX, dincx, dargs...
        end
        return CoDual(Cvoid(), zero_tangent(Cvoid)), dscal_pullback!!
    end
end

function _trans(flag, mat)
    flag === 'T' && return transpose(mat)
    flag === 'C' && return adjoint(mat)
    flag === 'N' && return mat
    throw(error("Unrecognised flag $flag"))
end

function t_char(x::UInt8)
    x == 0x4e && return 'N'
    x == 0x54 && return 'T'
    x == 0x43 && return 'C'
    throw(error("unrecognised char-code $x"))
end

function wrap_ptr_as_view(ptr::Ptr{T}, buffer_nrows::Int, nrows::Int, ncols::Int) where {T}
    return view(unsafe_wrap(Matrix{T}, ptr, (buffer_nrows, ncols)), 1:nrows, :)
end

for (gemm, elty) in (
    (:dgemm_, :Float64),
    (:sgemm_, :Float32),
)
    @eval function rrule!!(
        ::CoDual{typeof(__foreigncall__)},
        ::CoDual{Val{$(blas_name(gemm))}},
        RT::CoDual{Val{Cvoid}},
        AT::CoDual, # arg types
        ::CoDual, # nreq
        ::CoDual, # calling convention
        tA::CoDual{Ptr{UInt8}},
        tB::CoDual{Ptr{UInt8}},
        m::CoDual{Ptr{Int}},
        n::CoDual{Ptr{Int}},
        ka::CoDual{Ptr{Int}},
        alpha::CoDual{Ptr{$elty}},
        A::CoDual{Ptr{$elty}},
        LDA::CoDual{Ptr{Int}},
        B::CoDual{Ptr{$elty}},
        LDB::CoDual{Ptr{Int}},
        beta::CoDual{Ptr{$elty}},
        C::CoDual{Ptr{$elty}},
        LDC::CoDual{Ptr{Int}},
        args...,
    )
        _tA = t_char(unsafe_load(primal(tA)))
        _tB = t_char(unsafe_load(primal(tB)))
        _m = unsafe_load(primal(m))
        _n = unsafe_load(primal(n))
        _ka = unsafe_load(primal(ka))
        _alpha = unsafe_load(primal(alpha))
        _A = primal(A)
        _LDA = unsafe_load(primal(LDA))
        _B = primal(B)
        _LDB = unsafe_load(primal(LDB))
        _beta = unsafe_load(primal(beta))
        _C = primal(C)
        _LDC = unsafe_load(primal(LDC))

        A_mat = wrap_ptr_as_view(primal(A), _LDA, (_tA == 'N' ? (_m, _ka) : (_ka, _m))...)
        B_mat = wrap_ptr_as_view(primal(B), _LDB, (_tB == 'N' ? (_ka, _n) : (_n, _ka))...)
        C_mat = wrap_ptr_as_view(primal(C), _LDC, _m, _n)
        C_copy = collect(C_mat)

        BLAS.gemm!(_tA, _tB, _alpha, A_mat, B_mat, _beta, C_mat)

        function gemm!_pullback!!(
            _, df, dname, dRT, dAT, dnreq, dconvention,
            dtA, dtB, dm, dn, dka, dalpha, dA, dLDA, dB, dLDB, dbeta, dC, dLDC, dargs...,
        )
            # Restore previous state.
            C_mat .= C_copy

            # Convert pointers to views.
            dA_mat = wrap_ptr_as_view(dA, _LDA, (_tA == 'N' ? (_m, _ka) : (_ka, _m))...)
            dB_mat = wrap_ptr_as_view(dB, _LDB, (_tB == 'N' ? (_ka, _n) : (_n, _ka))...)
            dC_mat = wrap_ptr_as_view(dC, _LDC, _m, _n)

            # Increment cotangents.
            unsafe_store!(dbeta, unsafe_load(dbeta) + tr(dC_mat' * C_mat))
            dalpha_inc = tr(dC_mat' * _trans(_tA, A_mat) * _trans(_tB, B_mat))
            unsafe_store!(dalpha, unsafe_load(dalpha) + dalpha_inc)
            dA_mat .+= _alpha * transpose(_trans(_tA, _trans(_tB, B_mat) * transpose(dC_mat)))
            dB_mat .+= _alpha * transpose(_trans(_tB, transpose(dC_mat) * _trans(_tA, A_mat)))
            dC_mat .*= _beta

            return df, dname, dRT, dAT, dnreq, dconvention,
                dtA, dtB, dm, dn, dka, dalpha, dA, dLDA, dB, dLDB, dbeta, dC, dLDC, dargs...
        end
        return CoDual(Cvoid, zero_tangent(Cvoid)), gemm!_pullback!!
    end
end



#
# Performance-only rules. These should be able to be removed, and everything still works,
# just a bit slower. The effect of these is typically to remove many nodes from the tape.
#

for name in [
    :size,
    :(LinearAlgebra.lapack_size),
    :(Base.require_one_based_indexing),
    :in,
    :iszero,
    :isempty,
    :isbitstype,
    :sizeof,
    :promote_type,
]
    @eval isprimitive(::RMC, ::Core.Typeof($name), args...) = true
    @eval function rrule!!(::CoDual{Core.Typeof($name)}, args::CoDual...)
        v = $name(map(primal, args)...)
        return CoDual(v, zero_tangent(v)), NoPullback()
    end
end



#
# High-level -- AD functionality built on top of rules.
#

struct CoInstruction{Tinputs, Toutput, Tpb}
    inputs::Tinputs
    output::Toutput
    pb::Tpb
end

input_shadows(x::CoInstruction) = map(shadow ∘ getindex, x.inputs)
output_shadow(x::CoInstruction) = shadow(x.output[])

function build_coinstruction(inputs::CoInstruction...)
    input_refs = map(x -> x.output, inputs)
    input_values = map(getindex, input_refs)
    output_value, pb!! = rrule!!(input_values...)
    output_ref = Ref(output_value)
    pb_ref = Ref(pb!!)
    return CoInstruction(input_refs, output_ref, pb_ref)
end

function (instruction::CoInstruction)(inputs::CoInstruction...)
    input_refs = map(x -> x.output, inputs)
    input_values = map(getindex, input_refs)
    foreach(verify_codual_type, input_values)
    output_value, pb!! = rrule!!(input_values...)
    verify_codual_type(output_value)
    output_ref = instruction.output
    output_ref[] = output_value
    pb_ref = instruction.pb
    pb_ref[] = pb!!
    return CoInstruction(input_refs, output_ref, pb_ref)
end

function pullback!(instruction::CoInstruction)
    input_shadows = map(shadow ∘ getindex, instruction.inputs)
    output_shadow = shadow(instruction.output[])
    new_input_shadows = instruction.pb[](output_shadow, input_shadows...)
    foreach(update_shadow!, instruction.inputs, new_input_shadows)
    return nothing
end

pullback!(::CoInstruction{Nothing, <:Ref, Nothing}) = nothing

function update_shadow!(x::Ref{<:CoDual{Tx, Tdx}}, new_shadow::Tdx) where {Tx, Tdx}
    x_val = x[]
    x[] = CoDual(primal(x_val), new_shadow)
    return nothing
end

function to_reverse_mode_ad(tape::Tape{RMC}, ȳ, inputs::CoInstruction...)
    inputs!(tape, inputs...)

    new_tape = Tape(tape.c)

    # Transform forwards pass, replacing ops with associated rrule calls.
    for op in tape.ops
        push!(new_tape, to_reverse_mode_ad(op, new_tape))
    end
    new_tape.result = unbind(tape.result)

    # Seed reverse-pass and create operations to execute it.
    seed_op = mkcall(seed_return!, new_tape.result, ȳ)
    push!(new_tape, seed_op)

    Umlaut.exec!(new_tape, seed_op)
    for op in reverse(new_tape.ops[1:end-1])
        pb_op = mkcall(pullback!, Variable(op.id))
        push!(new_tape, pb_op)
        Umlaut.exec!(new_tape, pb_op)
    end

    return new_tape
end

const_coinstruction(x::CoDual) = CoInstruction(nothing, Ref(x), nothing)

to_reverse_mode_ad(x::Input, new_tape) = Input(x.val)
function to_reverse_mode_ad(x::Constant, new_tape)
    return Constant(const_coinstruction(CoDual(x.val, zero_tangent(x.val))))
end
function to_reverse_mode_ad(x::Call, new_tape)
    f = x.fn isa Variable ? new_tape[x.fn].val : x.fn
    f = f isa CoInstruction ? f : const_coinstruction(CoDual(f, zero_tangent(f)))
    raw_args = map(x -> x isa Variable ? new_tape[x].val : x, x.args)
    args = map(raw_args) do x
        x isa CoInstruction ? x : const_coinstruction(CoDual(x, zero_tangent(x)))
    end
    v = build_coinstruction(f, args...)
    return mkcall(v, f, args...; val=v)
end

function seed_return!(x::CoInstruction{T, V}, x̄) where {T, V}
    output = x.output[]
    x.output[] = CoDual(primal(output), x̄)
    return nothing
end

struct UnrolledFunction{Ttape}
    tape::Ttape
end


tangent_type(::Type{<:UnrolledFunction}) = NoTangent
randn_tangent(::AbstractRNG, ::UnrolledFunction) = NoTangent()
zero_tangent(::UnrolledFunction) = NoTangent()

(f::UnrolledFunction)(args...) = play!(f.tape, args...)

function seed_variable!(tape, var, ȳ)
    y_ref = tape[var].val.output
    dy = shadow(y_ref[])
    dy_new = increment!!(dy, ȳ)
    y_ref[] = CoDual(primal(y_ref[]), dy_new)
    return nothing
end

function rrule!!(f::CoDual{<:UnrolledFunction}, args...)
    tape = primal(f).tape
    wrapped_args = map(const_coinstruction, args)
    inputs!(tape, wrapped_args...)

    # display(tape)
    # println()

    new_tape = Tape(tape.c)

    # Transform forwards pass, replacing ops with associated rrule calls.
    for op in tape.ops
        new_op = to_reverse_mode_ad(op, new_tape)
        new_op_val = new_op.val.output[]
        @show new_op_val, typeof(new_op_val)
        if tangent_type(typeof(primal(new_op_val))) != typeof(shadow(new_op_val))
            inputs = map(getindex, new_op.val.inputs)
            display(inputs)
            println()
            display(new_op_val)
            println()
            display(which(rrule!!, map(Core.Typeof, inputs)))
            println()
            display("expected shadow type $(tangent_type(typeof(primal(new_op_val))))")
            println()
            throw(error("bad output types found in practice for op"))
        end
        push!(new_tape, new_op)
    end
    new_tape.result = unbind(tape.result)
    y_ref = new_tape[new_tape.result].val.output

    display(new_tape)
    println()

    # Run the reverse-pass.
    function unrolled_function_pb!!(ȳ, ::NoTangent, dargs...)

        # Initialise values on the tape.
        seed_variable!(new_tape, new_tape.result, ȳ)
        foreach((v, x̄) -> seed_variable!(new_tape, v, x̄), inputs(new_tape), dargs)

        # Run the tape backwards.
        for op in reverse(new_tape.ops)
            pullback!(new_tape[Variable(op.id)].val)
        end

        # Extract the results from the tape.
        return NoTangent(), map(v -> shadow(new_tape[v].val.output[]), inputs(new_tape))...
    end

    return y_ref[], unrolled_function_pb!!
end

tangent_type(::Type{<:Umlaut.Variable}) = NoTangent
