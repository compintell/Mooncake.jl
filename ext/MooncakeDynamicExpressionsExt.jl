module MooncakeDynamicExpressionsExt

using DynamicExpressions:
    DynamicExpressions as DE,
    AbstractExpressionNode,
    Expression,
    Nullable,
    branch_copy,
    leaf_copy,
    get_children
using Mooncake
using Mooncake: NoTangent
using Random: AbstractRNG

################################################################################
# Main tangent type
################################################################################

mutable struct TangentNode{Tv,D}
    degree::UInt8
    constant::Bool
    val::Tv
    children::NTuple{
        D,Union{@NamedTuple{null::NoTangent,x::TangentNode{Tv,D}},NoTangent}
    }

    TangentNode{Tv,D}() where {Tv,D} = new{Tv,D}()
end
function TangentNode{Tv,D}(
    val_tan::Union{Tv,Nothing}, children::Vararg{Union{TangentNode{Tv,D},NoTangent},deg}
) where {Tv,D,deg}
    n = TangentNode{Tv,D}()
    n.degree = UInt8(deg)
    n.children = ntuple(i -> i <= deg ? _wrap_nullable(children[i]) : NoTangent(), Val(D))
    n.constant = !isnothing(val_tan)
    if !isnothing(val_tan)
        n.val = val_tan
    end
    return n
end

function Mooncake.tangent_type(::Type{<:AbstractExpressionNode{T,D}}) where {T,D}
    Tv = Mooncake.tangent_type(T)
    return Tv === NoTangent ? NoTangent : TangentNode{Tv,D}
end
function Mooncake.tangent_type(::Type{Nullable{<:AbstractExpressionNode{T,D}}}) where {T,D}
    Tv = Mooncake.tangent_type(T)
    return Union{@NamedTuple{null::NoTangent,x::TangentNode{Tv,D}},NoTangent}
end
function Mooncake.tangent_type(::Type{TangentNode{Tv,D}}) where {Tv,D}
    return TangentNode{Tv,D}
end
function Mooncake.tangent_type(
    ::Type{TangentNode{Tv,D}}, ::Type{Mooncake.NoRData}
) where {Tv,D}
    return TangentNode{Tv,D}
end
function Mooncake.tangent(t::TangentNode, ::Mooncake.NoRData)
    return t
end

_unwrap_nullable(c::NoTangent) = c
_unwrap_nullable(c::NamedTuple{(:null, :x)}) = c.x
_wrap_nullable(c::NoTangent) = c
_wrap_nullable(c::TangentNode) = (; null=NoTangent(), x=c)

function get_child(t::TangentNode, i::Int)
    return _unwrap_nullable(t.children[i])
end
function _get_child(t, ::Val{i}) where {i}
    return get_child(t, i)
end
function DE.extract_gradient(
    gradient::Mooncake.Tangent{@NamedTuple{tree::TN,metadata::Mooncake.NoTangent}},
    tree::Expression{T},
) where {Tv,TN<:TangentNode{Tv},T}
    return DE.extract_gradient(gradient.fields.tree, DE.get_tree(tree))
end
function DE.extract_gradient(
    dtree::TangentNode{Tv,D}, tree::AbstractExpressionNode{T,D}
) where {Tv,D,T}
    num_constants = count(t -> t.degree == 0 && t.constant, tree)
    ar = Vector{Tv}(undef, num_constants)
    _extract_gradient!(ar, dtree, tree)
    return ar
end
function _extract_gradient!(
    ar, ::NoTangent, ::AbstractExpressionNode{T,D}, idx=firstindex(ar)
) where {D,T}
    return idx
end
@generated function _extract_gradient!(
    ar, gradient::TangentNode{Tv,D}, tree::AbstractExpressionNode{T,D}, idx=firstindex(ar)
) where {Tv,D,T}
    quote
        if tree.degree == 0
            if tree.constant
                ar[idx] = gradient.val::Tv
                idx = nextind(ar, idx)
            end
        else
            deg = tree.degree
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@nexprs(
                    i,
                    j -> begin
                        idx = _extract_gradient!(
                            ar, get_child(gradient, j), DE.get_child(tree, j), idx
                        )
                    end
                )
            )
        end
        return idx
    end
end

################################################################################
# zero_tangent / randn_tangent
################################################################################

struct InitHelper{F,ARGS<:Tuple,M<:Mooncake.MaybeCache}
    f::F
    args::ARGS
    dict::M
end
function (helper::InitHelper)(p::N) where {T,D,N<:AbstractExpressionNode{T,D}}
    Tv = Mooncake.tangent_type(T)
    Tv === NoTangent && return NoTangent()
    return get!(() -> helper_inner(helper, p), helper.dict, p)::TangentNode{Tv,D}
end
@generated function helper_inner(
    helper::InitHelper, p::N
) where {T,D,N<:AbstractExpressionNode{T,D}}
    quote
        Tv = Mooncake.tangent_type(T)
        deg = p.degree
        if deg == 0
            if p.constant
                TangentNode{Tv,D}(helper_call(helper, p.val))
            else
                TangentNode{Tv,D}(nothing)
            end
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> TangentNode{Tv,D}(nothing, map(helper, get_children(p, Val(i)))...),
            )
        end
    end
end
function helper_call(helper::InitHelper, val)
    return helper.f(helper.args..., val, helper.dict)
end

function Mooncake.zero_tangent_internal(
    p::N, dict::Mooncake.MaybeCache
) where {T,N<:AbstractExpressionNode{T}}
    return InitHelper(Mooncake.zero_tangent_internal, (), dict)(p)
end
function Mooncake.randn_tangent_internal(
    rng::AbstractRNG, p::N, dict::Mooncake.MaybeCache
) where {T,N<:AbstractExpressionNode{T}}
    return InitHelper(Mooncake.randn_tangent_internal, (rng,), dict)(p)
end

################################################################################
# In‑place mutation helpers
################################################################################

struct IncrementHelper{F,C<:Mooncake.IncCache}
    f::F
    cache::C
end
@generated function (helper::IncrementHelper)(t::TangentNode{Tv,D}, s...) where {Tv,D}
    quote
        if haskey(helper.cache, t) || (!isempty(s) && t === first(s))
            return t
        end
        helper.cache[t] = true
        ts = (t, s...)
        deg = t.degree
        if deg == 0
            if t.constant
                t.val = helper_call(helper, ts...)
            end
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> begin
                    Base.Cartesian.@nexprs(
                        i, c -> helper(map(Base.Fix2(_get_child, Val(c)), ts)...),
                    )
                end
            )
        end
        return t
    end
end
function helper_call(helper::IncrementHelper, t, s...)
    return helper.f(helper.cache, t.val, map(ti -> ti.val, s)...)
end

function Mooncake.increment_internal!!(c::Mooncake.IncCache, t::TangentNode, s::TangentNode)
    return IncrementHelper(Mooncake.increment_internal!!, c)(t, s)
end
function Mooncake.set_to_zero_internal!!(c::Mooncake.IncCache, t::TangentNode)
    return IncrementHelper(Mooncake.set_to_zero_internal!!, c)(t)
end

################################################################################
# Algebraic helpers (_dot / _scale / _add_to_primal / _diff)
################################################################################

Mooncake._dot_internal(c::Mooncake.MaybeCache, t::NoTangent, s::TangentNode) = 0.0
Mooncake._dot_internal(c::Mooncake.MaybeCache, t::TangentNode, s::NoTangent) = 0.0
@generated function Mooncake._dot_internal(
    c::Mooncake.MaybeCache, t::TangentNode{Tv,D}, s::TangentNode{Tv,D}
) where {Tv,D}
    quote
        key = (t, s)
        haskey(c, key) && return c[key]::Float64
        c[key] = 0.0
        deg = t.degree
        res = if deg == 0
            if t.constant && s.constant
                Mooncake._dot_internal(c, t.val, s.val)
            else
                0.0
            end
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@ncall(
                    i,
                    +,
                    j -> Mooncake._dot_internal(c, get_child(t, j), get_child(s, j))
                )
            )
        end
        c[key] = res
        return res
    end
end

function Mooncake._scale_internal(
    c::Mooncake.MaybeCache, a::Number, t::TangentNode{Tv,D}
) where {Tv,D}
    return get!(() -> _scale_internal_helper(c, a, t), c, t)::TangentNode{Tv,D}
end
@generated function _scale_internal_helper(
    c::Mooncake.MaybeCache, a::Number, t::TangentNode{Tv,D}
) where {Tv,D}
    quote
        deg = t.degree
        if deg == 0
            if t.constant
                TangentNode{Tv,D}(Mooncake._scale_internal(c, a, t.val))
            else
                TangentNode{Tv,D}(nothing)
            end
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@ncall(
                    i,
                    TangentNode{Tv,D},
                    nothing,
                    j -> Mooncake._scale_internal(c, a, get_child(t, j))
                )
            )
        end
    end
end

function Mooncake._add_to_primal_internal(
    c::Mooncake.MaybeCache, p::N, t::TangentNode{Tv,D}, unsafe::Bool
) where {T,D,N<:AbstractExpressionNode{T,D},Tv}
    key = (p, t, unsafe)
    return get!(() -> _add_to_primal_internal_helper(c, p, t, unsafe), c, key)::N
end
@generated function _add_to_primal_internal_helper(
    c::Mooncake.MaybeCache, p::N, t::TangentNode{Tv,D}, unsafe::Bool
) where {T,D,N<:AbstractExpressionNode{T,D},Tv}
    quote
        deg = p.degree
        if deg == 0
            new_leaf = leaf_copy(p)
            if p.constant
                new_leaf.val = Mooncake._add_to_primal_internal(c, p.val, t.val, unsafe)
            end
            new_leaf
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@ncall(
                    i,
                    branch_copy,
                    p,
                    j -> Mooncake._add_to_primal_internal(
                        c, DE.get_child(p, j), get_child(t, j), unsafe
                    )
                )
            )
        end
    end
end

function Mooncake._diff_internal(
    c::Mooncake.MaybeCache, p::N, q::N
) where {T,D,N<:AbstractExpressionNode{T,D}}
    Tv = Mooncake.tangent_type(T)
    Tv === NoTangent && return NoTangent()
    key = (p, q)
    return get!(() -> _diff_internal_helper(c, p, q), c, key)::TangentNode{Tv,D}
end

@generated function _diff_internal_helper(
    c::Mooncake.MaybeCache, p::N, q::N
) where {T,D,N<:AbstractExpressionNode{T,D}}
    quote
        Tv = Mooncake.tangent_type(T)
        deg = p.degree
        if p.degree == 0
            if p.constant
                TangentNode{Tv,D}(Mooncake._diff_internal(c, p.val, q.val))
            else
                TangentNode{Tv,D}(nothing)
            end
        else
            Base.Cartesian.@nif(
                $D,
                i -> i == deg,
                i -> Base.Cartesian.@ncall(
                    i,
                    TangentNode{Tv,D},
                    nothing,
                    j -> Mooncake._diff_internal(c, DE.get_child(p, j), DE.get_child(q, j))
                )
            )
        end
    end
end

################################################################################
# getfield / lgetfield / lsetfield! / _new_ rrules
################################################################################

@generated function _map_to_sym(::Type{N}, ::Val{F}) where {N<:AbstractExpressionNode,F}
    if F isa Symbol
        return Val(F)
    elseif F isa Int
        return Val(fieldname(N, F))
    else
        throw(ArgumentError("Unsupported field type: `$F::$(typeof(F))`"))
    end
end

struct Pullback{T,field_sym,n_args}
    pt::T
end
function (pb::Pullback{T,field_sym,n_args})(Δy_rdata) where {T,field_sym,n_args}
    if field_sym === :val && !(Δy_rdata isa Mooncake.NoRData)
        pb.pt.val = Mooncake.increment_rdata!!(pb.pt.val, Δy_rdata)
    end
    return ntuple(_ -> Mooncake.NoRData(), Val(n_args))
end

function _rrule_getfield_common(
    obj_cd::Mooncake.CoDual{N,TangentNode{Tv,D}}, ::Val{field_sym}, ::Val{n_args}
) where {T,D,N<:AbstractExpressionNode{T,D},Tv,field_sym,n_args}
    p = Mooncake.primal(obj_cd)
    pt = Mooncake.tangent(obj_cd)

    value_primal = getfield(p, field_sym)

    fdata_for_output = if field_sym === :val
        Mooncake.fdata(pt.val)
    elseif field_sym === :children
        map(value_primal, pt.children) do child_p, child_t
            if child_t isa Mooncake.NoTangent
                Mooncake.uninit_fdata(child_p)
            else
                Mooncake.FData(Mooncake.fdata(child_t))
            end
        end
    else
        Mooncake.NoFData()
    end
    y_cd = Mooncake.CoDual(value_primal, fdata_for_output)
    return y_cd, Pullback{typeof(pt),field_sym,n_args}(pt)
end

# lgetfield(AEN, Val{field})
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    typeof(Mooncake.lgetfield),AbstractExpressionNode,Val
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lgetfield)},
    obj_cd::Mooncake.CoDual{N,TangentNode{Tv,D}},
    ::Mooncake.CoDual{Val{FieldName}},
) where {T,D,N<:AbstractExpressionNode{T,D},Tv,FieldName}
    return _rrule_getfield_common(obj_cd, _map_to_sym(N, Val(FieldName)), Val(3))
end

# getfield(AEN, Symbol) or getfield(AEN, Int)
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    typeof(getfield),AbstractExpressionNode,Union{Symbol,Int}
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(getfield)},
    obj_cd::Mooncake.CoDual{N,TangentNode{Tv,D}},
    idx_or_sym_cd::Mooncake.CoDual{<:Union{Symbol,Int}},
) where {T,D,N<:AbstractExpressionNode{T,D},Tv}
    return _rrule_getfield_common(obj_cd, _map_to_sym(N, Val(Mooncake.primal(idx_or_sym_cd))), Val(3))
end



# lgetfield(AEN, Val{field}, Val{order})
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    typeof(Mooncake.lgetfield),AbstractExpressionNode,Val,Val
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lgetfield)},
    obj_cd::Mooncake.CoDual{N,TangentNode{Tv,D}},
    ::Mooncake.CoDual{Val{FieldName}},
    ::Mooncake.CoDual{Val{order}}
) where {T,D,N<:AbstractExpressionNode{T,D},Tv,FieldName,order}
    @assert order === :not_atomic "MooncakeDynamicExpressionsExt.jl does not support `order` other than `:not_atomic`"
    return _rrule_getfield_common(obj_cd, _map_to_sym(N, Val(FieldName)), Val(4))
end


# lsetfield!(AEN, Val{field}, Any)
Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    typeof(Mooncake.lsetfield!),AbstractExpressionNode,Val,Any
}
function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake.lsetfield!)},
    obj_cd::Mooncake.CoDual{N,TangentNode{Tv,D}},
    ::Mooncake.CoDual{Val{FieldName}},
    new_val_cd::Mooncake.CoDual
) where {T,D,N<:AbstractExpressionNode{T,D},Tv,FieldName}
    obj = Mooncake.primal(obj_cd)
    obj_t = Mooncake.tangent(obj_cd)
    new_val_primal = Mooncake.primal(new_val_cd)
    new_val_tangent = Mooncake.tangent(new_val_cd)

    v_field_sym = _map_to_sym(N, Val(FieldName))

    old_val = Mooncake.lgetfield(obj, v_field_sym)
    old_tangent = Mooncake.lgetfield(obj_t, v_field_sym)

    Mooncake.lsetfield!(obj, v_field_sym, new_val_primal)
    new_field_tangent = if (new_val_tangent isa Union{Mooncake.NoTangent,Mooncake.NoFData})
        Mooncake.zero_tangent(new_val_primal)
    else
        new_val_tangent
    end
    Mooncake.lsetfield!(obj_t, v_field_sym, new_field_tangent)

    y_fdata = Mooncake.fdata(new_field_tangent)
    y_cd = Mooncake.CoDual(new_val_primal, y_fdata)

    function lsetfield_node_pb(dy_rdata)
        Mooncake.lsetfield!(obj, v_field_sym, old_val)
        Mooncake.lsetfield!(obj_t, v_field_sym, old_tangent)
        return (Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), dy_rdata)
    end

    return y_cd, lsetfield_node_pb
end

# _new_
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{
    typeof(Mooncake._new_),Type{N},Vararg{Any,nargs}
} where {T,D,N<:AbstractExpressionNode{T,D},nargs}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(Mooncake._new_)},
    ::Mooncake.CoDual{Type{N}},
    args::Vararg{Mooncake.CoDual,nargs}
) where {T,D,N<:AbstractExpressionNode{T,D},nargs}
    @assert nargs == 0 "MooncakeDynamicExpressionsExt.jl does not support non-empty `new` for AbstractExpressionNode types"

    primal_node = N()

    Tv = Mooncake.tangent_type(T)
    fdt = Tv === Mooncake.NoTangent ? Mooncake.NoTangent() : TangentNode{Tv,D}(nothing)

    node_cd = Mooncake.CoDual(primal_node, fdt)

    new_node_pb(::Mooncake.NoRData) = (Mooncake.NoRData(), Mooncake.NoRData())

    return node_cd, new_node_pb
end

################################################################################
# Test‑utility helpers
################################################################################

function Mooncake.__verify_fdata_value(
    c::IdDict{Any,Nothing}, p::AbstractExpressionNode, f::TangentNode
)
    haskey(c, p) && return nothing
    c[p] = nothing

    deg = p.degree
    deg != f.degree &&
        throw(Mooncake.InvalidFDataException("degree mismatch between node and tangent"))

    if deg == 0
        p.constant != f.constant && throw(
            Mooncake.InvalidFDataException("constant mismatch between node and tangent")
        )
        if p.constant
            Mooncake._verify_fdata_value(c, p.val, Mooncake.fdata(f.val))
        end
    else
        for i in 1:deg
            Mooncake._verify_fdata_value(c, DE.get_child(p, i), get_child(f, i))
        end
    end
    return nothing
end

function Mooncake.__verify_fdata_value(
    c::IdDict{Any,Nothing}, p::AbstractExpressionNode, f::NoTangent
)
    !haskey(c, p) && (c[p] = nothing)
    return nothing
end

function Mooncake.TestUtils.populate_address_map_internal(
    m::Mooncake.TestUtils.AddressMap, p::N, t::TangentNode{Tv,D}
) where {T,D,N<:AbstractExpressionNode{T,D},Tv}
    kp = Base.pointer_from_objref(p)
    kt = Base.pointer_from_objref(t)
    !haskey(m, kp) && (m[kp] = kt)
    deg = p.degree
    if deg == 0
        if p.constant
            Mooncake.TestUtils.populate_address_map_internal(m, p.val, t.val)
        end
    else
        for i in 1:deg
            Mooncake.TestUtils.populate_address_map_internal(
                m, DE.get_child(p, i), get_child(t, i)
            )
        end
    end
    return m
end

function Mooncake.TestUtils.has_equal_data_internal(
    x::N, y::N, equndef::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {T,N<:AbstractExpressionNode{T}}
    idp = (objectid(x), objectid(y))
    # Just use regular `AbstractExpressionNode` Base.:(==)
    return get!(() -> x == y, d, idp)
end

function Mooncake.TestUtils.has_equal_data_internal(
    t::TangentNode{Tv,D},
    s::TangentNode{Tv,D},
    equndef::Bool,
    d::Dict{Tuple{UInt,UInt},Bool},
) where {Tv,D}
    idp = (objectid(t), objectid(s))
    return get!(() -> _has_equal_data_internal_helper(t, s, equndef, d), d, idp)
end
function _has_equal_data_internal_helper(
    t::TangentNode{Tv,D},
    s::TangentNode{Tv,D},
    equndef::Bool,
    d::Dict{Tuple{UInt,UInt},Bool},
) where {Tv,D}
    deg = t.degree
    deg == s.degree && if t.degree == 0
        if t.constant
            s.constant &&
                Mooncake.TestUtils.has_equal_data_internal(t.val, s.val, equndef, d)
        else
            !s.constant
        end
    else
        all(
            i -> Mooncake.TestUtils.has_equal_data_internal(
                get_child(t, i), get_child(s, i), equndef, d
            ),
            1:deg,
        )
    end
end

end
