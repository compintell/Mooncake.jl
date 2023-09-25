module TestResources

using ..Taped

using LinearAlgebra, Setfield

test_sin(x) = sin(x)

test_cos_sin(x) = cos(sin(x))

test_isbits_multiple_usage(x::Float64) = Core.Intrinsics.mul_float(x, x)

test_getindex(x::AbstractArray{<:Real}) = x[1]

function test_mutation!(x::AbstractVector{<:Real})
    x[1] = sin(x[2])
    return x[1]
end

function test_for_loop(x)
    for _ in 1:5
        x = sin(x)
    end
    return x
end

function test_while_loop(x)
    n = 3
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

mutable struct Foo
    x::Real
end

test_mutable_struct_basic(x) = Foo(x).x

test_mutable_struct_basic_sin(x) = sin(Foo(x).x)

function test_mutable_struct_setfield(x)
    foo = Foo(1.0)
    foo.x = x
    return foo.x
end

function test_mutable_struct(x)
    foo = Foo(x)
    foo.x = sin(foo.x)
    return foo.x
end

struct StructFoo
    a::Real
    b::Vector{Float64}
    StructFoo(a::Float64, b::Vector{Float64}) = new(a, b)
    StructFoo(a::Float64) = new(a)
end

function equal_field(a, b, f)
    (!isdefined(a, f) || !isdefined(b, f)) && return true
    return getfield(a, f) == getfield(b, f)
end

Base.:(==)(a::StructFoo, b::StructFoo) = equal_field(a, b, :a) && equal_field(a, b, :b)

Taped._add_to_primal(p::StructFoo, t::Tangent) = Taped._containerlike_add_to_primal(p, t)
Taped._diff(p::StructFoo, q::StructFoo) = Taped._containerlike_diff(p, q)

mutable struct MutableFoo
    a::Float64
    b::AbstractVector
    MutableFoo(a::Float64, b::Vector{Float64}) = new(a, b)
    MutableFoo(a::Float64) = new(a)
end

Base.:(==)(a::MutableFoo, b::MutableFoo) = equal_field(a, b, :a) && equal_field(a, b, :b)

function Taped._add_to_primal(p::MutableFoo, t::MutableTangent)
    return Taped._containerlike_add_to_primal(p, t)
end
Taped._diff(p::MutableFoo, q::MutableFoo) = Taped._containerlike_diff(p, q)

test_struct_partial_init(a::Float64) = StructFoo(a).a

test_mutable_partial_init(a::Float64) = MutableFoo(a).a

function test_naive_mat_mul!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}) where {T<:Real}
    for p in 1:size(C, 1)
        for q in 1:size(C, 2)
            C[p, q] = zero(T)
            for r in 1:size(A, 2)
                C[p, q] += A[p, r] * B[r, q]
            end
        end
    end
    return C
end

test_diagonal_to_matrix(D::Diagonal) = Matrix(D)

relu(x) = max(x, zero(x))

test_mlp(x, W1, W2) = W2 * relu.(W1 * x)

const TEST_FUNCTIONS = [
    (false, test_sin, 1.0),
    (false, test_cos_sin, 2.0),
    (false, test_isbits_multiple_usage, 5.0),
    (false, test_getindex, [1.0, 2.0]),
    (false, test_mutation!, [1.0, 2.0]),
    (false, test_while_loop, 2.0),
    (false, test_for_loop, 3.0),
    (false, test_mutable_struct_basic, 5.0),
    (false, test_mutable_struct_basic_sin, 5.0),
    (false, test_mutable_struct_setfield, 4.0),
    (false, test_mutable_struct, 5.0),
    (false, test_struct_partial_init, 3.5),
    (false, test_mutable_partial_init, 3.3),
    (false, test_naive_mat_mul!, randn(2, 1), randn(2, 1), randn(1, 1)),
    (false, (A, C) -> test_naive_mat_mul!(C, A, A), randn(2, 2), randn(2, 2)),
    (false, sum, randn(3)),
    (false, test_diagonal_to_matrix, Diagonal(randn(3))),
    (false, ldiv!, randn(2, 2), Diagonal(randn(2)), randn(2, 2)),
    (false, kron!, randn(4, 4), Diagonal(randn(2)), randn(2, 2)),
    (false, test_mlp, randn(5, 2), randn(7, 5), randn(3, 7)),
]

function value_dependent_control_flow(x, n)
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

my_setfield!(args...) = setfield!(args...)

function _setfield!(value::MutableTangent, name, x)
    @set value.fields.$name = x
    return x
end

function Taped.rrule!!(::Taped.CoDual{typeof(my_setfield!)}, value, name, x)
    _name = primal(name)
    old_x = isdefined(primal(value), _name) ? getfield(primal(value), _name) : nothing
    function setfield!_pullback(dy, df, dvalue, ::NoTangent, dx)
        new_dx = increment!!(dx, getfield(dvalue.fields, _name).tangent)
        set_field_to_zero!!(dvalue, _name)
        new_dx = increment!!(new_dx, dy)
        old_x !== nothing && setfield!(primal(value), _name, old_x)
        return df, dvalue, NoTangent(), new_dx
    end
    y = Taped.CoDual(
        setfield!(primal(value), _name, primal(x)),
        _setfield!(shadow(value), _name, shadow(x)),
    )
    return y, setfield!_pullback
end

end
