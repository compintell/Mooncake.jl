module TestResources

test_sin(x) = sin(x)

test_cos_sin(x) = cos(sin(x))

test_getindex(x::AbstractVector{<:Real}) = sin(x[1])

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

function test_mutable_struct(x)
    foo = Foo(x)
    foo.x = sin(foo.x)
    return foo.x
end

const UNARY_FUNCTIONS = [
    (test_sin, 1.0),
    (test_cos_sin, 2.0),
    (test_getindex, [1.0, 2.0]),
    (test_mutation!, [1.0, 2.0]),
    (test_while_loop, 2.0),
    (test_for_loop, 3.0),
    (test_mutable_struct, 5.0),
]

function value_dependent_control_flow(x, n)
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

end
