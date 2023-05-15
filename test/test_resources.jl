module TestResources

test_one(x) = sin(x)
test_two(x) = cos(sin(x))
function test_three(x)
    for _ in 1:5
        x = sin(x)
    end
    return x
end
function test_four(x)
    n = 3
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end
function test_five(x::AbstractVector)
    x[1] = 0.0
    return x[1] + x[2]
end

mutable struct Foo
    x::Real
end

function test_six(x)
    foo = Foo(x)
    foo.x = sin(foo.x)
    return foo.x
end

const UNARY_FUNCTIONS = [
    (test_one, 1.0),
    (test_two, 2.0),
    (test_three, 3.0),
    (test_four, 2.0),
    # (test_five, ones(3)),
    (test_six, 5.0),
]

function value_dependent_control_flow(x, n)
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

end
