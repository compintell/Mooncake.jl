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

const UNARY_FUNCTIONS = [
    (test_one, 1.0),
    (test_two, 2.0),
    # (test_three, 3.0),
    (test_four, 2.0),
]

function value_dependent_control_flow(x, n)
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

end
