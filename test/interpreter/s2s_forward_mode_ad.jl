using MistyClosures
using Mooncake
using Test
using Core.Compiler: SSAValue
const CC = Core.Compiler

@testset "s2s_forward_mode_ad" begin
    test_cases = collect(enumerate(TestResources.generate_test_functions()))[1:1]
    @testset "$(_typeof((f, x...)))" for (n, (interface_only, _, _, f, x...)) in test_cases
        sig = _typeof((f, x...))
        @info "$n: $sig"
        TestUtils.test_rule(
            Xoshiro(123456), f, x...; perf_flag, interface_only, is_primitive=false
        )
    end
end


#=
x, dx = 2.0, 3.0
xdual = Dual(x, dx)

sin_rule = build_frule(sin, x)
ydual = sin_rule(zero_dual(sin), xdual)

@test primal(ydual) == sin(x)
@test tangent(ydual) == dx * cos(x)
=#

function func2(x)
    if x > 0.0
        y = sin(x)
    else
        y = cos(x)
    end
    return y
end

x = 1.0
xdual = Dual(1.0, 2.0)

ir = Base.code_ircode(func2, (typeof(x),))[1][1]

func_rule = build_frule(func2, x)
ydual = func_rule(zero_dual(func2), xdual)

2cos(1)
