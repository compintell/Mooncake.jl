using MistyClosures
using Mooncake
using Test
using Core.Compiler: SSAValue
const CC = Core.Compiler

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
