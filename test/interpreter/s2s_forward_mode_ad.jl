using MistyClosures
using Mooncake
using Test

x, dx = 2.0, 3.0
xdual = Dual(x, dx)

sin_rule = build_frule(sin, x)
ydual = sin_rule(zero_dual(sin), xdual)

@test primal(ydual) == sin(x)
@test tangent(ydual) == dx * cos(x)

function func(x)
    z = cos(x)
    w = sin(z)
    return w
end

ir = Base.code_ircode(func, (typeof(x),))[1][1]
dual_ir = build_frule(func, x)
comp = CC.compact!(dual_ir)

dual_ir |> typeof

oc = MistyClosure(dual_ir)

oc.oc(zero_dual(func), xdual)

func_rule = build_frule(func, x)
ydual = func_rule(zero_dual(func), xdual)

@test primal(ydual) == cos(sin(x))
@test tangent(ydual) ≈ dx * -sin(sin(x)) * cos(x)
