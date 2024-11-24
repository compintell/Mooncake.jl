using Mooncake
using Test

x, dx = 2.0, 3.0
xdual = Dual(x, dx)

@testset "Manual frule" begin
    sin_rule = build_frule(sin, x)
    ydual = sin_rule(zero_dual(sin), xdual)

    @test primal(ydual) == sin(x)
    @test tangent(ydual) == dx * cos(x)
end

function func(x)
    y = sin(x)
    z = cos(y)
    return z
end

@testset "Automatic frule" begin
    func_rule = build_frule(func, x)
    ydual = func_rule(zero_dual(func), xdual)

    @test primal(ydual) == cos(sin(x))
    @test tangent(ydual) ≈ dx * -sin(sin(x)) * cos(x)
end
