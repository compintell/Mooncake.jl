function test_increment!!(z_target::T, x::T, y::T) where {T}
    function check_aliasing(z::T, x::T) where {T}
        if ismutabletype(T)
            return z === x
        else
            tmp = map(fieldnames(T)) do f
                return check_aliasing(getfield(z, f), getfield(x, f))
            end
            return all(tmp)
        end
    end

    z = Taped.increment!!(x, y)
    @test check_aliasing(z, x)
    @test z == z_target
end

function test_randn_tangent(rng::AbstractRNG, x)
    r = randn_tangent(rng, x)
    @test typeof(r) == typeof(x)
    if !(x isa NoTangent) # don't test for singleton type
        @test randn_tangent(rng, x) !== r
    end
end

function test_tangent(rng::AbstractRNG, z_target::T, x::T, y::T) where {T}
    test_increment!!(z_target, x, y)
    test_randn_tangent(rng, x)
end

@testset "tangents" begin
    rng = Xoshiro(123456)
    test_tangent(rng, NoTangent(), NoTangent(), NoTangent())

    for T in [Float16, Float32, Float64]
        test_tangent(rng, T(9), T(5), T(4))
    end

    x = randn(5)
    y = randn(5)
    test_tangent(rng, x + y, x, y)

    x = (5.0, randn(5))
    y = (4.0, randn(5))
    z = (9.0, x[2] + y[2])
    test_tangent(rng, z, x, y)

    x = (a=5.0, b=randn(5))
    y = (a=4.0, b=rand(5))
    z = (a=9.0, b=x.b + y.b)
    test_tangent(rng, z, x, y)

    x = Taped.Tangent((x=5.0, y=randn(5)))
    y = Taped.Tangent((x=4.0, y=randn(5)))
    z = Taped.Tangent((x=9.0, y=x.fields.y + y.fields.y))
    test_tangent(rng, z, x, y)

    x = Taped.MutableTangent((x=5.0, y=randn(5)))
    y = Taped.MutableTangent((x=4.0, y=rand(5)))
    z = Taped.MutableTangent((x=9.0, y=x.fields.y + y.fields.y))
    test_tangent(rng, z, x, y)

    @testset "increment_field!!" begin
        @testset "tuple" begin
            x = (5.0, 4.0)
            y = 3.0
            @test increment_field!!(x, y, 1) == (8.0, 4.0)
            @test increment_field!!(x, y, 2) == (5.0, 7.0)
        end
        @testset "NamedTuple" begin
            x = (a=5.0, b=4.0)
            y = 3.0
            @test increment_field!!(x, y, :a) == (a=8.0, b=4.0)
            @test increment_field!!(x, y, :b) == (a=5.0, b=7.0)
        end
        @testset "Tangent" begin
            x = Tangent((a=5.0, b=4.0))
            y = 3.0
            @test increment_field!!(x, y, :a) == Tangent((a=8.0, b=4.0))
            @test increment_field!!(x, y, :b) == Tangent((a=5.0, b=7.0))
        end
        @testset "MutableTangent" begin
            x = MutableTangent((a=5.0, b=4.0))
            y = 3.0
            @test increment_field!!(x, y, :a) == MutableTangent((a=8.0, b=4.0))
            @test increment_field!!(x, y, :a) === x

            x = MutableTangent((a=5.0, b=4.0))
            y = 3.0
            @test increment_field!!(x, y, :b) == MutableTangent((a=5.0, b=7.0))
            @test increment_field!!(x, y, :b) === x
        end
    end

    @testset "set_to_zero!!" begin
        @test set_to_zero!!(NoTangent()) === NoTangent()
        for T in [Float16, Float32, Float64]
            @test set_to_zero!!(T(5.0)) === T(0.0)
        end
        @test set_to_zero!!((5.0, NoTangent())) === (0.0, NoTangent())

        nt = (a=5.0, b=NoTangent())
        @test set_to_zero!!(nt) === (a=0.0, b=NoTangent())

        x = randn(5)
        @test set_to_zero!!(x) == zero(x)
        @test set_to_zero!!(x) === x

        @test set_to_zero!!(Tangent(nt)) == Tangent((a=0.0, b=NoTangent()))

        x = MutableTangent(nt)
        @test set_to_zero!!(x) === x
        @test x == MutableTangent((a=0.0, b=NoTangent()))
    end

    @testset "set_field_to_zero!!" begin
        nt = (a=5.0, b=4.0)
        nt2 = (a=0.0, b=4.0)
        @test set_field_to_zero!!(nt, :a) == nt2
        @test set_field_to_zero!!((5.0, 4.0), 2) == (5.0, 0.0)
        @test set_field_to_zero!!(Tangent(nt), :a) == Tangent(nt2)

        x = MutableTangent(nt)
        @test set_field_to_zero!!(x, :a) == MutableTangent(nt2)
        @test set_field_to_zero!!(x, :a) === x
    end
end
