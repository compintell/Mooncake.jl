@testset "test_utils" begin
    @testset "has_equal_data" begin
        @test !has_equal_data(5.0, 4.0)
        @test has_equal_data(5.0, 5.0)
        @test has_equal_data(Float64, Float64)
        @test !has_equal_data(Float64, Float32)
        @test has_equal_data(Float64.name, Float64.name)
        @test !has_equal_data(Float64.name, Float32.name)
        @test !has_equal_data(ones(1), ones(2))
        @test !has_equal_data(randn(5), randn(5))
        @test has_equal_data(ones(5), ones(5))
        @test has_equal_data(Complex(5.0, 4.0), Complex(5.0, 4.0))
        @test !has_equal_data(Complex(5.0, 4.0), Complex(5.0, 5.0))
        @test !has_equal_data(Diagonal(randn(5)), Diagonal(randn(5)))
        @test has_equal_data(Diagonal(ones(5)), Diagonal(ones(5)))
        @test has_equal_data("hello", "hello")
        @test !has_equal_data("hello", "goodbye")
        @test has_equal_data(trace(sin, 5.0), trace(sin, 5.0))
    end
end
