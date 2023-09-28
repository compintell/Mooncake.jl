@testset "reverse_mode_ad" begin
    @testset "rebind $(typeof(x))" for x in [
        5,
        5.0,
        randn(5),
        (5.0, randn(4), 3),
        (a=5.0, b=3, c=randn(2)),
        TestResources.StructFoo(5.0, randn(5)),
        TestResources.MutableFoo(5.0, randn(5)),
    ]
        test_rrule!!(Xoshiro(123456), Taped.rebind, x; interface_only=false)
    end
end
