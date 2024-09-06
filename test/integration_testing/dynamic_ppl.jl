@testset "DynamicPPLTapirExt" begin
    test_rule(Xoshiro(123456), DynamicPPL.istrans, DynamicPPL.VarInfo())
end
