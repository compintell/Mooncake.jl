@testset "DynamicPPLTapirExt" begin
    test_rule(sr(123456), DynamicPPL.istrans, DynamicPPL.VarInfo(); interface_only=true)
end
