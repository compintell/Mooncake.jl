module TapirJETExt

    using JET, Tapir

    Tapir.TestUtils.test_opt(::Tapir.TestUtils.Shim, args...) = JET.test_opt(args...)
    Tapir.TestUtils.report_opt(::Tapir.TestUtils.Shim, tt) = JET.report_opt(tt)
end
