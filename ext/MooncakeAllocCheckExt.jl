module MooncakeAllocCheckExt

using AllocCheck, Mooncake
import Mooncake.TestUtils: check_allocs, Shim

@check_allocs check_allocs(::Shim, f::F, x...) where {F} = f(x...)

end
