module MooncakeAllocCheckExt

using AllocCheck, Mooncake
import Mooncake.TestUtils: check_allocs, Shim

@check_allocs check_allocs(::Shim, f::F, x) where {F} = f(x)
@check_allocs check_allocs(::Shim, f::F, x, y) where {F} = f(x, y)
@check_allocs check_allocs(::Shim, f::F, x, y, z) where {F} = f(x, y, z)

end
