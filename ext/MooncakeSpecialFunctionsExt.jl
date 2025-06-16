module MooncakeSpecialFunctionsExt

using SpecialFunctions, Mooncake
using Base: IEEEFloat

import Mooncake: DefaultCtx, @from_chain_rule, @zero_derivative

@from_chain_rule DefaultCtx Mode Tuple{typeof(airyai),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(airyaix),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(airyaiprime),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(airybi),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(airybiprime),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(besselj0),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(besselj1),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(bessely0),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(bessely1),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(dawson),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(digamma),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(erf),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(erf),IEEEFloat,IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(erfc),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(logerfc),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(erfcinv),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(erfcx),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(logerfcx),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(erfi),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(erfinv),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(gamma),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(invdigamma),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(trigamma),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(polygamma),Integer,IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(beta),IEEEFloat,IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(logbeta),IEEEFloat,IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(logabsgamma),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(loggamma),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(expint),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(expintx),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(expinti),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(sinint),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(cosint),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(ellipk),IEEEFloat}
@from_chain_rule DefaultCtx Mode Tuple{typeof(ellipe),IEEEFloat}

@zero_derivative DefaultCtx Tuple{typeof(logfactorial),Integer}

end
