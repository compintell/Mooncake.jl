module MooncakeSpecialFunctionsExt

using SpecialFunctions, Mooncake
using Base: IEEEFloat

import Mooncake: @from_rrule, DefaultCtx, @zero_adjoint

@from_rrule DefaultCtx Tuple{typeof(airyai),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(airyaix),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(airyaiprime),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(airybi),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(airybiprime),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(besselj0),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(besselj1),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(bessely0),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(bessely1),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(dawson),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(digamma),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(erf),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(erf),IEEEFloat,IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(erfc),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(logerfc),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(erfcinv),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(erfcx),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(logerfcx),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(erfi),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(erfinv),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(gamma),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(invdigamma),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(trigamma),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(polygamma),Integer,IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(beta),IEEEFloat,IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(logbeta),IEEEFloat,IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(logabsgamma),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(loggamma),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(expint),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(expintx),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(expinti),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(sinint),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(cosint),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(ellipk),IEEEFloat}
@from_rrule DefaultCtx Tuple{typeof(ellipe),IEEEFloat}

@zero_adjoint DefaultCtx Tuple{typeof(logfactorial),Integer}

end
