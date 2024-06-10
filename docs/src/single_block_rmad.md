
# Reverse-Mode AD: _how_ does it do it?

As discussed in [Reverse-Mode AD: _what_ does it do?](@ref) the purpose of reverse-mode AD is to apply the adjoint of the derivative, ``D f (x)^\ast`` to a vector ``\bar{y}``.

In principle, AD achieves this by decomposing ``f`` into the composition ``f = f_N \circ \dots \circ f_1``, we assume that we can compute the adjoint of the derivative of each ``f_n``.
Then the adjoint is
```math
D f [x]^\ast (\bar{y}) = (D f_1 [x_1]^\ast \circ \dots \circ D f_N [x_N]^\ast)(\bar{y})
```
Reverse-mode AD is performed (roughly speaking) as follows.

Forwards-Pass:
1. ``x_1 = x``, ``n = 1``
2. construct ``D f_n [x_n]^\ast``
3. let ``x_{n+1} = f_n (x_n)``
4. let ``n = n + 1``
5. if ``n < N + 1`` then go to 2

Reverse-Pass:
1. let ``\bar{x}_{N+1} = \bar{y}``
2. let ``n = n - 1``
3. let ``\bar{x}_{n} = D f_n [x_n]^\ast (\bar{x}_{n+1})``
4. if ``n = 1`` return ``\bar{x}_1`` else go to 2.

_**How does this relate to vector-Jacobian products?**_

In Euclidean space we have the collection of Jacobians ``J_n[x_n]``. By the chain rule
```math
J[x] = J_N[x_N] \dots J_1[x_1] .
```
Taking the transpose and multiplying from the left by ``\bar{y}`` yields
```math
J[x]^\top \bar{y} = J[x_N]^\top_N \dots J[x_1]^\top_1 \bar{y} .
```
Comparing this with the expression in terms of adjoints and operators, we see that composition of adjoints of derivatives has been replaced with multiplying by transposed Jacobian matrices.
This expression is likely familiar to many readers.
