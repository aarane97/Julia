using LinearAlgebra
"""
`Aa = procrustes(B, A ; center=true, scale=true)`
In:
*`B` and `A` are `d x n` matrices
Option:
*`center=true/false` : consider centroids?
*`scale=true/false` : optimize alpha or leave scale as 1?
Your solution needs only to consider the defaults for these.
Out:*`Aa` `d x n` matrix containing `A` Procrustes−aligned to `B`
Returns `Aa = alpha*Q*(A−muA) + muB`, where `muB` and `muA` are
the `d x n` matrices whose rows contain copies of the centroids of
`B` and `A`, and `alpha` (scalar) and `Q` (`d x d` orthogonal matrix) are
the solutions to the Procrustes + scaling problem

`\\argmin_{alpha, muA, muB, Q: Q^T Q = I} \\|(B−muB)−alpha*Q (A−muA) \\|_F`
"""
function procrustes(B, A ; center=true, scale=true)
    d, n = size(A)
    C = ones(n,n)
    CB = (B * ones(n)) ./ n
    CA = (A * ones(n)) ./ n
    muB = zeros(d,n)
    muA = zeros(d,n)
    for i = 1:n
        muB[:,i] = CB
        muA[:,i] = CA
    end
    B0 = B - ((B ./ n) * C)
    A0 = A - ((A ./ n) * C)
    U, s, V = svd(B0*A0')
    Q = U*V'                 # Q rotation maitrx
    alpha = tr(B0*A0'*Q') / tr(A0*A0')    # alpha

    Aa = alpha * Q * (A - muA) + muB
    return Aa
end
