using LinearAlgebra
"""
Ac, r = compress_image(A, p)
In:
*`A` `m x n` matrix
*`p` scalar in `(0, 1]`
Out:
*`Ac` a `m x n` matrix containing a compressed version of `A`
that can be represented using at most `(100*p)%` as many bits
required to represent `A`*`r` the rank of `Ac`
"""
function compress_image(A, p)
    U, E, V = svd(A)
    m, n = size(A)
    r = length(E)
    nr = Int(floor((m*n*p)/(m+n+1)))
    U1 = U[:,1:nr]
    E1 = E[1:nr]
    V1 = V[:,1:nr]
    Ac = U1 * Diagonal(E1) * V1'
    r = nr
    return Ac, r
end
