using SparseArrays
using LinearAlgebra
"""
`A = first_diffs_2d_matrix(m, n)`
In:
− m and n are positive integers
Out:
− `A` is a `2mn x mn` sparse matrix such that `A * X[:]` computes the
first differences down the columns (along x direction)
and across the (along y direction) of the `m x n` matrix `X`
"""

function first_diffs_2d_matrix(m, n)
    Dm = spzeros(m,m)
    Dm[m,1]=1
    Dm[1,1]=-1
    for i=2:m
        Dm[i,i]=-1
        Dm[i-1,i]=1
    end
    Dn = spzeros(n,n)
    Dn[n,1]=1
    Dn[1,1]=-1
    for i=2:n
        Dn[i,i]=-1
        Dn[i-1,i]=1
    end
    A=[kron(sparse(I(n)),sparse(Dm));kron(sparse(Dn),sparse(I(m)))]
    return (A)
end
