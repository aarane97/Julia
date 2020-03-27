using LinearAlgebra
"""`z = orthcompnull(A, x)`
In:*`A` `m x n` matrix*`x` vector of length `n`,
 or matrix with `n` rows and many columns
 Out:*`z` : vector or matrix of size ??? (you determine this)
 Projects `x` onto the orthogonal complement of the null space
 of the input matrix `A`For full credit, your solution should be
 computationally efficient!"""
function orthcompnull(A, x)
    U, E, V= svd(A)
    N=rank(A)
    Vr=V[:,1:N]
    X=Vr*Vr'*x
    Y=pinv(A)*x
    return X,Y
end

A=[1 1 1;1 1 1;1 1 1]
x=[1;2;3]
X, Y=orthcompnull(A, x)

print(X,Y)
