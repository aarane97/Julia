using LinearAlgebra
"""
    haveCommonRoot = common_root(p1, p2)
Determine whether the input polynomials share a common root
In:
−`p1` is a vector of length `m + 1` with `p1[1] != 0`
that defines an `m`th degree polynomial of the form:
`P1(x) = p1[1] x^m + p1[2] x^(m−1) + ... + p1[m] x + p1[m + 1]`

−`p2` is a vector of length `n + 1` with `p2[1] != 0`
that defines an `n`th degree polynomial of the form:
`P2(x) = p2[1] x^n + p2[2] x^(n−1) + ... + p2[n] x + p2[n + 1]`

Out:
−`haveCommonRoot` = `true` when `P1` and `P2` share a common root, else `false`
"""
function common_root(p1, p2)
    m = length(p1) - 1
    n = length(p2) - 1
    c1 = zeros(m, m)
    c2 = zeros(n, n)
    p1 = p1 ./ p1[1]
    p2 = p2 ./ p2[1]
    I1 = Matrix(1I, m, m)
    I2 = Matrix(1I, n, n)

    for i = 1:m
        c1[1,i] = -p1[i+1]
    end
    for i = 1:(m-1)
        c1[i+1,i] = 1
    end
    for i = 1:n
        c2[1,i] = p2[i+1]
    end
    for i = 1:(n-1)
        c2[i+1,i] = -1
    end

    c3 = kron(c1, I2) + kron(I1, c2)
    if abs(det(c3)) <= 0.001
        haveCommonRoot = true
    else
        haveCommonRoot = false
    end

    return haveCommonRoot

end
