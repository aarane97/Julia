using LinearAlgebra
function outlying_zeros(p ; v0::Vector{<:Real}=randn(length(p)-1), nIters::Int=100)
    pflip = reverse(p,dims=1)
    p1 = p[2:end]./p[1]
    p1flip = pflip[2:end]./pflip[1]
    #compan = c -> [-reverse(c)'; [I zeros(length(c)-1)]]
    compan = (c) -> [-transpose(c); [I zeros(length(c)-1)]]

    A = compan(p1)
    B = compan(p1flip)

    v = v0
    w = v0
    for i = 1:nIters
        v = A*v./norm(A*v)
        w = B*w./norm(B*w)
    end
    #B = A*v
    zmax = v'*A*v
    #C = A-zmax*Matrix(I,(size(A)))

    N = w'*B*w
    zmin = 1/(N)
    if iszero(p1)
        zmax=0
    end
    if iszero(p1flip)||pflip[1]==0
        zmin=0
    end
    return (zmax, zmin)
end
