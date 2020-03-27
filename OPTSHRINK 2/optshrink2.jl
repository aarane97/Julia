using LinearAlgebra
"""
`Xh = optshrink2(Y::AbstractMatrix, r::Int)`
Perform rank−r denoising of data matrix Y using OptShrink
using the method by Prof. Raj Rao Nadakuditi in this May 2014
IEEE Transactions on Information paper:
http://doi.org/10.1109/TIT.2014.2311661
In:
−Y     2D array where Y = X + noise and goal is to estimate X
−r     estimated rank of X
Out:
−Xh    rank−r estimate of X using OptShrink weights for SVD components
This version works even if one of the dimensions of Y is large,
as long as the other is sufficiently small.
"""
function optshrink2(Y::AbstractMatrix, r::Int)
    U, E, V = svd(Y)
    Er = Diagonal(E[(r+1):end])
    n, m = size(Y)
    if n < m
        c1 = zeros(n-r, m-n)
        Er = hcat(Er, c1)
    elseif m < n
        r1 = zeros(n-m, m-r)
        Er = vcat(Er, r1)
    end

    p, q = size(Er)


    o = zeros(r)
    for i = 1:r
        zSTS = 0
        zSTS2 = 0
        z = E[i]
        z2 = z^2
        L = min(p,q)

        for j = 1:L
            zSTS = zSTS + (z/(z2 - Er[j,j]^2))
            zSTS2 = zSTS2 + (z2/(z2 - Er[j,j]^2)^2)
        end
        zSST = zSTS + (abs(p - q) / z)

        D = zSST * zSTS / (p*q)

        A = (zSTS / z) + (zSTS2 * -2)
        B = (zSST / z) + ((zSTS2 + (abs(p - q) / z2)) * -2)
        Dp = ((A * zSST) + (B * zSTS)) / (p * q)

        o[i] = -2 * D / Dp
    end
    S_opt = U[:,1:r] * Diagonal(o) * V[:,1:r]'

    return S_opt


end
# include("optshrink2.jl")
 # uncomment if needed
 using LinearAlgebra
 using Random: seed!
 seed!(0)
 X = randn(10^5)*randn(100)' / 8 # test large case now
 Y = X + randn(size(X))
 Xh_opt = optshrink2(Y, 1)
 U,E,V=svd(Y)
 r=rank(Y)
 o=zeros(r)
 o[1]=E[1]
 Xh_lr = U[:,1:r]*Diagonal(o)*V[:,1:r]'
 # Xh_lr =
 # you finish this part
 @show norm(Xh_opt-X)
 @show norm(Xh_lr-X)
