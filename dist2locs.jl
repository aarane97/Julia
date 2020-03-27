using LinearAlgebra
using Random: seed!
using Statistics
"""
Xr = dist2locs(D, d)

In:
* `D` is an `n x n` matrix such that `D[i, j]` is the distance from object `i` to object `j`
* `d` is the desired embedding dimension.

Out:
* `Xr` is an `n x d` matrix whose rows contains the relative coordinates of the `n` objects

Note: MDS is only unique up to rotation and translation,
so we enforce the following conventions on Xr in this order:
* [ORDER] `Xr[:,i]` corresponds to ith largest eigenpair of `C * C'`
* [CENTER] The centroid of the coordinates is zero
* [SIGN] The largest magnitude element of `Xr[:, i]` is positive
"""
function dist2locs(D, d)
    i,j=size(D)
    S=D.^2
    S = 0.5 * (S + S')
    P_orth=I(j)-(1/j)*ones(j)*ones(j)'
    G = -0.5 * (P_orth * S * P_orth)
    V,E,_=svd(G, full=true)
    r=d
    V=V[:,1:r]
    E=E[1:r]
    Xr1=(Diagonal(E).^0.5)'*V'
    Xr=transpose(Xr1)
    for i = 1:r
        if abs(minimum(Xr[:,i])) > maximum(Xr[:,i])
            Xr[:,i] = Xr[:,i] .* -1
        end
    end
    return Xr
end

D=[0.0     8.0     8.0    11.314  12.649   4.0    8.944  8.944  4.0
8.0     0.0    11.314   8.0    12.649   4.0    8.944  4.0    8.944
8.0    11.314   0.0     8.0     5.657   8.944  4.0    8.944  4.0
11.314   8.0     8.0     0.0     5.657   8.944  4.0    4.0    8.944
12.649  12.649   5.657   5.657   0.0    12.0    4.0    8.944  8.944
4.0     4.0     8.944   8.944  12.0     0.0    8.0    5.657  5.657
8.944   8.944   4.0     4.0     4.0     8.0    0.0    5.657  5.657
8.944   4.0     8.944   4.0     8.944   5.657  5.657  0.0    8.0
4.0     8.944   4.0     8.944   8.944   5.657  5.657  8.0    0.0]

d=2

Xr=dist2locs(D, d)

Xr2=dist2locs(D, d)
scatter(Xr2[:,1],Xr2[:,2], title="Xr",
    xlabel="coordinate 1", ylabel="coordinate 2", label="")
