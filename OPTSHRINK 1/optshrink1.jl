using LinearAlgebra
"""
`Xh = optshrink1(Y::AbstractMatrix, r::Int)
Perform rank−r denoising of data matrix Y using OptShrink
using the method by Prof. Raj Rao Nadakuditi in this May 2014
IEEE Transactions on Information paper:
http://doi.org/10.1109/TIT.2014.2311661
In:
−Y     2D array where Y = X + noise and goal is to estimate X
−r     estimated rank of X
Out:
−Xh    rank−r estimate of X using OptShrink weights for SVD components
This version works only if the size of Y is sufficiently small,
because it performs calculations involving arrays roughly of
size(Y'*Y) and size(Y*Y') so neither dimension of Y can be large.
"""
function optshrink1(Y::AbstractMatrix, r::Int)
    U,E,V=svd(Y)
    Er=Diagonal(E[(r+1):end])
    n,m=size(Y)
    p=n-r
    q=m-r
    if n<m
        c1=zeros(p,m-n)
        Er=hcat(Er,c1)
    elseif m<n
        r1=zeros(n-m,q)
        Er=vcat(Er,r1)
    end

    o=zeros(r)
    for i=1:r
        z=E[i]
        z2=z^2
        i1 = Matrix(1I, p, p)
        i2 = Matrix(1I, q, q)
        A=inv((z2*i1-(Er*Er')))
        B=inv((z2*i2-(Er'*Er)))

        t1=tr(z*A)/n
        t2=tr((-2*z2*B^2)+B)/m
        t3=tr(z*B)/m
        t4=tr((-2*z2*A^2)+A)/n

        D=t1*t3
        Ddash=(t1*t2)+(t3*t4)

        o[i]=-2*D/Ddash
    end
    Sopt_hat=U[:,1:r]*Diagonal(o)*V[:,1:r]'

    return Sopt_hat
end


# include("optshrink1.jl")
# uncomment if you need this
using Random: seed!
using LinearAlgebra
seed!(0)
X = randn(30)*randn(20)'
Y = X + 40*randn(size(X))
Xh_opt = optshrink1(Y, 1)
U,E,V=svd(Y, full=true)
r=rank(Y)
m,n=size(Y)
o=zeros(r)
o[1]=E[1]
Xh_lr = U[:,1:r]*Diagonal(o)*V[:,1:r]'
 # you finish this to make the conventional rank−1 approximation
@show norm(Xh_opt-X)
@show norm(Xh_lr-X)
