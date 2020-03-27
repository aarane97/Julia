using LinearAlgebra
using Plots
"""
`x = lsngd(A, b ; x0 = zeros(size(A,2)), nIters = 200, mu = 0)
Perform Nesterov−accelerated gradient descent to solve the LS problem
``\\argmin_x 0.5 \\| A x − b \\|_2``
In:
− `A` `m x n` matrix
− `b` vector of length `m`
Option:
− `x0` initial starting vector (of length n) to use; default 0 vector.
− `nIters` number of iterations to perform; default 200.
− `mu` step size, must satisfy ``0 < \\mu <= 1 / \\sigma_1(A)^2`
to guarantee convergence, where ``\\sigma_1(A)`` is the first (largest) singular value. Ch.5 will explain a default value for `mu`.
Out:
`x` vector of length `n` containing the approximate solution
"""
function lsngd(A::AbstractMatrix{<:Number}, b::AbstractVector{<:Number}; x0::AbstractVector{<:Number} = zeros(eltype(b), size(A,2)), nIters::Int = 300, mu::Real = 0)
    prevt=1
    prevz=x0
    prevx=x0
    x = zeros(size(x0))
    y = zeros(size(A,2),nIters)
    for i=1:nIters
        t=(1+sqrt(1+4*(prevt^2)))/2
        x=prevz-mu*transpose(A)*(A*prevz-b)
        prevz=x+((prevt-1)/t)*(x-prevx)
        prevx=x
        prevt=t
        y[:,i]=x
    end
    return x,y
    #return x
end

using Random: seed!
m = 100; n = 50; sigma = 0.1;
seed!(0); A = randn(m, n); xtrue = rand(n);
b = A * xtrue + sigma * randn(m);

P,E,Q=svd(A)
mu=(E[1,1])
display(mu)
(mu)=1/(mu)^2
M,Y=lsngd(A,b, mu=mu)
R=zeros(size(Y,2))
U=ones(size(Y))
Z=pinv(A)*b
U=U.*Z
for i=1:size(Y,2)
        R[i]=log10(norm(Y[:,i]-U[:,i],2)/norm(U[:,i]))
end
#display(Y)
#display((U))
#V=norm(R)
#display(R)
k=1:size(Y,2)
#display(R')
plot(,R)


function lsgd(A, b; mu::Real=0, x0=zeros(size(A,2)), nIters::Int=300)
        prevx=x0
        x = zeros(size(x0))
        y = zeros(size(A,2),nIters)
        for i=1:nIters
                x=prevx-mu*((transpose(A))*(A*prevx-b))
                y[:,i]=x
                #if x==prevx
                #        return x
                #end
                prevx=x
        end
        return x,y
        #return x
end


M,Y=lsgd(A,b, mu=mu)
R1=zeros(size(Y,2))
U=ones(size(Y))
Z=pinv(A)*b
U=U.*Z
for i=1:size(Y,2)
        R1[i]=log10(norm(Y[:,i]-U[:,i],2)/norm(U[:,i]))
end
#display(Y)
#display((U))
#V=norm(R)
#display(R)
k=1:size(Y,2)
#display(R')
plot!(k,R1, label="lsgd sigma=0.1")
