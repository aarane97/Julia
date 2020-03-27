using LinearAlgebra
using Plots
"""
    `x = lsgd(A, b ; mu=0, x0=zeros(size(A,2)), nIters::Int=200)`
Performs gradient descent to solve the least squares problem: ``\\argmin_x 0.5 \\| b − A x \\|_2``
In:
− `A` `m x n` matrix
− `b` vector of length `m`
Option:
− `mu` step size to use, and must satisfy ``0 < mu < 2 / \\sigma_1(A)^2``
    to guarantee convergence,
    where ``\\sigma_1(A)`` is the first (largest) singular value.
    Ch.5 will explain a default value for `mu`
− `x0` is the initial starting vector (of length `n`) to use. Its default value is all zeros for simplicity.
− `nIters` is the number of iterations to perform (default 200)
    Out:
    `x` vector of length `n` containing the approximate LS solution
    """
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

using Random: seed!
m = 100; n = 50; sigma = 0.1
seed!(0) # seed random number generator
A = randn(m, n); xtrue = rand(n)
b = A * xtrue + sigma * randn(m)
P,E,Q=svd(A)
mu=(E[1,1])
display(mu)
(mu)=1/(mu)^2
M,Y=lsgd(A,b, mu=mu)
R=zeros(size(Y,2))
U=ones(size(Y))
Z=pinv(A)*b
U=U.*Z
for i=1:size(Y,2)
        R[i]=norm(Y[:,i]-U[:,i],2)
end
#display(Y)
#display((U))
#V=norm(R)
#display(R)
k=1:size(Y,2)
#display(R')
plot(k,R, label="sigma=0.1", yscale=:log, ylims=(10e-5,5))


#using Random: seed!
m = 100; n = 50; sigma = 0.5
#seed!(0) # seed random number generator
#A = randn(m, n); xtrue = rand(n)
b = A * xtrue + sigma * randn(m)
P,E,Q=svd(A)
mu=(E[1,1])
display(mu)
(mu)=1/(mu)^2
M,Y=lsgd(A,b, mu=mu)
R=zeros(size(Y,2))
U=ones(size(Y))
Z=pinv(A)*b
U=U.*Z
for i=1:size(Y,2)
        R[i]=norm(Y[:,i]-U[:,i],2)
end
#display(Y)
#display((U))
#V=norm(R)
#display(R)
k=1:size(Y,2)
#display(R')
plot!(k,R, label="sigma=0.5", yscale=:log, ylims=(10e-5,5))


#using Random: seed!
m = 100; n = 50; sigma = 1
#seed!(0) # seed random number generator
#A = randn(m, n); xtrue = rand(n)
b = A * xtrue + sigma * randn(m)
P,E,Q=svd(A)
mu=(E[1,1])
display(mu)
(mu)=1/(mu)^2
M,Y=lsgd(A,b, mu=mu)
R=zeros(size(Y,2))
U=ones(size(Y))
Z=pinv(A)*b
U=U.*Z
for i=1:size(Y,2)
        R[i]=norm(Y[:,i]-U[:,i],2)
end
#display(Y)
#display((U))
#V=norm(R)
#display(R)
k=1:size(Y,2)
#display(R')
plot!(k,R, label="sigma=1", yscale=:log, ylims=(10e-5,5))


#using Random: seed!
m = 100; n = 50; sigma = 2
#seed!(0) # seed random number generator
#A = randn(m, n); xtrue = rand(n)
b = A * xtrue + sigma * randn(m)
P,E,Q=svd(A)
mu=(E[1,1])
display(mu)
(mu)=1/(mu)^2
M,Y=lsgd(A,b, mu=mu)
R=zeros(size(Y,2))
U=ones(size(Y))
Z=pinv(A)*b
U=U.*Z
for i=1:size(Y,2)
        R[i]=norm(Y[:,i]-U[:,i],2)
end
#display(Y)
#display((U))
#V=norm(R)
#display(R)
k=1:size(Y,2)
#display(R')
plot!(k,R, label="sigma=2", yscale=:log, ylims=(10e-5,5))
