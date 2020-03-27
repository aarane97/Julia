using LinearAlgebra
using Plots
"""`x = nnlsgd(A, b ; mu=0, x0=zeros(size(A,2)), nIters::Int=200)
`Performs projected gradient descent to solve the least squares problem:
``\\argmin_{x \\geq 0} 0.5 \\| b−A x \\|_2`` with nonnegativity constraint.
In:−`A` `m x n` matrix−`b` vector of length `m`
Option:−`mu` step size to use, and must satisfy ``0 < mu < 2 / \\sigma_1(A)^2``
to guarantee convergence,where ``\\sigma_1(A)`` is the first (largest) singular value.
Ch.5 will explain a default value for `mu`−`x0` is the initial starting vector (of length `n`) to use.
Its default value is all zeros for simplicity.−`nIters` is the number of iterations to perform (default 200)
Out:`x` vector of length `n` containing the approximate LS solution"""
function nnlsgd(A, b ; mu::Real=0, x0=zeros(size(A,2)), nIters::Int=200)
    prevx=x0
    x = zeros(size(x0))
    y = zeros(size(A,2),nIters)
    _,E,_=svd(A)
    if mu==0
        mu=1/(opnorm(A,Inf)*opnorm(A,1))
    end
    for i=1:nIters
     x=max.(0,prevx-mu*transpose(A)*(A*prevx-b))
     prevx=x
     y[:,i]=x
    end
    #return x
    return x,y
end


using Random: seed!
using Statistics: mean
seed!(0)
m = 100; n = 50; sigma = 0.3
A = randn(m,n)
xtrue = rand(n) # note that xtrue is non−negative
b = A*xtrue + sigma*randn(m)
x0 = A \ b; x0[x0 .<= 0] .= mean(x0[x0 .> 0]) # reasonable initial guess

#mu=opnorm(A'A,2)
"""mu=1/(opnorm(A,2)^2)
x=nnlsgd(A, b ;mu=mu, x0=x0, nIters=100)
display(x[1:3])

using Optim # you will likely need to add this package
using LinearAlgebra: norm
lower = zeros(n)

upper = fill(Inf, (n,))
inner_optimizer = GradientDescent()
f = x-> 1/2*norm(A*x-b)^2 # cost function
function grad!(g, x) # its gradient
    g[:] = A'*(A*x-b)
end
results = optimize(f, grad!, lower, upper, x0,
    Fminbox(inner_optimizer), Optim.Options(g_tol=1e-12))
xnnls = results.minimizer
display(xnnls[5:7])"""

mu=0.1/(opnorm(A,2)^2)
_,x=nnlsgd(A, b ;mu=mu, x0=x0, nIters=100)
display(x[1:3])

using Optim # you will likely need to add this package
using LinearAlgebra: norm
lower = zeros(n)

upper = fill(Inf, (n,))
inner_optimizer = GradientDescent()
f = x-> 1/2*norm(A*x-b)^2 # cost function
function grad!(g, x) # its gradient
    g[:] = A'*(A*x-b)
end
results = optimize(f, grad!, lower, upper, x0,
    Fminbox(inner_optimizer), Optim.Options(g_tol=1e-12))
xnnls = results.minimizer
xm=ones(size(x))
xm=xm.*xnnls
k=1:100
R=zeros(size(x,2))
for i=1:size(x,2)
        R[i]=log(norm(xm[:,i]-x[:,i],2))
end
plot(k,R, label="Mu=0.1/(sigma_1)^2")

mu=0.5/(opnorm(A,2)^2)
_,x=nnlsgd(A, b ;mu=mu, x0=x0, nIters=100)
display(x[1:3])

using Optim # you will likely need to add this package
using LinearAlgebra: norm
lower = zeros(n)

upper = fill(Inf, (n,))
inner_optimizer = GradientDescent()
f = x-> 1/2*norm(A*x-b)^2 # cost function
function grad!(g, x) # its gradient
    g[:] = A'*(A*x-b)
end
results = optimize(f, grad!, lower, upper, x0,
    Fminbox(inner_optimizer), Optim.Options(g_tol=1e-12))
xnnls = results.minimizer

xm=ones(size(x))
xm=xm.*xnnls
k=1:100
xm=ones(size(x))
xm=xm.*xnnls
k=1:100
R=zeros(size(x,2))
for i=1:size(x,2)
        R[i]=log(norm(xm[:,i]-x[:,i],2))
end
plot!(k, R, label="Mu=0.5/(sigma_1)^2")

mu=1/(opnorm(A,2)^2)
_,x=nnlsgd(A, b ;mu=mu, x0=x0, nIters=100)
display(x[1:3])

using Optim # you will likely need to add this package
using LinearAlgebra: norm
lower = zeros(n)

upper = fill(Inf, (n,))
inner_optimizer = GradientDescent()
f = x-> 1/2*norm(A*x-b)^2 # cost function
function grad!(g, x) # its gradient
    g[:] = A'*(A*x-b)
end
results = optimize(f, grad!, lower, upper, x0,
    Fminbox(inner_optimizer), Optim.Options(g_tol=1e-12))
xnnls = results.minimizer

xm=ones(size(x))
xm=xm.*xnnls
k=1:100
xm=ones(size(x))
xm=xm.*xnnls
k=1:100
R=zeros(size(x,2))
for i=1:size(x,2)
        R[i]=log(norm(xm[:,i]-x[:,i],2))
end
plot!(k, R, label="Mu=1/(sigma_1)^2")

mu=1.9/(opnorm(A,2)^2)
_,x=nnlsgd(A, b ;mu=mu, x0=x0, nIters=100)
display(x[1:3])

using Optim # you will likely need to add this package
using LinearAlgebra: norm
lower = zeros(n)

upper = fill(Inf, (n,))
inner_optimizer = GradientDescent()
f = x-> 1/2*norm(A*x-b)^2 # cost function
function grad!(g, x) # its gradient
    g[:] = A'*(A*x-b)
end
results = optimize(f, grad!, lower, upper, x0,
    Fminbox(inner_optimizer), Optim.Options(g_tol=1e-12))
xnnls = results.minimizer

xm=ones(size(x))
xm=xm.*xnnls
k=1:100
xm=ones(size(x))
xm=xm.*xnnls
k=1:100
R=zeros(size(x,2))
for i=1:size(x,2)
        R[i]=log(norm(xm[:,i]-x[:,i],2))
end
plot!(k, R, label="Mu=1.9/(sigma_1)^2")
