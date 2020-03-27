using LinearAlgebra
# TODO - complete this function; See HW 5 problem 7 for help

"""
``x, out = ngd(grad, x0; nIters::Int = 200, L::Real = 0, fun = (x, iter) -> 0)

Implementation of Nesterov's FGD (fast gradient descent),
given the gradient of the cost function

In:
- grad is a function that takes in x and calculates the gradient of the cost function with respect to x
- x0 is an initial point
Optional:
- nIters is the number of iterations
- L is the Lipschitz constant of the derivative of the cost function
- fun is a function to evaluate every iteration

Out: (x, out)
- x is the guess of the minimizer after running nIters iterations
- out is an Array of evaluations of the fun function
"""
function ngd(grad, x0; niter::Int = 200, L::Real = 0, fun = (x, iter) -> 0)

    # these lines initialize the output array to have the correct size/type
    fun_x0 = fun(x0, 0)
    out = similar(Array{typeof(fun_x0)}, niter+1)
    out[1] = fun_x0

    # set up some variables
    x = copy(x0)
    xold = copy(x0)
    told = 1

    # run the FGD for niter iterations
    for iter=1:niter
        t =  0.5 * (1 + sqrt(1 + 4 * told^2)) # TODO: update t
        z = x + ((told - 1) / t) * (x - xold)# TODO: update z
        x = z - (1/L) * (grad(x)) # TODO: update x
        told = copy(t)
        xold = copy(x)
        out[iter+1] = fun(x, iter) # compute cost each iteration
    end

    return x, out
end
