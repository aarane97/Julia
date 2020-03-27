using LinearAlgebra
function fista_schatten(Y, M, reg::Number, niter::Number)

    # Perform niter FISTA iterations to perform matrix completion
    # by seeking the minimizer over X of
    # 1/2 ||M .* (Y − X)|^2 + reg R(x)
    # where R(X) is the Schatten p−norm of X raised to the pth power, for p=1/2,
    # i.e., R(X) = sum_k (sig_k(X))^(1/2)

    # Input
    # Y: M by N matrix (with zeros in missing data locations)
    # M: M by N binary matrix (with ones in sampled data locations)
    # reg: regularization parameter
    # niter: # of iterations

    # Output
    # Xh M by N estimate of X after niter FISTA iterations

    X = copy(Y)
    Z = copy(X)
    Xold = copy(X)
    told = 1
    mask = convert(Array{Bool}, M .== 1)

    for k=1:niter
        Z[mask] = Y[mask]
        U, s, V = svd(Z)
        sh = shrink_p_1_2(s, reg)
        X = U * Diagonal(sh) * V'
        t = (1 + sqrt(1+4*told^2))/2
        Z = X + ((told-1)/t)*(X-Xold)
        Xold = copy(X)
        told = t
    end

    return X
end

"""function shrink_p_1_2(y, reg::Number)
    xh = zeros(size(y))
    fun = (y) -> 4/3. * y * cos(1/3. * acos(-(3^(3/2)*reg) / (4*y^(3/2))))^2
    big = y .> 3/2. * reg^(2/3.)
    xh[big] = fun.(y[big])
    return xh
end"""

using LinearAlgebra
function shrink_p_1_2(v, reg::Real)

    x = zeros(size(v))
    for i = 1:size(v,1)
        for j = 1:size(v,2)
            if v[i,j] > (reg^(2/3.) * 1.5)
                a = acos(-reg * sqrt(27)/(4 * v[i,j]^1.5)) / 3
                b = (4 * v[i,j] * cos(a)^2) / 3
                x[i,j] = b
            else
                x[i,j] = 0
            end
        end
    end
    return x
end
