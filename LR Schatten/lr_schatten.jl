using LinearAlgebra
function shrink_p_1_2(v, reg::Real)
    if size(v)==()
        m,n=1,1
    elseif size(v)==(length(v),)
        m = size(v)[1]
        n=1
    else
        m,n=size(v)

    end

    x = zeros(m, n)
    for i = 1:m
        for j = 1:n
            if v[i,j] > (reg^(2/3) * 1.5)
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

"""
    lr_schatten(Y, reg::Real)
Compute the regularized low−rank matrix approximation as the minimizer over `X`
of `1/2 \\|Y−X\\|^2 + reg R(x)`
where `R(X)` is the Schatten p−norm of `X` raised to the pth power, for `p=1/2`,
i.e., `R(X) = \\sum_k (\\sigma_k(X))^{1/2}`

In
:−`Y`   `M by N` matrix
−`reg` regularization parameter
Out:
−`Xh`  `M by N `solution to above minimization problem
"""
function lr_schatten(Y, reg::Real)
    U,Ed,V = svd(Y)
    E = diagm(0 => Ed)
    Esh = shrink_p_1_2(E, reg)
    Xh = U * Esh * V'
    return Xh

end
