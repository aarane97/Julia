function lssd(A, b; x0=zeros(size(A,2)), nIters=10)

    # Syntax: x = lssd(A, b; x0, nIters)

    # Inputs: A is a m x n matrix
    # b is a vector of length m
    # x0 is the initial starting vector (of length n) to use
    # (x0 is optional; the default is to initialize with 0 vector)
    # nIters is the number of iterations to perform (default 10)

    # Outputs: x is a vector of length n containing the approximate solution

    # Description: Performs steepest descent to solve the least squares problem
    # \min_x \|b − A x\|_2

    # Notes:
    # Because this is a quadratic cost function, there is a
    # closed−form solution for the step size each iteration,
    # so no "line search" procedure is needed.

    # A full−credit solution uses only *one* multiply by A and one by A' per iteration.

    A1 = A * x0                        # A_xk = 100x1
    xk = x0
    for i = 1:nIters
        d = (A' * (A1 - b)) .* -1    # 50x1
        A2 = A * d                   # 100x1
        alpha = (A2)' * (b - A1) / norm(A2)^2
        xk += alpha * d
        A1 += alpha * A2
    end

    return xk
end
