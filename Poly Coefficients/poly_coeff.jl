using LinearAlgebra

function poly_coeff(a, b, c, d)

    # Syntax: p, q = poly_coeff(a, b, c, d)

    # Inputs: (a, b, c, d) are positive integers

    # Outputs: p and q are vectors of length 5 with p[1] = 1 and q[1] = 1 such
    # that, in the notation defined below, P3(x3) = 0 and P4(x4) = 0

    # Description: Uses Kronecker products to construct integer−valued polynomials
    # with the given (algebraic) numbers as zeros

    # Notation:
    # x1 = a + sqrt(b)
    # x2 = c − sqrt(d)
    # x3 = x1 + x2
    # x4 = x1 * x2

    # P1(x) = a quadratic monic polynomial with integer−valued
    # coefficients with a zero at x1
    # P2(x) = a quadratic monic polynomial with integer−valued
    # coefficients with a zero at x2
    # P3(x) = x^4 + p[2] x^3 + p[3] x^2 + p[4] x + p[5]
    # P4(x) = x^4 + q[2] x^3 + q[3] x^2 + q[4] x + q[5]

    P3 = zeros(5)
    P4 = zeros(5)
    b1 = -2*a
    c1 = a^2-b
    b2 = -2*c
    c2 = c^2-d
    A = [-b1 -c1; 1 0]
    B = [-b2 -c2; 1 0]
    I2 = Matrix(1I, 2, 2)
    x3 = kron(A, I2) + kron(I2, B)
    x4 = kron(A, B)
    eig3 = eigvals(x3)
    eig4 = eigvals(x4)
    r1 = eig3[1]
    r2 = eig3[2]
    r3 = eig3[3]
    r4 = eig3[4]
    s1 = eig4[1]
    s2 = eig4[2]
    s3 = eig4[3]
    s4 = eig4[4]
    P3[1] = 1
    P4[1] = 1
    P3[2] = round((r1 + r2 + r3 + r4) * -1)
    P4[2] = round((s1 + s2 + s3 + s4) * -1)
    P3[3] = round(r1*r2 + r1*r3 + r1*r4 + r2*r3 + r2*r4 + r3*r4)
    P4[3] = round(s1*s2 + s1*s3 + s1*s4 + s2*s3 + s2*s4 + s3*s4)
    P3[4] = round((r1*r2*r3 + r1*r2*r4 + r1*r3*r4 + r2*r3*r4) * -1)
    P4[4] = round((s1*s2*s3 + s1*s2*s4 + s1*s3*s4 + s2*s3*s4) * -1)
    P3[5] = round(r1*r2*r3*r4)
    P4[5] = round(s1*s2*s3*s4)
    return P3, P4

end
