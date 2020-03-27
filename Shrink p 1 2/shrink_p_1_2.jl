using LinearAlgebra
using Plots
"""
out = shrink_p_1_2(v, reg::Real)
Compute minimizer of ``1/2 |v−x|^2 + reg |x|^p``
for `p=1/2` when `v` is real and nonnegative.
In:
*`v` scalar, vector, or array of (real, nonnegative) input values
*`reg` regularization parameter
Out:
*`xh`  solution to minimization problem for each element of `v`
(same size as `v`)
"""
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
reg=2
v = LinRange(0, 8*reg, 801)

xh=zeros(size(v))
xh=shrink_p_1_2(v, reg)
plot(v,xh,label="Shrinkage Function")

x1=zeros(size(v))
for i=1:801
    if v[i]>reg
        x1[i]=v[i]-reg
    elseif v[i]<-reg
        x1[i]=v[i]+reg
    else
        x1[i]=0
    end
end
plot!(v,x1,label="Soft Thresholding")

plot!(v,v,label="Identity Function")
