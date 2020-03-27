function convolution(h,x)
k = length(h)
n = length(x)
m = n+k-1
H = zeros(m,n)
y = zeros(eltype(H), m)
for i=1:m
    for j=1:n
        if 1<=(i-j+1)<=k
            H[i,j]=h[i-j+1]
        else
            H[i,j]=0
        end
    end
end
y = H*x
return H,y
end
