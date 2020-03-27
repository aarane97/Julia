# TODO
using LinearAlgebra
"""
    labels = classify_image(test, train, K::Int)

Classify `test` signals using `K`-dimensional subspaces
found from `train`ing data via SVD

In:
* `test` `n x p` matrix whose columns are vectorized test images to be classified
* `train` `n x m x 10` array containing `m` training images for each digit 0-9 (in ascending order)
* `K` in `[1, min(n, m)]` is the number of singular vectors to use during classification

Out:
`labels` vector of length `p` containing the classified digits (0-9) for each test image
"""
function classify_image(test, train, K::Int)
    (n,p) = size(test)
    (_, m, d) = size(train)
    Q=zeros(n,K,d)
    err=zeros(d, p)
    for i=1:d
        U,_,_=svd(train[:,:,i])
        Ur=U[:,1:K]
        err[i,:] = sum((test-Ur*(Ur'*test)).^2, dims=1)
    end
    idx = findmin(err, dims=1)[2] # These are linear indices!!
    idx = [CartesianIndices(size(err))[i][1] for i in vec(idx)] # think about this!
    labels = idx .-1 # Convert to digits in (0−9)
    return labels
end
