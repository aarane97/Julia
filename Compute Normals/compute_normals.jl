using LinearAlgebra
"""
`N = compute_normals(data, L)`
In:
− `data` `m x n x d` matrix whose `d` slices contain `m x n` images
of a common scene under different lighting conditions
− `L` `3 x d` matrix whose columns are the lighting direction vectors for the images in data, with `d >= 3`
Out:
− `N` `m x n x 3` matrix containing the unit−norm surface normal vectors
for each pixel in the scene
"""
function compute_normals(data, L)
    xlen,ylen,d=size(data)
    L=mapslices(normalize,L,dims=1)
    data=reshape(data,xlen*ylen,d)
    N=data*pinv(L)
    N=reshape(N,xlen,ylen,3)
    N=mapslices(normalize,N,dims=3)
    return N
end
