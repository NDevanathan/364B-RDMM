using LinearAlgebra
using Random
using Distributed
using FFTW

function radmm_ls(A,b,N,maxiter)
    lambda = 0
    x = 0
    return x,lambda
end

function radmm_ridge(A,b,N,maxiter)
    lambda = 0
    x = 0
    return x,lambda
end

function radmm_quadreg(A,b,N,maxiter)
    lambda = 0
    x = 0
    return x,lambda
end

function radmm_socp(A,b,N,maxiter)
    lambda = 0
    x = 0
    return x,lambda
end

"""
Utility function to preprocess A and b for RDMM LS
Inputs:
    A - A matrix of size n by d
    b - A column vector of size n
    N - The number of agents. N should divide n (the number of rows in A).

Outputs:
    SA - List of all S_i*A
    Sb - List of all S_i*b.
"""
function preprocessAb_ls(A,b,N)
    # TO DO: Allow for more general choices of N
    # TO DO: Check correctness
    n = size(A, 1)
    if n%N != 0
        throw(DimensionMismatch("N must divide the number of rows of A"))
    end
    d = size(A, 2)
    
    dividedindices = Iterators.partition(randperm(n), Int(ceil(n/N)))
    D = Diagonal(sign.(rand(n) .- 0.5))
    HDA = FFTW.r2r(D*A, FFTW.DHT, 1)
    HDb = FFTW.r2r(D*b, FFTW.DHT)
    
    SA = []
    Sb = []
    for indexcollection in dividedindices
        push!(SA, HDA[indexcollection, :])
        push!(Sb, HDb[indexcollection, :])
    end
    
    return SA, Sb
end

"""
Utility function to preprocess A and b for RDMM Regularized LS
Inputs:
    A - A matrix of size n by d
    b - A column vector of size n

Outputs:
    SA - List of all S_i*A
    Sb - List of all S_i*b.
"""
function preprocessAb_quadreg(A,b)
    return preprocessAb_ls(A,b,2)
end

