using LinearAlgebra
using Random
using Statisitics
using Distributed;
using Dagger
using FFTW

function radmm_ls(A,b,N,maxiter,mu)
    addprocs(N)
    n = size(A,1)
    d = size(A,2)
    
    x = [zeros(d) for i=1:N]
    lambda = [zeros(d) for i=1:N]
    
    for k=1:maxiter
        for i=1:N
            x[i] = Dagger.@spawn inv(SA[i]'*SA[i])*(SA[i]'*Sb[i]-lambda[i])
            lambda[i] = Dagger.@spawn lambda[i]+mu*A'*A*(x[i]-mean(x))
        end
    end
    
    xstar = fetch(x[1])
    lambdastar = fetch(lambda[1])
    
    rmprocs(workers())
    return xstar,lambdastar
end

function radmm_ridge(A,b,N,maxiter)
    addprocs(N)
    n = size(A,1)
    d = size(A,2)
    
    y = [zeros(n) for i=1:N]
    lambda = [zeros(n) for i=1:N]
    
    for k=1:maxiter
        for i=1:N
            y[i] = Dagger.@spawn inv(SAt[i]'*SAt[i]+I(n)/N)*(b/N-lambda[i])
            lambda[i] = Dagger.@spawn lambda[i]+mu*(A*A'+I(n)/N)*(y[i]-mean(y))
        end
    end
    
    ystar = fetch(y[1])
    lambdastar = fetch(lambda[1])
    
    rmprocs(workers())
    return ystar,lambdastar
end

function radmm_qr(A,b,N,maxiter)
    addprocs(N)
    n = size(A,1)
    d = size(A,2)
    
    y = [zeros(n) for i=1:N]
    lambda = [zeros(n) for i=1:N]
    
    for k=1:maxiter
        for i=1:N
            y[i] = Dagger.@spawn inv(SAt[i]'*SAt[i]+I(n)/N)*(b/N-lambda[i])
            lambda[i] = Dagger.@spawn lambda[i]+mu*(A*A'+I(n)/N)*(y[i]-mean(y))
        end
    end
    
    ystar = fetch(y[1])
    lambdastar = fetch(lambda[1])
    
    rmprocs(workers())
    return ystar,lambdastar
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

