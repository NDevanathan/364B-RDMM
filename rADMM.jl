using LinearAlgebra
using Random
using Statisitics
using Distributed
using Dagger
using FFTW

"""
"""
function radmm_ls(A,b,N,maxiter,mu,rflag=true)
    addprocs(N)
    n = size(A,1)
    d = size(A,2)
    
    SA, Sb = preprocessAb_ls(A,b,N)
    x = [zeros(d) for i=1:N]
    lambda = [zeros(d) for i=1:N]
    
    for k=1:maxiter
        for i=1:N
            x[i] = Dagger.@spawn (SA[i]'*SA[i]) \ (SA[i]'*Sb[i]-lambda[i])
            lambda[i] = Dagger.@spawn lambda[i]+mu*A'*A*(x[i]-mean(x))
        end
    end
    
    xstar = fetch(x[1])
    lambdastar = fetch(lambda[1])
    
    rmprocs(workers())
    return xstar,lambdastar
end

"""
"""
function radmm_ridge(A,b,eta,N,maxiter,rflag=true)
    addprocs(N)
    n = size(A,1)
    d = size(A,2)
    
    SAt = preprocessA_ridge(A,N)
    y = [zeros(n) for i=1:N]
    lambda = [zeros(n) for i=1:N]
    
    for k=1:maxiter
        for i=1:N
            y[i] = Dagger.@spawn (SAt[i]'*SAt[i]+I(n)/N) \ (b/N-lambda[i])
            lambda[i] = Dagger.@spawn lambda[i]+mu*(A*A'+I(n)/N)*(y[i]-mean(y))
        end
    end
    
    ystar = fetch(y[1])
    lambdastar = fetch(lambda[1])
    
    rmprocs(workers())
    return A'*ystar/eta,lambdastar
end

"""
"""
function radmm_qr(A,b,N,maxiter,rflag=true)
    addprocs(N)
    
    # unimplemented
    
    rmprocs(workers())
    return
end

"""
"""
function radmm_socp(A,wy,wx,N,maxiter,rflag=true)
    addprocs(N)
    n = size(A,1)
    d = size(A,2)
    
    z = [zeros(n) for i=1:N]
    lambda = [(A'*wy-wx)/N for i=1:N]
    
    for k=1:maxiter
        for i=1:N
            z[i] = Dagger.@spawn -(lambda[i])\(SAhat[i]'*SAhat[i])
            lambda[i] = Dagger.@spawn lambda[i]+mu*Ahat'*Ahat*(z[i]-mean(z))
        end
    end
    
    zstar = fetch(z[1])
    lambdastar = fetch(lambda[1])
    
    rmprocs(workers())
    return zstar,lambdastar
end

"""
Utility function to preprocess A and b for RDMM LS
Inputs:
    A - A matrix of size n by d
    b - A column vector of size n
    N - The number of agents. N should divide n (the number of rows in A),
        and should also satisfy N <= n/d.

Outputs:
    SA - List of all S_i*A
    Sb - List of all S_i*b.
"""
function preprocessAb_ls(A,b,N,rflag=true)
    # TO DO: Allow for more general choices of N
    # TO DO: Check correctness
    n = size(A, 1)
    d = size(A, 2)
    if n%N != 0
        throw(DimensionMismatch("N must divide the number of rows of A"))
    elseif N > n/d
        throw(DimensionMismatch("N can be at most n/d"))
    end
    
    dividedindices = Iterators.partition(randperm(n), Int(ceil(n/N)))
    D = Diagonal(sign.(rand(n) .- 0.5))
    HDA = FFTW.r2r(D*A, FFTW.DHT, 1)
    HDb = FFTW.r2r(D*b, FFTW.DHT)
    
    SA = []
    Sb = []
    for indexcollection in dividedindices
        push!(SA, HDA[indexcollection, :] / sqrt(N*length(indexcollection)))
        push!(Sb, HDb[indexcollection, :] / sqrt(N*length(indexcollection)))
    end
    
    return SA, Sb
end

"""
Utility function to preprocess A for RDMM Ridge
Inputs:
    A - A matrix of size n by d.
    N - The number of agents. N should divide n (the number of rows in A).

Outputs:
    SAt - List of all S_i*At.
"""
function preprocessA_ridge(A,N,rflag=true)
    # TO DO: Allow for more general choices of N
    # TO DO: Check correctness
    n = size(A, 1)
    d = size(A, 2)
    if n%N != 0
        throw(DimensionMismatch("N must divide the number of rows of A"))
    end
    
    dividedindices = Iterators.partition(randperm(n), Int(ceil(n/N)))
    D = Diagonal(sign.(rand(n) .- 0.5))
    HDAt = FFTW.r2r(D*A', FFTW.DHT, 1)
    
    SAt = []
    for indexcollection in dividedindices
        push!(SAt, HDAt[indexcollection, :] / sqrt(N*length(indexcollection)))
    end
    
    return SAt
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
function preprocessAb_quadreg(A,b,rflag=true)
    # TO DO: Check to make sure we don't need n >= 2*d
    n = size(A, 1)
    d = size(A, 2)
    if n%2 != 0
        throw(DimensionMismatch("N must divide the number of rows of A"))
    end
    
    dividedindices = Iterators.partition(randperm(n), Int(ceil(n/2)))
    D = Diagonal(sign.(rand(n) .- 0.5))
    HDA = FFTW.r2r(D*A, FFTW.DHT, 1)
    HDb = FFTW.r2r(D*b, FFTW.DHT)
    
    SA = []
    Sb = []
    for indexcollection in dividedindices
        push!(SA, HDA[indexcollection, :] / sqrt(N*length(indexcollection)))
        push!(Sb, HDb[indexcollection, :] / sqrt(N*length(indexcollection)))
    end
    
    return SA, Sb
end

