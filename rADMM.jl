using LinearAlgebra
using Random
using Statisitics
using Distributed
using Dagger
using FFTW
using Convex
using SCS

"""
"""
function rdmm_ls(A, b, N, maxiter, mu; rflag=true)
    addprocs(N)
    n = size(A,1)
    d = size(A,2)
    
    SA, Sb = preprocess_ls(A, b, N; rflag=rflag)
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
function rdmm_ridge(A, b, eta, N, maxiter; rflag=true)
    addprocs(N)
    n = size(A,1)
    d = size(A,2)
    
    SAt = preprocess_ridge(A, N; rflag=rflag)
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
function rdmm_quadreg(A, b, N, maxiter, rflag=true)
    addprocs(N)
    
    n = size(A,1)
    d = size(A,2)
    
    SA, Sb = preprocess_quadreg(A, b; rflag=true)
    x = zeros(d)
    y = zeros(d)
    lambda = [zeros(d) for i=1:N]
    
    for k=1:maxiter
        # temporarily implemented in convex while we search for a better solution
        varx = Variable(d)
        vary = Variable(d)
        
        probx = minimize(0.5*norm(SA[1]*xvar-Sb[1])^2+0.5*g(xvar)-lambda'*xvar)
        proby = minimize(0.5*norm(SA[1]*yvar-Sb[1])^2+0.5*g(yvar)-lambda'*yvar)
        
        solve!(probx, SCS.Optimizer(verbose=false)
        solve!(proby, SCS.Optimizer(verbose=false)
        
        x = evaluate(xvar)
        y = evaulate(yvar)
        
        # need Langarian
        lambda[i] = Dagger.@spawn lambda[i]+mu*(A'*A+L*I(d))*(y-x)
    end
    
    xstar = fetch(x)
    lambdastar = fetch(lambda[1])
    
    rmprocs(workers())
    return xstar, lambdastar
end

"""
"""
function rdmm_socp(A, wy, wx, N, maxiter; rflag=true)
    addprocs(N)
    n = size(A,1)
    d = size(A,2)
    
    Ahat = vcat(A, I(n))
    SAhat = preprocess_socp(Ahat)
    z = [zeros(n) for i=1:N]
    lambdavector = (A'*wy-wx)/N
    lambda = [lambdavector for i=1:N]
    
    for k=1:maxiter
        for i=1:N
            z[i] = Dagger.@spawn - (SAhat[i]'*SAhat[i]) \ (lambda[i])
            lambda[i] = Dagger.@spawn lambda[i]+mu*Ahat'*Ahat*(z[i]-mean(z))
        end
    end
    
    zstar = fetch(z[1])
    lambdastar = fetch(lambda[1])
    
    rmprocs(workers())
    return zstar,lambdastar
end

"""
Utility function to generate the random components needed for the delta-stable
decomposition of identity.
Inputs:
    n - Number of dimensions
    N - Number of agents
    
Outputs:
    D - Matrix with iid random Rademacher entries, i.e., +/- 1 entries.
    dividedindices - Partition of 1:n into N pieces roughly equal-sized pieces
"""
function generatePD(n, N; rflag=true)
    dividedindices = rflag ? Iterators.partition(randperm(n), Int(ceil(n/N))) :
        Iterators.partition(collect(1:n), Int(ceil(n/N))) 
    D = rflag ? Diagonal(sign.(rand(n) .- 0.5)) : I(n)
    return dividedindices, D
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
function preprocess_ls(A, b, N; rflag=true)
    # TO DO: Allow for more general choices of N
    # TO DO: Check correctness
    n = size(A, 1)
    d = size(A, 2)
    if n%N != 0
        throw(DimensionMismatch("N must divide the number of rows of A"))
    elseif N > n/d
        throw(DimensionMismatch("N can be at most n/d"))
    end
    
    dividedindices, D = generatePD(n, N; rflag=rflag)
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
function preprocess_ridge(A, N; rflag=true)
    # TO DO: Allow for more general choices of N
    # TO DO: Check correctness
    n = size(A, 1)
    d = size(A, 2)
    if n%N != 0
        throw(DimensionMismatch("N must divide the number of rows of A"))
    end
    
    dividedindices, D = generatePD(n, N; rflag=rflag)
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
function preprocess_quadreg(A, b; rflag=true)
    # TO DO: Check to make sure we don't need n >= 2*d
    n = size(A, 1)
    d = size(A, 2)
    if n%2 != 0
        throw(DimensionMismatch("2 must divide the number of rows of A"))
    end
    
    dividedindices, D = generatePD(n, N; rflag=rflag)
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
Utility function to preprocess A and b for RDMM Regularized LS
Inputs:
    Ahat - A matrix of size n by (d+n), where the right-most n by n block is the
           identity matrix.

Outputs:
    SAhat - List of all S_i*Ahat.
"""
function preprocess_quadreg(Ahat; rflag=true)
    # TO DO: Check to make sure we don't need n >= 2*d
    n = size(A, 1)
    d = size(A, 2)
    if n%2 != 0
        throw(DimensionMismatch("N must divide the number of rows of A"))
    end
    
    dividedindices, D = generatePD(n, N; rflag=rflag)
    HDAhat = FFTW.r2r(D*Ahat, FFTW.DHT, 1)
    
    SAhat = []
    for indexcollection in dividedindices
        push!(SAhat, HDAhat[indexcollection, :] / 
            sqrt(N*length(indexcollection)))
    end
    
    return SAhat
end

