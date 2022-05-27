using LinearAlgebra
using Random
using Statistics
using Distributed
using FFTW
using Convex
using SCS
using Random

"""
"""
function rdmm_ls(A, b, N, maxiter, mu; rflag=true)
    n = size(A,1)
    d = size(A,2)
    
    SA, Sb = preprocess_ls(A, b, N; rflag=rflag)
    x = [rand(d,1) for i=1:N]
    lambda = [zeros(d,1) for i=1:N]
    pieces = [zeros(d,1) for i=1:N,j=1:N]
        
    for k=1:maxiter
        Threads.@threads for i=1:N
            x[i] = (SA[i]'*SA[i]) \ (SA[i]'*Sb[i]-lambda[i])
        end
        
        meanx =  mean(x)
        
        Threads.@threads for i=1:N
            for j=1:N
                pieces[i,j] = SA[j]'*SA[j]*(x[i] - meanx)
            end
            
            lambda[i] += (mu/sqrt(k))*sum([pieces[i,j] for j=1:N])
        end
    end
    
    return mean(x), lambda[1]
end

"""
"""
function rdmm_ridge(A, b, eta, N, maxiter, mu; rflag=true)
    n = size(A,1)
    d = size(A,2)
    
    SAt = preprocess_ridge(A, N; rflag=rflag)
    y = [zeros(n,1) for i=1:N]
    lambda = [zeros(n,1) for i=1:N]
    pieces = [zeros(n,1) for i=1:N,j=1:N]
    
    for k=1:maxiter
        Threads.@threads for i=1:N
            y[i] = (SAt[i]'*SAt[i]/eta+I(n)/N) \ (b/N-lambda[i])
        end
        
        meany =  mean(y)
        
        Threads.@threads for i=1:N
            for j=1:N
                pieces[i,j] = (SAt[j]'*SAt[j]+I(n)/(N^2))*(y[i] - meany)
            end
            
            lambda[i] += (mu/sqrt(k))*sum([pieces[i,j] for j=1:N])
        end
    end
    
    return A'*mean(y)/eta,lambda[1]
end

"""
"""
function rdmm_qr(A, b, g, L, maxiter, mu; rflag=true)
    n = size(A,1)
    d = size(A,2)
    
    SA, Sb = preprocess_qr(A, b; rflag=true)
    x = [zeros(d,1), zeros(d,1)]
    lambda = zeros(d,1)
    
    for k=1:maxiter
        # temporarily implemented in convex while we search for a better solution
        Threads.@threads for i=1:2
            xvar = Variable(d)
            
            probx = minimize(0.5*square(norm(SA[1]*xvar-Sb[1],2))+0.5*g(xvar)-lambda'*xvar)
            solve!(probx, SCS.Optimizer(verbose=false))
            
            x[i] = evaluate(xvar)
        end
        
        lambda += (mu/sqrt(k))*(A'*A+L*I(d))*(x[2]-x[1])
    end
    
    return (x[1]+x[2])/2, lambda
end

"""
"""
function rdmm_socp(A, wy, wx, N, maxiter, mu; rflag=true)
    n = size(A,1)
    d = size(A,2)
    
    Ahat = vcat(A, I(d))
    SAhat = preprocess_socp(Ahat,rflag=rflag)
    z = [zeros(n,1) for i=1:N]
    lambdavector = (A'*wy-wx)/N
    lambda = [lambdavector for i=1:N]
    pieces = [zeros(d,1) for i=1:N,j=1:N]
    
    for k=1:maxiter
        Threads.@threads for i=1:N
            z[i] = -(SAhat[i]'*SAhat[i]) \ lambda[i]
        end
        
        meanz =  mean(z)
        
        Threads.@threads for i=1:N
            for j=1:N
                pieces[i,j] = SAhat[j]'*SAhat[j]*(z[i] - meanz)
            end
            
            lambda[i] += (mu/sqrt(k))*sum([pieces[i,j] for j=1:N])
        end
    end
    
    return mean(z),lambda[1]
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
    N - The number of agents. N should also satisfy N <= n/d.

Outputs:
    SA - List of all S_i*A
    Sb - List of all S_i*b.
"""
function preprocess_ls(A, b, N; rflag=true)
    # TO DO: Check correctness
    n = size(A, 1)
    d = size(A, 2)
    if N > n/d
        throw(DimensionMismatch("N can be at most n/d"))
    end
    
    dividedindices, D = generatePD(n, N; rflag=rflag)
    HDA = A
    HDb = b
    if rflag
        HDA = FFTW.r2r(D*A, FFTW.DHT, 1)
        HDb = FFTW.r2r(D*b, FFTW.DHT)
    end
    
    SA = []
    Sb = []
    normfactor = rflag ? sqrt(n) : 1
    for indexcollection in dividedindices
        push!(SA, HDA[indexcollection, :] / normfactor)
        push!(Sb, HDb[indexcollection, :] / normfactor)
    end
    
    return SA, Sb
end

"""
Utility function to preprocess A for RDMM Ridge
Inputs:
    A - A matrix of size n by d.
    N - The number of agents. N should divide d (the number of columns in A).

Outputs:
    SAt - List of all S_i*At.
"""
function preprocess_ridge(A, N; rflag=true)
    # TO DO: Check correctness
    n = size(A, 1)
    d = size(A, 2)
    
    dividedindices, D = generatePD(d, N; rflag=rflag)
    HDAt = A'
    if rflag
        HDAt = FFTW.r2r(D*A', FFTW.DHT, 1)
    end
    
    SAt = []
    normfactor = rflag ? sqrt(d) : 1
    for indexcollection in dividedindices
        push!(SAt, HDAt[indexcollection, :] / normfactor)
    end
    
    return SAt
end

"""
Utility function to preprocess A and b for RDMM Quadratic w/ Regularizer
Inputs:
    A - A matrix of size n by d
    b - A column vector of size n

Outputs:
    SA - List of all S_i*A
    Sb - List of all S_i*b.
"""
function preprocess_qr(A, b; rflag=true)
    # TO DO: Check to make sure we don't need n >= 2*d
    n = size(A, 1)
    d = size(A, 2)
    
    dividedindices, D = generatePD(n, 2; rflag=rflag)
    HDA = A
    HDb = b
    if rflag
        HDA = FFTW.r2r(D*A, FFTW.DHT, 1)
        HDb = FFTW.r2r(D*b, FFTW.DHT)
    end
    
    SA = []
    Sb = []
    normfactor = rflag ? sqrt(n) : 1
    for indexcollection in dividedindices
        push!(SA, HDA[indexcollection, :] / normfactor)
        push!(Sb, HDb[indexcollection, :] / normfactor)
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
function preprocess_socp(Ahat; rflag=true)
    # TO DO: Check to make sure we don't need n >= 2*d
    nplusd = size(Ahat, 1)
    d = size(Ahat, 2)
    
    dividedindices, D = generatePD(nplusd, N; rflag=rflag)
    HDAhat = Ahat
    if rflag
        HDAhat = FFTW.r2r(D*Ahat, FFTW.DHT, 1)
    end
    
    SAhat = []
    normfactor = rflag ? sqrt(nplusd) : 1
    for indexcollection in dividedindices
        push!(SAhat, HDAhat[indexcollection, :] / normfactor)
    end
    
    return SAhat
end

