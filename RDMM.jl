using LinearAlgebra
using Random
using Statistics
using FFTW
using Convex
using SCS
using Random


"""
Algorithm 1 from (Cand`es, Jiang, Pilanci, 2021). Utilizes RDMM to solve a
least squares problem of the form ||Ax-b||_2^2
Inputs:
    A - Data matrix
    b - Data vector
    N - Number of agents
    maxiter - number of RDMM iterations to perform
    abserr - desired absolute difference in norm of mean(x) between iterations.
    relerr - desired relative difference in norm of mean(x) between iterations. If the
             difference in norms is less than the sum of absolute and relative error, the
             function stops iterating and returns the current point.
    mu - step size (best to just leave as is unless you really know to use something else)
    rflag - determines whether to use randomized preprocessing. Without rflag=true,
            this function utilizes standard distributed ADMM instead of RDMM
    ret_lambda - boolean to determine if lambda is returned
    
Outputs:
    mean(x) - optimal primal variable for the least squares problem
    lambda - final dual coefficients for distributed problem
"""
function rdmm_ls(A, b, N; maxiter=100, abserr=0.001, relerr=0.01, mu=1, rflag=true, ret_lambda = false)
    n = size(A,1)
    d = size(A,2)
    
    SA, Sb = preprocess_ls(A, b, N; rflag=rflag)
    AtStSA = [zeros(d,d) for i=1:N]
    invAtStSA = [zeros(d,d) for i=1:N]
    AtStSb = [zeros(d,1) for i=1:N]
    Threads.@threads for i=1:N
        AtStSA[i] = SA[i]'*SA[i]
        invAtStSA[i] = inv(AtStSA[i])
        AtStSb[i] = SA[i]'*Sb[i]
    end
    
    error = abserr + relerr * norm(b)
    
    x = [zeros(d,1) for i=1:N]
    xold = zeros(d,1)
    lambda = [zeros(d,1) for i=1:N]
    pieces = [zeros(d,1) for i=1:N,j=1:N]
        
    for k=1:maxiter
        xold = mean(x)
        
        Threads.@threads for i=1:N
            x[i] = invAtStSA[i]*(AtStSb[i]-lambda[i])
        end
        
        if norm(mean(x)-xold) <= error
            break
        end
        
        meanx =  mean(x)
        
        Threads.@threads for i=1:N
            for j=1:N
                pieces[i,j] = AtStSA[j]*(x[i] - meanx)
            end
            
            lambda[i] += (mu/sqrt(k))*sum([pieces[i,j] for j=1:N])
        end
    end
    
    if ret_lambda
        return mean(x), lambda
    end
    return mean(x)
end

"""
Algorithm 2 from (Cand`es, Jiang, Pilanci, 2021). Utilizes RDMM to solve a
regularized least squares problem of the form 1/2*||Ax-b||_2^2 + eta/2*||x||_2^2
Inputs:
    A - Data matrix
    b - Data vector
    eta - regularization constant
    N - Number of agents
    maxiter - number of RDMM iterations to perform
    abserr - desired absolute difference in norm of mean(x) between iterations.
    relerr - desired relative difference in norm of mean(x) between iterations. If the
             difference in norms is less than the sum of absolute and relative error, the
             function stops iterating and returns the current point.
    mu - step size (best to just leave as is unless you really know to use something else)
    rflag - determines whether to use randomized preprocessing. Without rflag=true,
            this function utilizes standard distributed ADMM instead of RDMM
    ret_lambda - boolean to determine if lambda is returned
    
Outputs:
    A'*mean(y)/eta - optimal primal variable for the least squares problem
    lambda - final dual coefficients for distributed problem
"""
function rdmm_ridge(A, b, eta, N; maxiter=100, abserr=0.001, relerr=0.01, mu=1, rflag=true, ret_lambda = false)
    n = size(A,1)
    d = size(A,2)
    
    SAt = preprocess_ridge(A, N; rflag=rflag)
    ASStAt = [zeros(n,n) for i=1:N]
    invASStAt = [zeros(n,n) for i=1:N]
    Threads.@threads for i=1:N
        trueASStAt = SAt[i]'*SAt[i]
        ASStAt[i] = trueASStAt + I(n)/(N^2)
        invASStAt[i] = inv(trueASStAt/eta + I(n)/N)
    end
    
    error = abserr + relerr * norm(b)
    
    y = [zeros(n,1) for i=1:N]
    yold = zeros(n,1)
    lambda = [zeros(n,1) for i=1:N]
    pieces = [zeros(n,1) for i=1:N,j=1:N]
    
    for k=1:maxiter
        yold = mean(y)
    
        Threads.@threads for i=1:N
            y[i] = invASStAt[i]*(b/N-lambda[i])
        end
        
        if norm(mean(y)-yold) <= error
            break
        end
        
        meany =  mean(y)
        
        Threads.@threads for i=1:N
            for j=1:N
                pieces[i,j] = ASStAt[j]*(y[i] - meany)
            end
            
            lambda[i] += (mu/sqrt(k))*sum([pieces[i,j] for j=1:N])
        end
    end
    
    if ret_lambda
        return A'*mean(y)/eta,lambda
    end
    return A'*mean(y)/eta
end

"""
Algorithm 3 from (Cand`es, Jiang, Pilanci, 2021). Utilizes RDMM to solve a
quadratic problem with an L-smooth regularizer of the form 1/2*||Ax-b||_2^2+g(x)
Note: This problem only supports two workers as it isn't guaranteed to converge with more
Inputs:
    A - Data matrix
    b - Data vector
    g - L-smooth regularizer. Must be continuously differentiable, twice differentiable,
        and the derivative must be Lipschitz continuous with Lipschitz constant L.
    L - the Lipschitz constant for the derivative of g. Does not have to be tight.
    maxiter - number of RDMM iterations to perform
    abserr - desired absolute difference in norm of mean(x) between iterations.
    relerr - desired relative difference in norm of mean(x) between iterations. If the
             difference in norms is less than the sum of absolute and relative error, the
             function stops iterating and returns the current point.
    mu - step size (best to just leave as is unless you really know to use something else)
    rflag - determines whether to use randomized preprocessing. Without rflag=true,
            this function utilizes standard distributed ADMM instead of RDMM
    ret_lambda - boolean to determine if lambda is returned
    
Outputs:
    (x+y)/2 - optimal primal variable for the least squares problem
    lambda - final dual coefficients for distributed problem
"""
function rdmm_qr(A, b, g, L; maxiter=100, abserr=0.001, relerr=0.01, mu=1, rflag=true, ret_lambda = false)
    n = size(A,1)
    d = size(A,2)
    
    SA, Sb = preprocess_qr(A, b; rflag=true)
    
    # Need to add this stuff after parallelization. This is needed for the "pieces" approach
    #AtStSA = [zeros(d,d) for i=1:2]
    #Threads.@threads for i=1:2
    #    AtStSA[i] = SA[i]'*SA[i]
    #end
    
    error = abserr + relerr * norm(b)
    
    x = zeros(d,1)
    y = zeros(d,1)
    lambda = zeros(d,1)
    
    for k=1:maxiter
        xold = x
        yold = y
        
        # temporarily implemented in convex while we search for a better solution
        xvar = Variable(d)
        yvar = Variable(d)
        
        probx = minimize(0.5*square(norm(SA[1]*xvar-Sb[1],2))+0.5*g(xvar)-lambda'*xvar)
        proby = minimize(0.5*square(norm(SA[2]*yvar-Sb[2],2))+0.5*g(yvar)+lambda'*yvar)
        solve!(probx, SCS.Optimizer(verbose=false))
        solve!(proby, SCS.Optimizer(verbose=false))
        
        x = evaluate(xvar)
        y = evaluate(yvar)
        
        if norm((x+y-xold-yold)/2) <= error
            break
        end
        
        lambda += (mu/sqrt(k))*(A'*A+L*I(d))*(y-x)
    end
    
    if ret_lambda
        return (x+y)/2, lambda
    end
    return (x+y)/2
end

"""
Algorithm 4 from (Cand`es, Jiang, Pilanci, 2021). Utilizes RDMM to minimize equation (15).
This was implemented mainly for testing and also completeness. It solves a sub-problem
required in iterations of a splitting cone solver for general cone problems. The 
problem is of the form 1/2*||Az-(A^T)^(-1)wx-wy)||_2^2 + 1/2*||z||_2^2 where (A^T)^(-1)
is the pseudo inverse of A transpose.
Inputs:
    A - Data matrix
    wy - Data vector
    wx - Data vector
    N - Number of agents
    maxiter - number of RDMM iterations to perform
    abserr - desired absolute difference in norm of mean(x) between iterations.
    relerr - desired relative difference in norm of mean(x) between iterations. If the
             difference in norms is less than the sum of absolute and relative error, the
             function stops iterating and returns the current point.
    mu - step size (best to just leave as is unless you really know to use something else)
    rflag - determines whether to use randomized preprocessing. Without rflag=true,
            this function utilizes standard distributed ADMM instead of RDMM
    ret_lambda - boolean to determine if lambda is returned
    
Outputs:
    mean(z) - optimal primal variable for the least squares problem
    lambda - final dual coefficients for distributed problem
"""
function rdmm_socp(A, wy, wx, N; maxiter=100, abserr=0.001, relerr=0.01, mu=1, rflag=true, ret_lambda = false)
    n = size(A,1)
    d = size(A,2)
    
    Ahat = vcat(A, I(d))
    SAhat = preprocess_socp(Ahat,rflag=rflag)
    AhattStSAhat = [zeros(d,d) for i=1:N]
    invAhattStSAhat = [zeros(d,d) for i=1:N]
    Threads.@threads for i=1:N
        AhattStSAhat[i] = SAhat[i]'*SAhat[i]
        invAhattStSAhat[i] = inv(AhattStSAhat[i])
    end
    
    error = abserr + relerr * (norm(wx) + norm(wy))
    
    z = [zeros(n,1) for i=1:N]
    lambdavector = (A'*wy-wx)/N
    lambda = [lambdavector for i=1:N]
    pieces = [zeros(d,1) for i=1:N,j=1:N]
    
    for k=1:maxiter
        zold = z
        
        Threads.@threads for i=1:N
            z[i] = -invAhattStSAhat*lambda[i]
        end
        
        if norm(z-zold) <= error
            break
        end
        
        meanz =  mean(z)
        
        Threads.@threads for i=1:N
            for j=1:N
                pieces[i,j] = AhattStSAhat*(z[i] - meanz)
            end
            
            lambda[i] += (mu/sqrt(k))*sum([pieces[i,j] for j=1:N])
        end
    end
    
    if ret_lambda
        return mean(z),lambda
    end
    return mean(z)
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

