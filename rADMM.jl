using LinearAlgebra
using Random
using Statisitics
using Distributed;
using Dagger

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

function radmm_socp(A,b,N,maxiter)
    lambda = 0
    x = 0
    return x,lambda
end

function preprocessA(A,N)
    #fast walsh goes here

    return x,lambda
end

function preprocessb(b,N)

    return x,lambda
end
