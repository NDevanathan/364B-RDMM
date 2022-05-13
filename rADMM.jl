using LinearAlgebra
using Random
using Distributed

function rADMM_LS(A,b,N,maxiter)
    lambda = 0
    x = 0
    return x,lambda
end

function rADMM_Ridge(A,b,N,maxiter)
    lambda = 0
    x = 0
    return x,lambda
end

function rADMM_QuadReg(A,b,N,maxiter)
    lambda = 0
    x = 0
    return x,lambda
end

function rADMM_SOCP(A,b,N,maxiter)
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
