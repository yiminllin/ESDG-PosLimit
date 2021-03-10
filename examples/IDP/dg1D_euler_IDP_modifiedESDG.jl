using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack


push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
using EntropyStableEuler
#using EntropyStableEuler.Fluxes1D

function wavespeed_1D(rho,rhou,E)
    p = pfun_nd(rho,(rhou,),E)
    cvel = @. sqrt(γ*p/rho)
    return @. abs(rhou/rho) + cvel
end
unorm(U) = sum(map((x->x.^2),U))
function pfun_nd(rho, rhoU, E)
    rhoUnorm2 = unorm(rhoU)./rho
    return @. (γ-1)*(E - .5*rhoUnorm2)
end

function primitive_to_conservative_hardcode(rho,U,p)
    rhoU = rho.*U
    Unorm = unorm(U)
    E = @. p/(γ-1) + .5*rho*Unorm
    return (rho,rhoU,E)
end

function euler_fluxes_1D(rhoL,uL,betaL,rhologL,betalogL,
    rhoR,uR,betaR,rhologR,betalogR)

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = (@. .5*(rhoL+rhoR))
    uavg   = (@. .5*(uL+uR))

    unorm = (@. uL*uR)
    pa    = (@. rhoavg/(betaL+betaR))
    f4aux = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)

    FxS1 = (@. rholog*uavg)
    FxS2 = (@. FxS1*uavg + pa)
    FxS3 = (@. f4aux*uavg)

    return (FxS1,FxS2,FxS3)
end

function euler_fluxes(rhoL,uL,betaL,rhoR,uR,betaR)
    rhologL,betalogL,rhologR,betalogR = map(x->log.(x),(rhoL,betaL,rhoR,betaR))
    return euler_fluxes_1D(rhoL,uL,betaL,rhologL,betalogL,
                           rhoR,uR,betaR,rhologR,betalogR)
end



const TOL = 1e-16
"Approximation parameters"
N = 8 # The order of approximation
K = 10
T = 1.0
T = 6.0
#T = 0.0039

# # Sod shocktube
# const γ = 1.4
# const Bl = -0.5
# const Br = 0.5
# const rhoL = 1.0
# const rhoR = 0.125
# const pL = 1.0
# const pR = 0.1
# const xC = 0.0
# const GIFINTERVAL = 20
# T = 0.2

# Leblanc shocktube
const γ = 5/3
const Bl = 0.0
const Br = 9.0
const rhoL = 1.0
const rhoR = 0.001
const pL = 0.1
const pR = 1e-7
#const pR = 1e-15
const xC = 3.0
const GIFINTERVAL = 60
T = 6.0

"Mesh related variables"
VX = LinRange(Bl,Br,K+1)
EToV = transpose(reshape(sort([1:K; 2:K+1]),2,K))

"Initialize reference element"
r,_ = gauss_lobatto_quad(0,0,N)   # Reference nodes
VDM = vandermonde_1D(N,r)         # modal to nodal
Dr = grad_vandermonde_1D(N,r)/VDM # nodal differentiation
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1]) # nodal linear interpolation

Nq = N
rq,wq = gauss_lobatto_quad(0,0,Nq)
Vq = vandermonde_1D(N,rq)/VDM
M = Vq'*diagm(wq)*Vq
Mlump = zeros(size(M))
Mlump_inv = zeros(size(M))
for i = 1:Nq+1
    Mlump[i,i] = sum(M[i,:])
    Mlump_inv[i,i] = 1.0/Mlump[i,i]
end

# operators
Qr = M*Dr
B = zeros(N+1,N+1)
B[1,1] = -1
B[end,end] = 1
L = Array(spdiagm(0=>-2*ones(N+1), 1=>ones(N), -1=>ones(N)))
L[1,1] = -1
L[end,end] = -1
psi = pinv(L)*-1/2*B*ones(N+1)
S0 = zeros(N+1,N+1)
for i = 1:N+1
    for j = 1:N+1
        if L[i,j] != 0
            S0[i,j] = psi[j] - psi[i]
        end
    end
end
Qr0 = S0+1/2*B

# Drop zeros 
Qr = Matrix(droptol!(sparse(Qr),TOL))
Qr0 = Matrix(droptol!(sparse(Qr0),TOL))
B = Matrix(droptol!(sparse(B),TOL))
L = Matrix(droptol!(sparse(L),TOL))
S0 = Matrix(droptol!(sparse(S0),TOL))
S = Matrix(droptol!(sparse((Qr-Qr')/2),TOL))

rf = [-1.0;1.0]
nrJ = [-1.0;1.0]
Vf = vandermonde_1D(N,rf)/VDM

"""High order mesh"""
x = V1*VX[transpose(EToV)]
xf = Vf*x
mapM = reshape(1:2*K,2,K)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]

# """Periodic"""
# mapP[1] = mapM[end]
# mapP[end] = mapP[1]

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1.0


# """Low order mesh"""
# VX0 = [x[1:end-1,:][:]; x[end]]
# #EToV = transpose(reshape(sort([1:K; 2:K+1]),2,K)
# EToV0 = transpose(reshape(sort([1:N*K; 2:N*K+1]),2,N*K))
# x0 = VX0[transpose(EToV0)]
# xf0 = x0 
# mapM0 = reshape(1:2*N*K,2,N*K)
# mapP0 = copy(mapM0)
# mapP0[1,2:end] .= mapM0[2,1:end-1]
# mapP0[2,1:end-1] .= mapM0[1,2:end]

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1.0

"""Initial condition"""
rho_x(x) = (x <= xC) ? rhoL : rhoR
u_x(x) = 0.0
p_x(x) = (x <= xC) ? pL : pR 

rho = @. rho_x(x)
u = @. u_x(x)
p = @. p_x(x)
U = primitive_to_conservative_hardcode(rho,u,p)

function limiting_param(U_low, P_ij)
    l = 1.0
    # Limit density
    if U_low[1] + P_ij[1] < -TOL
        l = min(abs(U_low[1])/(abs(P_ij[1])+1e-14), 1.0)
    end

    # limiting internal energy (via quadratic function)
    a = P_ij[1]*P_ij[3]-1.0/2.0*P_ij[2]^2
    b = U_low[3]*P_ij[1]+U_low[1]*P_ij[3]-U_low[2]*P_ij[2]
    c = U_low[3]*U_low[1]-1.0/2.0*U_low[2]^2

    l_eps_ij = 1.0
    if b^2-4*a*c >= 0 
        r1 = (-b+sqrt(b^2-4*a*c))/(2*a)
        r2 = (-b-sqrt(b^2-4*a*c))/(2*a)
        if r1 > TOL && r2 > TOL
            l_eps_ij = min(r1,r2)
        elseif r1 > TOL && r2 < -TOL
            l_eps_ij = r1
        elseif r2 > TOL && r1 < -TOL
            l_eps_ij = r2
        end
    end 

    l = min(l,l_eps_ij)
    return l
end

function flux_lowIDP(U_i,U_j,f_i,f_j,c_ij,wavespd)
    return c_ij*(f_i+f_j)-abs(c_ij)*wavespd*(U_j-U_i)
end

function flux_high(f_i,f_j,c_ij)
    return c_ij*(f_i+f_j)
end

function flux_ES(rho_i,u_i,beta_i,rho_j,u_j,beta_j,c_ij)
    return 2*c_ij.*euler_fluxes(rho_i,u_i,beta_i,rho_j,u_j,beta_j)
end

function rhs_IDP(U,K,N,wq,S,S0,Mlump_inv)
    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[1]/(2*p)

    J = (Br-Bl)/K/2 # assume uniform interval

    # Low order and high order algebraic fluxes
    F_low       = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_high      = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_low_P     = [zeros(2),zeros(2),zeros(2)] # 1: left boundary, 2: right boundary
    F_high_P    = [zeros(2),zeros(2),zeros(2)]
    L           = zeros(N+1,N+1) # Array of limiting params
    L_P         = zeros(K-1) # limiting params at left/right boundary
    rhsU        = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]

    L_plot = zeros(K)
    L_plot2 = zeros(N+1,K)

    wavespd_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
        end
    end

    d_ii_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    wavespd = max(wavespd_arr[i,k],wavespd_arr[j,k])
                    d_ij = wavespd*abs(S0[i,j])
                    d_ii_arr[i,k] -= d_ij
                end
            end
            if i == 1
                U_left   = (k == 1) ? [rhoL; 0.0; pL/(γ-1)]    : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
                wavespd = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
                d_ij = wavespd/2.0
                d_ii_arr[i,k] -= d_ij
            end
            if i == N+1
                U_right  = (k == K) ? [rhoR; 0.0; pR/(γ-1)]    : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
                wavespd = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))
                d_ij = wavespd/2.0
                d_ii_arr[i,k] -= d_ij
            end
        end
    end
    
    dt = minimum(-J/2*M*(1 ./d_ii_arr))

    for k = 1:K
        # Assemble matrix of low and high order algebraic fluxes
        # interior of the element
        for i = 1:N+1
            for j = 1:N+1
                if i != j # skip diagonal
                    wavespd = max(wavespd_arr[i,k],wavespd_arr[j,k])
                    fluxS = flux_ES(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],S[i,j])
                    for c = 1:3
                        F_low[c][i,j]  = flux_lowIDP(U[c][i,k],U[c][j,k],flux[c][i,k],flux[c][j,k],S0[i,j],wavespd)
                        F_high[c][i,j] = fluxS[c]
                    end
                end
            end
        end

        # Assemble matrix of low and high order algebraic fluxes
        # interface of the element

        U_left   = (k == 1) ? [rhoL; 0.0; pL/(γ-1)]    : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; 0.0; rhoL/(2*pL)] : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [0.0; pL; 0.0]           : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; 0.0; pR/(γ-1)]    : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; 0.0; rhoR/(2*pR)] : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [0.0; pR; 0.0]           : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        # fluxS_l = flux_ES(Ub[1][1,k],Ub[2][1,k],Ub[3][1,k],Ub_left[1],Ub_left[2],Ub_left[3],-0.5)
        # fluxS_r = flux_ES(Ub[1][end,k],Ub[2][end,k],Ub[3][end,k],Ub_right[1],Ub_right[2],Ub_right[3],0.5)
        tmp = .-euler_fluxes(Ub[1][1,k],Ub[2][1,k],Ub[3][1,k],Ub_left[1],Ub_left[2],Ub_left[3])
        tmp2 = euler_fluxes(Ub[1][end,k],Ub[2][end,k],Ub[3][end,k],Ub_right[1],Ub_right[2],Ub_right[3])
        for c = 1:3
            F_low_P[c][1] = flux_lowIDP(U[c][1,k],U_left[c],flux[c][1,k],f_left[c],-0.5,wavespd_l) 
            F_low_P[c][2] = flux_lowIDP(U[c][end,k],U_right[c],flux[c][end,k],f_right[c],0.5,wavespd_r)

            # F_high_P[c][1] = fluxS_l[c]-wavespd_l/2*(Ub_left[c]-Ub[c][1,k])
            # F_high_P[c][2] = fluxS_r[c]-wavespd_r/2*(Ub_right[c]-Ub[c][end,k])
            F_high_P[c][1] = F_low_P[c][1]#tmp[c]-wavespd_l/2*(U_left[c]-U[c][1,k])#-euler_fluxes(Ub[c][1,k],Ub_left[c])#flux_high(Ub[c][1,k],Ub_left[c],-0.5)-wavespd_l/2*(Ub_left[c]-Ub[c][1,k])
            F_high_P[c][2] = F_low_P[c][2]#tmp2[c]-wavespd_r/2*(U_right[c]-U[c][end,k])#euler_fluxes(Ub[c][end,k],Ub_right[c])#flux_high(Ub[c][end,k],Ub_right[c],0.5)-wavespd_r/2*(Ub_right[c]-Ub[c][end,k])
        end

        # Calculate limiting parameters over interior of the element
        P_ij = zeros(3,1)
        # TODO: redundant
        U_low =  [zeros(N+1,1),zeros(N+1,1),zeros(N+1,1)]
        for c = 1:3
            U_low[c] .= sum(-F_low[c],dims=2)
            U_low[c][1] -= F_low_P[c][1]
            U_low[c][end] -= F_low_P[c][2]
            U_low[c] .= U[c][:,k]+dt/J*Mlump_inv*U_low[c]
        end
        for i = 1:N+1
            lambda_j = (i >= 2 && i <= N) ? 1/N : 1/(N+1)
            m_i = J*wq[i]
            for j = 1:N+1
                if i != j 
                    for c = 1:3
                        P_ij[c] = dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
                    end
                    L[i,j] = limiting_param([U_low[1][i]; U_low[2][i]; U_low[3][i]],P_ij)
                end
            end
        end
        
        # Symmetrize limiting parameters
        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    l_ij = min(L[i,j],L[j,i])
                    L[i,j] = l_ij
                    L[j,i] = l_ij
                end
            end
        end 

        # construct rhs
        for c = 1:3
            # With limiting
            rhsU[c][:,k] = sum((L.-1).*F_low[c] - L.*F_high[c],dims=2)

            if k > 1
                rhsU[c][1,k] += (L_P[k-1]-1)*F_low_P[c][1] - L_P[k-1]*F_high_P[c][1]
            else
                rhsU[c][1,k] += -F_high_P[c][1]#-F_low_P[c][1]
            end
            if k < K
                rhsU[c][N+1,k] += (L_P[k]-1)*F_low_P[c][2] - L_P[k]*F_high_P[c][2]
            else
                rhsU[c][N+1,k] += -F_high_P[c][2]#-F_low_P[c][2]
            end

            rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
        end

        L_plot[k] = sum(L)
        for i = 1:N+1
            L_plot2[i,k] = sum(L[i,:])
        end
        L_plot2[1,k] += 1.0
        L_plot2[end,k] += 1.0

        L_plot2[1,k] = L_plot2[1,k]/(N+1)
        L_plot2[end,k] = L_plot2[end,k]/(N+1)
        for i = 2:N
            L_plot2[i,k] = L_plot2[i,k]/N
        end
    end

    return rhsU,L_plot,L_plot2,dt
end

function rhs_IDPlow(U,K,N,Mlump_inv,p,flux,J)
    dfdx = (zeros(N+1,K),zeros(N+1,K),zeros(N+1,K))
    for i = 2:K*(N+1)-1
        for c = 1:3
            dfdx[c][i] = 1/2*(flux[c][mod1(i+1,K*(N+1))] - flux[c][mod1(i-1,K*(N+1))])
        end
    end
    dfdx[1][1] = 1/2*(flux[1][2] - 0.0)
    dfdx[2][1] = 1/2*(flux[2][2] - pL)
    dfdx[3][1] = 1/2*(flux[3][2] - 0.0)
    dfdx[1][end] = 1/2*(0.0 - flux[1][end-1])
    dfdx[2][end] = 1/2*(pR - flux[2][end-1])
    dfdx[3][end] = 1/2*(0.0 - flux[3][end-1])

    visc = (zeros(N+1,K),zeros(N+1,K),zeros(N+1,K))
    for i = 2:K*(N+1)-1
        wavespd_curr = wavespeed_1D(U[1][i],U[2][i],U[3][i])
        wavespd_R = wavespeed_1D(U[1][mod1(i+1,K*(N+1))],U[2][mod1(i+1,K*(N+1))],U[3][mod1(i+1,K*(N+1))])
        wavespd_L = wavespeed_1D(U[1][mod1(i-1,K*(N+1))],U[2][mod1(i-1,K*(N+1))],U[3][mod1(i-1,K*(N+1))])
        dL = 1/2*max(wavespd_curr,wavespd_L)
        dR = 1/2*max(wavespd_curr,wavespd_R)
        for c = 1:3
            visc[c][i] = dL*(U[c][mod1(i-1,K*(N+1))]-U[c][i]) + dR*(U[c][mod1(i+1,K*(N+1))]-U[c][i])
        end
    end

    # i = 1
    wavespd_curr = wavespeed_1D(U[1][1],U[2][1],U[3][1])
    wavespd_R = wavespeed_1D(U[1][2],U[2][2],U[3][2])
    wavespd_L = wavespeed_1D(rhoL,0.0,pL/(γ-1))
    dL = 1/2*max(wavespd_curr,wavespd_L)
    dR = 1/2*max(wavespd_curr,wavespd_R)
    visc[1][1] = dL*(rhoL-U[1][1]) + dR*(U[1][2]-U[1][1])
    visc[2][1] = dL*(0.0-U[2][1]) + dR*(U[2][2]-U[2][1])
    visc[3][1] = dL*(pL/(γ-1)-U[3][1]) + dR*(U[3][2]-U[3][1])

    # i = end
    wavespd_curr = wavespeed_1D(U[1][end],U[2][end],U[3][end])
    wavespd_R = wavespeed_1D(rhoR,0.0,pR/(γ-1))
    wavespd_L = wavespeed_1D(U[1][end-1],U[2][end-1],U[3][end-1])
    dL = 1/2*max(wavespd_curr,wavespd_L)
    dR = 1/2*max(wavespd_curr,wavespd_R)
    visc[1][end] = dL*(U[1][end-1]-U[1][end]) + dR*(rhoR-U[1][end])
    visc[2][end] = dL*(U[2][end-1]-U[2][end]) + dR*(0.0-U[2][end])
    visc[3][end] = dL*(U[3][end-1]-U[3][end]) + dR*(pR/(γ-1)-U[3][end])
    
    rhsU = (x->1/J*Mlump_inv*x).(.-dfdx.+visc)
    return rhsU
end

function rhs_high(U,K,N,Mlump_inv,S)
    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[1]/(2*p)

    J = (Br-Bl)/K/2 # assume uniform interval

    # Low order and high order algebraic fluxes
    F_high      = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_high_P    = [zeros(2),zeros(2),zeros(2)]
    rhsU        = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]

    # Determine dt
    wavespd_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
        end
    end

    for k = 1:K
        # Assemble matrix of low and high order algebraic fluxes
        # interior of the element
        for i = 1:N+1
            for j = 1:N+1
                if i != j # skip diagonal
                    # wavespd = max(wavespd_arr[i,k],wavespd_arr[j,k])
                    # for c = 1:3
                    #     F_high[c][i,j] = flux_ES(Ub[c][i,k],Ub[c][j,k],S[i,j])
                    # end
                    tmp = flux_ES(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],S[i,j])
                    for c = 1:3
                        F_high[c][i,j] = tmp[c]
                    end
                end
            end
        end
 
        # Assemble matrix of low and high order algebraic fluxes
        # interface of the element
        U_left   = (k == 1) ? [rhoL; 0.0; pL/(γ-1)]    : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; 0.0; rhoL/(2*pL)] : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [0.0; pL; 0.0]           : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; 0.0; pR/(γ-1)]    : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; 0.0; rhoR/(2*pR)] : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [0.0; pR; 0.0]           : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))
        
        tmp = .-euler_fluxes(Ub[1][1,k],Ub[2][1,k],Ub[3][1,k],Ub_left[1],Ub_left[2],Ub_left[3])
        tmp2 = euler_fluxes(Ub[1][end,k],Ub[2][end,k],Ub[3][end,k],Ub_right[1],Ub_right[2],Ub_right[3])
        for c = 1:3
            # F_high_P[c][1] = -1/2*(flux[c][1,k]+f_left[c])-wavespd_l/2*(U_left[c]-U[c][1,k])#tmp[c]-wavespd_l/2*(U_left[c]-U[c][1,k])#-euler_fluxes(Ub[c][1,k],Ub_left[c])#flux_high(Ub[c][1,k],Ub_left[c],-0.5)-wavespd_l/2*(Ub_left[c]-Ub[c][1,k])
            # F_high_P[c][2] = 1/2*(flux[c][end,k]+f_right[c])-wavespd_r/2*(U_right[c]-U[c][end,k])#tmp2[c]-wavespd_r/2*(U_right[c]-U[c][end,k])#euler_fluxes(Ub[c][end,k],Ub_right[c])#flux_high(Ub[c][end,k],Ub_right[c],0.5)-wavespd_r/2*(Ub_right[c]-Ub[c][end,k])
            F_high_P[c][1] = tmp[c]-wavespd_l/2*(U_left[c]-U[c][1,k])#-euler_fluxes(Ub[c][1,k],Ub_left[c])#flux_high(Ub[c][1,k],Ub_left[c],-0.5)-wavespd_l/2*(Ub_left[c]-Ub[c][1,k])
            F_high_P[c][2] = tmp2[c]-wavespd_r/2*(U_right[c]-U[c][end,k])#euler_fluxes(Ub[c][end,k],Ub_right[c])#flux_high(Ub[c][end,k],Ub_right[c],0.5)-wavespd_r/2*(Ub_right[c]-Ub[c][end,k])
        end
       
        # construct rhs
        for c = 1:3
            # With limiting
            rhsU[c][:,k] = -sum(F_high[c],dims=2)
            rhsU[c][1,k] += -F_high_P[c][1]
            rhsU[c][N+1,k] += -F_high_P[c][2]
            rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
        end

    end

    return rhsU
end


# Time stepping
"Time integration"
t = 0.0
U = collect(U)
resU = [zeros(size(x)),zeros(size(x)),zeros(size(x))]

# # Forward Euler
# while t < T
#     dt = 0.0001
#     rhsU = rhs_IDP(U,K,N,wq,S,S0,dt,Mlump_inv)
#     @. U = U + dt*rhsU
#     global t = t + dt
#     println("Current time $t with time step size $dt, and final time $T")
# end

# Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM
# gr(size=(300,300),ylims=(0,1.2),legend=false,markerstrokewidth=1,markersize=2)
# plt = plot(Vp*x,Vp*U[1])


Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM
gr(size=(300,300),ylims=(0,1.2),legend=false,markerstrokewidth=1,markersize=2)
plot()
dt_hist = []

while t < T
    rhsU,L_plot,L_plot2,dt = rhs_IDP(U,K,N,wq,S,S0,Mlump_inv)
    @. U = U + dt*rhsU
    push!(dt_hist,dt)
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T")  
end

plot(Vp*x,Vp*U[1])
savefig("~/Desktop/N=$N,K=$K,modifiedESDG.png")

gr(size=(300,300),ylims=(0,2*maximum(dt_hist)),legend=false,markerstrokewidth=1,markersize=2)
plot(1:length(dt_hist),dt_hist)
savefig("~/Desktop/N=$N,K=$K,modifiedESDG_dthist.png")

# dt = 0.001
# Nsteps = Int(T/dt)
# @gif for i = 1:Nsteps
#     rhsU,L_plot,L_plot2 = rhs_IDP(U,K,N,wq,S,S0,dt,Mlump_inv)
#     #rhsU = rhs_high(U,K,N,Mlump_inv,S)
#     @. U = U + dt*rhsU
#     global t = t + dt
#     println("Current time $t with time step size $dt, and final time $T")  
#     if i % GIFINTERVAL == 0  
#         plot(Vp*x,Vp*U[1])
#         # # plot(Vp*x,Vp*U[3])
#         # plot!(Bl+(Br-Bl)/K/2:(Br-Bl)/K:Br-(Br-Bl)/K/2,1 .-L_plot/N/(N+1),st=:bar,alpha=.2)
#         ptL = Bl+(Br-Bl)/K/(N+1)/2
#         ptR = Br-(Br-Bl)/K/(N+1)/2
#         hplot = (Br-Bl)/K/(N+1)
#         for k = 1:K
#             plot!(ptL+(k-1)*hplot*(N+1):hplot:ptL+k*hplot*(N+1), 1 .-L_plot2[:,k],st=:bar,alpha=0.2)
#         end
#         # #plot!(Bl+(Br-Bl)/K/(N+1)/2:(Br-Bl)/K/(N+1):Br-(Br-Bl)/K/(N+1)/2,1 .-L_plot2[:],st=:bar,alpha=.2)
#     end
# end every GIFINTERVAL

# #plot(Vp*x,Vp*U[1])