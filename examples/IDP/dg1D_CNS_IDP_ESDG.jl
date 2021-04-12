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
using EntropyStableEuler.Fluxes1D

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

function exact_sol_Leblanc(x,t)
    xi = (x-0.33)/t
    rhoLstar = 5.4079335349316249*1e-2
    rhoRstar = 3.9999980604299963*1e-3
    vstar    = 0.62183867139173454
    pstar    = 0.51557792765096996*1e-3
    lambda1  = 0.49578489518897934
    lambda3  = 0.82911836253346982
    if xi <= -1/3
        return rhoL, 0.0, pL
    elseif xi <= lambda1
        return (0.75-0.75*xi)^3, 0.75*(1/3+xi), 1/15*(0.75-0.75*xi)^5
    elseif xi <= vstar
        return rhoLstar, vstar, pstar
    elseif xi <= lambda3
        return rhoRstar, vstar, pstar
    else
        return rhoR, 0.0, pR
    end
end



const TOL = 1e-16
"Approximation parameters"
N = 4 # The order of approximation
K = 50
T = 1.0
T = 6.0
t0 = 0.0
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
# t0 = 0.0

# # Leblanc shocktube
# const γ = 5/3
# const Bl = 0.0
# const Br = 9.0
# const rhoL = 1.0
# const rhoR = 0.001
# const pL = 0.1
# const pR = 1e-7
# #const pR = 1e-15
# const xC = 3.0
# const GIFINTERVAL = 30000
# #const GIFINTERVAL = 3000
# T = 6.0

# # Ignacio - Leblanc shocktube
# const γ = 5/3
# const Bl = 0.0
# const Br = 1.0
# const rhoL = 1.0
# const rhoR = 1e-3
# const pL = (γ-1)*1e-1
# const pR = (γ-1)*1e-10
# const xC = 0.33
# T = 2/3

# Ignacio - Leblanc shocktube 2 
const γ = 5/3
const Bl = 0.0
const Br = 1.0
const rhoL = 1.0
const rhoR = 1e-3
const pL = (γ-1)*1e-1
const pR = (γ-1)*1e-7
const uL = 0.0
const uR = 0.0
const xC = 0.33
T = 2/3
t0 = 0.1
exact_sol = exact_sol_Leblanc

"Viscous parameters"
const Re = 10000
const Ma = 0.3
const mu = 1/Re
const lambda = -2/3*mu
const Pr = .71
const cv = 1/γ/(γ-1)/Ma^2
const kappa = γ*cv*mu/Pr

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

# U_init = @. exact_sol(x,t0)
# rho_init = [x[1] for x in U_init]
# u_init = [x[2] for x in U_init]
# p_init = [x[3] for x in U_init]
# U = primitive_to_conservative_hardcode.(rho_init,u_init,p_init)
# rho = [x[1] for x in U]
# rhou = [x[2] for x in U]
# E = [x[3] for x in U]
# U = (rho,rhou,E)

function rhs_inviscid(U,K,N,Mlump_inv,S)
    J = (Br-Bl)/K/2 # assume uniform interval
    Nc = 3
    rhsU        = [zeros(N+1,K) for i = 1:Nc]

    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[1]/(2*p)

    VU = v_ufun(U...)

    wavespd_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
        end
    end

    for k = 1:K
        # Volume term (flux differencing)
        for j = 1:N+1
            for i = 1:N+1
                if i != j
                    F = euler_fluxes(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k])
                    for c = 1:Nc
                        rhsU[c][i,k] += 2*S[i,j]*F[c]
                    end
                end
            end
        end

        # Surface term (numerical fluxes)
        U_left   = (k == 1) ? [rhoL; 0.0; pL/(γ-1)]    : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; 0.0; rhoL/(2*pL)] : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [0.0; pL; 0.0]           : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; 0.0; pR/(γ-1)]    : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; 0.0; rhoR/(2*pR)] : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [0.0; pR; 0.0]           : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))
        
        F_l = euler_fluxes(Ub[1][1,k],Ub[2][1,k],Ub[3][1,k],Ub_left[1],Ub_left[2],Ub_left[3])
        F_r = euler_fluxes(Ub[1][end,k],Ub[2][end,k],Ub[3][end,k],Ub_right[1],Ub_right[2],Ub_right[3])
        for c = 1:Nc
            rhsU[c][1,k] += -F_l[c]-wavespd_l/2*(U_left[c]-U[c][1,k])
            rhsU[c][end,k] += F_r[c]-wavespd_r/2*(U_right[c]-U[c][end,k])
        end

        for c = 1:Nc
            rhsU[c][:,k] = -1/J*Mlump_inv*rhsU[c][:,k]
        end
    end

    return rhsU
end

function rhs_viscous(U,K,N,Mlump_inv,Qr)
    J = (Br-Bl)/K/2 # assume uniform interval
    Nc = 3
    rhsU  = [zeros(N+1,K) for i = 1:Nc]
    theta = [zeros(N+1,K) for i = 1:Nc]
    sigma = [zeros(N+1,K) for i = 1:Nc]

    p = pfun_nd.(U[1],U[2],U[3])
    T = @. p/U[1]/(γ-1)/cv # temperature

    VU = v_ufun(U...)

    for k = 1:K
        # Construct theta \approx dv/dx 
        # Volume term
        for c = 1:Nc
            theta[c][:,k] = Qr*VU[c][:,k]
        end

        # Surface term (numerical fluxes)
        VU_left  = (k == 1) ? [v_ufun(rhoL,0.0,pL/(γ-1))...] : [VU[1][end,k-1]; VU[2][end,k-1]; VU[3][end,k-1]]
        VU_right = (k == K) ? [v_ufun(rhoR,0.0,pR/(γ-1))...] : [VU[1][1,k+1]; VU[2][1,k+1]; VU[3][1,k+1]]
        for c = 1:Nc
            theta[c][1,k] -= 1/2*(VU_left[c]-VU[c][1,k])
            theta[c][end,k] += 1/2*(VU_right[c]-VU[c][end,k])
            theta[c][:,k] = 1/J*Mlump_inv*theta[c][:,k]
        end

        # Construct sigma
        for i = 1:N+1
            Kx = zeros(3,3)
            Kx[2,2] = (lambda-2*mu)*VU[3][i]^2
            Kx[2,3] = (lambda+2*mu)*VU[2][i]*VU[3][i]
            Kx[3,2] = Kx[2,3]
            Kx[3,3] = (lambda+2*mu)*VU[2][i]^2-γ*mu*VU[3][i]/Pr
            Kx = 1/VU[3][i]^3*Kx
        
            sigma[2][i] = Kx[2,2]*theta[2][i] + Kx[2,3]*theta[3][i]
            sigma[3][i] = Kx[3,2]*theta[2][i] + Kx[3,3]*theta[3][i]
        end

        # Constuct rhs
        # Volume term
        for c = 1:Nc
            rhsU[c][:,k] = Qr*sigma[c][:,k]
        end

        # Surface term (numerical fluxes)
        # TODO: how to enforce BC?
        # sigma_left  = (k == 1) ? [0.0;0.0;0.0] : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        # sigma_right = (k == K) ? [0.0;0.0;0.0] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]]
        sigma_left  = (k == 1) ? [sigma[1][1,k];sigma[2][1,k];sigma[3][1,k]] : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [sigma[1][end,k];sigma[2][end,k];sigma[3][end,k]] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]]

        for c = 1:Nc
            rhsU[c][1,k] -= 1/2*(sigma_left[c]-sigma[c][1,k])
            rhsU[c][end,k] += 1/2*(sigma_right[c]-sigma[c][end,k])
            rhsU[c][:,k] = 1/J*Mlump_inv*rhsU[c][:,k]
        end

    end

    return rhsU
end

function rhs_ESDG(U,K,N,Mlump_inv,S,Qr)
    rhsI = rhs_inviscid(U,K,N,Mlump_inv,S)
    rhsV = rhs_viscous(U,K,N,Mlump_inv,Qr)
    return rhsI .+ rhsV
end

function rhs_low_inviscid(U,K,N,S0)
    J = (Br-Bl)/K/2 # assume uniform interval
    Nc = 3
    rhsU = [zeros(N+1,K) for i = 1:Nc]

    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    # Derivative
    for k = 1:K
        # Volume term (flux differencing)
        for j = 1:N+1
            for i = 1:N+1
                if i != j
                    for c = 1:Nc
                        rhsU[c][i,k] += S0[i,j]*(flux[c][i,k]+flux[c][j,k])
                    end
                end
            end
        end

        # Surface term (numerical fluxes)
        f_left   = (k == 1) ? [0.0; pL; 0.0]         : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        f_right  = (k == K) ? [0.0; pR; 0.0]         : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        
        for c = 1:Nc
            rhsU[c][1,k] += -1/2*(f_left[c]+flux[c][1,k])
            rhsU[c][end,k] += 1/2*(f_right[c]+flux[c][end,k])
        end
    end

    return rhsU
end

function rhs_low_viscous(U,K,N,Qr0)
    J = (Br-Bl)/K/2 # assume uniform interval
    Nc = 3
    rhsU  = [zeros(N+1,K) for i = 1:Nc]
    theta = [zeros(N+1,K) for i = 1:Nc]
    sigma = [zeros(N+1,K) for i = 1:Nc]

    p = pfun_nd.(U[1],U[2],U[3])
    T = @. p/U[1]/(γ-1)/cv # temperature

    VU = v_ufun(U...)

    for k = 1:K
        # Construct theta \approx dv/dx 
        # Volume term
        for c = 1:Nc
            theta[c][:,k] = Qr0*VU[c][:,k]
        end

        # Surface term (numerical fluxes)
        VU_left  = (k == 1) ? [v_ufun(rhoL,0.0,pL/(γ-1))...] : [VU[1][end,k-1]; VU[2][end,k-1]; VU[3][end,k-1]]
        VU_right = (k == K) ? [v_ufun(rhoR,0.0,pR/(γ-1))...] : [VU[1][1,k+1]; VU[2][1,k+1]; VU[3][1,k+1]]
        for c = 1:Nc
            theta[c][1,k] -= 1/2*(VU_left[c]-VU[c][1,k])
            theta[c][end,k] += 1/2*(VU_right[c]-VU[c][end,k])
            theta[c][:,k] = 1/J*Mlump_inv*theta[c][:,k]
        end

        # Construct sigma
        for i = 1:N+1
            Kx = zeros(3,3)
            Kx[2,2] = (lambda-2*mu)*VU[3][i]^2
            Kx[2,3] = (lambda+2*mu)*VU[2][i]*VU[3][i]
            Kx[3,2] = Kx[2,3]
            Kx[3,3] = (lambda+2*mu)*VU[2][i]^2-γ*mu*VU[3][i]/Pr
            Kx = 1/VU[3][i]^3*Kx
        
            sigma[2][i] = Kx[2,2]*theta[2][i] + Kx[2,3]*theta[3][i]
            sigma[3][i] = Kx[3,2]*theta[2][i] + Kx[3,3]*theta[3][i]
        end

        # Constuct rhs
        # Volume term
        for c = 1:Nc
            rhsU[c][:,k] = Qr0*sigma[c][:,k]
        end

        # Surface term (numerical fluxes)
        # TODO: how to enforce BC?
        # sigma_left  = (k == 1) ? [sigma[1][1,k];sigma[2][1,k];sigma[3][1,k]] : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        # sigma_right = (k == K) ? [sigma[1][end,k];sigma[2][end,k];sigma[3][end,k]] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]]
        sigma_left  = (k == 1) ? [0.0;0.0;0.0] : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [0.0;0.0;0.0] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]]


        for c = 1:Nc
            rhsU[c][1,k] -= 1/2*(sigma_left[c]-sigma[c][1,k])
            rhsU[c][end,k] += 1/2*(sigma_right[c]-sigma[c][end,k])
        end

    end
    
    return rhsU,sigma
end

function construct_artificial_wavespd(rhoL,mL,EL,rhoR,mR,ER,F1,F2,F3)
    dij = abs(F1/(rhoL+rhoR))
    rhosum = rhoL+rhoR
    msum = mL+mR
    Esum = EL+ER
    a = rhosum*Esum-1/2*msum^2
    b = rhosum*F3+Esum*F1-msum*F2
    c = F1*F3-1/2*F2^2

    if c >= 0 
        return max(dij,0.0)
    else
        return max(dij,(b+sqrt(b^2-4*a*c))/(2*a))
    end
end

function zhang_wavespd(rhoL,rhouL,EL,sigma2L,sigma3L,pL,rhoR,rhouR,ER,sigma2R,sigma3R,pR)
    uL   = rhouL/rhoL
    eL   = (EL - .5*rhoL*uL^2)/rhoL
    tauL = sigma2L
    qL   = uL*tauL-sigma3L
    uR   = rhouR/rhoR
    eR   = (ER - .5*rhoR*uR^2)/rhoR
    tauR = sigma2R
    qR   = uR*tauR-sigma3R
    wavespdL = abs(uL)+1/(2*rhoL^2*eL)*(sqrt(rhoL^2*qL^2+2*rhoL^2*eL*abs(tauL-pL)^2)+rhoL*abs(qL))
    wavespdR = abs(uR)+1/(2*rhoR^2*eR)*(sqrt(rhoR^2*qR^2+2*rhoR^2*eR*abs(tauR-pR)^2)+rhoR*abs(qR))
    return max(wavespdL,wavespdR)
end

function zhang_wavespd(rhoL,rhouL,EL,sigma2L,sigma3L,pL)
    uL   = rhouL/rhoL
    eL   = (EL - .5*rhoL*uL^2)/rhoL
    tauL = sigma2L
    qL   = uL*tauL-sigma3L
    wavespdL = abs(uL)+1/(2*rhoL^2*eL)*(sqrt(rhoL^2*qL^2+2*rhoL^2*eL*abs(tauL-pL)^2)+rhoL*abs(qL))
    return wavespdL
end

function rhs_low_graph_visc(U,K,N,sigma,S0,S,wq)
    J = (Br-Bl)/K/2
    Nc = 3
    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]
    graph_visc = [zeros(N+1,K) for i = 1:Nc]

    wavespd_arr = zeros(N+1,K)
    beta_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
            beta_arr[i,k] = zhang_wavespd(U[1][i,k],U[2][i,k],U[3][i,k],sigma[2][i,k],sigma[3][i,k],p[i,k])
        end
    end

    dt = Inf
    d_ii_arr = zeros(N+1,K)
    # Graph viscosity
    for k = 1:K
        for j = 1:N+1
            for i = 1:N+1
                if i != j
                    # F1 = S0[i,j]*(flux[1][i,k]+flux[1][j,k])-S0[i,j]*(sigma[1][i,k]+sigma[1][j,k])
                    # F2 = S0[i,j]*(flux[2][i,k]+flux[2][j,k])-S0[i,j]*(sigma[2][i,k]+sigma[2][j,k])
                    # F3 = S0[i,j]*(flux[3][i,k]+flux[3][j,k])-S0[i,j]*(sigma[3][i,k]+sigma[3][j,k])
                    # wavespd = 2*abs(S0[i,j])*construct_artificial_wavespd(U[1][i,k],U[2][i,k],U[3][i,k],U[1][j,k],U[2][j,k],U[3][j,k],F1,F2,F3)
                    # wavespd = abs(S0[i,j])*max(wavespd_arr[i,k],wavespd_arr[j,k]) 
                    wavespd = abs(S0[i,j])*max(beta_arr[i,k],beta_arr[j,k]) 
                    # wavespd = abs(S0[i,j])*max(wavespd_arr[i,k],wavespd_arr[j,k],beta_arr[i,k],beta_arr[j,k]) 
                    #wavespd = max(wavespd_arr[i,k],wavespd_arr[j,k]) 

                    for c = 1:Nc
                        graph_visc[c][i,k] += wavespd*(U[c][j,k]-U[c][i,k])
                    end
                    d_ii_arr[i,k] += wavespd
                end
            end
        end

        # Surface term (numerical fluxes)
        U_left     = (k == 1) ? [rhoL; 0.0; pL/(γ-1)]  : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        U_right    = (k == K) ? [rhoR; 0.0; pR/(γ-1)]  : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        p_left     = (k == 1) ? pL : p[end,k-1]
        p_right    = (k == K) ? pR : p[1,k+1]
        # Assume velocity doesn't change at boundary, isothermal
        TL = pL/rhoL/(γ-1)/cv
        TR = pR/rhoR/(γ-1)/cv
        sigma_left  = (k == 1) ? [0.0;0.0;0.0] : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [0.0;0.0;0.0] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]] 
        f_left   = (k == 1) ? [0.0; pL; 0.0]           : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        f_right  = (k == K) ? [0.0; pR; 0.0]           : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        # F1 = -1/2*(flux[1][1,k]+f_left[1])+1/2*(sigma[1][1,k]+sigma_left[1])
        # F2 = -1/2*(flux[2][1,k]+f_left[2])+1/2*(sigma[2][1,k]+sigma_left[2])
        # F3 = -1/2*(flux[3][1,k]+f_left[3])+1/2*(sigma[3][1,k]+sigma_left[3])
        # wavespd_l = construct_artificial_wavespd(U_left[1],U_left[2],U_left[3],U[1][1,k],U[2][1,k],U[3][1,k],F1,F2,F3)
        # F1 = 1/2*(flux[1][end,k]+f_right[1])-1/2*(sigma[1][end,k]+sigma_right[1])
        # F2 = 1/2*(flux[2][end,k]+f_right[2])-1/2*(sigma[2][end,k]+sigma_right[2])
        # F3 = 1/2*(flux[3][end,k]+f_right[3])-1/2*(sigma[3][end,k]+sigma_right[3])
        # wavespd_r = construct_artificial_wavespd(U_right[1],U_right[2],U_right[3],U[1][end,k],U[2][end,k],U[3][end,k],F1,F2,F3)

        # wavespd_l = 1/2*max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        # wavespd_r = 1/2*max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        wavespd_l = 1/2*max(beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left))
        wavespd_r = 1/2*max(beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right))

        # wavespd_l = 1/2*max(beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left),wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        # wavespd_r = 1/2*max(beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right),wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        for c = 1:Nc
            graph_visc[c][1,k] += wavespd_l*(U_left[c]-U[c][1,k])
            graph_visc[c][end,k] += wavespd_r*(U_right[c]-U[c][end,k])
        end
        d_ii_arr[1,k] += wavespd_l
        d_ii_arr[end,k] += wavespd_r
    end

    J = (Br-Bl)/K/2
    dt = minimum(J*wq./d_ii_arr./2.0)
    
    return graph_visc,dt
end

function rhs_low(U,K,N,Mlump_inv,S0,S,wq)
    J = (Br-Bl)/K/2 # assume uniform interval
    Nc = 3
    rhsU = [zeros(N+1,K) for i = 1:Nc]
    rhsI = rhs_low_inviscid(U,K,N,S0)
    rhsV,sigma = rhs_low_viscous(U,K,N,Qr0)

    graph_visc,dt = rhs_low_graph_visc(U,K,N,sigma,S0,S,wq)

    for k = 1:K
        for c = 1:Nc
            rhsU[c][:,k] = -1/J*Mlump_inv*(rhsI[c][:,k]-graph_visc[c][:,k]-rhsV[c][:,k])
            #rhsU[c][:,k] = -1/J*Mlump_inv*(rhsI[c][:,k]-rhsV[c][:,k])
            #rhsU[c][:,k] = -1/J*Mlump_inv*(rhsI[c][:,k]-graph_visc[c][:,k])
        end
    end

    return rhsU,dt
end

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

function flux_ES(rho_i,u_i,beta_i,rho_j,u_j,beta_j,c_ij)
    return 2*c_ij.*euler_fluxes(rho_i,u_i,beta_i,rho_j,u_j,beta_j)
end

function flux_viscous(sigma_i,sigma_j,c_ij)
    return c_ij*(sigma_i+sigma_j)
end

function rhs_IDP(U,K,N,Mlump_inv,S0,S,wq)
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
    rhsU        = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]
    
    # TODO: redundant!
    _,sigma = rhs_low_viscous(U,K,N,Qr)
    wavespd_arr = zeros(N+1,K)
    beta_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
            beta_arr[i,k] = zhang_wavespd(U[1][i,k],U[2][i,k],U[3][i,k],sigma[2][i,k],sigma[3][i,k],p[i,k])
        end
    end

    L_plot = zeros(N+1,K)
    
    #dt = Inf
    # TODO: hardcoded
    dt = 1e-4
    d_ii_arr = zeros(N+1,K)
    for k = 1:K
        L = zeros(N+1,N+1)
        for i = 1:N+1
            for j = 1:N+1
                if i != j 
                    wavespd = max(wavespd_arr[i,k],wavespd_arr[j,k],beta_arr[i,k],beta_arr[j,k])
                    fluxS = flux_ES(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],S[i,j])
                    for c = 1:3
                        F_low[c][i,j]  = -flux_viscous(sigma[c][i,k],sigma[c][j,k],S[i,j])+flux_lowIDP(U[c][i,k],U[c][j,k],flux[c][i,k],flux[c][j,k],S0[i,j],wavespd)
                        F_high[c][i,j] = -flux_viscous(sigma[c][i,k],sigma[c][j,k],S[i,j])+fluxS[c]
                    end
                    d_ii_arr[i,k] += wavespd
                end
            end
        end

        EL = pL/(γ-1)+0.5*rhoL*uL^2
        ER = pR/(γ-1)+0.5*rhoR*uR^2
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        p_left   = (k == 1) ? pL : p[end,k-1]
        p_right  = (k == K) ? pR : p[1,k+1]
        # Assume velocity doesn't change at boundary, isothermal
        TL = pL/rhoL/(γ-1)/cv
        TR = pR/rhoR/(γ-1)/cv
        sigma_left  = (k == 1) ? [0.0;0.0;0.0] : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [0.0;0.0;0.0] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]] 
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]),beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]),beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right))

        d_ii_arr[1,k] += wavespd_l
        d_ii_arr[end,k] += wavespd_r

        for c = 1:3
            F_low_P[c][1] = -flux_viscous(sigma[c][1,k],sigma_left[c],-1/2)+flux_lowIDP(U[c][1,k],U_left[c],flux[c][1,k],f_left[c],-0.5,wavespd_l) 
            F_low_P[c][2] = -flux_viscous(sigma[c][end,k],sigma_right[c],1/2)+flux_lowIDP(U[c][end,k],U_right[c],flux[c][end,k],f_right[c],0.5,wavespd_r)

            F_high_P[c][1] = F_low_P[c][1]
            F_high_P[c][2] = F_low_P[c][2]
        end

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
            lambda_j = 1/N
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
        
        # elementwise limiting
        l = 1.0
        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    l = min(l,L[i,j])
                end
            end
        end
        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    L[i,j] = l
                    L[j,i] = l
                end
            end
        end 

        for c = 1:3
            # With limiting
            rhsU[c][:,k] = sum((L.-1).*F_low[c] - L.*F_high[c],dims=2)

            rhsU[c][1,k] += -F_high_P[c][1]
            rhsU[c][N+1,k] += -F_high_P[c][2]

            rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
        end

        for i = 1:N+1
            L_plot[i,k] = l
        end
    end

    dt = min(dt,minimum(J*wq./d_ii_arr./2.0))
    
    return rhsU,dt,L_plot
end

function rhs_IDP(U,K,N,Mlump_inv,S0,S,wq,dt)
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
    rhsU        = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]
    
    # TODO: redundant!
    _,sigma = rhs_low_viscous(U,K,N,Qr)
    wavespd_arr = zeros(N+1,K)
    beta_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
            beta_arr[i,k] = zhang_wavespd(U[1][i,k],U[2][i,k],U[3][i,k],sigma[2][i,k],sigma[3][i,k],p[i,k])
        end
    end

    L_plot = zeros(N+1,K)
    
    for k = 1:K
        L = zeros(N+1,N+1)
        for i = 1:N+1
            for j = 1:N+1
                if i != j 
                    wavespd = max(wavespd_arr[i,k],wavespd_arr[j,k],beta_arr[i,k],beta_arr[j,k])
                    fluxS = flux_ES(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],S[i,j])
                    for c = 1:3
                        F_low[c][i,j]  = -flux_viscous(sigma[c][i,k],sigma[c][j,k],S[i,j])+flux_lowIDP(U[c][i,k],U[c][j,k],flux[c][i,k],flux[c][j,k],S0[i,j],wavespd)
                        F_high[c][i,j] = -flux_viscous(sigma[c][i,k],sigma[c][j,k],S[i,j])+fluxS[c]
                    end
                end
            end
        end

        EL = pL/(γ-1)+0.5*rhoL*uL^2
        ER = pR/(γ-1)+0.5*rhoR*uR^2
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        p_left   = (k == 1) ? pL : p[end,k-1]
        p_right  = (k == K) ? pR : p[1,k+1]
        # Assume velocity doesn't change at boundary, isothermal
        TL = pL/rhoL/(γ-1)/cv
        TR = pR/rhoR/(γ-1)/cv
        sigma_left  = (k == 1) ? [0.0;0.0;0.0] : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [0.0;0.0;0.0] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]] 
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]),beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]),beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right))

        for c = 1:3
            F_low_P[c][1] = -flux_viscous(sigma[c][1,k],sigma_left[c],-1/2)+flux_lowIDP(U[c][1,k],U_left[c],flux[c][1,k],f_left[c],-0.5,wavespd_l) 
            F_low_P[c][2] = -flux_viscous(sigma[c][end,k],sigma_right[c],1/2)+flux_lowIDP(U[c][end,k],U_right[c],flux[c][end,k],f_right[c],0.5,wavespd_r)

            F_high_P[c][1] = F_low_P[c][1]
            F_high_P[c][2] = F_low_P[c][2]
        end

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
            lambda_j = 1/N
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
        
        # elementwise limiting
        l = 1.0
        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    l = min(l,L[i,j])
                end
            end
        end
        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    L[i,j] = l
                    L[j,i] = l
                end
            end
        end 

        for c = 1:3
            # With limiting
            rhsU[c][:,k] = sum((L.-1).*F_low[c] - L.*F_high[c],dims=2)

            rhsU[c][1,k] += -F_high_P[c][1]
            rhsU[c][N+1,k] += -F_high_P[c][2]

            rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
        end

        for i = 1:N+1
            L_plot[i,k] = l
        end
    end

    return rhsU,L_plot
end


# Time stepping
"Time integration"
t = t0
U = collect(U)
resU = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
resW = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
resZ = [zeros(size(x)),zeros(size(x)),zeros(size(x))]

Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM
gr(size=(300,300),ylims=(0,1.2),legend=false,markerstrokewidth=1,markersize=2)
plot()

#dt = 0.00000001 # Leblanc
#dt = 0.000001 # Sod
#Nsteps = Int(ceil(T/dt))
#@gif for i = 1:Nsteps

anim = Animation()
const ptL = Bl+(Br-Bl)/K/(N+1)/2
const ptR = Br-(Br-Bl)/K/(N+1)/2
const hplot = (Br-Bl)/K/(N+1)
i = 1

while t < T

    # if abs(T-t) < dt
    #     global dt = T - t
    # end

    # # SSPRK(3,3)
    # rhsU,dt = rhs_low(U,K,N,Mlump_inv,S0,S,wq)
    # #dt = min(dt,T-t)
    # dt = min(dt,T-t)
    # @. resW = U + dt*rhsU
    # rhsU,_ = rhs_low(resW,K,N,Mlump_inv,S0,S,wq)
    # @. resZ = resW+dt*rhsU
    # @. resW = 3/4*U+1/4*resZ
    # rhsU,_ = rhs_low(resW,K,N,Mlump_inv,S0,S,wq)
    # @. resZ = resW+dt*rhsU
    # @. U = 1/3*U+2/3*resZ

    # dt = 0.0001
    # rhsU = rhs_ESDG(U,K,N,Mlump_inv,S,Qr)
    # @. resW = U + dt*rhsU
    # rhsU = rhs_ESDG(U,K,N,Mlump_inv,S,Qr)
    # @. resZ = resW+dt*rhsU
    # @. resW = 3/4*U+1/4*resZ
    # rhsU = rhs_ESDG(U,K,N,Mlump_inv,S,Qr)
    # @. resZ = resW+dt*rhsU
    # @. U = 1/3*U+2/3*resZ

    # SSPRK(3,3)
    rhsU,dt,_ = rhs_IDP(U,K,N,Mlump_inv,S0,S,wq)
    #dt = min(dt,T-t)
    dt = min(dt,T-t)
    @. resW = U + dt*rhsU
    rhsU,_ = rhs_IDP(resW,K,N,Mlump_inv,S0,S,wq,dt)
    @. resZ = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resZ
    rhsU,L_plot = rhs_IDP(resW,K,N,Mlump_inv,S0,S,wq,dt)
    @. resZ = resW+dt*rhsU
    @. U = 1/3*U+2/3*resZ


    global t = t + dt
    global i = i + 1
    println("Current time $t with time step size $dt, and final time $T")  
    if mod(i,600) == 1
        plot(Vp*x,Vp*U[1])
        for k = 1:K
            plot!(ptL+(k-1)*hplot*(N+1):hplot:ptL+k*hplot*(N+1), 1 .-L_plot[:,k],st=:bar,alpha=0.2)
        end
        frame(anim)
    end
    # if i % GIFINTERVAL == 0  
    #     plot(Vp*x,Vp*U[1])
    # end
end
#end every GIFINTERVAL

#plot(Vp*x,Vp*U[1])
gif(anim,"~/Desktop/tmp.gif",fps=15)