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



const TOL = 1e-16
"Approximation parameters"
N = 5 # The order of approximation
K = 20
T = 1.0
T = 6.0
#T = 0.0039

# Sod shocktube
const γ = 1.4
const Bl = -0.5
const Br = 0.5
const rhoL = 1.0
const rhoR = 0.125
const pL = 1.0
const pR = 0.1
const xC = 0.0
const GIFINTERVAL = 20
T = 0.2

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
# const GIFINTERVAL = 60
# T = 6.0

"Viscous parameters"
const Re = 1000
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

function rhs_viscous(U,K,N,Mlump_inv,S)
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
            Kx[2,2] = 4/3*mu*cv*T[i,k]
            Kx[3,2] = 4/3*mu*cv*T[i,k]*U[2][i,k]/U[1][i,k]
            Kx[3,3] = kappa*cv*T[i,k]^2

            sigma[2][i,k] = Kx[2,2]*theta[2][i,k]
            sigma[3][i,k] = Kx[3,2]*theta[2][i,k] + Kx[3,3]*theta[3][i,k]
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

function rhs_ESDG(U,K,N,Mlump_inv,S)
    rhsI = rhs_inviscid(U,K,N,Mlump_inv,S)
    rhsV = rhs_viscous(U,K,N,Mlump_inv,S)
    return rhsI .+ rhsV
end


# Time stepping
"Time integration"
t = 0.0
U = collect(U)
resU = [zeros(size(x)),zeros(size(x)),zeros(size(x))]

Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM
gr(size=(300,300),ylims=(0,1.2),legend=false,markerstrokewidth=1,markersize=2)
plot()

dt = 0.0001
Nsteps = Int(T/dt)
@gif for i = 1:Nsteps
    #rhsU = rhs_inviscid(U,K,N,Mlump_inv,S)
    rhsU = rhs_ESDG(U,K,N,Mlump_inv,S)
    @. U = U + dt*rhsU
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T")  
    if i % GIFINTERVAL == 0  
        plot(Vp*x,Vp*U[1])
    end
end every GIFINTERVAL

#plot(Vp*x,Vp*U[1])