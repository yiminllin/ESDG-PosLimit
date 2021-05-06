using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using StaticArrays



push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
include("../EntropyStableEuler.jl/src/logmean.jl")
using EntropyStableEuler
using EntropyStableEuler.Fluxes2D

function wavespeed_1D(rho,rhou,E)
    p = pfun_nd(rho,rhou,E)
    cvel = @. sqrt(γ*p/rho)
    return @. abs(rhou/rho) + cvel
end

function pfun_nd(rho, rhou, rhov, E)
    rhoUnorm2 = (rhou^2+rhov^2)/rho
    return @. (γ-1)*(E - .5*rhoUnorm2)
end

function pfun_nd(rho, rhou, E)
    rhoUnorm2 = (rhou^2)/rho
    return @. (γ-1)*(E - .5*rhoUnorm2)
end

function primitive_to_conservative_hardcode(rho,u,v,p)
    Unorm = u^2+v^2
    E = @. p/(γ-1) + .5*rho*Unorm
    return (rho,rho*u,rho*v,E)
end

function euler_fluxes_2D(rhoL,uL,vL,betaL,rhologL,betalogL,
    rhoR,uR,vR,betaR,rhologR,betalogR)

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = (@. .5*(rhoL+rhoR))
    uavg   = (@. .5*(uL+uR))
    vavg   = (@. .5*(vL+vR))

    unorm = (@. uL*uR + vL*vR)
    pa    = (@. rhoavg/(betaL+betaR))
    f4aux = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)

    FxS1 = (@. rholog*uavg)
    FxS2 = (@. FxS1*uavg + pa)
    FxS3 = (@. FxS1*vavg)
    FxS4 = (@. f4aux*uavg)

    FyS1 = (@. rholog*vavg)
    FyS2 = (@. FxS3)
    FyS3 = (@. FyS1*vavg + pa)
    FyS4 = (@. f4aux*vavg)
    return (FxS1,FxS2,FxS3,FxS4),(FyS1,FyS2,FyS3,FyS4)
end

function euler_fluxes(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR)
    rhologL,betalogL,rhologR,betalogR = map(x->log.(x),(rhoL,betaL,rhoR,betaR))
    return euler_fluxes_2D(rhoL,uL,vL,betaL,rhologL,betalogL,
                           rhoR,uR,vR,betaR,rhologR,betalogR)
end


const TOL = 1e-14
const Nc = 4 # number of components

"Approximation parameters"
N = 2
K1D = 16
T = 0.1

const NUM_THREADS = Threads.nthreads()

# Initial condition 2D shocktube
const γ = 1.4
const rhoL  = 8.0
const uL    = 8.25*cos(pi/6)
const vL    = -8.25*sin(pi/6)
const pL    = 116.5
const rhouL = rhoL*uL
const rhovL = rhoL*vL
const EL    = pL/(γ-1)+.5*rhoL*(uL^2+vL^2)
const betaL = rhoL/(2*pL)
const rhoR  = 1.4
const uR    = 0.0
const vR    = 0.0
const pR    = 1.0
const rhouR = rhoR*uR
const rhovR = rhoR*vR
const ER    = pR/(γ-1)+.5*rhoR*(uR^2+vR^2)
const betaR = rhoR/(2*pR)
const SHOCKSPD = 10.0/cos(pi/6)


"Mesh related variables"
VX, VY, EToV = uniform_quad_mesh(4*K1D,K1D)
@. VX = (VX+1)*2
@. VY = (VY+1)/2

rd = init_reference_quad(N,gauss_lobatto_quad(0,0,N))
"Initialize reference element"
r,_ = gauss_lobatto_quad(0,0,N)   # Reference nodes
VDM = vandermonde_1D(N,r)         # modal to nodal
Dr = grad_vandermonde_1D(N,r)/VDM # nodal differentiation
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1]) # nodal linear interpolation

Nq = N
rq,wq = gauss_lobatto_quad(0,0,Nq)
M = diagm(wq)
Mlump = zeros(size(M))
Mlump_inv = zeros(size(M))
for i = 1:Nq+1
    Mlump[i,i] = sum(M[i,:])
    Mlump_inv[i,i] = 1.0/Mlump[i,i]
end

# 1D operators
M1D = Mlump
M1D_inv = Mlump_inv
Q1D = M*Dr
B1D = zeros(N+1,N+1)
B1D[1,1] = -1
B1D[end,end] = 1
L = Array(spdiagm(0=>-2*ones(N+1), 1=>ones(N), -1=>ones(N)))
L[1,1] = -1
L[end,end] = -1
psi = pinv(L)*-1/2*B1D*ones(N+1)
S0 = zeros(N+1,N+1)
for i = 1:N+1
    for j = 1:N+1
        if L[i,j] != 0
            S0[i,j] = psi[j] - psi[i]
        end
    end
end
Q01D = S0+1/2*B1D



# Drop zeros
Q1D = Matrix(droptol!(sparse(Q1D),TOL))
Q01D = Matrix(droptol!(sparse(Q01D),TOL))
B1D = Matrix(droptol!(sparse(B1D),TOL))
S01D = Matrix(droptol!(sparse(S0),TOL))
S1D = Matrix(droptol!(sparse((Q1D-Q1D')/2),TOL))

# Tensor product operators
M = droptol!(sparse(kron(M1D,M1D)),TOL)
Minv = spdiagm(0 => 1 ./ diag(M))
Qr = droptol!(sparse(kron(M1D,Q1D)),TOL)
Qs = droptol!(sparse(kron(Q1D,M1D)),TOL)
Qr0 = droptol!(sparse(kron(M1D,Q01D)),TOL)
Qs0 = droptol!(sparse(kron(Q01D,M1D)),TOL)
Sr = droptol!(sparse((Qr-Qr')/2),TOL)
Ss = droptol!(sparse((Qs-Qs')/2),TOL)
S0r = droptol!(sparse(kron(M1D,S01D)),TOL)
S0s = droptol!(sparse(kron(S01D,M1D)),TOL)
Br = droptol!(sparse(Qr+Qr'),TOL)
Bs = droptol!(sparse(Qs+Qs'),TOL)

@unpack Vf,Dr,Ds,LIFT = rd


md = init_mesh((VX,VY),EToV,rd)
@unpack rxJ,sxJ,ryJ,syJ,sJ,J,xf,yf,mapM,mapP,mapB,nxJ,nyJ,x,y = md
xb,yb = (x->x[mapB]).((xf,yf))

# 2D shocktube
inflow   = mapB[findall(@. (abs(xb) < TOL) | ((xb < 1/6) & (abs(yb) < TOL)))]
outflow  = mapB[findall(@. abs(xb-4.) < TOL)]
topflow  = mapB[findall(@. abs(yb-1.) < TOL)]
wall     = mapB[findall(@. (xb >= 1/6) & (abs(yb) < TOL))]
nx_wall  = nxJ[wall]./sJ[wall]
ny_wall  = nyJ[wall]./sJ[wall]
const TOP_INIT = (1+sqrt(3)/6)/sqrt(3)

@unpack Nfaces,Vf = rd
@unpack xf,yf,K,mapM,mapP,mapB = md

function impose_BCs_inviscid_Ub!(UbP,Ubf,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,t)
    # inflow
    for i = inflow
        UbP[1][i] = rhoL
        UbP[2][i] = uL
        UbP[3][i] = vL
        UbP[4][i] = betaL
    end

    # outflow
    for i = outflow
        UbP[1][i] = Ubf[1][i]
        UbP[2][i] = Ubf[2][i]
        UbP[3][i] = Ubf[3][i]
        UbP[4][i] = Ubf[4][i]
    end

    # wall
    for i = 1:length(wall)
        iw = wall[i]
        u_1 = Ubf[2][iw]
        u_2 = Ubf[3][iw]
        n_1 = nx_wall[i]
        n_2 = ny_wall[i]

        Un = u_1*n_1+u_2*n_2
        Ut = u_1*n_2-u_2*n_1

        UbP[1][iw] = Ubf[1][iw]
        UbP[4][iw] = Ubf[4][iw]

        UbP[2][iw] = 1/(-n_1^2-n_2^2)*(n_1*Un-n_2*Ut)
        UbP[3][iw] = 1/(-n_1^2-n_2^2)*(n_2*Un+n_1*Ut)
    end

    # topflow
    breakpoint = TOP_INIT+t*SHOCKSPD
    for i = topflow
        if xf[i] < breakpoint
            UbP[1][i] = rhoL
            UbP[2][i] = uL
            UbP[3][i] = vL
            UbP[4][i] = betaL 
        else
            UbP[1][i] = rhoR
            UbP[2][i] = uR
            UbP[3][i] = vR
            UbP[4][i] = betaR 
        end
    end
end

function impose_BCs_inviscid_U!(UP,Uf,UbP,xf,inflow,outflow,topflow,wall,t)
    # inflow
    for i = inflow
        UP[1][i] = rhoL
        UP[2][i] = rhouL
        UP[3][i] = rhovL
        UP[4][i] = EL
    end

    # outflow
    for i = outflow
        UP[1][i] = Uf[1][i]
        UP[2][i] = Uf[2][i]
        UP[3][i] = Uf[3][i]
        UP[4][i] = Uf[4][i]
    end

    # wall
    for i = wall
        rho  = UbP[1][i]
        u    = UbP[2][i]
        v    = UbP[3][i]
        beta = UbP[4][i]
        p    = rho/beta/2.0
        E    = p/(γ-1) + .5*rho*(u^2+v^2)

        UP[1][i] = rho
        UP[2][i] = rho*u
        UP[3][i] = rho*v
        UP[4][i] = E
    end

    # topflow
    breakpoint = TOP_INIT+t*SHOCKSPD
    for i = topflow
        if xf[i] < breakpoint
            UbP[1][i] = rhoL
            UbP[2][i] = rhouL
            UbP[3][i] = rhovL
            UbP[4][i] = EL
        else
            UbP[1][i] = rhoR
            UbP[2][i] = rhouR
            UbP[3][i] = rhovR
            UbP[4][i] = ER 
        end
    end
end

function impose_BCs_flux!(flux_x_P,flux_y_P,flux_x_f,flux_y_f,UbP,xf,inflow,outflow,topflow,wall,t)
    # inflow
    for i = inflow
        flux_x_P[1][i] = rhoL*uL
        flux_x_P[2][i] = rhoL*uL^2+pL
        flux_x_P[3][i] = rhoL*uL*vL
        flux_x_P[4][i] = uL*(EL+pL)

        flux_y_P[1][i] = rhoL*vL
        flux_y_P[2][i] = rhoL*uL*vL
        flux_y_P[3][i] = rhoL*vL^2+pL
        flux_y_P[4][i] = vL*(EL+pL)
    end

    # Outflow
    for i = outflow
        flux_x_P[1][i] = flux_x_f[1][i]
        flux_x_P[2][i] = flux_x_f[2][i]
        flux_x_P[3][i] = flux_x_f[3][i]
        flux_x_P[4][i] = flux_x_f[4][i]

        flux_y_P[1][i] = flux_y_f[1][i]
        flux_y_P[2][i] = flux_y_f[2][i]
        flux_y_P[3][i] = flux_y_f[3][i]
        flux_y_P[4][i] = flux_y_f[4][i]
    end

    # wall
    for i = wall
        rho  = UbP[1][i]
        u    = UbP[2][i]
        v    = UbP[3][i]
        beta = UbP[4][i]
        p    = rho/beta/2.0
        E    = p/(γ-1) + .5*rho*(u^2+v^2)

        flux_x_P[1][i] = rho*u
        flux_x_P[2][i] = rho*u^2+p
        flux_x_P[3][i] = rho*u*v
        flux_x_P[4][i] = u*(E+p)

        flux_y_P[1][i] = rho*v
        flux_y_P[2][i] = rho*u*v
        flux_y_P[3][i] = rho*v^2+p
        flux_y_P[4][i] = v*(E+p)
    end

    # topwall
    breakpoint = TOP_INIT+t*SHOCKSPD
    for i = topflow
        if xf[i] < breakpoint
            flux_x_P[1][i] = rhoL*uL
            flux_x_P[2][i] = rhoL*uL^2+pL
            flux_x_P[3][i] = rhoL*uL*vL
            flux_x_P[4][i] = uL*(EL+pL)
    
            flux_y_P[1][i] = rhoL*vL
            flux_y_P[2][i] = rhoL*uL*vL
            flux_y_P[3][i] = rhoL*vL^2+pL
            flux_y_P[4][i] = vL*(EL+pL)
        else
            flux_x_P[1][i] = rhoR*uR
            flux_x_P[2][i] = rhoR*uR^2+pR
            flux_x_P[3][i] = rhoR*uR*vR
            flux_x_P[4][i] = uR*(ER+pR)
    
            flux_y_P[1][i] = rhoR*vR
            flux_y_P[2][i] = rhoR*uR*vR
            flux_y_P[3][i] = rhoR*vR^2+pR
            flux_y_P[4][i] = vR*(ER+pR)
        end
    end
end

function impose_BCs_lam!(lamP,lam,inflow,outflow,topflow,wall)
    for i = inflow
        lamP[i] = 0.0
        lam[i]  = 0.0
    end

    for i = outflow 
        lamP[i] = 0.0
        lam[i]  = 0.0
    end

    for i = wall
        lamP[i] = 0.0
        lam[i]  = 0.0
    end

    for i = topflow
        lamP[i] = 0.0
        lam[i]  = 0.0
    end
end


Np = (N+1)*(N+1)
face_idx = [1:N+1; (N+1):(N+1):Np; Np:-1:Np-N; Np-N:-(N+1):1]
x_idx    = [(N+2):(2*N+2); (3*N+4):(4*N+4)]
y_idx    = [1:(N+1); (2*N+3):(3*N+3)]
S0r1 = sum(S0r,dims=2)
S0s1 = sum(S0s,dims=2)
K      = size(x,2)
Np     = (N+1)*(N+1)
Nfaces = 4
Nfp    = Nfaces*(N+1)

# Initial condition 2D shocktube
at_left(x,y) = y-sqrt(3)*x+sqrt(3)/6 > 0.0
U = [zeros(size(x)) for i = 1:Nc]
for k = 1:K
    for i = 1:Np
        if at_left(x[i,k],y[i,k]) 
            U[1][i,k] = rhoL
            U[2][i,k] = rhouL
            U[3][i,k] = rhovL
            U[4][i,k] = EL
        else
            U[1][i,k] = rhoR
            U[2][i,k] = rhouR
            U[3][i,k] = rhovR
            U[4][i,k] = ER           
        end
    end
end

# Preallocation
VU     = zero.(U)
flux_x = zero.(U)
flux_y = zero.(U)
lam    = zeros(Nfp,K)
LFc    = zeros(Nfp,K)
Ub     = zero.(U)
sigma_x = zero.(U)
sigma_y = zero.(U)

function limiting_param(U_low, P_ij)
    l = 1.0
    # Limit density
    if U_low[1] + P_ij[1] < -TOL
        l = min(abs(U_low[1])/(abs(P_ij[1])+1e-14), 1.0)
    end

    # limiting internal energy (via quadratic function)
    a = P_ij[1]*P_ij[4]-1.0/2.0*(P_ij[2]^2+P_ij[3]^2)
    b = U_low[4]*P_ij[1]+U_low[1]*P_ij[4]-U_low[2]*P_ij[2]-U_low[3]*P_ij[3]
    c = U_low[4]*U_low[1]-1.0/2.0*(U_low[2]^2+U_low[3]^2)


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

function rhs_IDP_fixdt!(U,N,K1D,Minv,Vf,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,mapP,face_idx,x_idx,y_idx,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,flux_x,flux_y,lam,LFc,Ub,dt,t)
    # TODO: hardcoded!
    K      = size(U[1],2)
    Np     = (N+1)*(N+1)
    Nfaces = 4
    Nfp    = Nfaces*(N+1)

    J = (1/K1D/2)^2 # hardcoded!!
    Jf = 1/K1D/2
    rxJ = 1/K1D/2
    syJ = 1/K1D/2
    sJ = 1/K1D/2

    p = pfun_nd.(U[1],U[2],U[3],U[4])
    #@. p = (γ-1)*(U[4]-.5*(U[2]^2+U[3]^2)/U[1])
    @. flux_x[1] = U[2]
    @. flux_x[2] = U[2]^2/U[1]+p
    @. flux_x[3] = U[2]*U[3]/U[1]
    @. flux_x[4] = U[4]*U[2]/U[1]+p*U[2]/U[1]
    @. flux_y[1] = U[3]
    @. flux_y[2] = U[2]*U[3]/U[1]
    @. flux_y[3] = U[3]^2/U[1]+p
    @. flux_y[4] = U[4]*U[3]/U[1]+p*U[3]/U[1]

    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[3]/U[1]
    @. Ub[4] = U[1]/(2*p)
    Ubf = (x->Vf*x).(Ub)
    UbP = (x->x[mapP]).(Ubf)
    impose_BCs_inviscid_Ub!(UbP,Ubf,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,t)

    Uf = (x->Vf*x).(U)
    UP = (x->x[mapP]).(Uf)
    impose_BCs_inviscid_U!(UP,Uf,UbP,xf,inflow,outflow,topflow,wall,t)

    # simple lax friedrichs dissipation
    (rhoM,rhouM,rhovM,EM) = Uf
    rhoUM_n = @. (rhouM*nxJ + rhovM*nyJ)/sJ
    @. lam  = abs(sqrt(abs(rhoUM_n/rhoM))+sqrt(γ*(γ-1)*(EM-.5*rhoUM_n^2/rhoM)/rhoM))
    lamP = lam[mapP]
    impose_BCs_lam!(lamP,lam,inflow,outflow,topflow,wall)
    @. LFc = max(lam,lamP)*sJ

    flux_x_f = (x->Vf*x).(flux_x)
    flux_x_P = (x->x[mapP]).(flux_x_f)
    flux_y_f = (x->Vf*x).(flux_y)
    flux_y_P = (x->x[mapP]).(flux_y_f)
    impose_BCs_flux!(flux_x_P,flux_y_P,flux_x_f,flux_y_f,UbP,xf,inflow,outflow,topflow,wall,t)

    rhsU     = [zeros(Float64,Np,K) for i = 1:Nc]
    U_low    = [zeros(Float64,Np,NUM_THREADS) for i = 1:Nc]
    U_low_i  = zeros(Float64,Nc,NUM_THREADS)
    F_low    = [zeros(Float64,Np,Np,NUM_THREADS) for i = 1:Nc]
    F_high   = [zeros(Float64,Np,Np,NUM_THREADS) for i = 1:Nc]
    F_P      = [zeros(Float64,Nfp,NUM_THREADS) for i = 1:Nc]
    L     = ones(Float64,Np,Np,NUM_THREADS)
    P_ij  = zeros(Float64,Nc,NUM_THREADS)

    # =====================
    # Loop through elements
    # =====================
    #for k = 1:K
    Threads.@threads for k = 1:K
        tid = Threads.threadid()

        for c = 1:Nc
            F_low[c][:,:,tid]  .= 0.0
            F_high[c][:,:,tid] .= 0.0
            F_P[c][:,tid]      .= 0.0
            U_low[c][:,tid]    .= 0.0
        end
        L[:,:,tid] .= 1.0

        # Calculate low order algebraic flux
        for i = 1:Np
            for j = 1:Np
                c_ij_norm = sqrt(rxJ^2*S0r[i,j]^2+syJ^2*S0s[i,j]^2)
                if abs(c_ij_norm) >= TOL
                    n_ij_x = rxJ*S0r[i,j]/c_ij_norm
                    n_ij_y = syJ*S0s[i,j]/c_ij_norm
                    wavespd_i = wavespeed_1D(U[1][i,k],n_ij_x*U[2][i,k]+n_ij_y*U[3][i,k],U[4][i,k])
                    wavespd_j = wavespeed_1D(U[1][j,k],n_ij_x*U[2][j,k]+n_ij_y*U[3][j,k],U[4][j,k])
                    wavespd = max(wavespd_i,wavespd_j)
                    d_ij = wavespd*c_ij_norm

                    for c = 1:Nc
                        F_low[c][i,j,tid] = (rxJ*S0r[i,j]*(flux_x[c][i,k]+flux_x[c][j,k])
                                            +syJ*S0s[i,j]*(flux_y[c][i,k]+flux_y[c][j,k])
                                            -d_ij*(U[c][j,k]-U[c][i,k]))
                    end
                end
            end
        end

        # Calculate high order algebraic flux
        for i = 1:Np-1
            for j = i+1:Np
                # TODO: can preallocate nonzero entries
                if Sr[i,j] != 0.0 || Ss[i,j] != 0.0
                    F1,F2 = euler_fluxes(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[4][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],Ub[4][j,k])
                    for c = 1:Nc
                        val = (2*rxJ*Sr[i,j]*F1[c]+2*syJ*Ss[i,j]*F2[c])
                        F_high[c][i,j,tid] = val
                        F_high[c][j,i,tid] = -val
                    end
                end
            end
        end

        # Calculate interface fluxes
        for i = 1:Nfp
            S0r_ij = -S0r1[face_idx[i]]
            S0s_ij = -S0s1[face_idx[i]]

            # flux in x direction
            if i in x_idx
                wavespd_M = wavespeed_1D(Uf[1][i,k],Uf[2][i,k],Uf[4][i,k])
                wavespd_P = wavespeed_1D(UP[1][i,k],UP[2][i,k],UP[4][i,k])
                wavespd = max(wavespd_M,wavespd_P)
                d_ij = wavespd*abs(S0r_ij)
                for c = 1:Nc
                    F_P[c][i,tid] = (Jf*S0r_ij*(flux_x_f[c][i,k]+flux_x_P[c][i,k])
                                    -LFc[i,k]*abs(S0r_ij)*(UP[c][i,k]-Uf[c][i,k]))
                end
            end

            # flux in y direction
            if i in y_idx
                wavespd_M = wavespeed_1D(Uf[1][i,k],Uf[3][i,k],Uf[4][i,k])
                wavespd_P = wavespeed_1D(UP[1][i,k],UP[3][i,k],UP[4][i,k])
                wavespd = max(wavespd_M,wavespd_P)
                d_ij = wavespd*abs(S0s_ij)
                for c = 1:Nc
                    F_P[c][i,tid] = (Jf*S0s_ij*(flux_y_f[c][i,k]+flux_y_P[c][i,k])
                                    -LFc[i,k]*abs(S0s_ij)*(UP[c][i,k]-Uf[c][i,k]))
                end
            end
        end

        # Calculate low order solution
        for i = 1:Np
            for j = 1:Np
                for c = 1:Nc
                    U_low[c][i,tid] += F_low[c][i,j,tid]
                end
            end
        end
        for i = 1:Nfp
            for c = 1:Nc
                U_low[c][face_idx[i],tid] += F_P[c][i,tid]
            end
        end
        for c = 1:Nc
            for i = 1:Np
                U_low[c][i,tid] = U[c][i,k]-dt*Minv[i,i]/J*U_low[c][i,tid]
            end
        end

        # Calculate limiting parameters
        for i = 1:Np
            m_i = J/Minv[i,i]
            lambda_j = 1/(Np-1)
            for c = 1:Nc
                U_low_i[c,tid] = U_low[c][i,tid]
            end
            for j = 1:Np
                if i != j
                    for c = 1:Nc
                        P_ij[c,tid] = dt/(m_i*lambda_j)*(F_low[c][i,j,tid]-F_high[c][i,j,tid])
                    end
                    L[i,j,tid] = limiting_param(U_low_i[:,tid], P_ij[:,tid]) # TODO: rewrite expression
                end
            end
        end

        # Elementwise limiting
        l_e = 1.0
        for i = 1:Np
            for j = 1:Np
                if i != j
                    l_e = min(l_e,L[i,j,tid])
                end
            end
        end
        for i = 1:Np
            for j = 1:Np
                L[i,j,tid] = l_e
            end
        end

        for c = 1:Nc
            for i = 1:Np
                for j = 1:Np
                    rhsU[c][i,k] += (L[i,j,tid]-1)*F_low[c][i,j,tid]-L[i,j,tid]*F_high[c][i,j,tid]
                end
            end
            for i = 1:Nfp
                rhsU[c][face_idx[i],k] -= F_P[c][i,tid]
            end
        end
    end

    for c = 1:Nc
        rhsU[c] .= 1/J*Minv*rhsU[c]
    end

    return rhsU
end

function rhs_IDP_vardt!(U,N,K1D,Minv,Vf,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,mapP,face_idx,x_idx,y_idx,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,flux_x,flux_y,lam,LFc,Ub,t)
    # TODO: hardcoded!
    K      = size(U[1],2)
    Np     = (N+1)*(N+1)
    Nfaces = 4
    Nfp    = Nfaces*(N+1)

    J = (1/K1D/2)^2 # hardcoded!!
    Jf = 1/K1D/2
    rxJ = 1/K1D/2
    syJ = 1/K1D/2
    sJ = 1/K1D/2

    # =======================
    # Determine timestep size
    # =======================
    d_ii_arr = zeros(Np,K)
    for k = 1:K
        for i = 1:Np
            for j = 1:Np
                c_ij_norm = sqrt(rxJ^2*S0r[i,j]^2+syJ^2*S0s[i,j]^2)
                if abs(c_ij_norm) >= TOL
                    n_ij_x = rxJ*S0r[i,j]/c_ij_norm
                    n_ij_y = syJ*S0s[i,j]/c_ij_norm
                    wavespd_i = wavespeed_1D(U[1][i,k],n_ij_x*U[2][i,k]+n_ij_y*U[3][i,k],U[4][i,k])
                    wavespd_j = wavespeed_1D(U[1][j,k],n_ij_x*U[2][j,k]+n_ij_y*U[3][j,k],U[4][j,k])
                    wavespd = max(wavespd_i,wavespd_j)
                    d_ij = wavespd*c_ij_norm
                    d_ii_arr[i,k] -= d_ij
                end
            end
        end

        for i = 1:Nfp
            S0r_ij = -S0r1[face_idx[i]]
            S0s_ij = -S0s1[face_idx[i]]
            if i in x_idx
                d_ii_arr[face_idx[i]] -= LFc[i,k]*abs(S0r_ij)
            end
            if i in y_idx
                d_ii_arr[face_idx[i]] -= LFc[i,k]*abs(S0s_ij)
            end
        end
    end

    dt = minimum(-J/2*M*(1 ./d_ii_arr))

    rhsU = rhs_IDP_fixdt!(U,N,K1D,Minv,Vf,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,mapP,face_idx,x_idx,y_idx,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,flux_x,flux_y,lam,LFc,Ub,dt,t)
    return rhsU,dt
end


# Time stepping
"Time integration"
t = 0.0
U = collect(U)
resU = [zeros(size(x)),zeros(size(x)),zeros(size(x)),zeros(size(x))]
resW = [zeros(size(x)),zeros(size(x)),zeros(size(x)),zeros(size(x))]
resZ = [zeros(size(x)),zeros(size(x)),zeros(size(x)),zeros(size(x))]

gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2,axis=nothing)
const GIFINTERVAL = 100
#plot()

#plotting nodes
@unpack VDM = rd
rp,sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/VDM
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)

# dt = 1e-4
# rhs_IDP_fixdt!(U,N,K1D,Minv,Vf,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,mapP,face_idx,x_idx,y_idx,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,flux_x,flux_y,lam,LFc,Ub,dt,t);
# @btime rhs_IDP_fixdt!(U,N,K1D,Minv,Vf,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,mapP,face_idx,x_idx,y_idx,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,flux_x,flux_y,lam,LFc,Ub,dt,t);
# # @profiler rhs_IDP_fixdt!(U,N,K1D,Minv,Vf,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,mapP,face_idx,x_idx,y_idx,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,flux_x,flux_y,lam,LFc,Ub,dt,t);

dt_hist = []
anim = Animation()
i = 1

while t < T
    # SSPRK(3,3)
    dt = min(1e-4,T-t)
    rhsU = rhs_IDP_fixdt!(U,N,K1D,Minv,Vf,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,mapP,face_idx,x_idx,y_idx,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,flux_x,flux_y,lam,LFc,Ub,dt,t);
    # rhsU,dt = rhs_IDP_vardt!(U,N,K1D,Minv,Vf,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,mapP,face_idx,x_idx,y_idx,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,flux_x,flux_y,lam,LFc,Ub,t);
    # dt = min(dt,T-dt)
    @. resW = U + dt*rhsU
    rhsU = rhs_IDP_fixdt!(resW,N,K1D,Minv,Vf,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,mapP,face_idx,x_idx,y_idx,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,flux_x,flux_y,lam,LFc,Ub,dt,t);
    @. resZ = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resZ
    rhsU = rhs_IDP_fixdt!(resW,N,K1D,Minv,Vf,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,mapP,face_idx,x_idx,y_idx,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,flux_x,flux_y,lam,LFc,Ub,dt,t);
    @. resZ = resW+dt*rhsU
    @. U = 1/3*U+2/3*resZ

    push!(dt_hist,dt)
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T, at step $i")
    global i = i + 1

    if mod(i,100) == 1
        xp = x
        yp = y
        vv = U[1]
        scatter(xp,yp,vv,zcolor=vv,camera=(0,90),colorbar=:right)
        frame(anim)
    end
end

################
### Plotting ###
################

gif(anim,"~/Desktop/tmp.gif",fps=15)

rho = U[1]
rhou = U[2]
rhov = U[3]
E = U[4]

# # scatter(x,y,U[1],zcolor=U[1],camera=(0,90),colorbar=:right)
# xp = Vp*x
# yp = Vp*y
# vv = Vp*U[1]
# scatter(xp,yp,vv,zcolor=vv,camera=(0,90),colorbar=:right)
scatter(x,y,rho,zcolor=rho,camera=(0,90),colorbar=:right)
savefig("/expanse/lustre/scratch/yiminlin/temp_project/N=$N,K1D=$K1D,T=$T,doubleMachReflection.png")
#savefig("~/Desktop/N=$N,K1D=$K1D,T=$T,doubleMachReflection.png")
#scatter(xp,yp,exact_rho_p,zcolor=exact_rho_p,camera=(0,90),colorbar=:right)