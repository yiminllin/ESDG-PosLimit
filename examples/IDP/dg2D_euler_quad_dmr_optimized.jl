using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using StaticArrays
using DelimitedFiles
using Polyester
using MuladdMacro

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

@muladd begin

@inline function pfun(rho,rhou,E)
    return (γ-1)*(E-.5*rhou^2/rho)
end

@inline function pfun(rho,rhou,rhov,E)
    return (γ-1)*(E-.5*(rhou^2+rhov^2)/rho)
end

@inline function wavespeed_1D(rho,rhou,E)
    p = pfun(rho,rhou,E)
    return abs(rhou/rho) + sqrt(γ*p/rho)
end

@inline function logmean(aL,aR,logL,logR)

    # "from: Entropy stable num. approx. for the isothermal and polytropic Euler"

    da = aR-aL;
    aavg = .5*(aR+aL);
    f = da/aavg;
    v = f^2;
    if abs(f)<1e-4
        # numbers assume the specific value γ = 1.4
        return aavg*(1 + v*(-.2-v*(.0512 - v*0.026038857142857)))
    else
        return -da/(logL-logR)
    end
end

@inline function euler_fluxes_2D(rhoL,uL,vL,betaL,rhologL,betalogL,
                                 rhoR,uR,vR,betaR,rhologR,betalogR)

    rholog  = logmean(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = .5*(rhoL+rhoR)
    uavg   = .5*(uL+uR)
    vavg   = .5*(vL+vR)

    unorm = uL*uR + vL*vR
    pa    = rhoavg/(betaL+betaR)
    f4aux = rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm

    FxS1 = rholog*uavg
    FxS2 = FxS1*uavg + pa
    FxS3 = FxS1*vavg
    FxS4 = f4aux*uavg

    FyS1 = rholog*vavg
    FyS2 = FxS3
    FyS3 = FyS1*vavg + pa
    FyS4 = f4aux*vavg

    return FxS1,FxS2,FxS3,FxS4,FyS1,FyS2,FyS3,FyS4
end

@inline function limiting_param(rhoL,rhouL,rhovL,EL,rhoP,rhouP,rhovP,EP)
    # L - low order, P - P_ij

    l = 1.0
    # Limit density
    if rhoL + rhoP < -TOL
        l = max((-rhoL+POSTOL)/rhoP, 0)
    end

    # limiting internal energy (via quadratic function)
    a = rhoP*EP-(rhouP^2+rhovP^2)/2.0
    b = rhoP*EL+rhoL*EP-rhouL*rhouP-rhovL*rhovP
    c = rhoL*EL-(rhouL^2+rhovL^2)/2.0-POSTOL

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




const TOL = 5e-16
const POSTOL = 1e-14
const Nc = 4 # number of components

"Approximation parameters"
N = 2
K1D = 16
T = 0.1
dt0 = 1e-4
XLENGTH = 7/2
const CFL = 1.0
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
VX, VY, EToV = uniform_quad_mesh(Int(round(XLENGTH*K1D)),K1D)
@. VX = (VX+1)/2*XLENGTH
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

function impose_BCs_inviscid_Ub!(UbP,Ubf,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,t)
    # inflow
    for i = inflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)
        UbP[1,j,k] = rhoL
        UbP[2,j,k] = uL
        UbP[3,j,k] = vL
        UbP[4,j,k] = betaL
    end

    # outflow
    for i = outflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)
        UbP[1,j,k] = Ubf[1,j,k]
        UbP[2,j,k] = Ubf[2,j,k]
        UbP[3,j,k] = Ubf[3,j,k]
        UbP[4,j,k] = Ubf[4,j,k]
    end

    # wall
    for i = 1:length(wall)
        iw = wall[i]
        kw = fld1(iw,Nfp)
        jw = mod1(iw,Nfp)
        u_1 = Ubf[2,jw,kw]
        u_2 = Ubf[3,jw,kw]
        n_1 = nx_wall[i]
        n_2 = ny_wall[i]

        Un = u_1*n_1+u_2*n_2
        Ut = u_1*n_2-u_2*n_1

        UbP[1,jw,kw] = Ubf[1,jw,kw]
        UbP[4,jw,kw] = Ubf[4,jw,kw]
        
        UbP[2,jw,kw] = 1/(-n_1^2-n_2^2)*(n_1*Un-n_2*Ut)
        UbP[3,jw,kw] = 1/(-n_1^2-n_2^2)*(n_2*Un+n_1*Ut)
    end

    # topflow
    breakpoint = TOP_INIT+t*SHOCKSPD
    for i = topflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)
        if xf[i] < breakpoint
            UbP[1,j,k] = rhoL
            UbP[2,j,k] = uL
            UbP[3,j,k] = vL
            UbP[4,j,k] = betaL 
        else
            UbP[1,j,k] = rhoR
            UbP[2,j,k] = uR
            UbP[3,j,k] = vR
            UbP[4,j,k] = betaR 
        end
    end
end

function impose_BCs_inviscid_U!(UP,Uf,UbP,xf,inflow,outflow,topflow,wall,t)
    # inflow
    for i = inflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        UP[1,j,k] = rhoL
        UP[2,j,k] = rhouL
        UP[3,j,k] = rhovL
        UP[4,j,k] = EL
    end

    # outflow
    for i = outflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        UP[1,j,k] = Uf[1,j,k]
        UP[2,j,k] = Uf[2,j,k]
        UP[3,j,k] = Uf[3,j,k]
        UP[4,j,k] = Uf[4,j,k]
    end

    # wall
    for i = wall
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        rho  = UbP[1,j,k]
        u    = UbP[2,j,k]
        v    = UbP[3,j,k]
        beta = UbP[4,j,k]
        p    = rho/beta/2.0
        E    = p/(γ-1) + .5*rho*(u^2+v^2)

        UP[1,j,k] = rho
        UP[2,j,k] = rho*u
        UP[3,j,k] = rho*v
        UP[4,j,k] = E
    end

    # topflow
    breakpoint = TOP_INIT+t*SHOCKSPD
    for i = topflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        if xf[i] < breakpoint
            UbP[1,j,k] = rhoL
            UbP[2,j,k] = rhouL
            UbP[3,j,k] = rhovL
            UbP[4,j,k] = EL
        else
            UbP[1,j,k] = rhoR
            UbP[2,j,k] = rhouR
            UbP[3,j,k] = rhovR
            UbP[4,j,k] = ER 
        end
    end
end

function impose_BCs_flux!(flux_x_P,flux_y_P,flux_x_f,flux_y_f,UbP,xf,inflow,outflow,topflow,wall,t)
    # inflow
    for i = inflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        flux_x_P[1,j,k] = rhoL*uL
        flux_x_P[2,j,k] = rhoL*uL^2+pL
        flux_x_P[3,j,k] = rhoL*uL*vL
        flux_x_P[4,j,k] = uL*(EL+pL)
        
        flux_y_P[1,j,k] = rhoL*vL
        flux_y_P[2,j,k] = rhoL*uL*vL
        flux_y_P[3,j,k] = rhoL*vL^2+pL
        flux_y_P[4,j,k] = vL*(EL+pL)
    end

    # Outflow
    for i = outflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        flux_x_P[1,j,k] = flux_x_f[1,j,k]
        flux_x_P[2,j,k] = flux_x_f[2,j,k]
        flux_x_P[3,j,k] = flux_x_f[3,j,k]
        flux_x_P[4,j,k] = flux_x_f[4,j,k]
        
        flux_y_P[1,j,k] = flux_y_f[1,j,k]
        flux_y_P[2,j,k] = flux_y_f[2,j,k]
        flux_y_P[3,j,k] = flux_y_f[3,j,k]
        flux_y_P[4,j,k] = flux_y_f[4,j,k]
    end

    # wall
    for i = wall
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        rho  = UbP[1,j,k]
        u    = UbP[2,j,k]
        v    = UbP[3,j,k]
        beta = UbP[4,j,k]
        p    = rho/beta/2.0
        E    = p/(γ-1) + .5*rho*(u^2+v^2)

        flux_x_P[1,j,k] = rho*u
        flux_x_P[2,j,k] = rho*u^2+p
        flux_x_P[3,j,k] = rho*u*v
        flux_x_P[4,j,k] = u*(E+p)
        
        flux_y_P[1,j,k] = rho*v
        flux_y_P[2,j,k] = rho*u*v
        flux_y_P[3,j,k] = rho*v^2+p
        flux_y_P[4,j,k] = v*(E+p)
    end

    # topwall
    breakpoint = TOP_INIT+t*SHOCKSPD
    for i = topflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        if xf[i] < breakpoint
            flux_x_P[1,j,k] = rhoL*uL
            flux_x_P[2,j,k] = rhoL*uL^2+pL
            flux_x_P[3,j,k] = rhoL*uL*vL
            flux_x_P[4,j,k] = uL*(EL+pL)
            
            flux_y_P[1,j,k] = rhoL*vL
            flux_y_P[2,j,k] = rhoL*uL*vL
            flux_y_P[3,j,k] = rhoL*vL^2+pL
            flux_y_P[4,j,k] = vL*(EL+pL)
        else
            flux_x_P[1,j,k] = rhoR*uR
            flux_x_P[2,j,k] = rhoR*uR^2+pR
            flux_x_P[3,j,k] = rhoR*uR*vR
            flux_x_P[4,j,k] = uR*(ER+pR)
            
            flux_y_P[1,j,k] = rhoR*vR
            flux_y_P[2,j,k] = rhoR*uR*vR
            flux_y_P[3,j,k] = rhoR*vR^2+pR
            flux_y_P[4,j,k] = vR*(ER+pR)
        end
    end
end

function impose_BCs_lam!(lamP,lam,inflow,outflow,topflow,wall)
    for i = inflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        lamP[j,k] = 0.0
        lam[j,k]  = 0.0
    end

    for i = outflow 
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        lamP[j,k] = 0.0
        lam[j,k]  = 0.0
    end

    for i = wall
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        lamP[j,k] = 0.0
        lam[j,k]  = 0.0
    end

    for i = topflow
        k = fld1(i,Nfp)
        j = mod1(i,Nfp)

        lamP[j,k] = 0.0
        lam[j,k]  = 0.0
    end
end

function rhs_IDP_fixdt!(U,rhsU,t,dt,prealloc,ops,geom)
    # TODO: previous RHS time 12 ms
    # TODO: debug RHS time 2.88 ms
    # TODO: current RHS time 4.8 ms
    # TODO: optimize boundary condition enforcement
    # TODO: hardcoded variables!
    # TODO: diagonal matrix to vec
    f_x,f_y,lam,lamP,LFc,Ubf,UbP,f_x,f_xM,f_xP,f_y,f_yM,f_yP,Uf,UP,rholog,betalog,U_low,F_low,F_high,F_P,L = prealloc
    S0r,S0s,Sr,Ss,S0r_sq,S0s_sq,Minv,MJ_inv,Br_halved,Bs_halved,coeff_arr = ops
    mapP,Fmask,Fxmask,Fymask,x,y,nxJ,nyJ,sJ,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall = geom

    fill!(rhsU,0.0)
    for k = 1:K
        for i = 1:Np
            rho  = U[1,i,k]
            rhou = U[2,i,k]
            rhov = U[3,i,k]
            E    = U[4,i,k]
            p          = pfun(rho,rhou,rhov,E)
            f_x[1,i,k] = rhou
            f_x[2,i,k] = rhou^2/rho+p
            f_x[3,i,k] = rhou*rhov/rho
            f_x[4,i,k] = E*rhou/rho+p*rhou/rho
            f_y[1,i,k] = rhov
            f_y[2,i,k] = rhou*rhov/rho
            f_y[3,i,k] = rhov^2/rho+p
            f_y[4,i,k] = E*rhov/rho+p*rhov/rho
            rholog[i,k]  = log(rho)
            betalog[i,k] = log(rho/(2*p))
        end

        for i = 1:Nfp
            iM = Fmask[i]
            rhoM  = U[1,iM,k]
            rhouM = U[2,iM,k]
            rhovM = U[3,iM,k]
            EM    = U[4,iM,k]
            uM    = rhouM/rhoM
            vM    = rhovM/rhoM
            pM    = pfun(rhoM,rhouM,rhovM,EM)
            rhoUM_n = (rhouM*nxJ[i,k]+rhovM*nyJ[i,k])/sJ
            lambda  = abs(rhoUM_n/rhoM)+sqrt(γ*(γ-1)*(EM-.5*rhoUM_n^2/rhoM)/rhoM)#wavespeed_1D(rhoM,rhoUM_n,EM)

            Ubf[1,i,k] = rhoM
            Ubf[2,i,k] = uM
            Ubf[3,i,k] = vM
            Ubf[4,i,k] = rhoM/(2*pM)
            Uf[1,i,k] = rhoM
            Uf[2,i,k] = rhouM
            Uf[3,i,k] = rhovM
            Uf[4,i,k] = EM
            lam[i,k]  = lambda

            f_xM[1,i,k] = f_x[1,iM,k]
            f_xM[2,i,k] = f_x[2,iM,k]
            f_xM[3,i,k] = f_x[3,iM,k]
            f_xM[4,i,k] = f_x[4,iM,k]
            f_yM[1,i,k] = f_y[1,iM,k]
            f_yM[2,i,k] = f_y[2,iM,k]
            f_yM[3,i,k] = f_y[3,iM,k]
            f_yM[4,i,k] = f_y[4,iM,k]           
        end
    end

    for k = 1:K
        for i = 1:Nfp
            gP = mapP[i,k]           # exterior global face node number
            kP = fld1(gP,Nfp)        # exterior element number
            iP = mod1(gP,Nfp)        # exterior node number
 
            f_xP[1,i,k] = f_xM[1,iP,kP]
            f_xP[2,i,k] = f_xM[2,iP,kP]
            f_xP[3,i,k] = f_xM[3,iP,kP]
            f_xP[4,i,k] = f_xM[4,iP,kP]
            f_yP[1,i,k] = f_yM[1,iP,kP]
            f_yP[2,i,k] = f_yM[2,iP,kP]
            f_yP[3,i,k] = f_yM[3,iP,kP]
            f_yP[4,i,k] = f_yM[4,iP,kP]

            UbP[1,i,k] = Ubf[1,iP,kP]
            UbP[2,i,k] = Ubf[2,iP,kP]
            UbP[3,i,k] = Ubf[3,iP,kP]
            UbP[4,i,k] = Ubf[4,iP,kP]

            UP[1,i,k] = Uf[1,iP,kP]
            UP[2,i,k] = Uf[2,iP,kP]
            UP[3,i,k] = Uf[3,iP,kP]
            UP[4,i,k] = Uf[4,iP,kP]

            lamP[i,k]  = lam[iP,kP]
        end
    end

    impose_BCs_inviscid_Ub!(UbP,Ubf,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall,t)
    impose_BCs_inviscid_U!(UP,Uf,UbP,xf,inflow,outflow,topflow,wall,t)
    impose_BCs_lam!(lamP,lam,inflow,outflow,topflow,wall)
    @. LFc = max(lam,lamP)*sJ
    impose_BCs_flux!(f_xP,f_yP,f_x,f_y,UbP,xf,inflow,outflow,topflow,wall,t)
   
    # =====================
    # Loop through elements
    # =====================
    for k = 1:K
        tid = Threads.threadid()
        
        fill!(F_low ,0.0)
        fill!(F_high,0.0)
        fill!(F_P   ,0.0)
        fill!(U_low ,0.0)
        fill!(L     ,1.0)

        # Calculate low order algebraic flux
        for j = 2:Np
            rho_j  = U[1,j,k]
            rhou_j = U[2,j,k]
            rhov_j = U[3,j,k]
            E_j    = U[4,j,k]
            fx_1_j = f_x[1,j,k]
            fx_2_j = f_x[2,j,k]
            fx_3_j = f_x[3,j,k]
            fx_4_j = f_x[4,j,k]
            fy_1_j = f_y[1,j,k]
            fy_2_j = f_y[2,j,k]
            fy_3_j = f_y[3,j,k]
            fy_4_j = f_y[4,j,k]
            for i = 1:j-1
                S0r_ij    = S0r[i,j]
                S0s_ij    = S0s[i,j]
                S0r_ij_sq = S0r_sq[i,j]
                S0s_ij_sq = S0s_sq[i,j]
                rho_i  = U[1,i,k]
                rhou_i = U[2,i,k]
                rhov_i = U[3,i,k]
                E_i    = U[4,i,k]
                fx_1_i = f_x[1,i,k]
                fx_2_i = f_x[2,i,k]
                fx_3_i = f_x[3,i,k]
                fx_4_i = f_x[4,i,k]
                fy_1_i = f_y[1,i,k]
                fy_2_i = f_y[2,i,k]
                fy_3_i = f_y[3,i,k]
                fy_4_i = f_y[4,i,k]

                if S0r_ij != 0 || S0s_ij != 0
                    # TODO: store n_ij?
                    n_ij_norm = sqrt(rxJ_sq*S0r_ij_sq+syJ_sq*S0s_ij_sq)
                    # TODO: reuse wavespd?
                    n_ij_x = rxJ*S0r_ij/n_ij_norm
                    n_ij_y = syJ*S0s_ij/n_ij_norm
                    λ_i = wavespeed_1D(rho_i,n_ij_x*rhou_i+n_ij_y*rhov_i,E_i)
                    λ_j = wavespeed_1D(rho_j,n_ij_x*rhou_j+n_ij_y*rhov_j,E_j)
                    λ_ij = max(λ_i,λ_j)*n_ij_norm

                    FL1 = (rxJ*S0r_ij*(fx_1_i+fx_1_j)
                          +syJ*S0s_ij*(fy_1_i+fy_1_j)
                          -λ_ij*(rho_j-rho_i) )
                    FL2 = (rxJ*S0r_ij*(fx_2_i+fx_2_j)
                          +syJ*S0s_ij*(fy_2_i+fy_2_j)
                          -λ_ij*(rhou_j-rhou_i) )
                    FL3 = (rxJ*S0r_ij*(fx_3_i+fx_3_j)
                          +syJ*S0s_ij*(fy_3_i+fy_3_j)
                          -λ_ij*(rhov_j-rhov_i) )                    
                    FL4 = (rxJ*S0r_ij*(fx_4_i+fx_4_j)
                          +syJ*S0s_ij*(fy_4_i+fy_4_j)
                          -λ_ij*(E_j-E_i) )

 
                    F_low[1,i,j,tid] = FL1
                    F_low[2,i,j,tid] = FL2
                    F_low[3,i,j,tid] = FL3
                    F_low[4,i,j,tid] = FL4

                    F_low[1,j,i,tid] = -FL1
                    F_low[2,j,i,tid] = -FL2
                    F_low[3,j,i,tid] = -FL3
                    F_low[4,j,i,tid] = -FL4
                end
            end
        end

        # Calculate high order algebraic flux
        # TODO: use Ub?
        for j = 2:Np
            rho_j     = U[1,j,k]
            rhou_j    = U[2,j,k]
            rhov_j    = U[3,j,k]
            E_j       = U[4,j,k]
            p_j       = pfun(rho_j,rhou_j,rhov_j,E_j)
            u_j       = rhou_j/rho_j
            v_j       = rhov_j/rho_j
            beta_j    = rho_j/(2*p_j)
            rholog_j  = rholog[j,k]
            betalog_j = betalog[j,k]
            for i = 1:j-1
                Sr_ij     = Sr[i,j]
                Ss_ij     = Ss[i,j]
                rho_i     = U[1,i,k]
                rhou_i    = U[2,i,k]
                rhov_i    = U[3,i,k]
                E_i       = U[4,i,k]
                p_i       = pfun(rho_i,rhou_i,rhov_i,E_i)
                u_i       = rhou_i/rho_i
                v_i       = rhov_i/rho_i
                beta_i    = rho_i/(2*p_i)
                rholog_i  = rholog[i,k]
                betalog_i = betalog[i,k]
                if Sr_ij != 0.0 || Ss_ij != 0.0
                    Fx1,Fx2,Fx3,Fx4,Fy1,Fy2,Fy3,Fy4 = euler_fluxes_2D(rho_i,u_i,v_i,beta_i,rholog_i,betalog_i,
                                                                      rho_j,u_j,v_j,beta_j,rholog_j,betalog_j)

                    FH1 = rxJ_db*Sr_ij*Fx1+syJ_db*Ss_ij*Fy1
                    FH2 = rxJ_db*Sr_ij*Fx2+syJ_db*Ss_ij*Fy2
                    FH3 = rxJ_db*Sr_ij*Fx3+syJ_db*Ss_ij*Fy3
                    FH4 = rxJ_db*Sr_ij*Fx4+syJ_db*Ss_ij*Fy4

                    F_high[1,i,j,tid] = FH1
                    F_high[2,i,j,tid] = FH2
                    F_high[3,i,j,tid] = FH3
                    F_high[4,i,j,tid] = FH4

                    F_high[1,j,i,tid] = -FH1
                    F_high[2,j,i,tid] = -FH2
                    F_high[3,j,i,tid] = -FH3
                    F_high[4,j,i,tid] = -FH4
                end
            end
        end

        # Calculate interface fluxes
        # TODO: optimize
        for i = 1:Nfp
            S0r_ij = Br_halved[Fmask[i]]
            S0s_ij = Bs_halved[Fmask[i]]

            # flux in x direction
            if i in Fxmask
                d_ij = LFc[i,k]*abs(S0r_ij)
                for c = 1:Nc
                    F_P[c,i,tid] = (Jf*S0r_ij*(f_xM[c,i,k]+f_xP[c,i,k])
                                   -d_ij*(UP[c,i,k]-Uf[c,i,k]))
                end
            end

            # flux in y direction
            if i in Fymask
                d_ij = LFc[i,k]*abs(S0s_ij)
                for c = 1:Nc
                    F_P[c,i,tid] = (Jf*S0s_ij*(f_yM[c,i,k]+f_yP[c,i,k])
                                   -d_ij*(UP[c,i,k]-Uf[c,i,k]))
                end
            end
        end


        # Calculate low order solution
        for i = 1:Np
            for j = 1:Np
                for c = 1:Nc
                    U_low[c,i,tid] = U_low[c,i,tid] + F_low[c,i,j,tid]
                end
            end
        end

        for i = 1:Nfp
            iM = Fmask[i]
            for c = 1:Nc
                U_low[c,iM,tid] = U_low[c,iM,tid] + F_P[c,i,tid]
            end
        end

        for c = 1:Nc
            for i = 1:Np
                U_low[c,i,tid] = U[c,i,k] - dt*MJ_inv[i]*U_low[c,i,tid]
            end
        end

        # Calculate limiting parameters
        # TODO: skip diagonal
        for i = 1:Np
            coeff = coeff_arr[i]
            rhoL  = U_low[1,i,tid]
            rhouL = U_low[2,i,tid]
            rhovL = U_low[3,i,tid]
            EL    = U_low[4,i,tid]
            for j = 1:Np
                if i != j
                    rhoP  = coeff*(F_low[1,i,j,tid]-F_high[1,i,j,tid])
                    rhouP = coeff*(F_low[2,i,j,tid]-F_high[2,i,j,tid])
                    rhovP = coeff*(F_low[3,i,j,tid]-F_high[3,i,j,tid])
                    EP    = coeff*(F_low[4,i,j,tid]-F_high[4,i,j,tid])
                    L[i,j,tid] = limiting_param(rhoL,rhouL,rhovL,EL,rhoP,rhouP,rhovP,EP)
                end
            end
        end

        # Elementwise limiting
        l_e = 1.0
        for j = 1:Np
            for i = 1:Np
                if i != j
                    l_e = min(l_e,L[i,j,tid])
                end
            end
        end
        for j = 1:Np
            for i = 1:Np
                L[i,j,tid] = l_e
            end
        end

        for c = 1:Nc
            for i = 1:Np
                for j = 1:Np
                    rhsU[c,i,k] = rhsU[c,i,k] + (L[i,j,tid]-1)*F_low[c,i,j,tid]-L[i,j,tid]*F_high[c,i,j,tid]
                end
            end
            for i = 1:Nfp
                rhsU[c,Fmask[i],k] -= F_P[c,i,tid]
            end
        end
    end

    for k = 1:K
        for i = 1:Np
            for c = 1:Nc
                rhsU[c,i,k] = MJ_inv[i]*rhsU[c,i,k]
            end
        end
    end
end

@unpack Vf,Dr,Ds,LIFT = rd
md = init_mesh((VX,VY),EToV,rd)
@unpack xf,yf,mapM,mapP,mapB,nxJ,nyJ,x,y = md
xb,yb = (x->x[mapB]).((xf,yf))

const Np = (N+1)*(N+1)
const K  = size(x,2)
const Nfaces = 4
const Nfp    = Nfaces*(N+1)
const J   = 1.0/K1D/K1D/4
const Jf  = 1.0/K1D/2
const rxJ = 2*K1D*J
const syJ = 2*K1D*J
const sJ  = Jf
const rxJ_sq = rxJ^2
const syJ_sq = syJ^2
const rxJ_db = 2*rxJ
const syJ_db = 2*syJ

# Convert diagonal matrices to vectors
Br   = Array(diag(Br))
Bs   = Array(diag(Bs))
M    = Array(diag(M))
Minv = Array(diag(Minv))

MJ_inv    = Minv./J
Br_halved = -sum(S0r,dims=2)
Bs_halved = -sum(S0s,dims=2)
S0r_sq    = S0r.^2
S0s_sq    = S0s.^2
coeff_arr = dt0*(Np-1).*Minv/J

Fmask  = [1:N+1; (N+1):(N+1):Np; Np:-1:Np-N; Np-N:-(N+1):1]
Fxmask = [(N+2):(2*N+2); (3*N+4):(4*N+4)]
Fymask = [1:(N+1); (2*N+3):(3*N+3)]

# 2D shocktube
inflow   = mapB[findall(@. (abs(xb) < TOL) | ((xb < 1/6) & (abs(yb) < TOL)))]
outflow  = mapB[findall(@. abs(xb-XLENGTH) < TOL)]
topflow  = mapB[findall(@. abs(yb-1.) < TOL)]
wall     = mapB[findall(@. (xb >= 1/6) & (abs(yb) < TOL))]
nx_wall  = nxJ[wall]/sJ
ny_wall  = nyJ[wall]/sJ
const TOP_INIT = (1+sqrt(3)/6)/sqrt(3)


# Initial condition 2D shocktube
at_left(x,y) = y-sqrt(3)*x+sqrt(3)/6 > 0.0
U = zeros(Nc,Np,K)
for k = 1:K
    for i = 1:Np
        if at_left(x[i,k],y[i,k]) 
            U[1,i,k] = rhoL
            U[2,i,k] = rhouL
            U[3,i,k] = rhovL
            U[4,i,k] = EL
        else
            U[1,i,k] = rhoR
            U[2,i,k] = rhouR
            U[3,i,k] = rhovR
            U[4,i,k] = ER           
        end
    end
end

# Preallocation
rhsU   = zeros(Float64,size(U))
f_x    = zeros(Float64,size(U))
f_y    = zeros(Float64,size(U))
lam    = zeros(Float64,Nfp,K)
lamP   = zeros(Float64,Nfp,K)
LFc    = zeros(Float64,Nfp,K)
Ubf    = zeros(Float64,Nc,Nfp,K)
UbP    = zeros(Float64,Nc,Nfp,K)
f_xM   = zeros(Float64,Nc,Nfp,K)
f_xP   = zeros(Float64,Nc,Nfp,K)
f_yM   = zeros(Float64,Nc,Nfp,K)
f_yP   = zeros(Float64,Nc,Nfp,K)
Uf     = zeros(Float64,Nc,Nfp,K)
UP     = zeros(Float64,Nc,Nfp,K)

rholog  = zeros(Float64,Np,K)
betalog = zeros(Float64,Np,K)
U_low   = zeros(Float64,Nc,Np,NUM_THREADS)
F_low   = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_high  = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_P     = zeros(Float64,Nc,Nfp,NUM_THREADS)
L       =  ones(Float64,Np,Np,NUM_THREADS)

prealloc = (f_x,f_y,lam,lamP,LFc,Ubf,UbP,f_x,f_xM,f_xP,f_y,f_yM,f_yP,Uf,UP,rholog,betalog,U_low,F_low,F_high,F_P,L)
ops      = (S0r,S0s,Sr,Ss,S0r_sq,S0s_sq,Minv,MJ_inv,Br_halved,Bs_halved,coeff_arr)
geom     = (mapP,Fmask,Fxmask,Fymask,x,y,nxJ,nyJ,sJ,xf,inflow,outflow,topflow,wall,nx_wall,ny_wall)


# Time stepping
"Time integration"
t = 0.0
U = collect(U)
resW = zeros(size(U))

#plotting nodes
@unpack VDM = rd
rp,sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/VDM
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)

# dt = dt0
# @btime rhs_IDP_fixdt!(U,rhsU,t,dt,prealloc,ops,geom);
# # rhs_IDP_fixdt!(U,rhsU,t,dt,prealloc,ops,geom);

dt_hist = []
i = 1

@time while t < T
#while i < 2
    # SSPRK(3,3)
    dt = min(dt0,T-t)
    rhs_IDP_fixdt!(U,rhsU,t,dt,prealloc,ops,geom);
    @. resW = U + dt*rhsU
    rhs_IDP_fixdt!(resW,rhsU,t,dt,prealloc,ops,geom);
    @. resW = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resW
    rhs_IDP_fixdt!(resW,rhsU,t,dt,prealloc,ops,geom);
    @. resW = resW+dt*rhsU
    @. U = 1/3*U+2/3*resW

    push!(dt_hist,dt)
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T, at step $i")
    flush(stdout)
    global i = i + 1
end

xp = Vp*x
yp = Vp*y
vv = Vp*U[1,:,:]
scatter(xp,yp,vv,zcolor=vv,camera=(0,90),colorbar=:right)
savefig("~/Desktop/N=$N,K1D=$K1D,T=$T,doubleMachReflection.png")






end #muladd