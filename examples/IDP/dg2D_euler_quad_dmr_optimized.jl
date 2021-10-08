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

@muladd @inbounds begin

@inline function pfun(rho,rhou,E)
    return (γ-1)*(E-.5*rhou^2/rho)
end

@inline function pfun(rho,rhou,rhov,E)
    return (γ-1)*(E-.5*(rhou^2+rhov^2)/rho)
end

@inline function Efun(rho,u,v,p)
    return p/(γ-1) + .5*rho*(u^2+v^2)
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

@inline function euler_fluxes_2D_x(rhoL,uL,vL,betaL,rhologL,betalogL,
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

    return FxS1,FxS2,FxS3,FxS4
end

@inline function euler_fluxes_2D_y(rhoL,uL,vL,betaL,rhologL,betalogL,
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

    FyS1 = rholog*vavg
    FyS2 = FyS1*uavg
    FyS3 = FyS1*vavg + pa
    FyS4 = f4aux*vavg

    return FyS1,FyS2,FyS3,FyS4
end

@inline function inviscid_flux_prim(rho,u,v,p)
    E = Efun(rho,u,v,p)

    rhou  = rho*u
    rhov  = rho*v
    rhouv = rho*u*v
    Ep    = E+p

    fx1 = rhou
    fx2 = rhou*u+p
    fx3 = rhouv
    fx4 = u*Ep

    fy1 = rhov
    fy2 = rhouv
    fy3 = rhov*v+p
    fy4 = v*Ep

    return fx1,fx2,fx3,fx4,fy1,fy2,fy3,fy4
end

@inline function limiting_param(rhoL,rhouL,rhovL,EL,rhoP,rhouP,rhovP,EP)
    # L - low order, P - P_ij
    l = 1.0
    # Limit density
    if rhoL + rhoP < -TOL
        l = max((-rhoL+POSTOL)/rhoP, 0.0)
    end

    # limiting internal energy (via quadratic function)
    a = rhoP*EP-(rhouP^2+rhovP^2)/2.0
    b = rhoP*EL+rhoL*EP-rhouL*rhouP-rhovL*rhovP
    c = rhoL*EL-(rhouL^2+rhovL^2)/2.0-POSTOL

    d = 1.0/(2.0*a)
    e = b^2-4.0*a*c
    g = -b*d

    l_eps_ij = 1.0
    if e >= 0
        f = sqrt(e)
        h = f*d
        r1 = g+h
        r2 = g-h
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
const WALLPT = 1.0/6.0
const Nc = 4 # number of components
"Approximation parameters"
const N = 3
const K1D = 250
const T = 0.2
const dt0 = 1e-3
const XLENGTH = 7.0/2.0
const CFL = 0.75
const NUM_THREADS = Threads.nthreads()
const BOTTOMRIGHT = N+1
const TOPRIGHT    = 2*(N+1)
const TOPLEFT     = 3*(N+1)

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
const fx1L,fx2L,fx3L,fx4L,fy1L,fy2L,fy3L,fy4L = inviscid_flux_prim(rhoL,uL,vL,pL)
const fx1R,fx2R,fx3R,fx4R,fy1R,fy2R,fy3R,fy4R = inviscid_flux_prim(rhoR,uR,vR,pR)
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

@inline function get_infoP(mapP,Fmask,i,k)
    gP = mapP[i,k]           # exterior global face node number
    kP = fld1(gP,Nfp)        # exterior element number
    iP = Fmask[mod1(gP,Nfp)] # exterior node number
    return iP,kP
end

@inline function check_BC(xM,yM,i)
    inflow  = ((abs(xM) < TOL) | ((xM < WALLPT) & (abs(yM) < TOL)) & ((i <= BOTTOMRIGHT) | (i > TOPLEFT)))
    outflow = ((abs(xM-XLENGTH) < TOL) & (abs(yM) > TOL) & (abs(yM-1.0) > TOL) & (i > BOTTOMRIGHT) & (i <= TOPRIGHT))
    topflow = ((abs(yM-1.0) < TOL) & (i > TOPRIGHT) & (i <= TOPLEFT))
    wall    = ((xM >= WALLPT) & (abs(yM) < TOL) & (i <= BOTTOMRIGHT))
    has_bc  = (inflow | outflow | topflow | wall)
    return inflow,outflow,topflow,wall,has_bc
end

@inline function get_consP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k,inflow,outflow,topflow,wall,has_bc)
    if inflow
        rhoP   = rhoL
        rhouP  = rhouL
        rhovP  = rhovL
        EP     = EL
    elseif outflow
        rhoP   = U[1,iM,k]
        rhouP  = U[2,iM,k]
        rhovP  = U[3,iM,k]
        EP     = U[4,iM,k]
    elseif wall
        # We assume the normals are [0;-1] here
        # Un = -u2 = -vM
        # Ut = -u1 = -uM
        # uP = -Ut = uM
        # vP = -(-Un) = -vM
        rhoP   = U[1,iM,k]
        uP     = uM
        vP     = -vM
        rhouP  = rhoP*uP
        rhovP  = rhoP*vP
        EP     = U[4,iM,k]
    elseif topflow
        breakpoint = TOP_INIT+t*SHOCKSPD
        if xM < breakpoint
            rhoP   = rhoL
            rhouP  = rhouL
            rhovP  = rhovL
            EP     = EL
        else
            rhoP   = rhoR
            rhouP  = rhouR
            rhovP  = rhovR
            EP     = ER
        end
    else                         # if not on the physical boundary
        iP,kP = get_infoP(mapP,Fmask,i,k)

        rhoP  = U[1,iP,kP]
        rhouP = U[2,iP,kP]
        rhovP = U[3,iP,kP]
        EP    = U[4,iP,kP]
    end

    return rhoP,rhouP,rhovP,EP
end

@inline function get_fluxP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k,inflow,outflow,topflow,wall,has_bc,rhoP,rhouP,rhovP,EP)
    if inflow
        fx_1_P = fx1L
        fx_2_P = fx2L
        fx_3_P = fx3L
        fx_4_P = fx4L
        fy_1_P = fy1L
        fy_2_P = fy2L
        fy_3_P = fy3L
        fy_4_P = fy4L
    elseif outflow
        fx_1_P = fx1R
        fx_2_P = fx2R
        fx_3_P = fx3R
        fx_4_P = fx4R
        fy_1_P = fy1R
        fy_2_P = fy2R
        fy_3_P = fy3R
        fy_4_P = fy4R
    elseif wall
        # We assume the normals are [0;-1] here
        # Un = -u2 = -vM
        # Ut = -u1 = -uM
        # uP = -Ut = uM
        # vP = -(-Un) = -vM
        uP = rhouP/rhoP
        vP = rhovP/rhoP
        pP = pfun(rhoP,rhouP,rhovP,EP)
        fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P = inviscid_flux_prim(rhoP,uP,vP,pP)
    elseif topflow
        breakpoint = TOP_INIT+t*SHOCKSPD
        if xM < breakpoint
            fx_1_P = fx1L
            fx_2_P = fx2L
            fx_3_P = fx3L
            fx_4_P = fx4L
            fy_1_P = fy1L
            fy_2_P = fy2L
            fy_3_P = fy3L
            fy_4_P = fy4L
        else
            fx_1_P = fx1R
            fx_2_P = fx2R
            fx_3_P = fx3R
            fx_4_P = fx4R
            fy_1_P = fy1R
            fy_2_P = fy2R
            fy_3_P = fy3R
            fy_4_P = fy4R
        end
    else                         # if not on the physical boundary
        iP,kP = get_infoP(mapP,Fmask,i,k)

        fx_1_P = f_x[1,iP,kP]
        fx_2_P = f_x[2,iP,kP]
        fx_3_P = f_x[3,iP,kP]
        fx_4_P = f_x[4,iP,kP]
        fy_1_P = f_y[1,iP,kP]
        fy_2_P = f_y[2,iP,kP]
        fy_3_P = f_y[3,iP,kP]
        fy_4_P = f_y[4,iP,kP]
    end

    return fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P
end

@inline function get_valP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k)
    inflow,outflow,topflow,wall,has_bc = check_BC(xM,yM,i)
    rhoP,rhouP,rhovP,EP = get_consP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k,inflow,outflow,topflow,wall,has_bc)
    fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P = get_fluxP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k,inflow,outflow,topflow,wall,has_bc,rhoP,rhouP,rhovP,EP)
    return rhoP,rhouP,rhovP,EP,fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P,has_bc
end

@inline function is_face_x(i)
    return (((i > BOTTOMRIGHT) & (i <= TOPRIGHT)) | (i > TOPLEFT))
end

@inline function is_face_y(i)
    return (((i > TOPRIGHT) & (i <= TOPLEFT)) | (i <= BOTTOMRIGHT))
end

@inline function update_F_low!(F_low,k,tid,i,j,λ,S0J_ij,U,f)
    # Tensor product elements: f is f_x or f_y
    # S0J_ij - S0xJ_ij or S0yJ_ij
    rho_j  = U[1,j,k]
    rhou_j = U[2,j,k]
    rhov_j = U[3,j,k]
    E_j    = U[4,j,k]
    f_1_j  = f[1,j,k]
    f_2_j  = f[2,j,k]
    f_3_j  = f[3,j,k]
    f_4_j  = f[4,j,k]
    rho_i  = U[1,i,k]
    rhou_i = U[2,i,k]
    rhov_i = U[3,i,k]
    E_i    = U[4,i,k]
    f_1_i  = f[1,i,k]
    f_2_i  = f[2,i,k]
    f_3_i  = f[3,i,k]
    f_4_i  = f[4,i,k]

    FL1 = (S0J_ij*(f_1_i+f_1_j) - λ*(rho_j-rho_i))
    FL2 = (S0J_ij*(f_2_i+f_2_j) - λ*(rhou_j-rhou_i))
    FL3 = (S0J_ij*(f_3_i+f_3_j) - λ*(rhov_j-rhov_i))
    FL4 = (S0J_ij*(f_4_i+f_4_j) - λ*(E_j-E_i))

    F_low[1,i,j,tid] = FL1
    F_low[2,i,j,tid] = FL2
    F_low[3,i,j,tid] = FL3
    F_low[4,i,j,tid] = FL4

    F_low[1,j,i,tid] = -FL1
    F_low[2,j,i,tid] = -FL2
    F_low[3,j,i,tid] = -FL3
    F_low[4,j,i,tid] = -FL4
end

@inline function update_F_high!(F_high,k,tid,i,j,SJ_ij_db,U,rholog,betalog,direction)
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
    
    if (direction == 0)
        # x direction
        F1,F2,F3,F4 = euler_fluxes_2D_x(rho_i,u_i,v_i,beta_i,rholog_i,betalog_i,
                                        rho_j,u_j,v_j,beta_j,rholog_j,betalog_j)
    else
        # y direction
        F1,F2,F3,F4 = euler_fluxes_2D_y(rho_i,u_i,v_i,beta_i,rholog_i,betalog_i,
                                        rho_j,u_j,v_j,beta_j,rholog_j,betalog_j)
    end
    FH1 = SJ_ij_db*F1
    FH2 = SJ_ij_db*F2
    FH3 = SJ_ij_db*F3
    FH4 = SJ_ij_db*F4

    F_high[1,i,j,tid] = FH1
    F_high[2,i,j,tid] = FH2
    F_high[3,i,j,tid] = FH3
    F_high[4,i,j,tid] = FH4

    F_high[1,j,i,tid] = -FH1
    F_high[2,j,i,tid] = -FH2
    F_high[3,j,i,tid] = -FH3
    F_high[4,j,i,tid] = -FH4
end

function rhs_IDP!(U,rhsU,t,dt,prealloc,ops,geom,in_s1)
    f_x,f_y,rholog,betalog,U_low,F_low,F_high,F_P,L,wspd_arr,λ_arr,λf_arr,dii_arr = prealloc
    S0xJ_vec,S0yJ_vec,S0r_nnzi,S0r_nnzj,S0s_nnzi,S0s_nnzj,SxJ_db_vec,SyJ_db_vec,Sr_nnzi,Sr_nnzj,Ss_nnzi,Ss_nnzj,MJ_inv,BrJ_halved,BsJ_halved,coeff_arr,S_nnzi,S_nnzj = ops
    mapP,Fmask,x,y = geom

    fill!(rhsU,0.0)
    @batch for k = 1:K
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
    end

    # Precompute wavespeeds
    @batch for k = 1:K
        tid = Threads.threadid()

        # Interior wavespd, leading 2 - x and y directions 
        for i = 1:Np
            rho_i  = U[1,i,k]
            rhou_i = U[2,i,k]
            rhov_i = U[3,i,k]
            E_i    = U[4,i,k]
            wspd_arr[i,1,k] = wavespeed_1D(rho_i,rhou_i,E_i)
            wspd_arr[i,2,k] = wavespeed_1D(rho_i,rhov_i,E_i)
        end

        # Interior dissipation coeff
        for c_r = 1:S0r_nnz_hv
            i = S0r_nnzi[c_r]
            j = S0r_nnzj[c_r]
            λ = abs(S0xJ_vec[c_r])*max(wspd_arr[i,1,k],wspd_arr[j,1,k])
            λ_arr[c_r,1,k] = λ
            if in_s1
                dii_arr[i,k] = dii_arr[i,k] + λ
            end
        end

        for c_s = 1:S0s_nnz_hv
            i = S0s_nnzi[c_s]
            j = S0s_nnzj[c_s]
            λ = abs(S0yJ_vec[c_s])*max(wspd_arr[i,2,k],wspd_arr[j,2,k])
            λ_arr[c_s,2,k] = λ
            if in_s1
                dii_arr[i,k] = dii_arr[i,k] + λ
            end
        end

        # Interface dissipation coeff 
        for i = 1:Nfp
            iM = Fmask[i]
            BrJ_ii_halved_abs = abs(BrJ_halved[iM])
            BsJ_ii_halved_abs = abs(BsJ_halved[iM])
            xM    = x[iM,k]
            yM    = y[iM,k]

            inflow,outflow,topflow,wall,has_bc = check_BC(xM,yM,i)
            iP,kP = get_infoP(mapP,Fmask,i,k)
            
            if is_face_x(i)
                if has_bc
                    λf_arr[i,k] = 0.0
                else
                    λM = wspd_arr[iM,1,k]
                    λP = wspd_arr[iP,1,kP]
                    λf = max(λM,λP)*BrJ_ii_halved_abs
                    λf_arr[i,k] = λf
                    if in_s1
                        dii_arr[iM,k] = dii_arr[iM,k] + λf
                    end
                end
            end

            if is_face_y(i)
                if has_bc
                    λf_arr[i,k] = 0.0
                else
                    λM = wspd_arr[iM,2,k]
                    λP = wspd_arr[iP,2,kP]
                    λf = max(λM,λP)*BsJ_ii_halved_abs
                    λf_arr[i,k] = λf
                    if in_s1
                        dii_arr[iM,k] = dii_arr[iM,k] + λf
                    end
                end
            end
        end
    end

    # If at the first stage, calculate the time step
    if in_s1
        @batch for k = 1:K
            for i = 1:Np
                dt = min(dt,1.0/MJ_inv[i]/2.0/dii_arr[i,k])
            end
        end
    end

    # =====================
    # Loop through elements
    # =====================
    @batch for k = 1:K
        tid = Threads.threadid()
        fill!(U_low ,0.0)

        # Calculate low order algebraic flux
        for c_r = 1:S0r_nnz_hv
            i = S0r_nnzi[c_r]
            j = S0r_nnzj[c_r]
            λ = λ_arr[c_r,1,k]
            S0xJ_ij = S0xJ_vec[c_r]
            update_F_low!(F_low,k,tid,i,j,λ,S0xJ_ij,U,f_x)
        end

        for c_s = 1:S0s_nnz_hv
            i = S0s_nnzi[c_s]
            j = S0s_nnzj[c_s]
            λ = λ_arr[c_s,2,k]
            S0yJ_ij = S0yJ_vec[c_s]
            update_F_low!(F_low,k,tid,i,j,λ,S0yJ_ij,U,f_y)
        end

        # Calculate high order algebraic flux
        for c_r = 1:Sr_nnz_hv
            i         = Sr_nnzi[c_r]
            j         = Sr_nnzj[c_r]
            SxJ_ij_db = SxJ_db_vec[c_r]
            update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,U,rholog,betalog,0)
        end

        for c_r = 1:Sr_nnz_hv
            i         = Ss_nnzi[c_r]
            j         = Ss_nnzj[c_r]
            SyJ_ij_db = SyJ_db_vec[c_r]
            update_F_high!(F_high,k,tid,i,j,SyJ_ij_db,U,rholog,betalog,1)
        end

        # Calculate interface fluxes
        for i = 1:Nfp
            iM    = Fmask[i]
            BrJ_ii_halved = BrJ_halved[iM]
            BsJ_ii_halved = BsJ_halved[iM]
            xM    = x[iM,k]
            yM    = y[iM,k]
            rhoM  = U[1,iM,k]
            rhouM = U[2,iM,k]
            rhovM = U[3,iM,k]
            EM    = U[4,iM,k]
            uM    = rhouM/rhoM
            vM    = rhovM/rhoM
            fx_1_M = f_x[1,iM,k]
            fx_2_M = f_x[2,iM,k]
            fx_3_M = f_x[3,iM,k]
            fx_4_M = f_x[4,iM,k]
            fy_1_M = f_y[1,iM,k]
            fy_2_M = f_y[2,iM,k]
            fy_3_M = f_y[3,iM,k]
            fy_4_M = f_y[4,iM,k]

            rhoP,rhouP,rhovP,EP,fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P,has_bc = get_valP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k)
            λ = λf_arr[i,k]

            # flux in x direction
            if is_face_x(i)
                F_P[1,i,tid] = (BrJ_ii_halved*(fx_1_M+fx_1_P)
                               -λ*(rhoP-rhoM) )
                F_P[2,i,tid] = (BrJ_ii_halved*(fx_2_M+fx_2_P)
                               -λ*(rhouP-rhouM) )
                F_P[3,i,tid] = (BrJ_ii_halved*(fx_3_M+fx_3_P)
                               -λ*(rhovP-rhovM) )
                F_P[4,i,tid] = (BrJ_ii_halved*(fx_4_M+fx_4_P)
                               -λ*(EP-EM) )
            end

            # flux in y direction
            if is_face_y(i)
                F_P[1,i,tid] = (BsJ_ii_halved*(fy_1_M+fy_1_P)
                               -λ*(rhoP-rhoM) )
                F_P[2,i,tid] = (BsJ_ii_halved*(fy_2_M+fy_2_P)
                               -λ*(rhouP-rhouM) )
                F_P[3,i,tid] = (BsJ_ii_halved*(fy_3_M+fy_3_P)
                               -λ*(rhovP-rhovM) )
                F_P[4,i,tid] = (BsJ_ii_halved*(fy_4_M+fy_4_P)
                               -λ*(EP-EM) )
            end
        end

        # Calculate low order solution
        for j = 1:Np
            for i = 1:Np
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

        for i = 1:Np
            for c = 1:Nc
                U_low[c,i,tid] = U[c,i,k] - dt*MJ_inv[i]*U_low[c,i,tid]
            end
        end

        # Calculate limiting parameters
        for ni = 1:S_nnz_hv
            i = S_nnzi[ni]
            j = S_nnzj[ni]
            coeff = coeff_arr[i]
            rhoi  = U_low[1,i,tid]
            rhoui = U_low[2,i,tid]
            rhovi = U_low[3,i,tid]
            Ei    = U_low[4,i,tid]
            rhoj  = U_low[1,j,tid]
            rhouj = U_low[2,j,tid]
            rhovj = U_low[3,j,tid]
            Ej    = U_low[4,j,tid]
            rhoP  = coeff*(F_low[1,i,j,tid]-F_high[1,i,j,tid])
            rhouP = coeff*(F_low[2,i,j,tid]-F_high[2,i,j,tid])
            rhovP = coeff*(F_low[3,i,j,tid]-F_high[3,i,j,tid])
            EP    = coeff*(F_low[4,i,j,tid]-F_high[4,i,j,tid])
            L[ni,tid] = min(limiting_param(rhoi,rhoui,rhovi,Ei, rhoP, rhouP, rhovP, EP),
                            limiting_param(rhoj,rhouj,rhovj,Ej,-rhoP,-rhouP,-rhovP,-EP))
        end

        # Elementwise limiting
        l_e = 1.0
        for i = 1:S_nnz_hv
            l_e = min(l_e,L[i,tid])
        end
        l_em1 = l_e-1.0

        # TODO: reorder, use l_e directly
        for ni = 1:S_nnz_hv
            i     = S_nnzi[ni]
            j     = S_nnzj[ni]
            for c = 1:Nc
                FL_ij = F_low[c,i,j,tid]
                FH_ij = F_high[c,i,j,tid]
                rhsU[c,i,k] = rhsU[c,i,k] + l_em1*FL_ij - l_e*FH_ij
                rhsU[c,j,k] = rhsU[c,j,k] - l_em1*FL_ij + l_e*FH_ij
            end
        end

        for i = 1:Nfp
            iM = Fmask[i]
            for c = 1:Nc
                rhsU[c,iM,k] = rhsU[c,iM,k] - F_P[c,i,tid]
            end
        end
    end

    @batch for k = 1:K
        for i = 1:Np
            for c = 1:Nc
                rhsU[c,i,k] = MJ_inv[i]*rhsU[c,i,k]
            end
        end
    end

    return dt
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
coeff_arr = dt0*(Np-1).*Minv/J

Fmask  = [1:N+1; (N+1):(N+1):Np; Np:-1:Np-N; Np-N:-(N+1):1]
Fxmask = [(N+2):(2*N+2); (3*N+4):(4*N+4)]
Fymask = [1:(N+1); (2*N+3):(3*N+3)]


S0r_nnz = length(nonzeros(S0r))
S0s_nnz = length(nonzeros(S0s))
Sr_nnz  = length(nonzeros(Sr))
Ss_nnz  = length(nonzeros(Ss))
const S0r_nnz_hv = div(S0r_nnz,2)
const S0s_nnz_hv = div(S0s_nnz,2)
const S0_nnz_hv  = S0r_nnz_hv+S0s_nnz_hv
const Sr_nnz_hv  = div(Sr_nnz,2)
const Ss_nnz_hv  = div(Ss_nnz,2)
const S_nnz_hv   = Sr_nnz_hv+Ss_nnz_hv
S0r_vec  = zeros(S0r_nnz_hv)
S0s_vec  = zeros(S0s_nnz_hv)
S0r_nnzi = zeros(Int32,S0r_nnz_hv)
S0s_nnzi = zeros(Int32,S0s_nnz_hv)
S0r_nnzj = zeros(Int32,S0r_nnz_hv)
S0s_nnzj = zeros(Int32,S0s_nnz_hv)
Sr_vec   = zeros(Sr_nnz_hv)
Ss_vec   = zeros(Ss_nnz_hv)
Sr_nnzi  = zeros(Int32,Sr_nnz_hv)
Ss_nnzi  = zeros(Int32,Ss_nnz_hv)
Sr_nnzj  = zeros(Int32,Sr_nnz_hv)
Ss_nnzj  = zeros(Int32,Ss_nnz_hv)
S_nnzi   = zeros(Int32,S_nnz_hv)
S_nnzj   = zeros(Int32,S_nnz_hv)
global count_r0 = 1
global count_s0 = 1
global count_r  = 1
global count_s  = 1
global count    = 1
for j = 2:Np
    for i = 1:j-1
        S0r_ij = S0r[i,j]
        S0s_ij = S0s[i,j]
        Sr_ij  = Sr[i,j]
        Ss_ij  = Ss[i,j]
        if S0r_ij != 0
            global S0r_vec[count_r0]  = S0r_ij
            global S0r_nnzi[count_r0] = i
            global S0r_nnzj[count_r0] = j
            global count_r0 = count_r0+1
        end
        if S0s_ij != 0
            global S0s_vec[count_s0]  = S0s_ij
            global S0s_nnzi[count_s0] = i
            global S0s_nnzj[count_s0] = j
            global count_s0 = count_s0+1
        end
        if Sr_ij != 0
            global Sr_vec[count_r]  = Sr_ij
            global Sr_nnzi[count_r] = i
            global Sr_nnzj[count_r] = j
            global S_nnzi[count]    = i
            global S_nnzj[count]    = j
            global count_r = count_r+1
            global count   = count+1
        end
        if Ss_ij != 0
            global Ss_vec[count_s]  = Ss_ij
            global Ss_nnzi[count_s] = i
            global Ss_nnzj[count_s] = j
            global S_nnzi[count]    = i
            global S_nnzj[count]    = j
            global count_s = count_s+1
            global count   = count+1
        end
    end
end

# Scale by Jacobian
S0xJ_vec   = rxJ*S0r_vec
S0yJ_vec   = syJ*S0s_vec
SxJ_db_vec = 2*rxJ*Sr_vec
SyJ_db_vec = 2*syJ*Ss_vec
BrJ_halved = Jf*Br_halved
BsJ_halved = Jf*Bs_halved

# 2D shocktube
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
rholog  = zeros(Float64,Np,K)
betalog = zeros(Float64,Np,K)
U_low   = zeros(Float64,Nc,Np,NUM_THREADS)
F_low   = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_high  = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_P     = zeros(Float64,Nc,Nfp,NUM_THREADS)
L       =  ones(Float64,S_nnz_hv,NUM_THREADS)
wspd_arr = zeros(Float64,Np,2,K)
λ_arr    = zeros(Float64,S0r_nnz_hv,2,K) # Assume S0r and S0s has same number of nonzero entries
λf_arr   = zeros(Float64,Nfp,K)
dii_arr  = zeros(Float64,Np,K)

prealloc = (f_x,f_y,rholog,betalog,U_low,F_low,F_high,F_P,L,wspd_arr,λ_arr,λf_arr,dii_arr)
ops      = (S0xJ_vec,  S0yJ_vec,  S0r_nnzi,S0r_nnzj,S0s_nnzi,S0s_nnzj,
            SxJ_db_vec,SyJ_db_vec,Sr_nnzi, Sr_nnzj, Ss_nnzi, Ss_nnzj,
            MJ_inv,BrJ_halved,BsJ_halved,coeff_arr,S_nnzi,S_nnzj)
geom     = (mapP,Fmask,x,y)


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

#Timing
dt = dt0
#rhs_IDP!(U,rhsU,t,dt,prealloc,ops,geom,true);
@btime rhs_IDP!($U,$rhsU,$t,$dt,$prealloc,$ops,$geom,true);
# @profiler rhs_IDP!(U,rhsU,t,dt,prealloc,ops,geom,true);

# dt_hist = []
# i = 1

# mapN = collect(reshape(1:Np*K,Np,K))
# inflow_nodal = mapN[findall(@. (abs(x) < TOL) | ((x < WALLPT) & (abs(y) < TOL)))]
# outflow_nodal = mapN[findall(@. abs(x-XLENGTH) < TOL)]
# topflow_nodal = mapN[findall(@. abs(y-1.) < TOL)]

# @inline function enforce_BC_timestep!(U,inflow_nodal,topflow_nodal,t)
#     for i = inflow_nodal
#         k = fld1(i,Np)
#         j = mod1(i,Np)
#         U[1,j,k] = rhoL
#         U[2,j,k] = rhouL
#         U[3,j,k] = rhovL
#         U[4,j,k] = EL
#     end
#     breakpoint = TOP_INIT+t*SHOCKSPD
#     for i = topflow_nodal
#         k = fld1(i,Np)
#         j = mod1(i,Np)
#         if x[i] < breakpoint
#             U[1,j,k] = rhoL
#             U[2,j,k] = rhouL
#             U[3,j,k] = rhovL
#             U[4,j,k] = EL
#         else
#             U[1,j,k] = rhoR
#             U[2,j,k] = rhouR
#             U[3,j,k] = rhovR
#             U[4,j,k] = ER
#         end
#     end
# end


# @time while t < T
#     # SSPRK(3,3)
#     fill!(dii_arr,0.0)
#     # dt = min(dt0,T-t)
#     dt = dt0
#     dt = rhs_IDP!(U,rhsU,t,dt,prealloc,ops,geom,true);
#     dt = min(CFL*dt,T-t)
#     @. resW = U + dt*rhsU
#     rhs_IDP!(resW,rhsU,t+dt,dt,prealloc,ops,geom,false);
#     @. resW = resW+dt*rhsU
#     @. resW = 3/4*U+1/4*resW
#     rhs_IDP!(resW,rhsU,t+dt/2,dt,prealloc,ops,geom,false);
#     @. resW = resW+dt*rhsU
#     @. U = 1/3*U+2/3*resW
#     enforce_BC_timestep!(U,inflow_nodal,topflow_nodal,t+dt);

#     push!(dt_hist,dt)
#     global t = t + dt
#     println("Current time $t with time step size $dt, and final time $T, at step $i")
#     flush(stdout)
#     global i = i + 1
# end

# xp = Vp*x
# yp = Vp*y
# vv = Vp*U[1,:,:]
# xp = vec(xp)
# yp = vec(yp)
# vv = vec(vv)
# scatter(xp,yp,vv,zcolor=vv,camera=(0,90),colorbar=:right)
# savefig("~/Desktop/N=$N,K1D=$K1D,T=$T,doubleMachReflection.png")






end #muladd
