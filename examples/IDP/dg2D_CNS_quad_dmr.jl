using Pkg
Pkg.activate("Project.toml")
using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using StaticArrays
using DelimitedFiles
using Polyester
using MuladdMacro
using DataFrames
using JLD2
using FileIO
using WriteVTK

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh

using SetupDG

@muladd begin

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

@inline function limiting_param(rhoL,rhouL,rhovL,EL,rhoP,rhouP,rhovP,EP,Lrho,Lrhoe)
    # L - low order, P - P_ij
    l = 1.0
    # Limit density
    if rhoL + rhoP < Lrho
        l = max((Lrho-rhoL)/rhoP, 0.0)
    end

    p = pfun(rhoL+l*rhoP,rhouL+l*rhouP,rhovL+l*rhovP,EL+l*EP)
    if p/(γ-1) > Lrhoe
        return l
    end

    # limiting internal energy (via quadratic function)
    a = rhoP*EP-(rhouP^2+rhovP^2)/2.0
    b = rhoP*EL+rhoL*EP-rhouL*rhouP-rhovL*rhovP-rhoP*Lrhoe
    c = rhoL*EL-(rhouL^2+rhovL^2)/2.0-rhoL*Lrhoe

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

@inline function zhang_wavespd(rhoi,rhoui,rhovi,Ei,sigmax2,sigmax3,sigmax4,sigmay2,sigmay3,sigmay4,nx,ny)
    pl = pfun(rhoi,rhoui,rhovi,Ei)
    ui = rhoui/rhoi
    vi = rhovi/rhoi
    ei = (Ei-.5*rhoi*(ui^2+vi^2))/rhoi
    tau_xx = sigmax2
    tau_yx = sigmax3
    tau_xy = sigmay2
    tau_yy = sigmay3
    q_x = ui*tau_xx+vi*tau_yx-sigmax4
    q_y = ui*tau_xy+vi*tau_yy-sigmay4

    v_vec = ui*nx+vi*ny
    q_vec = q_x*nx+q_y*ny
    tau_vec_x = nx*tau_xx+ny*tau_yx
    tau_vec_y = nx*tau_xy+ny*tau_yy

    return POSTOL+abs(v_vec)+1/(2*rhoi^2*ei)*(sqrt(rhoi^2*q_vec^2+2*rhoi^2*ei*((tau_vec_x-pl*nx)^2+(tau_vec_y-pl*ny)^2))+rhoi*abs(q_vec))
end

@inline function get_Kvisc(Kxx,Kyy,Kxy,tid,v1,v2,v3,v4)
    lam = -lambda
    λ = -lambda
    μ   = mu
    v2_sq = v2^2
    v3_sq = v3^2
    v4_sq = v4^2
    λ2μ = (λ+2.0*μ)
    inv_v4_cubed = 1/(v4^3)

    Kxx[tid][2,2] = inv_v4_cubed*-λ2μ*v4_sq
    Kxx[tid][2,4] = inv_v4_cubed*λ2μ*v2*v4
    Kxx[tid][3,3] = inv_v4_cubed*-μ*v4_sq
    Kxx[tid][3,4] = inv_v4_cubed*μ*v3*v4
    Kxx[tid][4,2] = inv_v4_cubed*λ2μ*v2*v4
    Kxx[tid][4,3] = inv_v4_cubed*μ*v3*v4
    Kxx[tid][4,4] = inv_v4_cubed*-(λ2μ*v2_sq + μ*v3_sq - γ*μ*v4/Pr)

    Kxy[tid][2,3] = inv_v4_cubed*-λ*v4_sq
    Kxy[tid][2,4] = inv_v4_cubed*λ*v3*v4
    Kxy[tid][3,2] = inv_v4_cubed*-μ*v4_sq
    Kxy[tid][3,4] = inv_v4_cubed*μ*v2*v4
    Kxy[tid][4,2] = inv_v4_cubed*μ*v3*v4
    Kxy[tid][4,3] = inv_v4_cubed*λ*v2*v4
    Kxy[tid][4,4] = inv_v4_cubed*(λ+μ)*(-v2*v3)

    Kyy[tid][2,2] = inv_v4_cubed*-μ*v4_sq
    Kyy[tid][2,4] = inv_v4_cubed*μ*v2*v4
    Kyy[tid][3,3] = inv_v4_cubed*-λ2μ*v4_sq
    Kyy[tid][3,4] = inv_v4_cubed*λ2μ*v3*v4
    Kyy[tid][4,2] = inv_v4_cubed*μ*v2*v4
    Kyy[tid][4,3] = inv_v4_cubed*λ2μ*v3*v4
    Kyy[tid][4,4] = inv_v4_cubed*-(λ2μ*v3_sq + μ*v2_sq - γ*μ*v4/Pr)
end

@inline function entropyvar(rho,rhou,rhov,E)
    p       = pfun(rho,rhou,rhov,E)
    s       = log(p/(rho^γ))
    gm1divp = (γ-1)/p
    v1      = (γ+1-s)-gm1divp*E 
    v2      = gm1divp*rhou
    v3      = gm1divp*rhov
    v4      = -gm1divp*rho
    return v1,v2,v3,v4
end

@inline function entropyvar(rho,rhou,rhov,E,p)
    s       = log(p/(rho^γ))
    gm1divp = (γ-1)/p
    v1      = (γ+1-s)-gm1divp*E 
    v2      = gm1divp*rhou
    v3      = gm1divp*rhov
    v4      = -gm1divp*rho
    return v1,v2,v3,v4
end

const OUTPUTPATH = "/home/yiminlin/Desktop/dg2D_CNS_quad_dmr_output"
const SAVEINT    = 1000

const LIMITOPT   = 2 # 1 if elementwise limiting lij, 2 if elementwise limiting li
const POSDETECT  = 0 # 1 if turn on detection, 0 otherwise
const LBOUNDTYPE = 0.1 # 0 if use POSTOL as lower bound, 1 if use 0.1*loworder
const BCFLUXTYPE = 2 # 0 - Central, 1 - Nondissipative, 2 - dissipative
const VISCPENTYPE = 0.1 # \in [0,1]: alpha = VISCPENTYPE, -1: -1/Re/v_4
const IFSHOCKCAPTURE = true
const TOL = 1e-14
const POSTOL = 1e-14
const Nc = 4 # number of components
const GIFINT = 100
const USEPLOTPT = false#true
const NUMPLOTPT = 12

const γ = 1.4
const Re = 500
const mu = 1/Re
const lambda = 2/3*mu
const Pr = 0.73
const cp = γ/(γ-1)
const cv = 1/(γ-1)
const kappa = mu*cp/Pr

"Approximation parameters"
const N = 3
const K1D = 250
const T = 0.2
const XLENGTH = 7.0/2.0
const CFL = 0.75
const NUM_THREADS = Threads.nthreads()
const BOTTOMRIGHT = N+1
const TOPRIGHT    = 2*(N+1)
const TOPLEFT     = 3*(N+1)

const TN       = 0.5*10^(-1.8*(N+1)^0.25)
const alphamin = 0.001
const alphamax = 1.0
const alphaE0  = 0.0001
const s_factor = log((1-alphaE0)/alphaE0)

# DMR initial condition
const WALLPT = 1.0/6.0
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
const v1L,v2L,v3L,v4L = entropyvar(rhoL,rhouL,rhovL,EL)
const v1R,v2R,v3R,v4R = entropyvar(rhoR,rhouR,rhovR,ER)
const SHOCKSPD = 10.0/cos(pi/6)

"Mesh related variables"
VX, VY, EToV = uniform_quad_mesh(Int(round(XLENGTH*K1D)),K1D)
@. VX = (VX+1)/2*XLENGTH
@. VY = (VY+1)/2
# @. VX = VX*2
# @. VY = 2*((VY+1)/2)^(2) - 1

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

@unpack Vf,VDM = rd
VDMinv = inv(VDM)

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

@inline function is_inflow(i,xM,yM)
    return ((abs(xM) < TOL) | ((xM < WALLPT) & (abs(yM) < TOL)) & ((i <= BOTTOMRIGHT) | (i > TOPLEFT)))
end

@inline function is_outflow(i,xM,yM)
    return ((abs(xM-XLENGTH) < TOL) & (abs(yM) > TOL) & (abs(yM-1.0) > TOL) & (i > BOTTOMRIGHT) & (i <= TOPRIGHT))
end

@inline function is_topflow(i,yM)
    return ((abs(yM-1.0) < TOL) & (i > TOPRIGHT) & (i <= TOPLEFT))
end

@inline function is_bottom_wall(i,xM,yM)
    return ((xM >= WALLPT) & (abs(yM) < TOL) & (i <= BOTTOMRIGHT))
end

@inline function is_boundary(i,xM,yM)
    return is_inflow(i,xM,yM) || is_outflow(i,xM,yM) || is_topflow(i,yM) || is_bottom_wall(i,xM,yM)
end

@inline function noslip_flux(rhoM,uM,vM,pM,nx,ny)
    # Assume n = (n1,n2) normalized normal
    c     = sqrt(γ*pM/rhoM)
    vn    = uM*nx+vM*ny
    Ma_n  = vn/c
    Pstar = pM
    if (BCFLUXTYPE == 2)
        if (vn > 0)
            Pstar = (1+γ*Ma_n*((γ+1)/4*Ma_n + sqrt(((γ+1)/4*Ma_n)^2+1)))*pM
        else
            Pstar = max((1+1/2*(γ-1)*Ma_n)^(2*γ/(γ-1)), 1e-4)*pM
        end
    end

    return 0.0, Pstar*nx, Pstar*ny, 0.0
end

@inline function get_infoP(mapP,Fmask,i,k)
    gP = mapP[i,k]           # exterior global face node number
    kP = fld1(gP,Nfp)        # exterior element number
    iP = Fmask[mod1(gP,Nfp)] # exterior node number
    return iP,kP
end

@inline function get_consP(UP,U,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k,tid)
    if is_bottom_wall(i,xM,yM)
        uP        = uM
        vP        = -vM
        rhoP      = U[1,iM,k]
        UP[1,tid] = rhoP
        UP[2,tid] = rhoP*uP
        UP[3,tid] = rhoP*vP
        UP[4,tid] = U[4,iM,k]
    elseif is_inflow(i,xM,yM)
        UP[1,tid] = rhoL
        UP[2,tid] = rhouL
        UP[3,tid] = rhovL
        UP[4,tid] = EL
    elseif is_outflow(i,xM,yM)
        UP[1,tid] = rhoR
        UP[2,tid] = rhouR
        UP[3,tid] = rhovR
        UP[4,tid] = ER
    elseif is_topflow(i,yM)
        breakpoint = TOP_INIT+t*SHOCKSPD
        if xM < breakpoint
            UP[1,tid] = rhoL
            UP[2,tid] = rhouL
            UP[3,tid] = rhovL
            UP[4,tid] = EL
        else
            UP[1,tid] = rhoR
            UP[2,tid] = rhouR
            UP[3,tid] = rhovR
            UP[4,tid] = ER
        end
    else
        iP,kP = get_infoP(mapP,Fmask,i,k)

        UP[1,tid] = U[1,iP,kP]
        UP[2,tid] = U[2,iP,kP]
        UP[3,tid] = U[3,iP,kP]
        UP[4,tid] = U[4,iP,kP]
    end
end

@inline function get_fluxP(fP,f_x,f_y,t,mapP,Fmask,i,xM,yM,k,UP,tid)
    if is_bottom_wall(i,xM,yM)
        rhoP  = UP[1,tid]
        rhouP = UP[2,tid]
        rhovP = UP[3,tid]
        EP    = UP[4,tid]
        uP = rhouP/rhoP
        vP = rhovP/rhoP
        pP = pfun(rhoP,rhouP,rhovP,EP)
        fP[1,1,tid],fP[2,1,tid],fP[3,1,tid],fP[4,1,tid],
        fP[1,2,tid],fP[2,2,tid],fP[3,2,tid],fP[4,2,tid] = inviscid_flux_prim(rhoP,uP,vP,pP)
    elseif is_inflow(i,xM,yM)
        fP[1,1,tid] = fx1L
        fP[2,1,tid] = fx2L
        fP[3,1,tid] = fx3L
        fP[4,1,tid] = fx4L
        fP[1,2,tid] = fy1L
        fP[2,2,tid] = fy2L
        fP[3,2,tid] = fy3L
        fP[4,2,tid] = fy4L
    elseif is_outflow(i,xM,yM)
        fP[1,1,tid] = fx1R
        fP[2,1,tid] = fx2R
        fP[3,1,tid] = fx3R
        fP[4,1,tid] = fx4R
        fP[1,2,tid] = fy1R
        fP[2,2,tid] = fy2R
        fP[3,2,tid] = fy3R
        fP[4,2,tid] = fy4R
    elseif is_topflow(i,yM)
        breakpoint = TOP_INIT+t*SHOCKSPD
        if xM < breakpoint
            fP[1,1,tid] = fx1L
            fP[2,1,tid] = fx2L
            fP[3,1,tid] = fx3L
            fP[4,1,tid] = fx4L
            fP[1,2,tid] = fy1L
            fP[2,2,tid] = fy2L
            fP[3,2,tid] = fy3L
            fP[4,2,tid] = fy4L
        else
            fP[1,1,tid] = fx1R
            fP[2,1,tid] = fx2R
            fP[3,1,tid] = fx3R
            fP[4,1,tid] = fx4R
            fP[1,2,tid] = fy1R
            fP[2,2,tid] = fy2R
            fP[3,2,tid] = fy3R
            fP[4,2,tid] = fy4R
        end
    else
        iP,kP = get_infoP(mapP,Fmask,i,k)

        fP[1,1,tid] = f_x[1,iP,kP]
        fP[2,1,tid] = f_x[2,iP,kP]
        fP[3,1,tid] = f_x[3,iP,kP]
        fP[4,1,tid] = f_x[4,iP,kP]
        fP[1,2,tid] = f_y[1,iP,kP]
        fP[2,2,tid] = f_y[2,iP,kP]
        fP[3,2,tid] = f_y[3,iP,kP]
        fP[4,2,tid] = f_y[4,iP,kP]
    end
end

@inline function get_vP(VUP,VU,t,mapP,Fmask,i,iM,xM,yM,k,tid)
    if is_bottom_wall(i,xM,yM)
        VUP[1,tid] =  VU[1,iM,k]
        VUP[2,tid] =  VU[2,iM,k]
        VUP[3,tid] = -VU[3,iM,k]
        VUP[4,tid] =  VU[4,iM,k]
    elseif is_inflow(i,xM,yM)
        VUP[1,tid] = v1L
        VUP[2,tid] = v2L
        VUP[3,tid] = v3L
        VUP[4,tid] = v4L
    elseif is_outflow(i,xM,yM)
        VUP[1,tid] = v1R
        VUP[2,tid] = v2R
        VUP[3,tid] = v3R
        VUP[4,tid] = v4R
    elseif is_topflow(i,yM)
        breakpoint = TOP_INIT+t*SHOCKSPD
        if xM < breakpoint
            VUP[1,tid] = v1L
            VUP[2,tid] = v2L
            VUP[3,tid] = v3L
            VUP[4,tid] = v4L
        else
            VUP[1,tid] = v1R
            VUP[2,tid] = v2R
            VUP[3,tid] = v3R
            VUP[4,tid] = v4R
        end
    else
        iP,kP = get_infoP(mapP,Fmask,i,k)

        VUP[1,tid] = VU[1,iP,kP]
        VUP[2,tid] = VU[2,iP,kP]
        VUP[3,tid] = VU[3,iP,kP]
        VUP[4,tid] = VU[4,iP,kP]
    end
end

@inline function get_sigmaP(sigmaP,sigma_x,sigma_y,t,i,iM,xM,yM,k,mapP,Fmask,tid)
    if is_bottom_wall(i,xM,yM)
        sigmaP[1,1,tid] =  0.0
        sigmaP[2,1,tid] = -sigma_x[2,iM,k]
        sigmaP[3,1,tid] =  sigma_x[3,iM,k]
        sigmaP[4,1,tid] = -sigma_x[4,iM,k]
        sigmaP[1,2,tid] =  0.0
        sigmaP[2,2,tid] = -sigma_y[2,iM,k]
        sigmaP[3,2,tid] =  sigma_y[3,iM,k]
        sigmaP[4,2,tid] = -sigma_y[4,iM,k]
    elseif (is_outflow(i,xM,yM) || is_inflow(i,xM,yM) || is_topflow(i,yM))
        sigmaP[1,1,tid] =  0.0#0.0
        sigmaP[2,1,tid] =  0.0#sigma_x[2,iM,k]
        sigmaP[3,1,tid] =  0.0#sigma_x[3,iM,k]
        sigmaP[4,1,tid] =  0.0#sigma_x[4,iM,k]
        sigmaP[1,2,tid] =  0.0#0.0
        sigmaP[2,2,tid] =  0.0#sigma_y[2,iM,k]
        sigmaP[3,2,tid] =  0.0#sigma_y[3,iM,k]
        sigmaP[4,2,tid] =  0.0#sigma_y[4,iM,k]
    else
        iP,kP = get_infoP(mapP,Fmask,i,k)
        sigmaP[1,1,tid] = 0.0
        sigmaP[2,1,tid] = sigma_x[2,iP,kP]
        sigmaP[3,1,tid] = sigma_x[3,iP,kP]
        sigmaP[4,1,tid] = sigma_x[4,iP,kP]
        sigmaP[1,2,tid] = 0.0
        sigmaP[2,2,tid] = sigma_y[2,iP,kP]
        sigmaP[3,2,tid] = sigma_y[3,iP,kP]
        sigmaP[4,2,tid] = sigma_y[4,iP,kP]
    end
end

@inline function get_valP(UP,fP,sigmaP,U,f_x,f_y,sigma_x,sigma_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k,tid)
    get_consP(UP,U,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k,tid)
    get_fluxP(fP,f_x,f_y,t,mapP,Fmask,i,xM,yM,k,UP,tid)
    get_sigmaP(sigmaP,sigma_x,sigma_y,t,i,iM,xM,yM,k,mapP,Fmask,tid)
end

@inline function is_face_x(i)
    return (((i > BOTTOMRIGHT) & (i <= TOPRIGHT)) | (i > TOPLEFT))
end

@inline function is_face_y(i)
    return (((i > TOPRIGHT) & (i <= TOPLEFT)) | (i <= BOTTOMRIGHT))
end

@inline function update_F_low!(F_low,k,tid,i,j,λ,S0J_ij,U,f,sigma)
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

    sigma1_j = sigma[1,j,k]
    sigma2_j = sigma[2,j,k]
    sigma3_j = sigma[3,j,k]
    sigma4_j = sigma[4,j,k]
    sigma1_i = sigma[1,i,k]
    sigma2_i = sigma[2,i,k]
    sigma3_i = sigma[3,i,k]
    sigma4_i = sigma[4,i,k]

    FL1 = (S0J_ij*(f_1_i+f_1_j-sigma1_i-sigma1_j) - λ*(rho_j-rho_i))
    FL2 = (S0J_ij*(f_2_i+f_2_j-sigma2_i-sigma2_j) - λ*(rhou_j-rhou_i))
    FL3 = (S0J_ij*(f_3_i+f_3_j-sigma3_i-sigma3_j) - λ*(rhov_j-rhov_i))
    FL4 = (S0J_ij*(f_4_i+f_4_j-sigma4_i-sigma4_j) - λ*(E_j-E_i))

    F_low[1,i,j,tid] = FL1
    F_low[2,i,j,tid] = FL2
    F_low[3,i,j,tid] = FL3
    F_low[4,i,j,tid] = FL4

    F_low[1,j,i,tid] = -FL1
    F_low[2,j,i,tid] = -FL2
    F_low[3,j,i,tid] = -FL3
    F_low[4,j,i,tid] = -FL4
end

@inline function update_F_high!(F_high,k,tid,i,j,SJ_ij_db,U,rholog,betalog,direction,sigma)
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

    sigma1_i  = sigma[1,i,k]
    sigma2_i  = sigma[2,i,k]
    sigma3_i  = sigma[3,i,k]
    sigma4_i  = sigma[4,i,k]
    sigma1_j  = sigma[1,j,k]
    sigma2_j  = sigma[2,j,k]
    sigma3_j  = sigma[3,j,k]
    sigma4_j  = sigma[4,j,k]
    
    if (direction == 0)
        # x direction
        F1,F2,F3,F4 = euler_fluxes_2D_x(rho_i,u_i,v_i,beta_i,rholog_i,betalog_i,
                                        rho_j,u_j,v_j,beta_j,rholog_j,betalog_j)
    else
        # y direction
        F1,F2,F3,F4 = euler_fluxes_2D_y(rho_i,u_i,v_i,beta_i,rholog_i,betalog_i,
                                        rho_j,u_j,v_j,beta_j,rholog_j,betalog_j)
    end
    SJ_ij = SJ_ij_db/2
    FH1 = SJ_ij_db*F1 - SJ_ij*(sigma1_i+sigma1_j)
    FH2 = SJ_ij_db*F2 - SJ_ij*(sigma2_i+sigma2_j)
    FH3 = SJ_ij_db*F3 - SJ_ij*(sigma3_i+sigma3_j)
    FH4 = SJ_ij_db*F4 - SJ_ij*(sigma4_i+sigma4_j)

    F_high[1,i,j,tid] = FH1
    F_high[2,i,j,tid] = FH2
    F_high[3,i,j,tid] = FH3
    F_high[4,i,j,tid] = FH4

    F_high[1,j,i,tid] = -FH1
    F_high[2,j,i,tid] = -FH2
    F_high[3,j,i,tid] = -FH3
    F_high[4,j,i,tid] = -FH4
end

function compute_sigma(prealloc,ops,geom)
    f_x,f_y,theta_x,theta_y,sigma_x,sigma_y,VU,rholog,betalog,U_low,U_high,F_low,F_high,F_P,L,wspd_arr,λ_arr,λf_arr,dii_arr,L_plot,Kxx,Kyy,Kxy,UP,VUP,fP,sigmaP,viscpen,rhop,epsN,blending = prealloc
    S0r_vec,S0s_vec,S0r_nnzi,S0r_nnzj,S0s_nnzi,S0s_nnzj,
    Sr_vec,Ss_vec,Sr_nnzi,Sr_nnzj,Ss_nnzi,Ss_nnzj,
    Minv,Br_halved,Bs_halved,S_nnzi,S_nnzj,VDMinv = ops
    mapP,Fmask,x,y,rxJ,syJ,J,sJ = geom

    @batch for k = 1:K
        J_k   = J[1,k]
        rxJ_k = rxJ[1,k]
        syJ_k = syJ[1,k]

        tid = Threads.threadid()
        for i = 1:Np
            for c = 1:Nc
                theta_x[c,i,k] = 0.0
                theta_y[c,i,k] = 0.0
                sigma_x[c,i,k] = 0.0
                sigma_y[c,i,k] = 0.0
            end
        end

        for c_r = 1:Sr_nnz_hv
            i = Sr_nnzi[c_r]
            j = Sr_nnzj[c_r]
            for c = 1:Nc
                theta_x[c,i,k] = theta_x[c,i,k] + rxJ_k*Sr_vec[c_r]*VU[c,j,k]
                theta_x[c,j,k] = theta_x[c,j,k] - rxJ_k*Sr_vec[c_r]*VU[c,i,k]
            end
        end

        for c_s = 1:Ss_nnz_hv
            i = Ss_nnzi[c_s]
            j = Ss_nnzj[c_s]
            for c = 1:Nc
                theta_y[c,i,k] = theta_y[c,i,k] + syJ_k*Ss_vec[c_s]*VU[c,j,k]
                theta_y[c,j,k] = theta_y[c,j,k] - syJ_k*Ss_vec[c_s]*VU[c,i,k]
            end
        end

        for i = 1:Nfp
            sJ_ik  = sJ[i,k]
            iM = Fmask[i]
            xM = x[iM,k]
            yM = y[iM,k]
            get_vP(VUP,VU,t,mapP,Fmask,i,iM,xM,yM,k,tid)

            if is_face_x(i)
                BrJ_ii_halved = sJ_ik*Br_halved[iM]
                for c = 1:Nc
                    theta_x[c,iM,k] = theta_x[c,iM,k]+ BrJ_ii_halved*(VUP[c,tid])
                end
            end                                                      
                                                                     
            if is_face_y(i)                                          
                BsJ_ii_halved = sJ_ik*Bs_halved[iM]                       
                for c = 1:Nc
                    theta_y[c,iM,k] = theta_y[c,iM,k]+ BsJ_ii_halved*(VUP[c,tid])
                end
            end

        end

        for i = 1:Np
            mJ_inv_ii = Minv[i]/J_k
            for c = 1:Nc
                theta_x[c,i,k] = mJ_inv_ii*theta_x[c,i,k]
                theta_y[c,i,k] = mJ_inv_ii*theta_y[c,i,k]
            end
        end

        for i = 1:Np
            get_Kvisc(Kxx,Kyy,Kxy,tid,VU[1,i,k],VU[2,i,k],VU[3,i,k],VU[4,i,k])
            for ci = 2:Nc
                for cj = 2:Nc
                    sigma_x[ci,i,k] = sigma_x[ci,i,k] + Kxx[tid][ci,cj]*theta_x[cj,i,k] + Kxy[tid][ci,cj]*theta_y[cj,i,k]
                    sigma_y[ci,i,k] = sigma_y[ci,i,k] + Kxy[tid][cj,ci]*theta_x[cj,i,k] + Kyy[tid][ci,cj]*theta_y[cj,i,k]
                end
            end
        end
    end
 
end

function rhs_IDP!(U,rhsU,t,dtl,prealloc,ops,geom,in_s1)
    f_x,f_y,theta_x,theta_y,sigma_x,sigma_y,VU,rholog,betalog,U_low,U_high,F_low,F_high,F_P,L,wspd_arr,λ_arr,λf_arr,dii_arr,L_plot,Kxx,Kyy,Kxy,UP,VUP,fP,sigmaP,viscpen,rhop,epsN,blending = prealloc
    S0r_vec,S0s_vec,S0r_nnzi,S0r_nnzj,S0s_nnzi,S0s_nnzj,
    Sr_vec,Ss_vec,Sr_nnzi,Sr_nnzj,Ss_nnzi,Ss_nnzj,
    Minv,Br_halved,Bs_halved,S_nnzi,S_nnzj,VDMinv = ops
    mapP,Fmask,x,y,rxJ,syJ,J,sJ = geom

    fill!(rhsU,0.0)
    @batch for k = 1:K
        for i = 1:Np
            rho  = U[1,i,k]
            rhou = U[2,i,k]
            rhov = U[3,i,k]
            E    = U[4,i,k]
            p           = pfun(rho,rhou,rhov,E)
            v1,v2,v3,v4 = entropyvar(rho,rhou,rhov,E,p)
            rhop[i,k]  = rho*p
            f_x[1,i,k] = rhou
            f_x[2,i,k] = rhou^2/rho+p
            f_x[3,i,k] = rhou*rhov/rho
            f_x[4,i,k] = E*rhou/rho+p*rhou/rho
            f_y[1,i,k] = rhov
            f_y[2,i,k] = rhou*rhov/rho
            f_y[3,i,k] = rhov^2/rho+p
            f_y[4,i,k] = E*rhov/rho+p*rhov/rho
            VU[1,i,k]   = v1
            VU[2,i,k]   = v2
            VU[3,i,k]   = v3
            VU[4,i,k]   = v4
            rholog[i,k]  = log(rho)
            betalog[i,k] = log(rho/(2*p))
        end
    end

    if (IFSHOCKCAPTURE)
        @batch for k = 1:K
            @views mul!(epsN[:,k],VDMinv,rhop[:,k])
        end

        @batch for k = 1:K
            count = 1
            modeN_energy = 0.0
            total_energy = 0.0
            for j = 0:N
                for i = 0:N
                    energy = epsN[count,k]^2
                    if ((i == N) || (j == N))
                        modeN_energy += energy
                    end
                    total_energy += energy
                    count += 1
                end
            end
            alpha = 1/(1+exp(-s_factor/TN*(modeN_energy/total_energy-TN)))
            if (alpha < alphamin)
                blending[k] = 0.0
            elseif (alpha >= alphamin) && (alpha <= 1-alphamin)
                blending[k] = alpha
            else
                blending[k] = 1.0
            end
            blending[k] = min(1.0-blending[k],alphamax)
        end
    end

    if (IFSHOCKCAPTURE)
        @batch for k = 1:K
            @views mul!(epsN[:,k],VDMinv,rhop[:,k])
        end

        @batch for k = 1:K
            count = 1
            modeN_energy = 0.0
            total_energy = 0.0
            for j = 0:N
                for i = 0:N
                    energy = epsN[count,k]^2
                    if ((i == N) || (j == N))
                        modeN_energy += energy
                    end
                    total_energy += energy
                    count += 1
                end
            end
            alpha = 1/(1+exp(-s_factor/TN*(modeN_energy/total_energy-TN)))
            if (alpha < alphamin)
                blending[k] = 0.0
            elseif (alpha >= alphamin) && (alpha <= 1-alphamin)
                blending[k] = alpha
            else
                blending[k] = 1.0
            end
            blending[k] = min(1.0-blending[k],alphamax)
        end
    end

    compute_sigma(prealloc,ops,geom)
    
    # Precompute wavespeeds
    @batch for k = 1:K
        rxJ_k = rxJ[1,k]
        syJ_k = syJ[1,k]
        tid = Threads.threadid()

        # Interior wavespd, leading 2 - x and y directions 
        for i = 1:Np
            rho_i    = U[1,i,k]
            rhou_i   = U[2,i,k]
            rhov_i   = U[3,i,k]
            E_i      = U[4,i,k]
            sigma1_2 = sigma_x[2,i,k]
            sigma1_3 = sigma_x[3,i,k]
            sigma1_4 = sigma_x[4,i,k]
            sigma2_2 = sigma_y[2,i,k]
            sigma2_3 = sigma_y[3,i,k]
            sigma2_4 = sigma_y[4,i,k]
            wspd_arr[i,1,k] = max(zhang_wavespd(rho_i,rhou_i,rhov_i,E_i,
                                                sigma1_2,sigma1_3,sigma1_4,
                                                sigma2_2,sigma2_3,sigma2_4,
                                                1,0),
                                  wavespeed_1D(rho_i,rhou_i,E_i))#wavespeed_1D(rho_i,rhou_i,E_i)
            wspd_arr[i,2,k] = max(zhang_wavespd(rho_i,rhou_i,rhov_i,E_i,
                                                sigma1_2,sigma1_3,sigma1_4,
                                                sigma2_2,sigma2_3,sigma2_4,
                                                0,1),
                                  wavespeed_1D(rho_i,rhov_i,E_i))#wavespeed_1D(rho_i,rhov_i,E_i)
        end

        # Interior dissipation coeff
        for c_r = 1:S0r_nnz_hv
            i = S0r_nnzi[c_r]
            j = S0r_nnzj[c_r]
            λ = abs(rxJ_k*S0r_vec[c_r])*max(wspd_arr[i,1,k],wspd_arr[j,1,k])
            λ_arr[c_r,1,k] = λ
            if in_s1
                dii_arr[i,k] = dii_arr[i,k] + λ
            end
        end

        for c_s = 1:S0s_nnz_hv
            i = S0s_nnzi[c_s]
            j = S0s_nnzj[c_s]
            λ = abs(syJ_k*S0s_vec[c_s])*max(wspd_arr[i,2,k],wspd_arr[j,2,k])
            λ_arr[c_s,2,k] = λ
            if in_s1
                dii_arr[i,k] = dii_arr[i,k] + λ
            end
        end
    end

    # Interface dissipation coeff 
    @batch for k = 1:K
        for i = 1:Nfp
            sJ_ik  = sJ[i,k]
            iM = Fmask[i]
            BrJ_ii_halved_abs = abs(sJ_ik*Br_halved[iM])
            BsJ_ii_halved_abs = abs(sJ_ik*Bs_halved[iM])
            xM    = x[iM,k]
            yM    = y[iM,k]

            iP,kP = get_infoP(mapP,Fmask,i,k)
            
            if is_face_x(i)
                if (is_boundary(i,xM,yM))
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
                if (is_boundary(i,xM,yM))
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
        for k = 1:K
        J_k   = J[1,k]
            for i = 1:Np
                dtl = min(dtl,1.0/Minv[i]/2.0/dii_arr[i,k]*J_k)
            end
        end
        dtl = min(CFL*dtl,T-t)
    end

    # =====================
    # Loop through elements
    # =====================
    @batch for k = 1:K
        J_k   = J[1,k]
        rxJ_k = rxJ[1,k]
        syJ_k = syJ[1,k]
        tid = Threads.threadid()
        for i = 1:Np
            for c = 1:Nc
                U_low[c,i,tid]  = 0.0
                U_high[c,i,tid] = 0.0
            end
        end

        # Calculate low order algebraic flux
        for c_r = 1:S0r_nnz_hv
            i = S0r_nnzi[c_r]
            j = S0r_nnzj[c_r]
            λ = λ_arr[c_r,1,k]
            S0xJ_ij = rxJ_k*S0r_vec[c_r]
            update_F_low!(F_low,k,tid,i,j,λ,S0xJ_ij,U,f_x,sigma_x)
        end

        for c_s = 1:S0s_nnz_hv
            i = S0s_nnzi[c_s]
            j = S0s_nnzj[c_s]
            λ = λ_arr[c_s,2,k]
            S0yJ_ij = syJ_k*S0s_vec[c_s]
            update_F_low!(F_low,k,tid,i,j,λ,S0yJ_ij,U,f_y,sigma_y)
        end

        # Calculate high order algebraic flux
        for c_r = 1:Sr_nnz_hv
            i         = Sr_nnzi[c_r]
            j         = Sr_nnzj[c_r]
            SxJ_ij_db = 2*rxJ_k*Sr_vec[c_r]
            update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,U,rholog,betalog,0,sigma_x)
        end

        for c_r = 1:Ss_nnz_hv
            i         = Ss_nnzi[c_r]
            j         = Ss_nnzj[c_r]
            SyJ_ij_db = 2*syJ_k*Ss_vec[c_r]
            update_F_high!(F_high,k,tid,i,j,SyJ_ij_db,U,rholog,betalog,1,sigma_y)
        end

        # Calculate interface fluxes
        for i = 1:Nfp
            sJ_ik  = sJ[i,k]
            iM    = Fmask[i]
            BrJ_ii_halved = sJ_ik*Br_halved[iM]
            BsJ_ii_halved = sJ_ik*Bs_halved[iM]
            xM    = x[iM,k]
            yM    = y[iM,k]
            rhoM  = U[1,iM,k]
            rhouM = U[2,iM,k]
            rhovM = U[3,iM,k]
            EM    = U[4,iM,k]
            v2M   = VU[2,iM,k]
            v3M   = VU[3,iM,k]
            v4M   = VU[4,iM,k]
            uM    = rhouM/rhoM
            vM    = rhovM/rhoM
            pM    = pfun(rhoM,rhouM,rhovM,EM)

            get_valP(UP,fP,sigmaP,U,f_x,f_y,sigma_x,sigma_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k,tid)
            get_vP(VUP,VU,t,mapP,Fmask,i,iM,xM,yM,k,tid)
            λ = λf_arr[i,k]

            alpha = 0.0
            if (VISCPENTYPE == -1)
                alpha = -1/Re/VU[4,iM,k]
            elseif ((VISCPENTYPE >= 0.0) && (VISCPENTYPE <= 1.0))
                alpha = VISCPENTYPE
            end
            for c = 2:Nc
                viscpen[c,tid] = alpha*(VUP[c,tid]-VU[c,iM,k])
            end
            if (is_bottom_wall(i,xM,yM))
                viscpen[2,tid] = -alpha*(VUP[2,tid]-VU[2,iM,k])
                viscpen[3,tid] = -alpha*(VUP[3,tid]-VU[3,iM,k])
                viscpen[4,tid] =  alpha*((VUP[2,tid]+v2M)*(VUP[2,tid]-v2M) 
                                       + (VUP[3,tid]+v3M)*(VUP[3,tid]-v3M) 
                                       + (VUP[4,tid]-v4M)*(VUP[4,tid]-v4M))/2/v4M
            end

            # flux in x direction
            if is_face_x(i)
                for c = 1:Nc
                    F_P[c,i,tid] = (BrJ_ii_halved*(f_x[c,iM,k]+fP[c,1,tid]-sigma_x[c,iM,k]-sigmaP[c,1,tid])
                                   -λ*(UP[c,tid]-U[c,iM,k]) - 2*abs(BrJ_ii_halved)*viscpen[c,tid]) 
                end
            end

            # flux in y direction
            if is_face_y(i)
                for c = 1:Nc
                    F_P[c,i,tid] = (BsJ_ii_halved*(f_y[c,iM,k]+fP[c,2,tid]-sigma_y[c,iM,k]-sigmaP[c,2,tid])
                                   -λ*(UP[c,tid]-U[c,iM,k]) - 2*abs(BsJ_ii_halved)*viscpen[c,tid]) 
                end
            end

            if (BCFLUXTYPE >= 1)
                if is_bottom_wall(i,xM,yM)
                    nx = 0.0
                    ny = -1.0
                    f1s,f2s,f3s,f4s = noslip_flux(rhoM,uM,vM,pM,nx,ny)
                    wfi = 0.0
                    if is_face_x(i)
                        wfi = abs(BrJ_ii_halved)*2
                    end
                    if is_face_y(i)
                        wfi = abs(BsJ_ii_halved)*2
                    end
                    F_P[1,i,tid] = wfi*f1s
                    F_P[2,i,tid] = wfi*f2s
                    F_P[3,i,tid] = wfi*f3s
                    F_P[4,i,tid] = wfi*f4s

                    # flux in x direction
                    if is_face_x(i)
                        for c = 1:Nc
                            F_P[c,i,tid] -= BrJ_ii_halved*(sigma_x[c,iM,k]+sigmaP[c,1,tid]) + 2*abs(BrJ_ii_halved)*viscpen[c,tid]
                        end
                    end

                    # flux in y direction
                    if is_face_y(i)
                        for c = 1:Nc
                            F_P[c,i,tid] -= BsJ_ii_halved*(sigma_y[c,iM,k]+sigmaP[c,2,tid]) + 2*abs(BsJ_ii_halved)*viscpen[c,tid]
                        end
                    end
                end
            end
        end

        # Calculate low order solution
        for j = 1:Np
            for i = 1:Np
                for c = 1:Nc
                    U_low[c,i,tid]  = U_low[c,i,tid]  + F_low[c,i,j,tid]
                    U_high[c,i,tid] = U_high[c,i,tid] + F_high[c,i,j,tid]
                end
            end
        end

        for i = 1:Nfp
            iM = Fmask[i]
            for c = 1:Nc
                U_low[c,iM,tid]  = U_low[c,iM,tid]  + F_P[c,i,tid]
                U_high[c,iM,tid] = U_high[c,iM,tid] + F_P[c,i,tid]
            end
        end

        for i = 1:Np
            for c = 1:Nc
                U_low[c,i,tid]  = U[c,i,k] - dtl*Minv[i]*U_low[c,i,tid]/J_k
                U_high[c,i,tid] = U[c,i,k] - dtl*Minv[i]*U_high[c,i,tid]/J_k
            end
        end

        is_H_positive = true
        for i = 1:Np
            rhoH_i  = U_high[1,i,tid]
            rhouH_i = U_high[2,i,tid]
            rhovH_i = U_high[3,i,tid]
            EH_i    = U_high[4,i,tid]
            pH_i    = pfun(rhoH_i,rhouH_i,rhovH_i,EH_i)
            if pH_i < POSTOL || rhoH_i < POSTOL
                is_H_positive = false
            end
        end

        if POSDETECT == 0
            is_H_positive = false
        end

        if (LIMITOPT == 1)
            # Calculate limiting parameters
            if !is_H_positive
                for ni = 1:S_nnz_hv
                    i = S_nnzi[ni]
                    j = S_nnzj[ni]
                    coeff_i = dtl*2*N*Minv[i]/J_k
                    coeff_j = dtl*2*N*Minv[j]/J_k
                    rhoi  = U_low[1,i,tid]
                    rhoui = U_low[2,i,tid]
                    rhovi = U_low[3,i,tid]
                    Ei    = U_low[4,i,tid]
                    rhoj  = U_low[1,j,tid]
                    rhouj = U_low[2,j,tid]
                    rhovj = U_low[3,j,tid]
                    Ej    = U_low[4,j,tid]
                    rhoP  = (F_low[1,i,j,tid]-F_high[1,i,j,tid])
                    rhouP = (F_low[2,i,j,tid]-F_high[2,i,j,tid])
                    rhovP = (F_low[3,i,j,tid]-F_high[3,i,j,tid])
                    EP    = (F_low[4,i,j,tid]-F_high[4,i,j,tid])
                    rhoP_i  = coeff_i*rhoP  
                    rhouP_i = coeff_i*rhouP 
                    rhovP_i = coeff_i*rhovP 
                    EP_i    = coeff_i*EP    
                    rhoP_j  = -coeff_j*rhoP  
                    rhouP_j = -coeff_j*rhouP 
                    rhovP_j = -coeff_j*rhovP 
                    EP_j    = -coeff_j*EP
                    Lrho      = POSTOL
                    Lrhoe     = POSTOL
                    L[ni,tid] = min(limiting_param(rhoi,rhoui,rhovi,Ei,rhoP_i,rhouP_i,rhovP_i,EP_i,Lrho,Lrhoe),
                                    limiting_param(rhoj,rhouj,rhovj,Ej,rhoP_j,rhouP_j,rhovP_j,EP_j,Lrho,Lrhoe))
                end
            end
        elseif (LIMITOPT == 2)
            # li limiting
            li_min = 1.0 # limiting parameter for element
            for i = 1:Np
                rhoi  = U_low[1,i,tid]
                rhoui = U_low[2,i,tid]
                rhovi = U_low[3,i,tid]
                Ei    = U_low[4,i,tid]
                rhoP_i  = 0.0
                rhouP_i = 0.0
                rhovP_i = 0.0
                EP_i    = 0.0
                coeff   = dtl*Minv[i]/J_k
                for j = 1:Np
                    rhoP   = (F_low[1,i,j,tid]-F_high[1,i,j,tid])
                    rhouP  = (F_low[2,i,j,tid]-F_high[2,i,j,tid])
                    rhovP  = (F_low[3,i,j,tid]-F_high[3,i,j,tid])
                    EP     = (F_low[4,i,j,tid]-F_high[4,i,j,tid])
                    rhoP_i  = rhoP_i  + coeff*rhoP 
                    rhouP_i = rhouP_i + coeff*rhouP 
                    rhovP_i = rhovP_i + coeff*rhovP 
                    EP_i    = EP_i    + coeff*EP 
                end
                if (LBOUNDTYPE == 0)
                    Lrho  = POSTOL
                    Lrhoe = POSTOL
                elseif (LBOUNDTYPE > 0)
                    Lrho  = LBOUNDTYPE*rhoi
                    Lrhoe = LBOUNDTYPE*pfun(rhoi,rhoui,rhovi,Ei)/(γ-1)
                end
                l = limiting_param(rhoi,rhoui,rhovi,Ei,rhoP_i,rhouP_i,rhovP_i,EP_i,Lrho,Lrhoe)
                li_min = min(li_min,l)
            end

            for ni = 1:S_nnz_hv
                L[ni,tid] = li_min
            end
        end

        # Elementwise limiting
        l_e = 1.0
        for i = 1:S_nnz_hv
            l_e = min(l_e,L[i,tid])
        end
        if (IFSHOCKCAPTURE)
            l_e = min(l_e,blending[k])
        end
        # TODO: hardcoded
        blending[k] = l_e
        l_em1 = l_e-1.0
        
        if is_H_positive
            l_e = 1.0
            l_em1 = 0.0
        end

        if in_s1
            L_plot[k] = l_e
        end

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
        J_k   = J[1,k]
        for i = 1:Np
            for c = 1:Nc
                rhsU[c,i,k] = Minv[i]*rhsU[c,i,k]/J_k
            end
        end
    end

    # @show sum(L_plot)/K

    return dtl
end

@unpack Vf,Dr,Ds,LIFT = rd
md = init_mesh((VX,VY),EToV,rd)
@unpack xf,yf,mapM,mapP,mapB,nxJ,nyJ,x,y = md
xb,yb = (x->x[mapB]).((xf,yf))

const K  = size(x,2)
const Nfaces = 4

# Make domain periodic
@unpack Vf = rd
@unpack xf,yf,mapM,mapP,mapB,J,rxJ,syJ,sJ = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

const Np = (N+1)*(N+1)
const Nfp = Nfaces*(N+1)

# Convert diagonal matrices to vectors
Br   = Array(diag(Br))
Bs   = Array(diag(Bs))
M    = Array(diag(M))
Minv = Array(diag(Minv))

Br_halved = -sum(S0r,dims=2)
Bs_halved = -sum(S0s,dims=2)

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


# Initial condition 2D shocktube
const TOP_INIT = (1+sqrt(3)/6)/sqrt(3)
at_left(x,y) = y-sqrt(3)*x+sqrt(3)/6 > 0.0
function initial_cond(xi,yi,t)

    if at_left(xi,yi)
        rho0  = rhoL
        rhou0 = rhouL
        rhov0 = rhovL
        E0    = EL
    else
        rho0  = rhoR
        rhou0 = rhouR
        rhov0 = rhovR
        E0    = ER
    end

    return (rho0, rhou0, rhov0, E0)
end



U = zeros(Nc,Np,K)
for k = 1:K
    for i = 1:Np
        rho0,rhou0,rhov0,E0 = initial_cond(x[i,k],y[i,k],0.0)
        U[1,i,k] = rho0
        U[2,i,k] = rhou0
        U[3,i,k] = rhov0
        U[4,i,k] = E0 
    end
end

# Preallocation
rhsU   = zeros(Float64,size(U))
f_x    = zeros(Float64,size(U))
f_y    = zeros(Float64,size(U))
rhop    = zeros(Float64,Np,K)
epsN    = zeros(Float64,Np,K)
sigma_x = zeros(Float64,size(U))
sigma_y = zeros(Float64,size(U))
theta_x = zeros(Float64,size(U))
theta_y = zeros(Float64,size(U))
VU      = zeros(Float64,size(U))
rholog  = zeros(Float64,Np,K)
betalog = zeros(Float64,Np,K)
U_low   = zeros(Float64,Nc,Np,NUM_THREADS)
U_high  = zeros(Float64,Nc,Np,NUM_THREADS)
F_low   = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_high  = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_P     = zeros(Float64,Nc,Nfp,NUM_THREADS)
UP      = zeros(Float64,Nc,NUM_THREADS)
VUP     = zeros(Float64,Nc,NUM_THREADS)
fP      = zeros(Float64,Nc,2,NUM_THREADS)
sigmaP  = zeros(Float64,Nc,2,NUM_THREADS)
L       =  ones(Float64,S_nnz_hv,NUM_THREADS)
wspd_arr = zeros(Float64,Np,2,K)
λ_arr    = zeros(Float64,S0r_nnz_hv,2,K) # Assume S0r and S0s has same number of nonzero entries
λf_arr   = zeros(Float64,Nfp,K)
dii_arr  = zeros(Float64,Np,K)
L_plot   = zeros(Float64,K)
Kxx      = [zeros(MMatrix{Nc,Nc,Float64}) for _ in 1:NUM_THREADS]
Kyy      = [zeros(MMatrix{Nc,Nc,Float64}) for _ in 1:NUM_THREADS]
Kxy      = [zeros(MMatrix{Nc,Nc,Float64}) for _ in 1:NUM_THREADS]
viscpen  = zeros(Float64,Nc,NUM_THREADS)
blending = ones(Float64,K)

prealloc = (f_x,f_y,theta_x,theta_y,sigma_x,sigma_y,VU,rholog,betalog,U_low,U_high,F_low,F_high,F_P,L,wspd_arr,λ_arr,λf_arr,dii_arr,L_plot,Kxx,Kyy,Kxy,UP,VUP,fP,sigmaP,viscpen,rhop,epsN,blending)
ops      = (S0r_vec,S0s_vec,S0r_nnzi,S0r_nnzj,S0s_nnzi,S0s_nnzj,
            Sr_vec,Ss_vec,Sr_nnzi,Sr_nnzj,Ss_nnzi,Ss_nnzj,
            Minv,Br_halved,Bs_halved,S_nnzi,S_nnzj,VDMinv)
geom     = (mapP,Fmask,x,y,rxJ,syJ,J,sJ)


# Time stepping
"Time integration"
t = 0.0
U = collect(U)
resW = zeros(size(U))

#plotting nodes
@unpack VDM = rd
rp,sp = equi_nodes_2D(NUMPLOTPT)
Vp = vandermonde_2D(N,rp,sp)/VDM
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)
anim = Animation()

function construct_vtk_mesh!(x_vtk,y_vtk)
    # TODO: assume sturctured Mesh
    for k = 1:K
        iK = mod1(k,Int(XLENGTH*K1D))
        jK = div(k-1,Int(XLENGTH*K1D))+1
        xk = reshape(x[:,k],N+1,N+1)
        yk = reshape(y[:,k],N+1,N+1)
        x_vtk[(iK-1)*(N+1)+1:iK*(N+1),(jK-1)*(N+1)+1:jK*(N+1)] = xk
        y_vtk[(iK-1)*(N+1)+1:iK*(N+1),(jK-1)*(N+1)+1:jK*(N+1)] = yk
    end
end

function construct_vtk_file!(U,blending,Uvtk,blending_vtk,pvdfile,xvtk,yvtk,t)
    vtk_grid("$(OUTPUTPATH)/dg2D_euler_quad_dmr_LBOUNDTYPE=$(LBOUNDTYPE)_t=$t",xvtk,yvtk) do vtk
        for k = 1:K
            iK = mod1(k,Int(XLENGTH*K1D))
            jK = div(k-1,Int(XLENGTH*K1D))+1
            for c = 1:Nc
                Uvtk[c,(iK-1)*(N+1)+1:iK*(N+1),(jK-1)*(N+1)+1:jK*(N+1)] = U[c,:,k]
            end
            blending_vtk[(iK-1)*(N+1)+1:iK*(N+1),(jK-1)*(N+1)+1:jK*(N+1)] .= blending[k]
        end    
        vtk["rho"]  = Uvtk[1,:,:] 
        vtk["rhou"] = Uvtk[2,:,:] 
        vtk["rhov"] = Uvtk[3,:,:] 
        vtk["E"]    = Uvtk[4,:,:] 
        vtk["blending"] = blending_vtk

        pvdfile[t]  = vtk
    end
end

x_vtk    = zeros(Float64,Int(XLENGTH*(N+1)*K1D),(N+1)*K1D)    # TODO: hardcoded domain size
y_vtk    = zeros(Float64,Int(XLENGTH*(N+1)*K1D),(N+1)*K1D)
U_vtk    = zeros(Float64,Nc,Int(XLENGTH*(N+1)*K1D),(N+1)*K1D)
blending_vtk = zeros(Float64,Int(XLENGTH*(N+1)*K1D),(N+1)*K1D)

construct_vtk_mesh!(x_vtk,y_vtk)
pvd = paraview_collection("$(OUTPUTPATH)/dg2D_CNS_quad_dmr_LBOUNDTYPE=$(LBOUNDTYPE).pvd")

xp   = Vp*x
yp   = Vp*y

dt_hist = []
i = 1
Uhist = []
blendinghist = []
thist = []

mapN = collect(reshape(1:Np*K,Np,K))
inflow_nodal = mapN[findall(@. (abs(x) < TOL) | ((x < WALLPT) & (abs(y) < TOL)))]
outflow_nodal = mapN[findall(@. abs(x-XLENGTH) < TOL)]
topflow_nodal = mapN[findall(@. abs(y-1.) < TOL)]
wall_nodal = mapN[findall(@. ((x >= WALLPT) & (abs(y) < TOL)))]

@inline function enforce_BC_timestep!(U,inflow_nodal,topflow_nodal,wall_nodal,t)
    for i = inflow_nodal
        k = fld1(i,Np)
        j = mod1(i,Np)
        U[1,j,k] = rhoL
        U[2,j,k] = rhouL
        U[3,j,k] = rhovL
        U[4,j,k] = EL
    end
    for i = outflow_nodal
        k = fld1(i,Np)
        j = mod1(i,Np)
        U[1,j,k] = rhoR
        U[2,j,k] = rhouR
        U[3,j,k] = rhovR
        U[4,j,k] = ER
    end
    breakpoint = TOP_INIT+t*SHOCKSPD
    for i = topflow_nodal
        k = fld1(i,Np)
        j = mod1(i,Np)
        if x[i] < breakpoint
            U[1,j,k] = rhoL
            U[2,j,k] = rhouL
            U[3,j,k] = rhovL
            U[4,j,k] = EL
        else
            U[1,j,k] = rhoR
            U[2,j,k] = rhouR
            U[3,j,k] = rhovR
            U[4,j,k] = ER
        end
    end
end



# for i = 1:5
#     dt = dt0
#     @btime rhs_IDP!($U,$rhsU,$t,$dt,$prealloc,$ops,$geom,$true);
# end

# try

const dx  = min(minimum([x[end,j]-x[1,j] for j in 1:K]),minimum([y[end,j]-y[1,j] for j in 1:K]))
const dt0 = CFL*Re*dx^2
@time while t < T
#while i < 2
    # SSPRK(3,3)
    fill!(dii_arr,0.0)
    # dt = min(dt0,T-t)
    dt = dt0
    dt = rhs_IDP!(U,rhsU,t,dt,prealloc,ops,geom,true);
    @. resW = U + dt*rhsU
    rhs_IDP!(resW,rhsU,t+dt,dt,prealloc,ops,geom,false);
    @. resW = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resW
    rhs_IDP!(resW,rhsU,t+dt/2,dt,prealloc,ops,geom,false);
    @. resW = resW+dt*rhsU
    @. U = 1/3*U+2/3*resW
    enforce_BC_timestep!(U,inflow_nodal,topflow_nodal,wall_nodal,t+dt);

    push!(dt_hist,dt)
    push!(thist,t)
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T, at step $i")
    flush(stdout)
    global i = i + 1
    if (mod(i,SAVEINT) == 1)
        # TODO: 
        for k = 1:K 
            for i = 1:Np
                if x[i,k] > 3.3
                    U[1,i,k] = rhoR
                    U[2,i,k] = rhouR
                    U[3,i,k] = rhovR
                    U[4,i,k] = ER
                end
            end
        end
        push!(Uhist,copy(U))
        push!(blendinghist,copy(blending))
        construct_vtk_file!(U,blending,U_vtk,blending_vtk,pvd,x_vtk,y_vtk,t)
    end
end
# catch err
# end

# gif(anim,"fig/dg2D_CNS_quad_dmr/N=$N,K1D=$K1D,T=$T,LIMITOPT=$LIMITOPT,POSDETECT=$POSDETECT,LBOUNDTYPE=$LBOUNDTYPE,BCFLUXTYPE=$BCFLUXTYPE,VISCPENTYPE=$VISCPENTYPE,Re=$Re.gif")

push!(Uhist,copy(U))
push!(blendinghist,copy(blending))
construct_vtk_file!(U,blending,U_vtk,blending_vtk,pvd,x_vtk,y_vtk,t)

vtk_save(pvd)

df = DataFrame(N=Int64[],K=Int64[],T=Float64[],t=Float64[],
               CFL=Float64[],dt0=Float64[],
               γ=Float64[],
               LIMITOPT=Int64[],POSDETECT=Int64[],LBOUNDTYPE=Float64[],BCFLUXTYPE=Int64[],IFSHOCKCAPTURE=Bool[],
               POSTOL=Float64[],
               Uhist=Array{Any,1}[],blendinghist=Array{Any,1}[],
               thist=Array{Any,1}[],dt_hist=Array{Any,1}[])
# df = load("$(OUTPUTPATH)/dg2D_euler_quad_dmr.jld2","data")
push!(df,(N,K,T,t,CFL,dt0,γ,LIMITOPT,POSDETECT,LBOUNDTYPE,BCFLUXTYPE,IFSHOCKCAPTURE,POSTOL,Uhist,blendinghist,thist,dt_hist))
save("$(OUTPUTPATH)/dg2D_euler_quad_dmr.jld2","data",df)

end #muladd


