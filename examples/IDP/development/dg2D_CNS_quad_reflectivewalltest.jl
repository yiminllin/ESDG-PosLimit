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

push!(LOAD_PATH, "../examples/EntropyStableEuler.jl/src")
include("../../EntropyStableEuler.jl/src/logmean.jl")

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

    p = pfun(rhoL+l*rhoP,rhouL+l*rhouP,rhovL+l*rhovP,EL+l*EP)
    if p > TOL
        return l
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

# @inline function zhang_wavespd(rho,rhou,rhov,E,sigma1_2,sigma1_3,sigma1_4,sigma2_2,sigma2_3,sigma2_4,n_1,n_2)
#     u     = rhou/rho
#     v     = rhov/rho
#     p     = pfun(rho,rhou,rhov,E)
#     rhoe  = E-.5*rho*(u^2+v^2)
#     qn    = n_1*(u*sigma1_2+v*sigma1_3-sigma1_4) + n_2*(u*sigma2_2+v*sigma2_3-sigma2_4)
#     ntau1 = n_1*sigma1_2 + n_2*sigma1_3
#     ntau2 = n_1*sigma2_2 + n_2*sigma2_3
#     nm    = sqrt((ntau1-p*n_1)^2+(ntau2-p*n_2)^2)
#     rho2e_db = 2*rho*rhoe

#     return POSTOL+abs(u*n_1+v*n_2)+1.0/rho2e_db*(sqrt((rho*qn)^2+rho2e_db*nm) + rho*abs(qn))
# end

# TODO: previous zhang wavespd implementation
@inline function zhang_wavespd(rho,rhou,rhov,E,sigmax2,sigmax3,sigmax4,sigmay2,sigmay3,sigmay4,nx,ny)
    p = pfun(rho,rhou,rhov,E)
    u = rhou/rho
    v = rhov/rho
    e = (E-.5*rho*(u^2+v^2))/rho
    tau_xx = sigmax2
    tau_yx = sigmax3
    tau_xy = sigmay2
    tau_yy = sigmay3
    q_x = u*tau_xx+v*tau_yx-sigmax4
    q_y = u*tau_xy+v*tau_yy-sigmay4

    v_vec = u*nx+v*ny
    q_vec = q_x*nx+q_y*ny
    tau_vec_x = nx*tau_xx+ny*tau_yx
    tau_vec_y = nx*tau_xy+ny*tau_yy

    return abs(v_vec)+1/(2*rho^2*e)*(sqrt(rho^2*q_vec^2+2*rho^2*e*((tau_vec_x-p*nx)^2+(tau_vec_y-p*ny)^2))+rho*abs(q_vec))+POSTOL
end

# TODO: previous implementation of Kvisc 
function viscous_matrices!(Kxx,Kxy,Kyy,v)
    λ=-lambda
    μ=mu
    Pr=Pr
    v1,v2,v3,v4 = v
    inv_v4_cubed = @. 1/(v4^3)
    λ2μ = (λ+2.0*μ)
    Kxx[2,2] = inv_v4_cubed*-λ2μ*v4^2
    Kxx[2,4] = inv_v4_cubed*λ2μ*v2*v4
    Kxx[3,3] = inv_v4_cubed*-μ*v4^2
    Kxx[3,4] = inv_v4_cubed*μ*v3*v4
    Kxx[4,2] = inv_v4_cubed*λ2μ*v2*v4
    Kxx[4,3] = inv_v4_cubed*μ*v3*v4
    Kxx[4,4] = inv_v4_cubed*-(λ2μ*v2^2 + μ*v3^2 - γ*μ*v4/Pr)

    Kxy[2,3] = inv_v4_cubed*-λ*v4^2
    Kxy[2,4] = inv_v4_cubed*λ*v3*v4
    Kxy[3,2] = inv_v4_cubed*-μ*v4^2
    Kxy[3,4] = inv_v4_cubed*μ*v2*v4
    Kxy[4,2] = inv_v4_cubed*μ*v3*v4
    Kxy[4,3] = inv_v4_cubed*λ*v2*v4
    Kxy[4,4] = inv_v4_cubed*(λ+μ)*(-v2*v3)

    Kyy[2,2] = inv_v4_cubed*-μ*v4^2
    Kyy[2,4] = inv_v4_cubed*μ*v2*v4
    Kyy[3,3] = inv_v4_cubed*-λ2μ*v4^2
    Kyy[3,4] = inv_v4_cubed*λ2μ*v3*v4
    Kyy[4,2] = inv_v4_cubed*μ*v2*v4
    Kyy[4,3] = inv_v4_cubed*λ2μ*v3*v4
    Kyy[4,4] = inv_v4_cubed*-(λ2μ*v3^2 + μ*v2^2 - γ*μ*v4/Pr)
end


@inline function get_Kvisc(v1,v2,v3,v4)
    lam = -lambda
    μ   = mu
    v2_sq = v2^2
    v3_sq = v3^2
    v4_sq = v4^2
    v43_inv = 1/(v4_sq*v4)
    λ2μ = (lam+2.0*μ)

    v2v3  = v2*v3
    v2v4  = v2*v4
    v3v4  = v3*v4
    μv2v4 = μ*v2v4
    μv3v4 = μ*v3v4
    μv4_sq = μ*v4_sq
    gmv4Pr = γ*μ*v4/Pr
    λ2μv4_sq = λ2μ*v4_sq

    K11_22 = v43_inv*-λ2μv4_sq
    K11_24 = v43_inv*λ2μ*v2v4
    K11_33 = v43_inv*-μv4_sq
    K11_34 = v43_inv*μv3v4
    K11_44 = v43_inv*-(λ2μ*v2_sq + μ*v3_sq - gmv4Pr)
    
    K12_23 = v43_inv*-lam*v4_sq
    K12_24 = v43_inv*lam*v3v4
    K12_32 = v43_inv*-μv4_sq
    K12_34 = v43_inv*μv2v4
    K12_42 = v43_inv*μv3v4
    K12_43 = v43_inv*lam*v2v4
    K12_44 = v43_inv*(lam+μ)*(-v2v3)
    
    K22_22 = v43_inv*-μv4_sq
    K22_24 = v43_inv*μv2v4
    K22_33 = v43_inv*-λ2μv4_sq
    K22_34 = v43_inv*λ2μ*v3v4
    K22_44 = v43_inv*-(λ2μ*v3_sq + μ*v2_sq - gmv4Pr)

    return K11_22,K11_24,K11_33,K11_34,K11_44,
           K12_23,K12_24,K12_32,K12_34,K12_42,K12_43,K12_44,
           K22_22,K22_24,K22_33,K22_34,K22_44
end

const LIMITOPT = 1 # 1 if elementwise limiting lij, 2 if elementwise limiting li, 3 if nodewise
const TOL = 5e-16
const POSTOL = 1e-10
const WALLPT = 1.0/6.0
const Nc = 4 # number of components
"Approximation parameters"
const N = 3
const K1D = 20
const T = 0.02
# const dt0 = 1e-5
const Re = 500
const dt0 = min(.5/N/(N-1)/K1D,0.001*Re/K1D^2)#0.2
# dt0 = 1e+1
const XLENGTH = 1.0
const CFL = 0.75#0.95

const γ = 1.4

# Viscous parameters
const mu = 1/Re
const lambda = 2/3*mu
const Pr = 3/4
const cp = γ/(γ-1)
const cv = 1/(γ-1)
const kappa = mu*cp/Pr

const NUM_THREADS = Threads.nthreads()
const BOTTOMRIGHT = N+1
const TOPRIGHT    = 2*(N+1)
const TOPLEFT     = 3*(N+1)

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

@unpack Vf = rd
md = init_mesh((VX,VY),EToV,rd)
@unpack xf,yf,mapM,mapP,mapB,nxJ,nyJ,x,y = md
const K  = size(x,2)
const Nfaces = 4
# Make domain periodic
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

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

@inline function check_BC(xM,i)
    leftwall = ((abs(xM) < TOL) & (i <= TOPRIGHT))
    rightwall = ((abs(xM-1.0) < TOL) & (i >= BOTTOMRIGHT) & (i < TOPRIGHT))
    has_bc  = (leftwall | rightwall)
    return leftwall,rightwall,has_bc
end

@inline function get_vP(VU,mapP,Fmask,i,iM,k,leftwall,rightwall)
    if leftwall
        # We assume the normals are [0;-1] here
        v1P = VU[1,iM,k]
        v2P = -VU[2,iM,k]
        v3P = VU[3,iM,k]
        v4P = VU[4,iM,k]
    elseif rightwall
        # We assume the normals are [0;-1] here
        v1P = VU[1,iM,k]
        v2P = -VU[2,iM,k]
        v3P = VU[3,iM,k]
        v4P = VU[4,iM,k]
    else                         # if not on the physical boundary
        iP,kP = get_infoP(mapP,Fmask,i,k)

        v1P = VU[1,iP,kP]
        v2P = VU[2,iP,kP]
        v3P = VU[3,iP,kP]
        v4P = VU[4,iP,kP]
    end

    return v1P,v2P,v3P,v4P
end

@inline function get_consP(U,mapP,Fmask,i,iM,uM,vM,k,leftwall,rightwall)
    if leftwall
        # We assume the normals are [0;-1] here
        # Un = -u2 = -vM
        # Ut = -u1 = -uM
        # uP = -Ut = uM
        # vP = -(-Un) = -vM
        rhoP   = U[1,iM,k]
        uP     = -uM
        vP     = vM
        rhouP  = rhoP*uP
        rhovP  = rhoP*vP
        EP     = U[4,iM,k]
    elseif rightwall
        rhoP   = U[1,iM,k]
        uP     = -uM
        vP     = vM
        rhouP  = rhoP*uP
        rhovP  = rhoP*vP
        EP     = U[4,iM,k]
    else                         # if not on the physical boundary
        iP,kP = get_infoP(mapP,Fmask,i,k)

        rhoP  = U[1,iP,kP]
        rhouP = U[2,iP,kP]
        rhovP = U[3,iP,kP]
        EP    = U[4,iP,kP]
    end

    return rhoP,rhouP,rhovP,EP
end

@inline function get_fluxP(f_x,f_y,mapP,Fmask,i,k,leftwall,rightwall,rhoP,rhouP,rhovP,EP)
    if leftwall
        # We assume the normals are [0;-1] here
        # Un = -u2 = -vM
        # Ut = -u1 = -uM
        # uP = -Ut = uM
        # vP = -(-Un) = -vM
        uP = rhouP/rhoP
        vP = rhovP/rhoP
        pP = pfun(rhoP,rhouP,rhovP,EP)
        fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P = inviscid_flux_prim(rhoP,uP,vP,pP)
    elseif rightwall
        uP = rhouP/rhoP
        vP = rhovP/rhoP
        pP = pfun(rhoP,rhouP,rhovP,EP)
        fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P = inviscid_flux_prim(rhoP,uP,vP,pP)
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

@inline function get_sigmaP(sigma_x,sigma_y,mapP,Fmask,i,iM,k,leftwall,rightwall)
    if leftwall
        sigmax_1_P =  0.0
        sigmax_2_P =  sigma_x[2,iM,k]
        sigmax_3_P = -sigma_x[3,iM,k]
        sigmax_4_P = -sigma_x[4,iM,k]
        sigmay_1_P =  0.0
        sigmay_2_P =  sigma_y[2,iM,k]
        sigmay_3_P = -sigma_y[3,iM,k]
        sigmay_4_P = -sigma_y[4,iM,k]
    elseif rightwall
        sigmax_1_P =  0.0
        sigmax_2_P =  sigma_x[2,iM,k]
        sigmax_3_P = -sigma_x[3,iM,k]
        sigmax_4_P = -sigma_x[4,iM,k]
        sigmay_1_P =  0.0
        sigmay_2_P =  sigma_y[2,iM,k]
        sigmay_3_P = -sigma_y[3,iM,k]
        sigmay_4_P = -sigma_y[4,iM,k]
    else
        iP,kP = get_infoP(mapP,Fmask,i,k)
        sigmax_1_P = 0.0
        sigmax_2_P = sigma_x[2,iP,kP]
        sigmax_3_P = sigma_x[3,iP,kP]
        sigmax_4_P = sigma_x[4,iP,kP]
        sigmay_1_P = 0.0
        sigmay_2_P = sigma_y[2,iP,kP]
        sigmay_3_P = sigma_y[3,iP,kP]
        sigmay_4_P = sigma_y[4,iP,kP]
    end

    return sigmax_1_P,sigmax_2_P,sigmax_3_P,sigmax_4_P,sigmay_1_P,sigmay_2_P,sigmay_3_P,sigmay_4_P
end

@inline function get_valP(U,f_x,f_y,sigma_x,sigma_y,mapP,Fmask,i,iM,xM,uM,vM,k)
    leftwall,rightwall,has_bc = check_BC(xM,i)
    rhoP,rhouP,rhovP,EP = get_consP(U,mapP,Fmask,i,iM,uM,vM,k,leftwall,rightwall)
    fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P = get_fluxP(f_x,f_y,mapP,Fmask,i,k,leftwall,rightwall,rhoP,rhouP,rhovP,EP)
    sigmax_1_P,sigmax_2_P,sigmax_3_P,sigmax_4_P,sigmay_1_P,sigmay_2_P,sigmay_3_P,sigmay_4_P = get_sigmaP(sigma_x,sigma_y,mapP,Fmask,i,iM,k,leftwall,rightwall)
    return rhoP,rhouP,rhovP,EP,fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P,
           sigmax_1_P,sigmax_2_P,sigmax_3_P,sigmax_4_P,sigmay_1_P,sigmay_2_P,sigmay_3_P,sigmay_4_P,has_bc
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
    rho_j    = U[1,j,k]
    rhou_j   = U[2,j,k]
    rhov_j   = U[3,j,k]
    E_j      = U[4,j,k]
    f_1_j    = f[1,j,k]
    f_2_j    = f[2,j,k]
    f_3_j    = f[3,j,k]
    f_4_j    = f[4,j,k]
    sigma1_j = sigma[1,j,k]
    sigma2_j = sigma[2,j,k]
    sigma3_j = sigma[3,j,k]
    sigma4_j = sigma[4,j,k]
    rho_i    = U[1,i,k]
    rhou_i   = U[2,i,k]
    rhov_i   = U[3,i,k]
    E_i      = U[4,i,k]
    f_1_i    = f[1,i,k]
    f_2_i    = f[2,i,k]
    f_3_i    = f[3,i,k]
    f_4_i    = f[4,i,k]
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
    sigma1_j  = sigma[1,j,k]
    sigma2_j  = sigma[2,j,k]
    sigma3_j  = sigma[3,j,k]
    sigma4_j  = sigma[4,j,k]
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

    SJ_ij = SJ_ij_db/2
    
    if (direction == 0)
        # x direction
        F1,F2,F3,F4 = euler_fluxes_2D_x(rho_i,u_i,v_i,beta_i,rholog_i,betalog_i,
                                        rho_j,u_j,v_j,beta_j,rholog_j,betalog_j)
    else
        # y direction
        F1,F2,F3,F4 = euler_fluxes_2D_y(rho_i,u_i,v_i,beta_i,rholog_i,betalog_i,
                                        rho_j,u_j,v_j,beta_j,rholog_j,betalog_j)
    end
    FH1 = SJ_ij_db*F1-SJ_ij*(sigma1_i+sigma1_j)
    FH2 = SJ_ij_db*F2-SJ_ij*(sigma2_i+sigma2_j)
    FH3 = SJ_ij_db*F3-SJ_ij*(sigma3_i+sigma3_j)
    FH4 = SJ_ij_db*F4-SJ_ij*(sigma4_i+sigma4_j)

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
    f_x,f_y,VU,sigma_x,sigma_y,rholog,betalog,U_low,U_high,F_low,F_high,F_P,L,wspd_arr,λ_arr,λf_arr,dii_arr = prealloc
    S0xJ_vec,S0yJ_vec,S0r_nnzi,S0r_nnzj,S0s_nnzi,S0s_nnzj,SxJ_db_vec,SyJ_db_vec,Sr_nnzi,Sr_nnzj,Ss_nnzi,Ss_nnzj,SxJ_vec,SyJ_vec,MJ_inv,BrJ_halved,BsJ_halved,coeff_arr,S_nnzi,S_nnzj = ops
    mapP,Fmask,x,y = geom

    fill!(rhsU,0.0)
    fill!(sigma_x,0.0)
    fill!(sigma_y,0.0)
    @batch for k = 1:K
        for i = 1:Np
            rho  = U[1,i,k]
            rhou = U[2,i,k]
            rhov = U[3,i,k]
            E    = U[4,i,k]
            p           = pfun(rho,rhou,rhov,E)
            if p < 0.0 || rho < 0.0
                @show " ========= "
                @show i,k 
                @show p, rho
                @show x[i,k],y[i,k]
                throw(error("negative pressure/density"))
                break
            end
            v1,v2,v3,v4 = entropyvar(rho,rhou,rhov,E,p)
            f_x[1,i,k]  = rhou
            f_x[2,i,k]  = rhou^2/rho+p
            f_x[3,i,k]  = rhou*rhov/rho
            f_x[4,i,k]  = E*rhou/rho+p*rhou/rho
            f_y[1,i,k]  = rhov
            f_y[2,i,k]  = rhou*rhov/rho
            f_y[3,i,k]  = rhov^2/rho+p
            f_y[4,i,k]  = E*rhov/rho+p*rhov/rho
            VU[1,i,k]   = v1
            VU[2,i,k]   = v2
            VU[3,i,k]   = v3
            VU[4,i,k]   = v4
            rholog[i,k]  = log(rho)
            betalog[i,k] = log(rho/(2*p))
        end
    end

    # Compute sigma
    @batch for k = 1:K 
        # First construct theta \approx \pd{v}{x}
        for c_r = 1:Sr_nnz_hv
            i = Sr_nnzi[c_r]
            j = Sr_nnzj[c_r]
            for c = 1:Nc
                sigma_x[c,i,k] = sigma_x[c,i,k] + SxJ_vec[c_r]*VU[c,j,k]
                sigma_x[c,j,k] = sigma_x[c,j,k] - SxJ_vec[c_r]*VU[c,i,k]
            end
        end

        for c_s = 1:Ss_nnz_hv
            i = Ss_nnzi[c_s]
            j = Ss_nnzj[c_s]
            for c = 1:Nc
                sigma_y[c,i,k] = sigma_y[c,i,k] + SyJ_vec[c_s]*VU[c,j,k]
                sigma_y[c,j,k] = sigma_y[c,j,k] - SyJ_vec[c_s]*VU[c,i,k]
            end
        end

        for i = 1:Nfp
            iM = Fmask[i]
            xM = x[iM,k]
            yM = y[iM,k]
            v1M = VU[1,iM,k]
            v2M = VU[2,iM,k]
            v3M = VU[3,iM,k]
            v4M = VU[4,iM,k]

            leftwall,rightwall,has_bc = check_BC(xM,i)
            v1P,v2P,v3P,v4P = get_vP(VU,mapP,Fmask,i,iM,k,leftwall,rightwall)

            if is_face_x(i)
                BrJ_ii_halved = BrJ_halved[iM]
                sigma_x[1,iM,k] = sigma_x[1,iM,k] + BrJ_ii_halved*(v1M+v1P)/2
                sigma_x[2,iM,k] = sigma_x[2,iM,k] + BrJ_ii_halved*(v2M+v2P)/2
                sigma_x[3,iM,k] = sigma_x[3,iM,k] + BrJ_ii_halved*(v3M+v3P)/2
                sigma_x[4,iM,k] = sigma_x[4,iM,k] + BrJ_ii_halved*(v4M+v4P)/2
            end

            if is_face_y(i)
                BsJ_ii_halved = BsJ_halved[iM]
                sigma_y[1,iM,k] = sigma_y[1,iM,k] + BsJ_ii_halved*(v1M+v1P)/2
                sigma_y[2,iM,k] = sigma_y[2,iM,k] + BsJ_ii_halved*(v2M+v2P)/2
                sigma_y[3,iM,k] = sigma_y[3,iM,k] + BsJ_ii_halved*(v3M+v3P)/2
                sigma_y[4,iM,k] = sigma_y[4,iM,k] + BsJ_ii_halved*(v4M+v4P)/2
            end
        end

        for i = 1:Np
            mJ_inv_ii = MJ_inv[i]
            for c = 1:Nc
                sigma_x[c,i,k] = mJ_inv_ii*sigma_x[c,i,k]
                sigma_y[c,i,k] = mJ_inv_ii*sigma_y[c,i,k]
            end
        end

        for i = 1:Np
            v1 = VU[1,i,k]
            v2 = VU[2,i,k]
            v3 = VU[3,i,k]
            v4 = VU[4,i,k]
            theta1_1 = sigma_x[1,i,k]
            theta1_2 = sigma_x[2,i,k]
            theta1_3 = sigma_x[3,i,k]
            theta1_4 = sigma_x[4,i,k]
            theta2_1 = sigma_y[1,i,k]
            theta2_2 = sigma_y[2,i,k]
            theta2_3 = sigma_y[3,i,k]
            theta2_4 = sigma_y[4,i,k]

            K11_22,K11_24,K11_33,K11_34,K11_44,
            K12_23,K12_24,K12_32,K12_34,K12_42,K12_43,K12_44,
            K22_22,K22_24,K22_33,K22_34,K22_44 = get_Kvisc(v1,v2,v3,v4)

            K21_32,K21_42,K21_23,K21_43,K21_24,K21_34,K21_44 = K12_23,K12_24,K12_32,K12_34,K12_42,K12_43,K12_44

            sigma1_2 = K11_22*theta1_2 + K11_24*theta1_4
                     + K12_23*theta2_3 + K12_24*theta2_4
            sigma1_3 = K11_33*theta1_3 + K11_34*theta1_4
                     + K12_32*theta2_2 + K12_34*theta2_4
            sigma1_4 = K11_24*theta1_2 + K11_34*theta1_3 + K11_44*theta1_4
                     + K12_42*theta2_2 + K12_43*theta2_3 + K12_44*theta2_4

            sigma2_2 = K21_23*theta1_3 + K21_24*theta1_4
                     + K22_22*theta2_2 + K22_24*theta2_4
            sigma2_3 = K21_32*theta1_2 + K21_34*theta1_4
                     + K22_33*theta2_3 + K22_34*theta2_4
            sigma2_4 = K21_42*theta1_2 + K21_43*theta1_3 + K21_44*theta1_4
                     + K22_24*theta2_2 + K22_34*theta2_3 + K22_44*theta2_4

            sigma_x[1,i,k] = 0.0
            sigma_x[2,i,k] = sigma1_2
            sigma_x[3,i,k] = sigma1_3
            sigma_x[4,i,k] = sigma1_4
            sigma_y[1,i,k] = 0.0 
            sigma_y[2,i,k] = sigma2_2
            sigma_y[3,i,k] = sigma2_3
            sigma_y[4,i,k] = sigma2_4 
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
                                  wavespeed_1D(rho_i,rhou_i,E_i))
            wspd_arr[i,2,k] = max(zhang_wavespd(rho_i,rhou_i,rhov_i,E_i,
                                            sigma1_2,sigma1_3,sigma1_4,
                                            sigma2_2,sigma2_3,sigma2_4,
                                            0,1),
                                  wavespeed_1D(rho_i,rhov_i,E_i))
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
    end

    # Interface dissipation coeff 
    @batch for k = 1:K
        for i = 1:Nfp
            iM = Fmask[i]
            BrJ_ii_halved_abs = abs(BrJ_halved[iM])
            BsJ_ii_halved_abs = abs(BsJ_halved[iM])
            xM    = x[iM,k]

            leftwall,rightwall,_= check_BC(xM,i)
            iP,kP = get_infoP(mapP,Fmask,i,k)
            
            if is_face_x(i)
                λM = wspd_arr[iM,1,k]
                λP = wspd_arr[iP,1,kP]
                λf = max(λM,λP)*BrJ_ii_halved_abs
                λf_arr[i,k] = λf
                if in_s1
                    dii_arr[iM,k] = dii_arr[iM,k] + λf
                end
            end

            if is_face_y(i)
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

    # If at the first stage, calculate the time step
    if in_s1
        for k = 1:K
            for i = 1:Np
                dt = min(dt,1.0/MJ_inv[i]/2.0/dii_arr[i,k])
            end
        end
        dt= min(CFL*dt,T-t)
    end

    # =====================
    # Loop through elements
    # =====================
    @batch for k = 1:K
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
            S0xJ_ij = S0xJ_vec[c_r]
            update_F_low!(F_low,k,tid,i,j,λ,S0xJ_ij,U,f_x,sigma_x)
        end

        for c_s = 1:S0s_nnz_hv
            i = S0s_nnzi[c_s]
            j = S0s_nnzj[c_s]
            λ = λ_arr[c_s,2,k]
            S0yJ_ij = S0yJ_vec[c_s]
            update_F_low!(F_low,k,tid,i,j,λ,S0yJ_ij,U,f_y,sigma_y)
        end

        # Calculate high order algebraic flux
        for c_r = 1:Sr_nnz_hv
            i         = Sr_nnzi[c_r]
            j         = Sr_nnzj[c_r]
            SxJ_ij_db = SxJ_db_vec[c_r]
            update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,U,rholog,betalog,0,sigma_x)
        end

        for c_r = 1:Sr_nnz_hv
            i         = Ss_nnzi[c_r]
            j         = Ss_nnzj[c_r]
            SyJ_ij_db = SyJ_db_vec[c_r]
            update_F_high!(F_high,k,tid,i,j,SyJ_ij_db,U,rholog,betalog,1,sigma_y)
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
            if rhoM < 1e-6
                @show "==== !!! ===="
                @show "small rhoM in interface flux"
                @show i,k
                @show rhoM
                # uM = 0.0
                # vM = 0.0
            end
            fx_1_M = f_x[1,iM,k]
            fx_2_M = f_x[2,iM,k]
            fx_3_M = f_x[3,iM,k]
            fx_4_M = f_x[4,iM,k]
            fy_1_M = f_y[1,iM,k]
            fy_2_M = f_y[2,iM,k]
            fy_3_M = f_y[3,iM,k]
            fy_4_M = f_y[4,iM,k]
            sigmax_1_M = sigma_x[1,iM,k]
            sigmax_2_M = sigma_x[2,iM,k]
            sigmax_3_M = sigma_x[3,iM,k]
            sigmax_4_M = sigma_x[4,iM,k]
            sigmay_1_M = sigma_y[1,iM,k]
            sigmay_2_M = sigma_y[2,iM,k]
            sigmay_3_M = sigma_y[3,iM,k]
            sigmay_4_M = sigma_y[4,iM,k]

            rhoP,rhouP,rhovP,EP,fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P,
            sigmax_1_P,sigmax_2_P,sigmax_3_P,sigmax_4_P,sigmay_1_P,sigmay_2_P,sigmay_3_P,sigmay_4_P = get_valP(U,f_x,f_y,sigma_x,sigma_y,mapP,Fmask,i,iM,xM,uM,vM,k)
            λ = λf_arr[i,k]

            # flux in x direction
            if is_face_x(i)
                F_P[1,i,tid] = (BrJ_ii_halved*(fx_1_M+fx_1_P-sigmax_1_M-sigmax_1_P)
                               -λ*(rhoP-rhoM) )
                F_P[2,i,tid] = (BrJ_ii_halved*(fx_2_M+fx_2_P-sigmax_2_M-sigmax_2_P)
                               -λ*(rhouP-rhouM) )
                F_P[3,i,tid] = (BrJ_ii_halved*(fx_3_M+fx_3_P-sigmax_3_M-sigmax_3_P)
                               -λ*(rhovP-rhovM) )
                F_P[4,i,tid] = (BrJ_ii_halved*(fx_4_M+fx_4_P-sigmax_4_M-sigmax_4_P)
                               -λ*(EP-EM) )
            end

            # flux in y direction
            if is_face_y(i)
                F_P[1,i,tid] = (BsJ_ii_halved*(fy_1_M+fy_1_P-sigmay_1_M-sigmay_1_P)
                               -λ*(rhoP-rhoM) )
                F_P[2,i,tid] = (BsJ_ii_halved*(fy_2_M+fy_2_P-sigmay_2_M-sigmay_2_P)
                               -λ*(rhouP-rhouM) )
                F_P[3,i,tid] = (BsJ_ii_halved*(fy_3_M+fy_3_P-sigmay_3_M-sigmay_3_P)
                               -λ*(rhovP-rhovM) )
                F_P[4,i,tid] = (BsJ_ii_halved*(fy_4_M+fy_4_P-sigmay_4_M-sigmay_4_P)
                               -λ*(EP-EM) )
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
                U_low[c,i,tid]  = U[c,i,k] - dt*MJ_inv[i]*U_low[c,i,tid]
                U_high[c,i,tid] = U[c,i,k] - dt*MJ_inv[i]*U_high[c,i,tid]
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

        # if (k < XLENGTH*K1D)
        #     is_H_positive = false
        # end

        if !is_H_positive
            # Calculate limiting parameters
            for ni = 1:S_nnz_hv
                i = S_nnzi[ni]
                j = S_nnzj[ni]
                coeff_i = dt*coeff_arr[i]
                coeff_j = dt*coeff_arr[j]
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
                L[ni,tid] = min(limiting_param(rhoi,rhoui,rhovi,Ei,rhoP_i,rhouP_i,rhovP_i,EP_i),
                                limiting_param(rhoj,rhouj,rhovj,Ej,rhoP_j,rhouP_j,rhovP_j,EP_j))
            end
        end

        # Elementwise limiting
        l_e = 1.0
        for i = 1:S_nnz_hv
            l_e = min(l_e,L[i,tid])
        end
        l_em1 = l_e-1.0

        if is_H_positive
            l_e = 1.0
            l_em1 = 0.0
        end

        for ni = 1:S_nnz_hv
            i     = S_nnzi[ni]
            j     = S_nnzj[ni]
            for c = 1:Nc
                FL_ij = F_low[c,i,j,tid]
                FH_ij = F_high[c,i,j,tid]
                # if (LIMITOPT == 3)
                #     l_e   = L[ni,tid]
                #     l_em1 = L[ni,tid]-1.0 
                # end
                rhsU[c,i,k] = rhsU[c,i,k] + l_em1*FL_ij - l_e*FH_ij
                rhsU[c,j,k] = rhsU[c,j,k] - l_em1*FL_ij + l_e*FH_ij
                # l_e   = L[ni,tid]
                # l_em1 = L[ni,tid]-1.0 
                # # l_e = 0.0
                # # l_em1 = -1.0
                # rhsU[c,i,k] = rhsU[c,i,k] + l_em1*FL_ij - l_e*FH_ij
                # rhsU[c,j,k] = rhsU[c,j,k] - l_em1*FL_ij + l_e*FH_ij
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

    # @show "min density"
    # @show minimum(U[1,:,:])
    # @show maximum(U[1,:,:])
    # @show minimum(U[2,:,:])
    # @show maximum(U[2,:,:])
    # @show minimum(U[3,:,:])
    # @show maximum(U[3,:,:])
    # @show minimum(U[4,:,:])
    # @show maximum(U[4,:,:])

    return dt
end

@unpack Vf,Dr,Ds,LIFT = rd
md = init_mesh((VX,VY),EToV,rd)
@unpack xf,yf,mapM,mapB,nxJ,nyJ,x,y = md
xb,yb = (x->x[mapB]).((xf,yf))

const Np = (N+1)*(N+1)
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
coeff_arr = 2*N*MJ_inv

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
SxJ_vec    = rxJ*Sr_vec
SyJ_vec    = syJ*Ss_vec
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
        if x[i,k] < 0.5
            U[1,i,k] = 8.0
            U[2,i,k] = 8.25
            U[3,i,k] = 0.0
            U[4,i,k] = 116.5
        else
            U[1,i,k] = 1.4
            U[2,i,k] = 0.0
            U[3,i,k] = 0.0
            U[4,i,k] = 1.0
        end
    end
end

# Preallocation
prevU  = zeros(Float64,size(U))
rhsU   = zeros(Float64,size(U))
f_x    = zeros(Float64,size(U))
f_y    = zeros(Float64,size(U))
sigma_x = zeros(Float64,size(U))
sigma_y = zeros(Float64,size(U))
VU      = zeros(Float64,Nc,Np,K)
rholog  = zeros(Float64,Np,K)
betalog = zeros(Float64,Np,K)
U_low   = zeros(Float64,Nc,Np,NUM_THREADS)
U_high  = zeros(Float64,Nc,Np,NUM_THREADS)
F_low   = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_high  = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_P     = zeros(Float64,Nc,Nfp,NUM_THREADS)
L       =  ones(Float64,S_nnz_hv,NUM_THREADS)
wspd_arr = zeros(Float64,Np,2,K)
λ_arr    = zeros(Float64,S0r_nnz_hv,2,K) # Assume S0r and S0s has same number of nonzero entries
λf_arr   = zeros(Float64,Nfp,K)
dii_arr  = zeros(Float64,Np,K)

prealloc = (f_x,f_y,VU,sigma_x,sigma_y,rholog,betalog,U_low,U_high,F_low,F_high,F_P,L,wspd_arr,λ_arr,λf_arr,dii_arr)
ops      = (S0xJ_vec,  S0yJ_vec,  S0r_nnzi,S0r_nnzj,S0s_nnzi,S0s_nnzj,
            SxJ_db_vec,SyJ_db_vec,Sr_nnzi, Sr_nnzj, Ss_nnzi, Ss_nnzj,
            SxJ_vec,   SyJ_vec,
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

#=
#Timing
dt = dt0
#rhs_IDP!(U,rhsU,t,dt,prealloc,ops,geom,true);
@btime rhs_IDP!($U,$rhsU,$t,$dt,$prealloc,$ops,$geom,true);
# @profiler rhs_IDP!(U,rhsU,t,dt,prealloc,ops,geom,true);
=#

dt_hist = []
i = 1

@time while t < T
#while i < 2
    # SSPRK(3,3)
    fill!(dii_arr,0.0)
    dt = min(CFL*dt0,T-t)
    
    dt = rhs_IDP!(U,rhsU,t,dt,prealloc,ops,geom,true);
    @. resW = U + dt*rhsU
    rhs_IDP!(resW,rhsU,t+dt,dt,prealloc,ops,geom,false);
    @. resW = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resW
    rhs_IDP!(resW,rhsU,t+dt/2,dt,prealloc,ops,geom,false);
    @. resW = resW+dt*rhsU
    @. U = 1/3*U+2/3*resW

    push!(dt_hist,dt)
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T, at step $i")
    flush(stdout)
    global i = i + 1

    # if ((mod(i,100) == 1) | (i >= 55))
    # @show U[1,:,351]
    # @show sigma_x[1,:,351]
    if (mod(i,500) == 1)
    end
end

xp = Vp*x
yp = Vp*y
vv = Vp*U[1,:,:]
xp = vec(xp)
yp = vec(yp)
vv = vec(vv)
scatter(xp,yp,vv,zcolor=vv,camera=(0,90),colorbar=:right)
savefig("~/Desktop/N=$N,K1D=$K1D,T=$T,CNSreflectivewalltest.png")

end #muladd
