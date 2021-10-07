using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using StaticArrays
using Polyester
using MuladdMacro
using Setfield

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

const TOL     = 1e-15
const POSTOL  = 1e-14   # Tolerance of positivity
const Nc      = 4       # number of components
const WALLPT  = 1.0/6.0 # Starting point of the wall in x direction
# const XLENGTH = 4.0     # length of domain TODO: hardcoded now

"Approximation parameters"
const N   = 3
const K1D = 200
const T   = 0.1

const Nq  = N
const Np  = (N+1)*(N+1)
const Nfp = 4*(N+1)

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

"Initialize reference element"
rd = init_reference_quad(N,gauss_lobatto_quad(0,0,N))
@unpack Vf,Dr,Ds,LIFT,Nfaces = rd

md = init_mesh((VX,VY),EToV,rd)
@unpack sJ,xf,yf,mapM,mapP,mapB,nxJ,nyJ,x,y = md
# TODO: assume uniform mesh, hardcoded geometric factors
const K   = 4*K1D*K1D
const J   = 1.0/K
const Jf  = 1.0/K1D/2
const rxJ = 2*K1D*J
const syJ = 2*K1D*J

xb,yb = (x->x[mapB]).((xf,yf))
r,_ = gauss_lobatto_quad(0,0,N)   # Reference nodes
VDM = vandermonde_1D(N,r)         # modal to nodal
Dr = grad_vandermonde_1D(N,r)/VDM # nodal differentiation
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1]) # nodal linear interpolation

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

# Convert diagonal matrices to vectors
Br   = Array(diag(Br))
Bs   = Array(diag(Bs))
M    = Array(diag(M))
Minv = Array(diag(Minv))

Fmask  = [1:N+1; (N+1):(N+1):Np; Np:-1:Np-N; Np-N:-(N+1):1]
Fxmask = [(N+2):(2*N+2); (3*N+4):(4*N+4)]
Fymask = [1:(N+1); (2*N+3):(3*N+3)]

# Boundary conditions
inflow   = mapB[findall(@. (abs(xb) < TOL) | ((xb < 1/6) & (abs(yb) < TOL)))]
outflow  = mapB[findall(@. abs(xb-4.) < TOL)]
topflow  = mapB[findall(@. abs(yb-1.) < TOL)]
wall     = mapB[findall(@. (xb >= 1/6) & (abs(yb) < TOL))]
nx_wall  = nxJ[wall]./sJ[wall]
ny_wall  = nyJ[wall]./sJ[wall]
const TOP_INIT = (1+sqrt(3)/6)/sqrt(3)


@inline function get_valP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k)
    inflow  = ((abs(xM) < TOL) & (abs(yM-1.) > TOL)) | ((xM <= WALLPT) & (abs(yM) < TOL))
    outflow = ((abs(xM-4.0) < TOL) & (abs(yM) > TOL) & (abs(yM-1.0) > TOL))
    topflow = (abs(yM-1.0) < TOL)
    wall    = ((xM > WALLPT) & (abs(yM) < TOL))
    has_bc  = (inflow | outflow | topflow | wall)

    if inflow
        rhoP   = rhoL
        rhouP  = rhouL
        rhovP  = rhovL
        EP     = EL
        fx_1_P = rhoL*uL
        fx_2_P = rhoL*uL^2+pL
        fx_3_P = rhoL*uL*vL
        fx_4_P = uL*(EL+pL)
        fy_1_P = rhoL*vL
        fy_2_P = rhoL*uL*vL
        fy_3_P = rhoL*vL^2+pL
        fy_4_P = vL*(EL+pL)

    elseif outflow
        rhoP   = U[1,iM,k]
        rhouP  = U[2,iM,k]
        rhovP  = U[3,iM,k]
        EP     = U[4,iM,k]
        fx_1_P = f_x[1,iM,k]
        fx_2_P = f_x[2,iM,k]
        fx_3_P = f_x[3,iM,k]
        fx_4_P = f_x[4,iM,k]
        fy_1_P = f_y[1,iM,k]
        fy_2_P = f_y[2,iM,k]
        fy_3_P = f_y[3,iM,k]
        fy_4_P = f_y[4,iM,k]

    elseif wall
        # TODO: we assume the normals are [0;-1] here
        # Un = -vM
        # Ut = -uM
        rhoP   = U[1,iM,k]
        rhouP  = rhoP*uM   # rhoP = rhoM, TODO: refactor
        rhovP  = -rhoP*vM
        uP     = rhouP/rhoP
        vP     = rhovP/rhoP
        EP     = U[4,iM,k]
        pP     = pfun(rhoP,rhouP,rhovP,EP)
        fx_1_P = rhoP*uP
        fx_2_P = rhoP*uP^2+pP
        fx_3_P = rhoP*uP*vP
        fx_4_P = uP*(EP+pP)
        fy_1_P = rhoP*vP
        fy_2_P = rhoP*uP*vP
        fy_3_P = rhoP*vP^2+pP
        fy_4_P = vP*(EP+pP)

    elseif topflow
        breakpoint = TOP_INIT+t*SHOCKSPD
        if xM < breakpoint
            rhoP   = rhoL
            rhouP  = rhouL
            rhovP  = rhovL
            EP     = EL
            fx_1_P = rhoL*uL
            fx_2_P = rhoL*uL^2+pL
            fx_3_P = rhoL*uL*vL
            fx_4_P = uL*(EL+pL)
            fy_1_P = rhoL*vL
            fy_2_P = rhoL*uL*vL
            fy_3_P = rhoL*vL^2+pL
            fy_4_P = vL*(EL+pL)
        else
            rhoP   = rhoR
            rhouP  = rhouR
            rhovP  = rhovR
            EP     = ER
            fx_1_P = rhoR*uR
            fx_2_P = rhoR*uR^2+pR
            fx_3_P = rhoR*uR*vR
            fx_4_P = uR*(ER+pR)
            fy_1_P = rhoR*vR
            fy_2_P = rhoR*uR*vR
            fy_3_P = rhoR*vR^2+pR
            fy_4_P = vR*(ER+pR)
        end
    else                         # if not on the physical boundary
        gP = mapP[i,k]           # exterior global face node number
        kP = fld1(gP,Nfp)        # exterior element number
        iP = Fmask[mod1(gP,Nfp)] # exterior node number

        rhoP  = U[1,iP,kP]
        rhouP = U[2,iP,kP]
        rhovP = U[3,iP,kP]
        EP    = U[4,iP,kP]
        fx_1_P = f_x[1,iP,kP]
        fx_2_P = f_x[2,iP,kP]
        fx_3_P = f_x[3,iP,kP]
        fx_4_P = f_x[4,iP,kP]
        fy_1_P = f_y[1,iP,kP]
        fy_2_P = f_y[2,iP,kP]
        fy_3_P = f_y[3,iP,kP]
        fy_4_P = f_y[4,iP,kP]
    end

    return rhoP,rhouP,rhovP,EP,fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P,has_bc
end


# initial condition
at_left(x,y) = y-sqrt(3)*x+sqrt(3)/6 > 0.0
U = zeros(Float64,Nc,Np,K)
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


# Preallocation
f_x     = zeros(Float64,size(U))
f_y     = zeros(Float64,size(U))
rholog  = zeros(Float64,Np,K)
betalog = zeros(Float64,Np,K)
U_low   = zeros(Float64,Nc,Np,NUM_THREADS)
F_low   = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_high  = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_P     = zeros(Float64,Nc,Nfp,NUM_THREADS)
L       =  ones(Float64,Np,Np,NUM_THREADS)


const DEBUGELEM = 3
function rhs_IDP_fixdt!(U,rhsU,t,prealloc,ops,geom)
    f_x,f_y,U_low,F_low,F_high,F_P,L,rholog,betalog = prealloc
    S0r,S0s,Sr,Ss,Br_halved,Bs_halved,MJ_inv,coeff_arr = ops
    mapP,Fmask,Fxmask,Fymask,x,y = geom

    # Preallocate inviscid flux TODO: shall we?
    # Current time: 107 ms
    # Current time: 27.5 ms
    # TODO: rewrite arithmetic expressions
    # TODO: BC
    # TODO: BC dissipation
    # TODO: @inbounds @turbo
    fill!(rhsU, 0.0)   # TODO: fill rhsU with zeros?
    for e = 1:K
        for i = 1:Np
            rho  = U[1,i,e]
            rhou = U[2,i,e]
            rhov = U[3,i,e]
            E    = U[4,i,e]
            p          = pfun(rho,rhou,rhov,E)
            f_x[1,i,e] = rhou
            f_x[2,i,e] = rhou^2/rho+p
            f_x[3,i,e] = rhou*rhov/rho
            f_x[4,i,e] = E*rhou/rho+p*rhou/rho
            f_y[1,i,e] = rhov
            f_y[2,i,e] = rhou*rhov/rho
            f_y[3,i,e] = rhov^2/rho+p
            f_y[4,i,e] = E*rhov/rho+p*rhov/rho
            rholog[i,e]  = log(rho)
            betalog[i,e] = log(rho/(2*p))
        end
    end

    # =====================
    # Loop through elements
    # =====================
    for  k = 1:K
        tid = Threads.threadid()

        # ===========================================
        # Reinitialize arrays TODO: maybe unnecessary
        # ===========================================
        fill!(F_low , 0.0)
        fill!(F_high, 0.0)
        fill!(U_low , 0.0)
        fill!(F_P   , 0.0)
        fill!(L     , 1.0)

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
                S0r_ij = S0r[i,j]
                S0s_ij = S0s[i,j]
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
                    n_ij_norm = sqrt(rxJ_sq*S0r_ij^2+syJ_sq*S0s_ij^2)
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
        for j = 2:Np
            rho_j     = U[1,j,k]
            u_j       = U[2,j,k]/rho_j
            v_j       = U[3,j,k]/rho_j
            beta_j    = rho_j/(2*pfun(rho_j,rho_j*u_j,rho_j*v_j,U[4,j,k]))
            rholog_j  = rholog[j,k]
            betalog_j = betalog[j,k]
            for i = 1:j-1
                Sr_ij     = Sr[i,j]
                Ss_ij     = Ss[i,j]
                rho_i     = U[1,i,k]
                u_i       = U[2,i,k]/rho_i
                v_i       = U[3,i,k]/rho_i
                beta_i    = rho_i/(2*pfun(rho_i,rho_i*u_i,rho_i*v_i,U[4,i,k]))
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
        for i = 1:Nfp
            Br_ii_halved = Br_halved[Fmask[i]]
            Bs_ii_halved = Bs_halved[Fmask[i]]

            iM    = Fmask[i]
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

            # flux in x direction
            if i in Fxmask
                λM = wavespeed_1D(rhoM,rhouM,EM)
                λP = wavespeed_1D(rhoP,rhouP,EP)
                if has_bc
                    λ = 0
                else
                    λ  = max(λM,λP)*abs(Br_ii_halved)
                end

                F_P[1,i,tid] = (Jf*Br_ii_halved*(fx_1_M+fx_1_P)
                               -λ*(rhoP-rhoM) )
                F_P[2,i,tid] = (Jf*Br_ii_halved*(fx_2_M+fx_2_P)
                               -λ*(rhouP-rhouM) )
                F_P[3,i,tid] = (Jf*Br_ii_halved*(fx_3_M+fx_3_P)
                               -λ*(rhovP-rhovM) )
                F_P[4,i,tid] = (Jf*Br_ii_halved*(fx_4_M+fx_4_P)
                               -λ*(EP-EM) )
            end

            # flux in y direction
            if i in Fymask
                λM = wavespeed_1D(rhoM,rhovM,EM)
                λP = wavespeed_1D(rhoP,rhovP,EP)
                if has_bc
                    λ = 0
                else
                    λ  = max(λM,λP)*abs(Bs_ii_halved)
                end

                F_P[1,i,tid] = (Jf*Bs_ii_halved*(fy_1_M+fy_1_P)
                               -λ*(rhoP-rhoM) )
                F_P[2,i,tid] = (Jf*Bs_ii_halved*(fy_2_M+fy_2_P)
                               -λ*(rhouP-rhouM) )
                F_P[3,i,tid] = (Jf*Bs_ii_halved*(fy_3_M+fy_3_P)
                               -λ*(rhovP-rhovM) )
                F_P[4,i,tid] = (Jf*Bs_ii_halved*(fy_4_M+fy_4_P)
                               -λ*(EP-EM) )
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
                # TODO: debug
                L[i,j,tid] = 0.0#l_e
            end
        end

        # Accumulate RHS
        # for j = 1:Np
        #     for i = 1:Np
        #         for c = 1:Nc
        #             rhsU[c,i,k] = rhsU[c,i,k] + (L[i,j,tid]-1)*F_low[c,i,j,tid]-L[i,j,tid]*F_high[c,i,j,tid]
        #         end
        #     end
        # end
        # for i = 1:Nfp
        #     for c = 1:Nc
        #         rhsU[c,Fmask[i],k] = rhsU[c,Fmask[i],k] - F_P[c,i,tid]
        #     end
        # end

        for c = 1:Nc
            for i = 1:Np
                for j = 1:Np
                    rhsU[c,i,k] = rhsU[c,i,k] + (L[i,j,tid]-1)*F_low[c,i,j,tid]-L[i,j,tid]*F_high[c,i,j,tid]
                    # rhsU[c,i,k] = rhsU[c,i,k] - F_low[c,i,j,tid]
                end
            end
            for i = 1:Nfp
                rhsU[c,Fmask[i],k] = rhsU[c,Fmask[i],k] - F_P[c,i,tid]
            end
        end

        # if k == DEBUGELEM
        #     for c = 1:4
        #         println(c)
        #         println("=== F_low ===", c)
        #         display(F_low[c,:,:,1])
        #         println("=== F_P ===", c)
        #         display(F_P[c,:,1])
        #         println("=== U_low ===", c)
        #         display(U_low[c,:,1])
        #         println("=== rhsU ===", c)
        #         display(rhsU[c,:,DEBUGELEM])
        #     end
        # end
    end

    for k = 1:K
        for i = 1:Np
            for c = 1:Nc
                rhsU[c,i,k] = MJ_inv[i]*rhsU[c,i,k]
            end
        end
    end
end

t = 0.0
const dt = 1e-4
rhsU = zeros(size(U))

# Define some constants for optimizations
const rxJ_sq = rxJ^2
const syJ_sq = syJ^2
const rxJ_db = 2*rxJ
const syJ_db = 2*syJ
MJ_inv    = Minv./J
# Br_halved = Br./2.0
# Bs_halved = Bs./2.0
Br_halved = -sum(S0r,dims=2)
Bs_halved = -sum(S0s,dims=2)
coeff_arr = dt*(Np-1).*Minv*J

prealloc = (f_x,f_y,U_low,F_low,F_high,F_P,L,rholog,betalog)
ops      = (S0r,S0s,Sr,Ss,Br_halved,Bs_halved,MJ_inv,coeff_arr)
geom     = (mapP,Fmask,Fxmask,Fymask,x,y)

t = 0.0
#@btime rhs_IDP_fixdt!($U,$rhsU,$t,$prealloc,$ops,$geom)
@profiler rhs_IDP_fixdt!(U,rhsU,t,prealloc,ops,geom)
#rhs_IDP_fixdt!(U,rhsU,t,prealloc,ops,geom)

# for c = 1:4
#     println("=== rhsU ===", c)
#     display(rhsU[c,:,:])
# end

# # Time stepping
# i = 1
# t = 0.0
# rhsU = zeros(size(U))
# resW = zeros(size(U))

# # while t < T
# while i < 2
#     # TODO: fix
#     # dt = min(1e-4,T-t)

#     # println("===== First step =====")
#     # for c = 1:4
#     #     println("=== U ===", c)
#     #     display(U[c,:,:])
#     # end
#     rhs_IDP_fixdt!(U,rhsU,t,prealloc,ops,geom)
#     @. resW = U + dt*rhsU
#     println("===== Second step =====")
#     for c = 1:4
#         println("=== resW ===", c)
#         display(resW[c,:,:])
#     end
#     rhs_IDP_fixdt!(resW,rhsU,t,prealloc,ops,geom)
#     println("===== Second RHSU =====")
#     for c = 1:4
#         println("=== rhsU ===", c)
#         display(rhsU[c,:,:])
#     end
#     # println("===== Look at first resW =====")
#     # for c = 1:4
#     #     println("=== resW ===", c)
#     #     display(resW[c,:,:])
#     # end
#     @. resW = resW+dt*rhsU
#     # println("===== Look at second resW =====")
#     # for c = 1:4
#     #     println("=== resW ===", c)
#     #     display(resW[c,:,:])
#     # end
#     @. resW = 0.75*U+0.25*resW
#     # println("===== Look at third resW =====")
#     # for c = 1:4
#     #     println("=== resW ===", c)
#     #     display(resW[c,:,:])
#     # end
#     # println("===== Third step =====")
#     # for c = 1:4
#     #     println("=== resW ===", c)
#     #     display(resW[c,:,:])
#     # end
#     rhs_IDP_fixdt!(resW,rhsU,t,prealloc,ops,geom)
#     @. resW = resW+dt*rhsU
#     @. U = 1/3*U + 2/3*resW
#     # println("===== Last step =====")
#     # for c = 1:4
#     #     println("=== U ===", c)
#     #     display(U[c,:,:])
#     # end
#     global t = t + dt
#     println("Current time $t with time step size $dt, and final time $T, at step $i")
#     global i = i + 1
# end

# # rp,sp = equi_nodes_2D(10)
# # @unpack VDM = rd
# # Vp = vandermonde_2D(N,rp,sp)/VDM
# # gr(aspect_ratio=:equal,legend=false,markerstrokewidth=0,markersize=2)
# # xp = Vp*x
# # yp = Vp*y
# # rho = Vp*U[1,:,:]

# # # scatter(xp,yp,zcolor=rho,camera=(0,90))
# # # savefig("~/Desktop/tmp.png")

end # muladd
