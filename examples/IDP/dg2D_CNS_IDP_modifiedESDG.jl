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
K1D = 8
T = 0.1

# # Initial condition 2D shocktube
# const γ = 1.4
# const rhoL = 6.0#120.0
# const uL   = 0.0
# const vL   = 0.0
# const pL   = rhoL/γ
# const rhoR = 1.2
# const uR   = 0.0
# const vR   = 0.0
# const pR   = rhoR/γ

# const mu = 0.01
# const lambda = 0.0
# const Pr = 0.73
# const cp = γ/(γ-1)
# const cv = 1/(γ-1)
# const kappa = mu*cp/Pr

# Becker viscous shocktube
const γ = 1.4
const M_0 = 3.0
const mu = 0.01
const lambda = 2/3*mu
const Pr = 3/4
const cp = γ/(γ-1)
const cv = 1/(γ-1)
const kappa = mu*cp/Pr

const v_inf = 0.2
const rho_0 = 1.0
const m_0 = 1.0
const v_0 = 1.0
const v_1 = (γ-1+2/M_0^2)/(γ+1)
const v_01 = sqrt(v_0*v_1)

const uL = v_0+v_inf
const uR = v_1+v_inf
const vL = 0.0
const vR = 0.0
const rhoL = m_0/v_0
const rhoR = m_0/v_1
const eL = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_0^2)
const eR = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_1^2)
const pL = (γ-1)*rhoL*eL
const pR = (γ-1)*rhoR*eR
const EL = pL/(γ-1)+0.5*rhoL*uL^2
const ER = pR/(γ-1)+0.5*rhoR*uR^2


# # Becker viscous shocktube 2
# const γ = 1.4
# const M_0 = 20.0
# const mu = 0.001
# const lambda = 2/3*mu
# const Pr = 3/4
# const cp = γ/(γ-1)
# const cv = 1/(γ-1)
# const kappa = mu*cp/Pr

# const v_inf = 0.2
# const rho_0 = 1.0
# const v_0 = 1.0
# const m_0 = rho_0*v_0
# const v_1 = (γ-1+2/M_0^2)/(γ+1)
# const v_01 = sqrt(v_0*v_1)

# const uL = v_0+v_inf
# const uR = v_1+v_inf
# const vL = 0.0
# const vR = 0.0
# const rhoL = m_0/v_0
# const rhoR = m_0/v_1
# const eL = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_0^2)
# const eR = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_1^2)
# const pL = (γ-1)*rhoL*eL
# const pR = (γ-1)*rhoR*eR
# const EL = pL/(γ-1)+0.5*rhoL*uL^2
# const ER = pR/(γ-1)+0.5*rhoR*uR^2

function Riemann_2D(x,y)
    if x >= 0 && y >= 0
        return .5313, 0.0, 0.0, 0.4
    elseif x < 0 && y >= 0
        return 1.0, .7276, 0.0, 1.0
    elseif x < 0 && y < 0
        return .8, 0.0, 0.0, 1.0
    else
        return 1.0, 0.0, .7276, 1.0
    end
end

"Mesh related variables"
VX, VY, EToV = uniform_quad_mesh(Int(K1D/2*3),K1D)
# @. VX = (VX+1)/2
# @. VY = (VY+1)/4
@. VX = (VX/4*3)+1/4
@. VY = (VY+1)/2
# VX, VY, EToV = uniform_quad_mesh(K1D,K1D)

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
topwall  = mapB[findall(@. abs(yb-1/2) < TOL)]
wall     = mapB[findall(@. abs(yb-1/2) > TOL)]
boundary = mapB
nx_top  = nxJ[topwall]./sJ[topwall]
ny_top  = nyJ[topwall]./sJ[topwall]
nx_wall = nxJ[wall]./sJ[wall]
ny_wall = nyJ[wall]./sJ[wall]
nx_b    = nxJ[boundary]./sJ[boundary]
ny_b    = nyJ[boundary]./sJ[boundary]

# Becker's viscous shocktube
leftwall     = mapB[findall(@. abs(xb+0.5) < 1e-12)]
rightwall    = mapB[findall(@. abs(xb-1.0) < 1e-12)]
topwall      = mapB[findall(@. abs(yb-1.0) < 1e-12)]
bottomwall   = mapB[findall(@. abs(yb) < 1e-12)]
vwall = [leftwall;rightwall]

# Make domain periodic
@unpack Nfaces,Vf = rd
@unpack xf,yf,K,mapM,mapP,mapB = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP


function impose_BCs_inviscid_Ub!(QP,Qf,leftwall,rightwall)
    # # No-slip at walls
    # u_1 = Qf[2][boundary]
    # u_2 = Qf[3][boundary]
    # n_1 = nx_b
    # n_2 = ny_b

    # Un = @. u_1*n_1 + u_2*n_2
    # Ut = @. u_1*n_2 - u_2*n_1

    # # ρ^+ = ρ, p^+ = p (beta^+ = beta)
    # @. QP[1][boundary] = Qf[1][boundary]
    # @. QP[4][boundary] = Qf[4][boundary]

    # # u_n^+ = -u_n, u_t^+ = u_t
    # @. QP[2][boundary] = 1/(-n_1^2-n_2^2)*(n_1*Un-n_2*Ut)
    # @. QP[3][boundary] = 1/(-n_1^2-n_2^2)*(n_2*Un+n_1*Ut)


    # QP[1][leftwall] .= rhoL
    # QP[2][leftwall] .= uL
    # QP[3][leftwall] .= vL
    # QP[4][leftwall] .= rhoL/(2*pL)
    #
    # QP[1][rightwall] .= Qf[1][rightwall]
    # QP[2][rightwall] .= Qf[2][rightwall]
    # QP[3][rightwall] .= Qf[3][rightwall]
    # QP[4][rightwall] .= Qf[4][rightwall]

    for li = leftwall
        QP[1][li] = rhoL
        QP[2][li] = uL
        QP[3][li] = vL
        QP[4][li] = rhoL/(2*pL)
    end

    for ri = rightwall
        QP[1][ri] = Qf[1][ri]
        QP[2][ri] = Qf[2][ri]
        QP[3][ri] = Qf[3][ri]
        QP[4][ri] = Qf[4][ri]
    end
end

function impose_BCs_inviscid_U!(QP,Qf,leftwall,rightwall)
    # QP[1][leftwall] .= rhoL
    # QP[2][leftwall] .= rhoL*uL
    # QP[3][leftwall] .= rhoL*vL
    # QP[4][leftwall] .= EL
    #
    # QP[1][rightwall] .= Qf[1][rightwall]
    # QP[2][rightwall] .= Qf[2][rightwall]
    # QP[3][rightwall] .= Qf[3][rightwall]
    # QP[4][rightwall] .= Qf[4][rightwall]

    for li = leftwall
        QP[1][li] = rhoL
        QP[2][li] = rhoL*uL
        QP[3][li] = rhoL*vL
        QP[4][li] = EL
    end

    for ri = rightwall
        QP[1][ri] = Qf[1][ri]
        QP[2][ri] = Qf[2][ri]
        QP[3][ri] = Qf[3][ri]
        QP[4][ri] = Qf[4][ri]
    end
end

function impose_BCs_flux!(flux_x_P,flux_y_P,leftwall,rightwall)
    # flux_x_P[1][leftwall] .= rhoL*uL
    # flux_x_P[2][leftwall] .= rhoL*uL^2+pL
    # flux_x_P[3][leftwall] .= rhoL*uL*vL
    # flux_x_P[4][leftwall] .= uL*(EL+pL)
    #
    # flux_y_P[1][leftwall] .= rhoL*vL
    # flux_y_P[2][leftwall] .= rhoL*uL*vL
    # flux_y_P[3][leftwall] .= rhoL*vL^2+pL
    # flux_y_P[4][leftwall] .= vL*(EL+pL)
    #
    # flux_x_P[1][rightwall] .= rhoR*uR
    # flux_x_P[2][rightwall] .= rhoR*uR^2+pR
    # flux_x_P[3][rightwall] .= rhoR*uR*vR
    # flux_x_P[4][rightwall] .= uR*(ER+pR)
    #
    # flux_y_P[1][rightwall] .= rhoR*vR
    # flux_y_P[2][rightwall] .= rhoR*uR*vR
    # flux_y_P[3][rightwall] .= rhoR*vR^2+pR
    # flux_y_P[4][rightwall] .= vR*(ER+pR)

    for li = leftwall
        flux_x_P[1][li] = rhoL*uL
        flux_x_P[2][li] = rhoL*uL^2+pL
        flux_x_P[3][li] = rhoL*uL*vL
        flux_x_P[4][li] = uL*(EL+pL)

        flux_y_P[1][li] = rhoL*vL
        flux_y_P[2][li] = rhoL*uL*vL
        flux_y_P[3][li] = rhoL*vL^2+pL
        flux_y_P[4][li] = vL*(EL+pL)
    end

    for ri = rightwall
        flux_x_P[1][ri] = rhoR*uR
        flux_x_P[2][ri] = rhoR*uR^2+pR
        flux_x_P[3][ri] = rhoR*uR*vR
        flux_x_P[4][ri] = uR*(ER+pR)

        flux_y_P[1][ri] = rhoR*vR
        flux_y_P[2][ri] = rhoR*uR*vR
        flux_y_P[3][ri] = rhoR*vR^2+pR
        flux_y_P[4][ri] = vR*(ER+pR)
    end
end

function impose_BCs_lam!(lamP,lam,leftwall,rightwall)
    # lam[leftwall]   .= 0.0
    # lam[rightwall]  .= 0.0
    # lamP[leftwall]  .= 0.0
    # lamP[rightwall] .= 0.0

    for li = leftwall
        lam[li]  = 0.0
        lamP[li] = 0.0
    end

    for ri = leftwall
        lam[ri]  = 0.0
        lamP[ri] = 0.0
    end
end

function impose_BCs_entropyvars!(VUP,VUf,leftwall,rightwall)
    # # adiabatic on wall
    # @. VUP[2][wall] = -VUf[2][wall]
    # @. VUP[3][wall] = -VUf[3][wall]
    # @. VUP[4][wall] =  VUf[4][wall]

    # # Reflective on top
    # v_1 = VUf[2][topwall]
    # v_2 = VUf[3][topwall]
    # n_1 = nx_top
    # n_2 = ny_top

    # VUn = @. v_1*n_1 + v_2*n_2
    # VUt = @. v_1*n_2 - v_2*n_1

    # # v_4^+ = v_4
    # @. VUP[4][topwall] = VUf[4][topwall]
    # # v_n^+ = -v_n, v_t^+ = v_t
    # @. VUP[2][topwall] = VUf[2][topwall] - 2*VUn*n_1
    # @. VUP[3][topwall] = VUf[3][topwall] - 2*VUn*n_2

    VL = v_ufun(rhoL,rhoL*uL,rhoL*vL,EL)
    VR = v_ufun(rhoR,rhoR*uR,rhoR*vR,ER)

    # VUP[1][leftwall] .= VL[1]
    # VUP[2][leftwall] .= VL[2]
    # VUP[3][leftwall] .= VL[3]
    # VUP[4][leftwall] .= VL[4]
    #
    # VUP[1][rightwall] .= VUf[1][rightwall]
    # VUP[2][rightwall] .= VUf[2][rightwall]
    # VUP[3][rightwall] .= VUf[3][rightwall]
    # VUP[4][rightwall] .= VUf[4][rightwall]

    for li = leftwall
        VUP[1][li] = VL[1]
        VUP[2][li] = VL[2]
        VUP[3][li] = VL[3]
        VUP[4][li] = VL[4]
    end

    for ri = rightwall
        VUP[1][ri] = VUf[1][ri]
        VUP[2][ri] = VUf[2][ri]
        VUP[3][ri] = VUf[3][ri]
        VUP[4][ri] = VUf[4][ri]
    end
end

function impose_BCs_stress!(σxP,σyP,σxf,σyf,vwall)
    # # Adiabatic no-slip BC
    # @. σxP[2][wall] = σxf[2][wall]
    # @. σyP[2][wall] = σyf[2][wall]
    # @. σxP[3][wall] = σxf[3][wall]
    # @. σyP[3][wall] = σyf[3][wall]
    # @. σxP[4][wall] = -σxf[4][wall]
    # @. σyP[4][wall] = -σyf[4][wall]

    # # Reflective on top
    # sigma_x_1 = σxf[2][topwall]
    # sigma_x_2 = σxf[3][topwall]
    # sigma_y_1 = σyf[2][topwall]
    # sigma_y_2 = σyf[3][topwall]
    # n_1 = nx_top
    # n_2 = ny_top

    # σn_x = @. sigma_x_1*n_1 + sigma_x_2*n_2
    # σn_y = @. sigma_y_1*n_1 + sigma_y_2*n_2

    # @. σxP[2][topwall] = -σxf[2][topwall]+2*n_1*σn_x
    # @. σyP[2][topwall] = -σyf[2][topwall]+2*n_1*σn_y
    # @. σxP[3][topwall] = -σxf[3][topwall]+2*n_2*σn_x
    # @. σyP[3][topwall] = -σyf[3][topwall]+2*n_2*σn_y
    # @. σxP[4][topwall] = -σxf[4][topwall]
    # @. σyP[4][topwall] = -σyf[4][topwall]

    @. σxP[1][vwall] = σxf[1][vwall]
    @. σyP[1][vwall] = σyf[1][vwall]
    @. σxP[2][vwall] = σxf[2][vwall]
    @. σyP[2][vwall] = σyf[2][vwall]
    @. σxP[3][vwall] = σxf[3][vwall]
    @. σyP[3][vwall] = σyf[3][vwall]
    @. σxP[4][vwall] = σxf[4][vwall]
    @. σyP[4][vwall] = σyf[4][vwall]

    for vi = vwall
        σxP[1][vi] = σxf[1][vi]
        σyP[1][vi] = σyf[1][vi]
        σxP[2][vi] = σxf[2][vi]
        σyP[2][vi] = σyf[2][vi]
        σxP[3][vi] = σxf[3][vi]
        σyP[3][vi] = σyf[3][vi]
        σxP[4][vi] = σxf[4][vi]
        σyP[4][vi] = σyf[4][vi]
    end
end


function init_visc_fxn(λ,μ,Pr)
    let λ=-λ,μ=μ,Pr=Pr
        function viscous_matrices!(Kxx,Kxy,Kyy,v)
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
        return viscous_matrices!
    end
end

viscous_matrices! = init_visc_fxn(lambda,mu,Pr)



# # Initial condition 2D shocktube
# rho_init(x) = (x < 0.5) ? rhoL : rhoR
# p_init(x)   = (x < 0.5) ? pL   : pR

# rho = @. rho_init(x)
# u   = zeros(size(x))
# v   = zeros(size(x))
# p   = @. p_init(x)

# U = primitive_to_conservative_hardcode.(rho,u,v,p)
# rho = [x[1] for x in U]
# rhou = [x[2] for x in U]
# rhov = [x[3] for x in U]
# E = [x[4] for x in U]
# U = (rho,rhou,rhov,E)

# Initial condition Becker's viscous shocktube
function bisection_solve_velocity(x,max_iter,tol)
    v_L = v_1
    v_R = v_0
    num_iter = 0

    L_k = kappa/m_0/cv
    f(v) = -x+2*L_k/(γ+1)*(v_0/(v_0-v_1)*log((v_0-v)/(v_0-v_01))-v_1/(v_0-v_1)*log((v-v_1)/(v_01-v_1)))

    v_new = (v_L+v_R)/2
    while num_iter < max_iter
        v_new = (v_L+v_R)/2

        if abs(f(v_new)) < tol
            return v_new
        elseif sign(f(v_L)) == sign(f(v_new))
            v_L = v_new
        else
            v_R = v_new
        end
        num_iter += 1
    end

    return v_new
end

const max_iter = 100
const tol = 1e-14

function exact_sol_viscous_shocktube(x,t)
    u   = bisection_solve_velocity(x-v_inf*t,max_iter,tol)
    rho = m_0/u
    e   = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-u^2)
    return rho, rho*(v_inf+u), 0.0, rho*(e+1/2*(v_inf+u)^2)
end

U = exact_sol_viscous_shocktube.(x,0.0)
U = ([x[1] for x in U], [x[2] for x in U], [x[3] for x in U], [x[4] for x in U])
# U = @. Riemann_2D(x,y)
# rho,u,v,p = [(x->x[i]).(U) for i = 1:Nc]
# U = @. primitive_to_conservative_hardcode(rho,u,v,p)
# rho,rhou,rhov,E = [(x->x[i]).(U) for i = 1:Nc]
# U = (rho,rhou,rhov,E)

Np = (N+1)*(N+1)
face_idx = [1:N+1; (N+1):(N+1):Np; Np:-1:Np-N; Np-N:-(N+1):1]
x_idx    = [(N+2):(2*N+2); (3*N+4):(4*N+4)]
y_idx    = [1:(N+1); (2*N+3):(3*N+3)]
S0r1 = sum(S0r,dims=2)
S0s1 = sum(S0s,dims=2)
K      = size(U[1],2)
Np     = (N+1)*(N+1)
Nfaces = 4
Nfp    = Nfaces*(N+1)

# Preallocation
p      = zero(U[1])
VU     = zero.(U)
flux_x = zero.(U)
flux_y = zero.(U)
lam    = zeros(Nfp,K)
LFc    = zeros(Nfp,K)
Ub     = zero.(U)
sigma_x = zero.(U)
sigma_y = zero.(U)


function zhang_wavespd(rhoL,rhouL,rhovL,EL,sigma2L,sigma3L,sigma4L,pL)
    uL   = rhouL/rhoL
    eL   = (EL - .5*rhoL*uL^2)/rhoL
    tauL = sigma2L
    qL   = uL*tauL-sigma3L
    wavespdL = abs(uL)+1/(2*rhoL^2*eL)*(sqrt(rhoL^2*qL^2+2*rhoL^2*eL*abs(tauL-pL)^2)+rhoL*abs(qL))
    return wavespdL
end

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


function rhs_IDP!(U,N,K1D,Minv,Vf,Dr,Ds,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,LIFT,mapP,face_idx,x_idx,y_idx,leftwall,rightwall,vwall,dt,p,VU,flux_x,flux_y,lam,LFc,Ub,sigma_x,sigma_y)
    # TODO: hardcoded!
    K      = size(U[1],2)
    Np     = (N+1)*(N+1)
    Nfaces = 4
    Nfp    = Nfaces*(N+1)

    # J = (1/K1D)^2 # hardcoded!!
    # Jf = 1/K1D
    # rxJ = 1/K1D
    # syJ = 1/K1D
    # sJ = 1/K1D

    J   = (1/K1D/2)^2
    Jf  = 1/K1D/2
    rxJ = 1/K1D/2
    syJ = 1/K1D/2
    sJ  = 1/K1D/2

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

    Uf = (x->Vf*x).(U)
    UP = (x->x[mapP]).(Uf)
    impose_BCs_inviscid_U!(UP,Uf,leftwall,rightwall)

    # simple lax friedrichs dissipation
    (rhoM,rhouM,rhovM,EM) = Uf
    rhoUM_n = @. (rhouM*nxJ + rhovM*nyJ)/sJ
    @. lam  = abs(sqrt(abs(rhoUM_n/rhoM))+sqrt(γ*(γ-1)*(EM-.5*rhoUM_n^2/rhoM)/rhoM))
    lamP = lam[mapP]
    impose_BCs_lam!(lamP,lam,leftwall,rightwall)
    @. LFc = max(lam,lamP)*sJ

    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[3]/U[1]
    @. Ub[4] = U[1]/(2*p)
    Ubf = (x->Vf*x).(Ub)
    UbP = (x->x[mapP]).(Ubf)
    impose_BCs_inviscid_Ub!(UbP,Ubf,leftwall,rightwall)

    flux_x_f = (x->Vf*x).(flux_x)
    flux_x_P = (x->x[mapP]).(flux_x_f)
    flux_y_f = (x->Vf*x).(flux_y)
    flux_y_P = (x->x[mapP]).(flux_y_f)
    impose_BCs_flux!(flux_x_P,flux_y_P,leftwall,rightwall)

    U_low    = [zeros(Float64,Np) for i = 1:Nc]
    U_low_i  = zeros(Float64,Nc)
    F_low    = [zeros(Float64,Np,Np) for i = 1:Nc]
    F_high   = [zeros(Float64,Np,Np) for i = 1:Nc]
    F_P      = [zeros(Float64,Nfp) for i = 1:Nc]
    rhsU  = [zeros(Float64,Np,K) for i = 1:Nc]
    L     = zeros(Float64,Np,Np)
    P_ij  = zeros(Float64,Nc)

    # =======================
    # Calculate viscous part
    # =======================
    # VU  = v_ufun(U...)
    # VUf = (x->Vf*x).(VU)
    # VUP = (x->x[mapP]).(VUf)
    vector_norm(U) = sum((x->x.^2).(U))
    @. VU[3] = U[2]^2+U[3]^2 # rhoUnorm
    @. VU[4] = U[4]-.5*VU[3]/U[1] # rhoe
    @. VU[1] = log((γ-1)*VU[4]/(U[1]^γ)) # sU
    @. VU[1] = (-U[4]+VU[4]*(γ+1-VU[1]))/VU[4]
    @. VU[2] = U[2]/VU[4]
    @. VU[3] = U[3]/VU[4]
    @. VU[4] = -U[1]/VU[4]
    VUf = (x->Vf*x).(VU)
    VUP = (x->x[mapP]).(VUf)
    impose_BCs_entropyvars!(VUP,VUf,leftwall,rightwall)

    surfx(uP,uf) = LIFT*(@. .5*(uP-uf)*nxJ)
    surfy(uP,uf) = LIFT*(@. .5*(uP-uf)*nyJ)
    VUx = rxJ.*(x->Dr*x).(VU) .+ surfx.(VUP,VUf)
    VUy = syJ.*(x->Ds*x).(VU) .+ surfy.(VUP,VUf)
    # VUx = (x->x./J).(VUx)
    # VUy = (x->x./J).(VUy)
    for c = 1:Nc
        @. VUx[c] = VUx[c]/J
        @. VUy[c] = VUy[c]/J
    end

    # initialize sigma_x,sigma_y = viscous rhs
    Kxx,Kxy,Kyy = ntuple(x->MMatrix{4,4}(zeros(Nc,Nc)),3)
    sigma_x_e = zero.(getindex.(VU,:,1))
    sigma_y_e = zero.(getindex.(VU,:,1))
    for e = 1:K
        fill!.(sigma_x_e,0.0)
        fill!.(sigma_y_e,0.0)

        # mult by matrices and perform local projections
        for i = 1:Np
            vxi = getindex.(VUx,i,e)
            vyi = getindex.(VUy,i,e)
            viscous_matrices!(Kxx,Kxy,Kyy,getindex.(VU,i,e))

            for col = 1:Nc
                vxi_col = vxi[col]
                vyi_col = vyi[col]
                for row = 1:Nc
                    sigma_x_e[row][i] += Kxx[row,col]*vxi_col + Kxy[row,col]*vyi_col
                    sigma_y_e[row][i] += Kxy[col,row]*vxi_col + Kyy[row,col]*vyi_col
                end
            end
        end
        setindex!.(sigma_x,sigma_x_e,:,e)
        setindex!.(sigma_y,sigma_y_e,:,e)
    end

    sxf = (x->Vf*x).(sigma_x)
    syf = (x->Vf*x).(sigma_y)
    sxP = (x->x[mapP]).(sxf)
    syP = (x->x[mapP]).(syf)
    impose_BCs_stress!(sxP,syP,sxf,syf,vwall)

    # =====================
    # Loop through elements
    # =====================
    for k = 1:K
        for c = 1:Nc
            F_low[c]  .= 0.0
            F_high[c] .= 0.0
            F_P[c]    .= 0.0
            U_low[c]  .= 0.0
        end
        L .= 0.0

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
                        F_low[c][i,j] = (rxJ*S0r[i,j]*(flux_x[c][i,k]+flux_x[c][j,k])
                                        +syJ*S0s[i,j]*(flux_y[c][i,k]+flux_y[c][j,k])
                                        -rxJ*S0r[i,j]*(sigma_x[c][i,k]+sigma_x[c][j,k])
                                        -syJ*S0s[i,j]*(sigma_y[c][i,k]+sigma_y[c][j,k])
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
                        val = (2*rxJ*Sr[i,j]*F1[c]+2*syJ*Ss[i,j]*F2[c]
                                -rxJ*S0r[i,j]*(sigma_x[c][i,k]+sigma_x[c][j,k])
                                -syJ*S0s[i,j]*(sigma_y[c][i,k]+sigma_y[c][j,k]))
                        F_high[c][i,j] = val
                        F_high[c][j,i] = -val
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
                    F_P[c][i] = (Jf*S0r_ij*(flux_x_f[c][i,k]+flux_x_P[c][i,k])
                                -Jf*S0r_ij*(sxf[c][i,k]+sxP[c][i,k])
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
                    F_P[c][i] = (Jf*S0s_ij*(flux_y_f[c][i,k]+flux_y_P[c][i,k])
                                -Jf*S0s_ij*(syf[c][i,k]+syP[c][i,k])
                                -LFc[i,k]*abs(S0s_ij)*(UP[c][i,k]-Uf[c][i,k]))
                end
            end
        end

        # Calculate low order solution
        for i = 1:Np
            for j = 1:Np
                for c = 1:Nc
                    U_low[c][i] += F_low[c][i,j]
                end
            end
        end
        for i = 1:Nfp
            for c = 1:Nc
                U_low[c][face_idx[i]] += F_P[c][i]
            end
        end
        for c = 1:Nc
            for i = 1:Np
                U_low[c][i] = U[c][i,k]-dt*Minv[i,i]/J*U_low[c][i]
            end
        end

        # Calculate limiting parameters
        for i = 1:Np
            m_i = J/Minv[i,i]
            lambda_j = 1/(Np-1)
            for c = 1:Nc
                U_low_i[c] = U_low[c][i]
            end
            for j = 1:Np
                if i != j
                    for c = 1:Nc
                        P_ij[c] = dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
                    end
                    L[i,j] = limiting_param(U_low_i, P_ij)
                end
            end
        end

        # Symmetrize limiting parameters
        for i = 1:Np
            for j = i+1:Np
                l = min(L[i,j],L[j,i])
                L[i,j] = l
                L[j,i] = l
            end
        end

        # L = ones(size(L))

        for c = 1:Nc
            for i = 1:Np
                for j = 1:Np
                    rhsU[c][i,k] += (L[i,j]-1)*F_low[c][i,j]-L[i,j]*F_high[c][i,j]
                end
            end
            for i = 1:Nfp
                rhsU[c][face_idx[i],k] -= F_P[c][i]
            end
        end
    end

    for c = 1:Nc
        rhsU[c] .= 1/J*Minv*rhsU[c]
    end

    return rhsU
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

# dt = 1e-4
# rhs_IDP!(U,N,K1D,Minv,Vf,Dr,Ds,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,LIFT,mapP,face_idx,x_idx,y_idx,leftwall,rightwall,vwall,dt,p,VU,flux_x,flux_y,lam,LFc,Ub,sigma_x,sigma_y);
# @btime rhs_IDP!(U,N,K1D,Minv,Vf,Dr,Ds,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,LIFT,mapP,face_idx,x_idx,y_idx,leftwall,rightwall,vwall,dt,p,VU,flux_x,flux_y,lam,LFc,Ub,sigma_x,sigma_y);
# @profiler rhs_IDP!(U,N,K1D,Minv,Vf,Dr,Ds,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,LIFT,mapP,face_idx,x_idx,y_idx,leftwall,rightwall,vwall,dt,p,VU,flux_x,flux_y,lam,LFc,Ub,sigma_x,sigma_y);

dt_hist = []
anim = Animation()
i = 1

while t < T
    # SSPRK(3,3)
    dt = min(1e-4,T-t)
    rhsU = rhs_IDP!(U,N,K1D,Minv,Vf,Dr,Ds,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,LIFT,mapP,face_idx,x_idx,y_idx,leftwall,rightwall,vwall,dt,p,VU,flux_x,flux_y,lam,LFc,Ub,sigma_x,sigma_y);
    @. resW = U + dt*rhsU
    rhsU = rhs_IDP!(U,N,K1D,Minv,Vf,Dr,Ds,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,LIFT,mapP,face_idx,x_idx,y_idx,leftwall,rightwall,vwall,dt,p,VU,flux_x,flux_y,lam,LFc,Ub,sigma_x,sigma_y);
    @. resZ = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resZ
    rhsU = rhs_IDP!(U,N,K1D,Minv,Vf,Dr,Ds,nxJ,nyJ,Sr,Ss,S0r,S0s,S0r1,S0s1,LIFT,mapP,face_idx,x_idx,y_idx,leftwall,rightwall,vwall,dt,p,VU,flux_x,flux_y,lam,LFc,Ub,sigma_x,sigma_y);
    @. resZ = resW+dt*rhsU
    @. U = 1/3*U+2/3*resZ

    push!(dt_hist,dt)
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T, at step $i")
    global i = i + 1
    # if mod(i,50) == 1
    #     scatter(Vp*x,Vp*y,Vp*U[1],zcolor=Vp*U[1],camera=(0,90))
    #     frame(anim)
    # end
end

################
### Plotting ###
################

#plotting nodes
@unpack VDM = rd
rp,sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/VDM
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)
#vv = Vp*(@. (Q[2]/Q[1])^2 + (Q[3]/Q[1])^2)


exact_U = @. exact_sol_viscous_shocktube.(x,T)
exact_rho = [x[1] for x in exact_U]
exact_rhou = [x[2] for x in exact_U]
exact_rhov = [x[3] for x in exact_U]
exact_E = [x[4] for x in exact_U]

rho = U[1]
rhou = U[2]
rhov = U[3]
E = U[4]

exact_rho_p = Vp*exact_rho

Linferr = maximum(abs.(exact_rho-rho))/maximum(abs.(rho)) +
          maximum(abs.(exact_rhou-rhou))/maximum(abs.(rhou)) +
          #maximum(abs.(exact_rhov-rhov))/maximum(abs.(rhov).+1e-14) +
          maximum(abs.(exact_E-E))/maximum(abs.(E))

J = J[1] # TODO: assume uniform mesh
L1err = sum(J.*abs.(exact_rho-rho))/sum(J.*abs.(rho)) +
        sum(J.*abs.(exact_rhou-rhou))/sum(J.*abs.(rhou)) +
        #sum(J.*abs.(exact_rhov-rhov))/sum(J.*abs.(rhov).+1e-14) +
        sum(J.*abs.(exact_E-E))/sum(J.*abs.(E))
println("N = $N, K = $K")
println("L1 error is $L1err")
println("Linf error is $Linferr")

scatter(x,y,U[1],zcolor=U[1],camera=(0,90),colorbar=:right)
xp = Vp*x
yp = Vp*y
vv = Vp*U[1]
scatter(xp,yp,vv,zcolor=vv,camera=(0,90),colorbar=:right)
#scatter(xp,yp,exact_rho_p,zcolor=exact_rho_p,camera=(0,90),colorbar=:right)
