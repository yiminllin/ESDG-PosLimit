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
using Basis1D
using CommonUtils
# using Basis2DTri
# using UniformTriMesh
# using NodesAndModes
# using NodesAndModes.Tri
# using CommonUtils
# using Basis1D
using Basis2DQuad
using UniformQuadMesh



using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
include("../../EntropyStableEuler.jl/src/logmean.jl")
include("../SBP_quad_data.jl")
using EntropyStableEuler
using EntropyStableEuler.Fluxes2D


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

function build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)
    # [-1,1,0], [-1,-1,sqrt(4/3)]
    equilateral_map(r,s) = (@. .5*(2*r+1*s+1), @. sqrt(3)*(1+s)/2 - 1/sqrt(3) )
    req,seq = equilateral_map(rq,sq)
    ref,sef = equilateral_map(rf,sf)
    barycentric_coords(r,s) = ((@. (1+r)/2), (@. (1+s)/2), (@. -(r+s)/2))
    λ1,λ2,λ3 = barycentric_coords(rq,sq)
    λ1f,λ2f,λ3f = barycentric_coords(rf,sf)

    Br = diagm(nrJ.*wf)
    Bs = diagm(nsJ.*wf)

    # build extrapolation matrix
    E = zeros(length(rf),length(rq))
    for i = 1:length(rf)
        # d = @. (λ1 - λ1f[i])^2 + (λ2 - λ2f[i])^2 + (λ3 - λ3f[i])^2
        d2 = @. (req-ref[i])^2 + (seq-sef[i])^2
        p = sortperm(d2)
        h2 = (wf[i]/sum(wf))*2/pi # set so that h = radius of circle with area w_i = face weight
        nnbrs = min(4,max(3,count(d2[p] .< h2))) # find 3 closest points
        p = p[1:nnbrs]
        Ei = vandermonde_2D(1,[rf[i]],[sf[i]])/vandermonde_2D(1,rq[p],sq[p])
        E[i,p] = Ei
    end
    E = Matrix(droptol!(sparse(E),1e-13))

    # build stencil
    A = spzeros(length(req),length(req))
    for i = 1:length(req)
        d2 = @. (req-req[i])^2 + (seq-seq[i])^2
        p = sortperm(d2)

        # h^2 = wq[i]/pi = radius of circle with area wq[i]
        # h2 =     (sqrt(3)/sum(wq))*wq[i]/pi
        h2 = α^2*(sqrt(3)/sum(wq))*wq[i]/pi

        nnbrs = count(d2[p] .< h2)
        nbrs = p[1:nnbrs]
        A[i,nbrs] .= one(eltype(A))
    end
    A = (A+A')
    A.nzval .= one(eltype(A)) # bool-ish

    # build graph Laplacian
    L1 = (A-diagm(diag(A))) # ignore
    L1 -= diagm(vec(sum(L1,dims=2)))

    b1r = -sum(.5*E'*Br*E,dims=2)
    b1s = -sum(.5*E'*Bs*E,dims=2)
    ψ1r = pinv(L1)*b1r
    ψ1s = pinv(L1)*b1s

    function fillQ(adj,ψ)
        Np = length(ψ)
        S = zeros(Np,Np)
        for i = 1:Np
            for j = 1:Np
                if adj[i,j] != 0
                        S[i,j] += (ψ[j]-ψ[i])
                end
            end
        end
        return S
    end

    S1r,S1s = fillQ.((A,A),(ψ1r,ψ1s))
    Qr = Matrix(droptol!(sparse(S1r + .5*E'*Br*E),1e-14))
    Qs = Matrix(droptol!(sparse(S1s + .5*E'*Bs*E),1e-14))

    return Qr,Qs,E,Br,Bs,A
end

function init_reference_tri_sbp_GQ(N, qnode_choice)
    include("SBP_quad_data.jl")
    # initialize a new reference element data struct
    rd = RefElemData()

    fv = tri_face_vertices() # set faces for triangle
    Nfaces = length(fv)
    @pack! rd = fv, Nfaces

    # Construct matrices on reference elements
    r, s = nodes(Tri(),N)
    VDM = vandermonde(Tri(),N, r, s)
    Vr, Vs = grad_vandermonde(Tri(),N, r, s)
    Dr = Vr/VDM
    Ds = Vs/VDM
    @pack! rd = r,s,VDM,Dr,Ds

    # low order interpolation nodes
    r1,s1 = nodes(Tri(),1)
    V1 = vandermonde(Tri(),1,r,s)/vandermonde(Tri(),1,r1,s1)
    @pack! rd = V1

    #Nodes on faces, and face node coordinate
    if qnode_choice == "GQ"
        r1D, w1D = gauss_quad(0,0,N)
    elseif qnode_choice == "GL" || qnode_choice == "tri_diage"
        r1D, w1D = gauss_lobatto_quad(0,0,N+1)
    end
    Nfp = length(r1D) # number of points per face
    e = ones(Nfp) # vector of all ones
    z = zeros(Nfp) # vector of all zeros
    rf = [r1D; -r1D; -e];
    sf = [-e; r1D; -r1D];
    wf = vec(repeat(w1D,3,1));
    nrJ = [z; e; -e]
    nsJ = [-e; e; z]
    @pack! rd = rf,sf,wf,nrJ,nsJ

    if qnode_choice == "GQ"
        rq,sq,wq = GQ_SBP[N];
    elseif qnode_choice == "GL"
        rq,sq,wq = GL_SBP[N];
    elseif qnode_choice == "tri_diage"
        rq,sq,wq = Tri_diage[N];
    end
    # rq,sq,wq = GQ_SBP[N]
    # rq,sq,wq = GL_SBP[N]
    # rq,sq,wq = Tri_diage[N]
    Vq = vandermonde(Tri(),N,rq,sq)/VDM
    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))
    @pack! rd = rq,sq,wq,Vq,M,Pq

    Vf = vandermonde(Tri(),N,rf,sf)/VDM # interpolates from nodes to face nodes
    LIFT = M\(Vf'*diagm(wf)) # lift matrix used in rhs evaluation
    @pack! rd = Vf,LIFT

    # plotting nodes
    rp, sp = equi_nodes(Tri(),10)
    Vp = vandermonde(Tri(),N,rp,sp)/VDM
    @pack! rd = rp,sp,Vp

    return rd
end

const TOL = 5e-16
const POSTOL = 1e-14
const WALLPT = 1.0/6.0
const Nc = 4 # number of components
"Approximation parameters"
const N = 2
const K1D = 10
const T = 0.5#1e-4
const dt0 = 1e-3
const XLENGTH = 1.5#2.0
const CFL = 1.0
const NUM_THREADS = Threads.nthreads()

# Initial condition 2D shocktube
const γ = 1.4


"Mesh related variables"
# VX, VY, EToV = uniform_quad_mesh(Int(round(XLENGTH*K1D)),K1D)
# @. VX = (VX+1)/2*XLENGTH*5
# @. VY = (VY+1)/2*5
Kx = Int(round(XLENGTH*K1D))
Ky = 2*K1D
VX, VY, EToV = uniform_quad_mesh(Kx,Ky)
# @. VX = (VX+1)/2*XLENGTH*5
# @. VY = (VY+1)/2*5
@. VX = 15*(1+VX)/2
@. VY = 5*VY

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
Br_halved = -sum(S0r,dims=2)
Bs_halved = -sum(S0s,dims=2)

@unpack Vf,Dr,Ds,LIFT,nrJ,nsJ,wf = rd
md = init_mesh((VX,VY),EToV,rd)
@unpack xf,yf,mapM,mapP,mapB,nxJ,nyJ,x,y = md
xb,yb = (x->x[mapB]).((xf,yf))

const K  = size(x,2)
const Nfaces = 4

# Make domain periodic
@unpack Vf = rd
@unpack xf,yf,mapM,mapP,mapB,rxJ,ryJ,sxJ,syJ,J,sJ,nxJ,nyJ = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

xq = x
yq = y

nx = nxJ ./ sJ
ny = nyJ ./ sJ

E = Vf
Fmask = zeros(Int64,size(Vf,1))
for i = 1:size(Vf,1)
    tmparr = findall(abs.(E[i,:] .- 1) .< 1e-12)
    Fmask[i] = tmparr[1]
end



@inline function get_infoP(mapP,Fmask,i,k)
    gP = mapP[i,k]           # exterior global face node number
    kP = fld1(gP,Nfp)        # exterior element number
    iP = Fmask[mod1(gP,Nfp)] # exterior node number
    return iP,kP
end

function rhs_ESDG!(U,rhsU,ops,geom)
    Sr,Ss,S0r,S0s,Minv,Br_halved,Bs_halved,E = ops
    mapP,Fmask,xq,yq,rxJ,ryJ,sxJ,syJ,J,sJ,nx,ny,nrJ,nsJ,wf = geom

    FHx = zeros(Nc,Np,Np)
    FHy = zeros(Nc,Np,Np)
    FHPx = zeros(Nc,Nfp)
    FHPy = zeros(Nc,Nfp)
    Sx  = zeros(Np,Np)
    Sy  = zeros(Np,Np)
    EBx  = zeros(Np,Np)
    EBy  = zeros(Np,Np)
    rhsvec = zeros(Nc,Np)

    Br = diagm(wf.*nrJ)
    Bs = diagm(wf.*nsJ)
    for k = 1:K
        for j = 1:Np
            for i = 1:Np
                rhoM     = U[1,i,k]
                rhouM    = U[2,i,k]
                rhovM    = U[3,i,k]
                EM       = U[4,i,k]
                uM       = rhouM/rhoM
                vM       = rhovM/rhoM
                pM       = pfun(rhoM,rhouM,rhovM,EM)
                betaM    = rhoM/(2*pM)
                rhologM  = log(rhoM)
                betalogM = log(betaM)
                rhoP     = U[1,j,k]
                rhouP    = U[2,j,k]
                rhovP    = U[3,j,k]
                EP       = U[4,j,k]
                uP       = rhouP/rhoP
                vP       = rhovP/rhoP
                pP       = pfun(rhoP,rhouP,rhovP,EP)
                betaP    = rhoP/(2*pP)
                rhologP  = log(rhoP)
                betalogP = log(betaP)

                Fx1,Fx2,Fx3,Fx4,Fy1,Fy2,Fy3,Fy4 = euler_fluxes_2D(rhoM,uM,vM,betaM,rhologM,betalogM,
                                                                  rhoP,uP,vP,betaP,rhologP,betalogP)
                                                
                FHx[1,i,j] = Fx1 
                FHx[2,i,j] = Fx2 
                FHx[3,i,j] = Fx3 
                FHx[4,i,j] = Fx4 
                FHy[1,i,j] = Fy1 
                FHy[2,i,j] = Fy2 
                FHy[3,i,j] = Fy3 
                FHy[4,i,j] = Fy4
            end
        end

        for i = 1:Nfp
            iM       = Fmask[i]
            rhoM     = U[1,iM,k]
            rhouM    = U[2,iM,k]
            rhovM    = U[3,iM,k]
            EM       = U[4,iM,k]
            uM       = rhouM/rhoM
            vM       = rhovM/rhoM
            pM       = pfun(rhoM,rhouM,rhovM,EM)
            betaM    = rhoM/(2*pM)
            rhologM  = log(rhoM)
            betalogM = log(betaM)

            iP,kP = get_infoP(mapP,Fmask,i,k)
            rhoP     = U[1,iP,kP]
            rhouP    = U[2,iP,kP]
            rhovP    = U[3,iP,kP]
            EP       = U[4,iP,kP]
            uP       = rhouP/rhoP
            vP       = rhovP/rhoP
            pP       = pfun(rhoP,rhouP,rhovP,EP)
            betaP    = rhoP/(2*pP)
            rhologP  = log(rhoP)
            betalogP = log(betaP)

            Fx1,Fx2,Fx3,Fx4,Fy1,Fy2,Fy3,Fy4 = euler_fluxes_2D(rhoM,uM,vM,betaM,rhologM,betalogM,
                                                              rhoP,uP,vP,betaP,rhologP,betalogP)
            
            FHPx[1,i] = Fx1
            FHPx[2,i] = Fx2
            FHPx[3,i] = Fx3
            FHPx[4,i] = Fx4
            FHPy[1,i] = Fy1
            FHPy[2,i] = Fy2
            FHPy[3,i] = Fy3
            FHPy[4,i] = Fy4
        end
        
        Sx  =     rxJ[1,k]*Sr + sxJ[1,k]*Ss
        Sy  =     ryJ[1,k]*Sr + syJ[1,k]*Ss
        EBx = E'*(rxJ[1,k]*Br + sxJ[1,k]*Bs)
        EBy = E'*(ryJ[1,k]*Br + syJ[1,k]*Bs)

        for c = 1:Nc
            rhsvec[c,:] = 2*(Sx .* FHx[c,:,:])*ones(Np) + 2*(Sy .* FHy[c,:,:])*ones(Np) + EBx*FHPx[c,:] + EBy*FHPy[c,:]
            rhsvec[c,:] = -(1/J[1,k]*Minv)*rhsvec[c,:]
        end

        for c = 1:Nc
            rhsU[c,:,k] = rhsvec[c,:]
        end
    end
end




# Initial condition 2D shocktube
# at_left(x,y) = y-sqrt(3)*x+sqrt(3)/6 > 0.0
function vortex_sol(x,y,t)
    x0 = 5
    y0 = 0
    beta = 5
    r2 = @. (x-x0-t)^2 + (y-y0)^2

    u = @. 1 - beta*exp(1-r2)*(y-y0)/(2*pi)
    v = @. beta*exp(1-r2)*(x-x0-t)/(2*pi)
    rho = @. 1 - (1/(8*γ*pi^2))*(γ-1)/2*(beta*exp(1-r2))^2
    rho = @. rho^(1/(γ-1))
    p = @. rho^γ

    return (rho, u, v, p)
end




const Np  = size(xq,1)
const Nfp = size(Vf,1)
U = zeros(Nc,Np,K)
for k = 1:K
    for i = 1:Np
        rho,u,v,p = vortex_sol(xq[i,k],yq[i,k],0.0)
        U[1,i,k] = rho
        U[2,i,k] = rho*u
        U[3,i,k] = rho*v
        U[4,i,k] = Efun(rho,u,v,p)
    end
end

ops      = (Sr,Ss,S0r,S0s,Minv,Br_halved,Bs_halved,E)
geom     = (mapP,Fmask,xq,yq,rxJ,ryJ,sxJ,syJ,J,sJ,nx,ny,nrJ,nsJ,wf)


# Time stepping
"Time integration"
t = 0.0
U = collect(U)
rhsU = zeros(size(U))
resW = zeros(size(U))

dt_hist = []
i = 1

@time while t < T
#while i < 2
    @show t
    # SSPRK(3,3)
    dt = dt0
    rhs_ESDG!(U,rhsU,ops,geom)
    dt = min(CFL*dt,T-t)
    @. resW = U + dt*rhsU
    rhs_ESDG!(resW,rhsU,ops,geom)
    @. resW = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resW
    rhs_ESDG!(resW,rhsU,ops,geom)
    @. resW = resW+dt*rhsU
    @. U = 1/3*U+2/3*resW

    push!(dt_hist,dt)
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T, at step $i")
    flush(stdout)
    global i = i + 1
end

exact_U = @. vortex_sol.(xq,yq,T)
exact_rho = [x[1] for x in exact_U]
exact_u = [x[2] for x in exact_U]
exact_v = [x[3] for x in exact_U]
exact_rhou = exact_rho .* exact_u
exact_rhov = exact_rho .* exact_v
exact_p = [x[4] for x in exact_U]
exact_E = Efun.(exact_rho,exact_u,exact_v,exact_p)

rho = U[1,:,:]
u = U[2,:,:]./U[1,:,:]
v = U[3,:,:]./U[1,:,:]
rhou = U[2,:,:]
rhov = U[3,:,:]
E = U[4,:,:]

#plotting nodes
# @unpack Vp = rd
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)

@unpack r,s,rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd
@unpack VDM,V1,Vq,Vf,Dr,Ds,M,Pq,LIFT,Vp = rd
vv = Vp*U[1,:,:]
scatter(Vp*xq,Vp*yq,vv,zcolor=vv,camera=(0,90))
savefig("~/Desktop/tmpquad.png")