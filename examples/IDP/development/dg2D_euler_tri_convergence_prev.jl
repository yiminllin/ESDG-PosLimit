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
include("../EntropyStableEuler.jl/src/logmean.jl")
include("SBP_quad_data.jl")
using EntropyStableEuler
using EntropyStableEuler.Fluxes2D

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



#=
"Mesh related variables"
Kx = Int(round(XLENGTH*K1D))
Ky = K1D
VX, VY, EToV = uniform_tri_mesh(Kx,Ky)
# @. VX = (VX+1)/2*XLENGTH*5
# @. VY = (VY+1)/2*5
@. VX = 15*(1+VX)/2
@. VY = 5*VY
K = Kx*Ky*2
Nfaces = 3

r1D, w1D = gauss_quad(0,0,N)
e = ones(length(r1D)) # vector of all ones
z = zeros(length(r1D)) # vector of all zeros
rf = [r1D; -r1D; -e];
sf = [-e; r1D; -r1D];
wf = vec(repeat(w1D,3,1));
nrJ = [z; e; -e]
nsJ = [-e; e; z]

rq,sq,wq = GQ_SBP[N];
VDM = vandermonde(Tri(),N,rq,sq) 
# r1,s1 = nodes(Tri(),1)
r1 = [-1;-1;1]
s1 = [1;-1;-1]
V1 = vandermonde(Tri(),1,rq,sq)/vandermonde(Tri(),1,r1,s1)
Vr,Vs = grad_vandermonde(Tri(),N,rq,sq)
Dr = Vr/VDM
Ds = Vs/VDM
Vf = vandermonde(Tri(),N,rf,sf)/VDM

x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]
xq = x
yq = y

# "Connectivity maps"
xf,yf = (x->Vf*x).((x,y))
mapM = reshape(collect(1:K*length(wf)),length(wf),K)
mapP = reshape(collect(1:K*length(wf)),length(wf),K)

for k = 1:K
    Npf = div(length(wf),Nfaces)
    k1 = Int(mod1(k+2*3*(mod(k,2)-.5),2*Kx)+2*Kx*div(k-1,2*Kx))
    k2 = Int(k+2*(mod(k,2)-.5))
    k3 = Int(mod1(k+2*(2*Kx-1)*(mod(k-1,2)-.5),K))
    for i = 1:Npf
        mapP[i,k]       = mapM[Npf-i+1,k1]
        mapP[i+Npf,k]   = mapM[2*Npf-i+1,k2]
        mapP[i+2*Npf,k] = mapM[3*Npf-i+1,k3]
    end
end


# "Geometric factors and surface normals"
rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
rxJ = Matrix(droptol!(sparse(rxJ),1e-12)); sxJ = Matrix(droptol!(sparse(sxJ),1e-12))
ryJ = Matrix(droptol!(sparse(ryJ),1e-12)); syJ = Matrix(droptol!(sparse(syJ),1e-12))
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
sJ = @. sqrt(nxJ^2 + nyJ^2)
nx = nxJ./sJ; ny = nyJ./sJ;

M = diagm(wq)
Minv = diagm(1 ./ wq)

Qr = M*Dr
Qs = M*Ds
Sr = .5*(Qr-transpose(Qr))
Ss = .5*(Qs-transpose(Qs))

α = 3.5
Qr_ID,Qs_ID,E,Br,Bs,A = build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)
if (norm(sum(Qr_ID,dims=2)) > 1e-10) | (norm(sum(Qs_ID,dims=2)) > 1e-10)
    error("Qr_ID or Qs_ID doesn't sum to zero for α = $α")
end
Q0r = Matrix(droptol!(sparse(Qr_ID),1e-15))
Q0s = Matrix(droptol!(sparse(Qs_ID),1e-15))
S0r = .5*(Qr_ID-transpose(Qr_ID))
S0s = .5*(Qs_ID-transpose(Qs_ID))
Br_halved = -sum(S0r,dims=2)
Bs_halved = -sum(S0s,dims=2)

# E = Vf
# TODO: hardcoded
Fmask = zeros(Int64,size(Vf,1))
for i = 1:size(Vf,1)
    tmparr = findall(abs.(E[i,:] .- 1) .< 1e-12)
    Fmask[i] = tmparr[1]
end
=#



#=
rd = init_reference_tri(N)
md = init_mesh((VX,VY),EToV,rd)

FToF = connect_mesh(EToV,tri_face_vertices())
const Nfaces = 3
const K      = size(FToF,2)
rd = init_reference_tri_sbp_GQ(N,"GQ")
@unpack r,s,rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd
@unpack VDM,V1,Vq,Vf,Dr,Ds,M,Pq,LIFT = rd

# "Construct global coordinates"
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

# # "Connectivity maps"
# xf,yf = (x->Vf*x).((x,y))
# mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
# mapM = reshape(mapM,length(wf),K)
# mapP = reshape(mapP,length(wf),K)

# # "Make periodic"
# LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
# mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
# mapP[mapB] = mapPB

# "Geometric factors and surface normals"
# rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
@unpack rxJ,sxJ,ryJ,syJ,J = md
rxJ = Matrix(droptol!(sparse(rxJ),1e-14)); sxJ = Matrix(droptol!(sparse(sxJ),1e-14))
ryJ = Matrix(droptol!(sparse(ryJ),1e-14)); syJ = Matrix(droptol!(sparse(syJ),1e-14))
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
sJ = @. sqrt(nxJ^2 + nyJ^2)
nx = nxJ./sJ; ny = nyJ./sJ;
rxJ,sxJ,ryJ,syJ,J = (x->Vq*x).((rxJ,sxJ,ryJ,syJ,J))
xq,yq = (x->Vq*x).((x,y))


# Construct matrices on reference elements
# From philip code
V = vandermonde_2D(N, r, s)
Vr, Vs = grad_vandermonde_2D(N, r, s)
M = inv(V*V')
Dr_ES = Vr/V; Ds = Vs/V;
M = Vq'*diagm(wq)*Vq

Qr_ES = M*Dr;
Qs_ES = M*Ds;
Pq = M\(Vq'*diagm(wq));

# diff. matrices redefined in terms of quadrature points
Qr_ES = Pq'*Qr_ES*Pq;
Qs_ES = Pq'*Qs_ES*Pq;
E_ES = Vf*Pq;

# Need to choose α so that Qr, Qs have zero row sums (and maybe a minimum number of neighbors)
# α = 4 # for N=1
# α = 2.5 #for N=2
α = 3.5 # for N=3
if (norm(sum(Qr_ID,dims=2)) > 1e-10) | (norm(sum(Qs_ID,dims=2)) > 1e-10)
    error("Qr_ID or Qs_ID doesn't sum to zero for α = $α")
end
Qr_ID = Matrix(droptol!(sparse(Qr_ID),1e-15))
Qs_ID = Matrix(droptol!(sparse(Qs_ID),1e-15))
Qrskew_ID = .5*(Qr_ID-transpose(Qr_ID))
Qsskew_ID = .5*(Qs_ID-transpose(Qs_ID))

QNr = [Qr_ES - .5*E_ES'*Br*E_ES .5*E_ES'*Br;
    -.5*Br*E_ES .5*Br];
QNs = [Qs_ES - .5*E_ES'*Bs*E_ES .5*E_ES'*Bs;
    -.5*Bs*E_ES .5*Bs];

VN_sbp = [eye(length(wq));E];
QNr_sbp = VN_sbp'*QNr*VN_sbp;
@show norm(QNr_sbp+QNr_sbp' - E'*diagm(wf.*nrJ)*E)
QNs_sbp = VN_sbp'*QNs*VN_sbp;
@show norm(QNs_sbp+QNs_sbp' - E'*diagm(wf.*nsJ)*E)

Qrskew_ES = .5*(QNr_sbp-QNr_sbp');
Qsskew_ES = .5*(QNs_sbp-QNs_sbp');


Sr = Qrskew_ES
Ss = Qsskew_ES
S0r = Qrskew_ID
S0s = Qsskew_ID
Br  = Qr_ID+Qr_ID'
Bs  = Qs_ID+Qs_ID'
M    = Array(wq)
Minv = Array(1 ./ wq)
Br_halved = -sum(S0r,dims=2)
Bs_halved = -sum(S0s,dims=2)

Fmask = zeros(Int64,size(Vf,1))
for i = 1:size(Vf,1)
    tmparr = findall(abs.(E[i,:] .- 1) .< 1e-12)
    Fmask[i] = tmparr[1]
end


# # "Connectivity maps"
# VDM = Basis2DTri.vandermonde_2D(N,rq,sq)
# Vf  = Basis2DTri.vandermonde_2D(N,rf,sf)/VDM
# xf,yf = (x->Vf*x).((xq,yq))
# mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
# mapM = reshape(mapM,length(wf),K)
# mapP = reshape(mapP,length(wf),K)

# # "Make periodic"
# LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
# mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
# mapP[mapB] = mapPB

# TODO: hardcoded for uniform mesh
xf = E*xq
yf = E*yq
mapM = reshape(collect(1:K*length(wf)),length(wf),K)
mapP = reshape(collect(1:K*length(wf)),length(wf),K)
for k = 1:K
    Npf = div(length(wf),Nfaces)
    k1 = Int(k+2*(mod(k,2)-.5))
    k2 = Int(mod1(k+2*3*(mod(k,2)-.5),2*Kx)+2*Kx*div(k-1,2*Kx))
    k3 = Int(mod1(k+2*(2*Kx-1)*(mod(k-1,2)-.5),K))
    for i = 1:Npf
        mapP[i,k]       = mapM[Npf-i+1,k1]
        mapP[i+Npf,k]   = mapM[2*Npf-i+1,k2]
        mapP[i+2*Npf,k] = mapM[3*Npf-i+1,k3]
    end
end

=#


@inline function get_infoP(mapP,Fmask,i,k)
    gP = mapP[i,k]           # exterior global face node number
    kP = fld1(gP,Nfp)        # exterior element number
    iP = Fmask[mod1(gP,Nfp)] # exterior node number
    return iP,kP
end

@inline function get_consP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k)
    iP,kP = get_infoP(mapP,Fmask,i,k)

    rhoP  = U[1,iP,kP]
    rhouP = U[2,iP,kP]
    rhovP = U[3,iP,kP]
    EP    = U[4,iP,kP]

    return rhoP,rhouP,rhovP,EP
end

@inline function get_fluxP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k,rhoP,rhouP,rhovP,EP)
    iP,kP = get_infoP(mapP,Fmask,i,k)

    fx_1_P = f_x[1,iP,kP]
    fx_2_P = f_x[2,iP,kP]
    fx_3_P = f_x[3,iP,kP]
    fx_4_P = f_x[4,iP,kP]
    fy_1_P = f_y[1,iP,kP]
    fy_2_P = f_y[2,iP,kP]
    fy_3_P = f_y[3,iP,kP]
    fy_4_P = f_y[4,iP,kP]

    return fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P
end

@inline function get_valP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k)
    rhoP,rhouP,rhovP,EP = get_consP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k)
    fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P = get_fluxP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k,rhoP,rhouP,rhovP,EP)
    return rhoP,rhouP,rhovP,EP,fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P
end

@inline function update_F_low!(F_low,k,tid,i,j,λ,S0xJ_ij,S0yJ_ij,U,fx,fy)
    # Tensor product elements: f is f_x or f_y
    rho_j  = U[1,j,k]
    rhou_j = U[2,j,k]
    rhov_j = U[3,j,k]
    E_j    = U[4,j,k]
    fx_1_j  = fx[1,j,k]
    fx_2_j  = fx[2,j,k]
    fx_3_j  = fx[3,j,k]
    fx_4_j  = fx[4,j,k]
    fy_1_j  = fy[1,j,k]
    fy_2_j  = fy[2,j,k]
    fy_3_j  = fy[3,j,k]
    fy_4_j  = fy[4,j,k]
    rho_i  = U[1,i,k]
    rhou_i = U[2,i,k]
    rhov_i = U[3,i,k]
    E_i    = U[4,i,k]
    fx_1_i  = fx[1,i,k]
    fx_2_i  = fx[2,i,k]
    fx_3_i  = fx[3,i,k]
    fx_4_i  = fx[4,i,k]
    fy_1_i  = fy[1,i,k]
    fy_2_i  = fy[2,i,k]
    fy_3_i  = fy[3,i,k]
    fy_4_i  = fy[4,i,k]

    FL1 = (S0xJ_ij*(fx_1_i+fx_1_j) + S0yJ_ij*(fy_1_i+fy_1_j) - λ*(rho_j-rho_i))
    FL2 = (S0xJ_ij*(fx_2_i+fx_2_j) + S0yJ_ij*(fy_2_i+fy_2_j) - λ*(rhou_j-rhou_i))
    FL3 = (S0xJ_ij*(fx_3_i+fx_3_j) + S0yJ_ij*(fy_3_i+fy_3_j) - λ*(rhov_j-rhov_i))
    FL4 = (S0xJ_ij*(fx_4_i+fx_4_j) + S0yJ_ij*(fy_4_i+fy_4_j) - λ*(E_j-E_i))

    F_low[1,i,j,tid] = FL1
    F_low[2,i,j,tid] = FL2
    F_low[3,i,j,tid] = FL3
    F_low[4,i,j,tid] = FL4

    F_low[1,j,i,tid] = -FL1
    F_low[2,j,i,tid] = -FL2
    F_low[3,j,i,tid] = -FL3
    F_low[4,j,i,tid] = -FL4
end

@inline function update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,SyJ_ij_db,U,rholog,betalog)
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
    
    Fx1,Fx2,Fx3,Fx4,Fy1,Fy2,Fy3,Fy4 = euler_fluxes_2D(rho_i,u_i,v_i,beta_i,rholog_i,betalog_i,
                                                      rho_j,u_j,v_j,beta_j,rholog_j,betalog_j)

    FH1 = SxJ_ij_db*Fx1 + SyJ_ij_db*Fy1
    FH2 = SxJ_ij_db*Fx2 + SyJ_ij_db*Fy2
    FH3 = SxJ_ij_db*Fx3 + SyJ_ij_db*Fy3
    FH4 = SxJ_ij_db*Fx4 + SyJ_ij_db*Fy4

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
    f_x,f_y,rholog,betalog,U_low,F_low,F_high,F_P,L,λ_arr,λf_arr,dii_arr = prealloc
    Sr,Ss,S0r,S0s,Minv,Br_halved,Bs_halved = ops
    mapP,Fmask,xq,yq,rxJ,ryJ,sxJ,syJ,J,sJ,nx,ny = geom

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
        rxJ_k = rxJ[1,k]
        ryJ_k = ryJ[1,k]
        sxJ_k = sxJ[1,k]
        syJ_k = syJ[1,k]

        for j = 2:Np
            rho_j  = U[1,j,k]
            rhou_j = U[2,j,k]
            rhov_j = U[3,j,k]
            E_j    = U[4,j,k]
            for i = 1:j-1
                nij_x = rxJ_k*S0r[i,j] + sxJ_k*S0s[i,j]
                nij_y = ryJ_k*S0r[i,j] + syJ_k*S0s[i,j]
                nij_norm = sqrt(nij_x^2+nij_y^2)
                rho_i  = U[1,i,k]
                rhou_i = U[2,i,k]
                rhov_i = U[3,i,k]
                E_i    = U[4,i,k]
                if nij_norm != 0
                    λ = nij_norm*max(wavespeed_1D(rho_i,nij_x/nij_norm*rhou_i+nij_y/nij_norm*rhov_i,E_i),
                                    wavespeed_1D(rho_j,nij_x/nij_norm*rhou_j+nij_y/nij_norm*rhov_j,E_j))
                    λ_arr[i,j,k] = λ
                    λ_arr[j,i,k] = λ
                    if in_s1
                        dii_arr[i,k] = dii_arr[i,k] + λ
                        dii_arr[j,k] = dii_arr[j,k] + λ
                    end
                end    
            end
        end
    end

    # Interface dissipation coeff 
    @batch for k = 1:K
        rxJ_k = rxJ[1,k]
        ryJ_k = ryJ[1,k]
        sxJ_k = sxJ[1,k]
        syJ_k = syJ[1,k]

        for i = 1:Nfp
            sJ_ik = sJ[i,k]
            iM = Fmask[i]
            nij_x = abs(rxJ_k*Br_halved[iM]+sxJ_k*Bs_halved[iM]) 
            nij_y = abs(ryJ_k*Br_halved[iM]+syJ_k*Bs_halved[iM])
            nij_norm = sqrt(nij_x^2+nij_y^2)

            iP,kP = get_infoP(mapP,Fmask,i,k)

            rhoM  = U[1,iM,k]
            rhouM = U[2,iM,k]
            rhovM = U[3,iM,k]
            EM    = U[4,iM,k]
            rhoP  = U[1,iP,kP]
            rhouP = U[2,iP,kP]
            rhovP = U[3,iP,kP]
            EP    = U[4,iP,kP]

            λ = nij_norm*max(wavespeed_1D(rhoM,nij_x/nij_norm*rhouM+nij_y/nij_norm*rhovM,EM),
                             wavespeed_1D(rhoP,nij_x/nij_norm*rhouP+nij_y/nij_norm*rhovP,EP))
            λf_arr[i,k] = λ
            if in_s1
                dii_arr[iM,k] = dii_arr[iM,k] + λ
            end
        end
    end

    # If at the first stage, calculate the time step
    if in_s1
        for k = 1:K
            J_k = J[1,k]
            for i = 1:Np
                dt = min(dt,1.0/(Minv[i]/J_k)/2.0/dii_arr[i,k])
            end
        end
    end

    @show dt
    dt = 1e-4

    # =====================
    # Loop through elements
    # =====================
    @batch for k = 1:K
        tid = Threads.threadid()
        rxJ_k = rxJ[1,k]
        ryJ_k = ryJ[1,k]
        sxJ_k = sxJ[1,k]
        syJ_k = syJ[1,k]
        J_k   = J[1,k]

        for i = 1:Np
            for c = 1:Nc
                U_low[c,i,tid] = 0.0
            end
        end

        # Calculate low order algebraic flux
        # for c_r = 1:S0r_nnz_hv
        #     i = S0r_nnzi[c_r]
        #     j = S0r_nnzj[c_r]
        #     λ = λ_arr[c_r,1,k]
        #     S0xJ_ij = S0xJ_vec[c_r]
        #     update_F_low!(F_low,k,tid,i,j,λ,S0xJ_ij,U,f_x)
        # end

        # for c_s = 1:S0s_nnz_hv
        #     i = S0s_nnzi[c_s]
        #     j = S0s_nnzj[c_s]
        #     λ = λ_arr[c_s,2,k]
        #     S0yJ_ij = S0yJ_vec[c_s]
        #     update_F_low!(F_low,k,tid,i,j,λ,S0yJ_ij,U,f_y)
        # end

        for j = 2:Np
            for i = 1:j-1
                λ = λ_arr[i,j,k]
                S0xJ_ij = rxJ_k*S0r[i,j] + sxJ_k*S0s[i,j]
                S0yJ_ij = ryJ_k*S0r[i,j] + syJ_k*S0s[i,j]
                update_F_low!(F_low,k,tid,i,j,λ,S0xJ_ij,S0yJ_ij,U,f_x,f_y)
            end
        end

        # Calculate high order algebraic flux
        # for c_r = 1:Sr_nnz_hv
        #     i         = Sr_nnzi[c_r]
        #     j         = Sr_nnzj[c_r]
        #     SxJ_ij_db = SxJ_db_vec[c_r]
        #     update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,U,rholog,betalog,0)
        # end

        # for c_r = 1:Sr_nnz_hv
        #     i         = Ss_nnzi[c_r]
        #     j         = Ss_nnzj[c_r]
        #     SyJ_ij_db = SyJ_db_vec[c_r]
        #     update_F_high!(F_high,k,tid,i,j,SyJ_ij_db,U,rholog,betalog,1)
        # end

        for j = 2:Np
            for i = 1:j-1
                # λ = λ_arr[i,j,k]
                SxJ_ij_db = 2*(rxJ_k*Sr[i,j] + sxJ_k*Ss[i,j])
                SyJ_ij_db = 2*(ryJ_k*Sr[i,j] + syJ_k*Ss[i,j])
                update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,SyJ_ij_db,U,rholog,betalog)
            end
        end

        # Calculate interface fluxes
        for i = 1:Nfp
            iM    = Fmask[i]
            # BrJ_ii_halved = BrJ_halved[iM]
            # BsJ_ii_halved = BsJ_halved[iM]
            BxJ_ii_halved = rxJ_k*Br_halved[iM]+sxJ_k*Bs_halved[iM]
            ByJ_ii_halved = ryJ_k*Br_halved[iM]+syJ_k*Bs_halved[iM]
            xM    = xq[iM,k]
            yM    = yq[iM,k]
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

            rhoP,rhouP,rhovP,EP,fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P = get_valP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k)
            λ = λf_arr[i,k]

            # # flux in x direction
            # if is_face_x(i)
            #     F_P[1,i,tid] = (BrJ_ii_halved*(fx_1_M+fx_1_P)
            #                    -λ*(rhoP-rhoM) )
            #     F_P[2,i,tid] = (BrJ_ii_halved*(fx_2_M+fx_2_P)
            #                    -λ*(rhouP-rhouM) )
            #     F_P[3,i,tid] = (BrJ_ii_halved*(fx_3_M+fx_3_P)
            #                    -λ*(rhovP-rhovM) )
            #     F_P[4,i,tid] = (BrJ_ii_halved*(fx_4_M+fx_4_P)
            #                    -λ*(EP-EM) )
            # end

            # # flux in y direction
            # if is_face_y(i)
            #     F_P[1,i,tid] = (BsJ_ii_halved*(fy_1_M+fy_1_P)
            #                    -λ*(rhoP-rhoM) )
            #     F_P[2,i,tid] = (BsJ_ii_halved*(fy_2_M+fy_2_P)
            #                    -λ*(rhouP-rhouM) )
            #     F_P[3,i,tid] = (BsJ_ii_halved*(fy_3_M+fy_3_P)
            #                    -λ*(rhovP-rhovM) )
            #     F_P[4,i,tid] = (BsJ_ii_halved*(fy_4_M+fy_4_P)
            #                    -λ*(EP-EM) )
            # end

            F_P[1,i,tid] = (BxJ_ii_halved*(fx_1_M+fx_1_P) + ByJ_ii_halved*(fy_1_M+fy_1_P)
                            -λ*(rhoP-rhoM) )
            F_P[2,i,tid] = (BxJ_ii_halved*(fx_2_M+fx_2_P) + ByJ_ii_halved*(fy_2_M+fy_2_P)
                            -λ*(rhouP-rhouM) )
            F_P[3,i,tid] = (BxJ_ii_halved*(fx_3_M+fx_3_P) + ByJ_ii_halved*(fy_3_M+fy_3_P)
                            -λ*(rhovP-rhovM) )
            F_P[4,i,tid] = (BxJ_ii_halved*(fx_4_M+fx_4_P) + ByJ_ii_halved*(fy_4_M+fy_4_P)
                            -λ*(EP-EM) )

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
                U_low[c,i,tid] = U[c,i,k] - dt*Minv[i]/J_k*U_low[c,i,tid]
            end
        end

        # Calculate limiting parameters
        # for ni = 1:S_nnz_hv
        for j = 2:Np
            for i = 1:j-1
            # i = S_nnzi[ni]
            # j = S_nnzj[ni]
                # coeff_i = dt*coeff_arr[i]
                # coeff_j = dt*coeff_arr[j]
                coeff_i = dt*(Np-1)*Minv[i]/J_k
                coeff_j = dt*(Np-1)*Minv[j]/J_k
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
                # L[ni,tid] = min(limiting_param(rhoi,rhoui,rhovi,Ei,rhoP_i,rhouP_i,rhovP_i,EP_i),
                #                 limiting_param(rhoj,rhouj,rhovj,Ej,rhoP_j,rhouP_j,rhovP_j,EP_j))
                l = min(limiting_param(rhoi,rhoui,rhovi,Ei,rhoP_i,rhouP_i,rhovP_i,EP_i),
                        limiting_param(rhoj,rhouj,rhovj,Ej,rhoP_j,rhouP_j,rhovP_j,EP_j))
                L[i,j,tid] = l
                L[j,i,tid] = l
            end
        end

        # Elementwise limiting
        l_e = 1.0
        # for i = 1:S_nnz_hv
        for j = 1:Np
            for i = 1:Np
                l_e = min(l_e,L[i,j,tid])
            end
        end
        l_em1 = l_e-1.0

        # for ni = 1:S_nnz_hv
        for j = 2:Np
            for i = 1:j-1
                # i     = S_nnzi[ni]
                # j     = S_nnzj[ni]
                for c = 1:Nc
                    l_em1 = -0.0
                    l_e = 1.0
                    FL_ij = F_low[c,i,j,tid]
                    FH_ij = F_high[c,i,j,tid]
                    rhsU[c,i,k] = rhsU[c,i,k] + l_em1*FL_ij - l_e*FH_ij
                    rhsU[c,j,k] = rhsU[c,j,k] - l_em1*FL_ij + l_e*FH_ij
                    # l_e   = L[ni,tid]
                    # l_em1 = L[ni,tid]-1.0 
                    # rhsU[c,i,k] = rhsU[c,i,k] + l_em1*FL_ij - l_e*FH_ij
                    # rhsU[c,j,k] = rhsU[c,j,k] - l_em1*FL_ij + l_e*FH_ij
                end
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
        J_k = J[1,k]
        for i = 1:Np
            for c = 1:Nc
                rhsU[c,i,k] = Minv[i]/J_k*rhsU[c,i,k]
            end
        end
    end

    return dt
end

function rhs_ESDG!(U,rhsU,t,dt,prealloc,ops,geom)
    f_x,f_y,rholog,betalog,U_low,F_low,F_high,F_P,L,λ_arr,λf_arr,dii_arr = prealloc
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
                                                
                # fx1M,fx2M,fx3M,fx4M,fy1M,fy2M,fy3M,fy4M = inviscid_flux_prim(rhoM,uM,vM,pM)
                # fx1P,fx2P,fx3P,fx4P,fy1P,fy2P,fy3P,fy4P = inviscid_flux_prim(rhoP,uP,vP,pP)
                # Fx1 = (fx1M+fx1P)/2
                # Fx2 = (fx2M+fx2P)/2
                # Fx3 = (fx3M+fx3P)/2
                # Fx4 = (fx4M+fx4P)/2
                # Fy1 = (fy1M+fy1P)/2
                # Fy2 = (fy2M+fy2P)/2
                # Fy3 = (fy3M+fy3P)/2
                # Fy4 = (fy4M+fy4P)/2

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
            
            # fx1M,fx2M,fx3M,fx4M,fy1M,fy2M,fy3M,fy4M = inviscid_flux_prim(rhoM,uM,vM,pM)
            # fx1P,fx2P,fx3P,fx4P,fy1P,fy2P,fy3P,fy4P = inviscid_flux_prim(rhoP,uP,vP,pP)
            # Fx1 = (fx1M+fx1P)/2
            # Fx2 = (fx2M+fx2P)/2
            # Fx3 = (fx3M+fx3P)/2
            # Fx4 = (fx4M+fx4P)/2
            # Fy1 = (fy1M+fy1P)/2
            # Fy2 = (fy2M+fy2P)/2
            # Fy3 = (fy3M+fy3P)/2
            # Fy4 = (fy4M+fy4P)/2


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
    #=
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

    # =====================
    # Loop through elements
    # =====================
    @batch for k = 1:K
        tid = Threads.threadid()
        rxJ_k = rxJ[1,k]
        ryJ_k = ryJ[1,k]
        sxJ_k = sxJ[1,k]
        syJ_k = syJ[1,k]
        J_k   = J[1,k]

        for j = 2:Np
            for i = 1:j-1
                SxJ_ij_db = 2*(rxJ_k*Sr[i,j] + sxJ_k*Ss[i,j])
                SyJ_ij_db = 2*(ryJ_k*Sr[i,j] + syJ_k*Ss[i,j])
                update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,SyJ_ij_db,U,rholog,betalog)
            end
        end

        # Calculate interface fluxes
        for i = 1:Nfp
            iM    = Fmask[i]
            BxJ_ii_halved = rxJ_k*Br_halved[iM]+sxJ_k*Bs_halved[iM]
            ByJ_ii_halved = ryJ_k*Br_halved[iM]+syJ_k*Bs_halved[iM]
            xM    = xq[iM,k]
            yM    = yq[iM,k]
            rhoM  = U[1,iM,k]
            rhouM = U[2,iM,k]
            rhovM = U[3,iM,k]
            EM    = U[4,iM,k]
            uM    = rhouM/rhoM
            vM    = rhovM/rhoM
            pM    = pfun(rhoM,rhouM,rhovM,EM)
            rhologM  = rholog[iM,k]
            betalogM = betalog[iM,k]
            betaM = rhoM/(2*pM)
            
            rhoP,rhouP,rhovP,EP,fx_1_P,fx_2_P,fx_3_P,fx_4_P,fy_1_P,fy_2_P,fy_3_P,fy_4_P = get_valP(U,f_x,f_y,t,mapP,Fmask,i,iM,xM,yM,uM,vM,k)
            uP      = rhouP/rhoP
            vP      = rhovP/rhoP
            pP      = pfun(rhoP,rhouP,rhovP,EP)
            betaP   = rhoP/(2*pP)
            rhologP  = log(rhoP)
            betalogP = log(betaP)
            Fx1,Fx2,Fx3,Fx4,Fy1,Fy2,Fy3,Fy4 = euler_fluxes_2D(rhoM,uM,vM,betaM,rhologM,betalogM,
                                                              rhoP,uP,vP,betaP,rhologP,betalogP)

            F_P[1,i,tid] = (BxJ_ii_halved*(2*Fx1) + ByJ_ii_halved*(2*Fy1))
            F_P[2,i,tid] = (BxJ_ii_halved*(2*Fx2) + ByJ_ii_halved*(2*Fy2))
            F_P[3,i,tid] = (BxJ_ii_halved*(2*Fx3) + ByJ_ii_halved*(2*Fy3))
            F_P[4,i,tid] = (BxJ_ii_halved*(2*Fx4) + ByJ_ii_halved*(2*Fy4))

        end

        for j = 2:Np
            for i = 1:j-1
                for c = 1:Nc
                    FH_ij = F_high[c,i,j,tid]
                    rhsU[c,i,k] = rhsU[c,i,k] - FH_ij
                    rhsU[c,j,k] = rhsU[c,j,k] + FH_ij
                end
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
        J_k = J[1,k]
        for i = 1:Np
            for c = 1:Nc
                rhsU[c,i,k] = Minv[i]/J_k*rhsU[c,i,k]
            end
        end
    end

    return dt
    =#
end




# Initial condition 2D shocktube
# at_left(x,y) = y-sqrt(3)*x+sqrt(3)/6 > 0.0
function vortex_sol(x,y,t)

    # x0 = 4.5
    # y0 = 2.5
    # beta = 2#5#8
    # r2 = @. (x-x0-t)^2 + (y-y0)^2

    # u = @. 1 - beta*exp(1-r2)*(y-y0)/(2*pi)
    # v = @. beta*exp(1-r2)*(x-x0-t)/(2*pi)
    # rho = @. 1 - (1/(8*γ*pi^2))*(γ-1)/2*(beta*exp(1-r2))^2
    # rho = @. rho^(1/(γ-1))
    # p = @. rho^γ

    # return (rho, u, v, p)

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
L       =  ones(Float64,Np,Np,NUM_THREADS)
λ_arr    = zeros(Float64,Np,Np,K)
λf_arr   = zeros(Float64,Nfp,K)
dii_arr  = zeros(Float64,Np,K)

prealloc = (f_x,f_y,rholog,betalog,U_low,F_low,F_high,F_P,L,λ_arr,λf_arr,dii_arr)
ops      = (Sr,Ss,S0r,S0s,Minv,Br_halved,Bs_halved,E)
geom     = (mapP,Fmask,xq,yq,rxJ,ryJ,sxJ,syJ,J,sJ,nx,ny,nrJ,nsJ,wf)


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
# @btime rhs_IDP!($U,$rhsU,$t,$dt,$prealloc,$ops,$geom,true);
# @profiler rhs_IDP!(U,rhsU,t,dt,prealloc,ops,geom,true);

dt_hist = []
i = 1

@time while t < T
#while i < 2
    @show t
    # SSPRK(3,3)
    fill!(dii_arr,0.0)
    # dt = min(dt0,T-t)
    dt = dt0
    # dt = rhs_IDP!(U,rhsU,t,dt,prealloc,ops,geom,true);
    rhs_ESDG!(U,rhsU,t,dt,prealloc,ops,geom)
    dt = min(CFL*dt,T-t)
    @. resW = U + dt*rhsU
    # rhs_IDP!(resW,rhsU,t+dt,dt,prealloc,ops,geom,false);
    rhs_ESDG!(resW,rhsU,t,dt,prealloc,ops,geom)
    @. resW = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resW
    # rhs_IDP!(resW,rhsU,t+dt/2,dt,prealloc,ops,geom,false);
    rhs_ESDG!(resW,rhsU,t,dt,prealloc,ops,geom)
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

# rd = init_reference_tri_sbp_GQ(N,"GQ")
@unpack r,s,rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd
@unpack VDM,V1,Vq,Vf,Dr,Ds,M,Pq,LIFT = rd
# VDM = vandermonde(Tri(),N, rq, sq)
# rp,sp = equi_nodes_2D(10)
# Vp = vandermonde_2D(N,rp,sp)/VDM
# @show size(VDM)
vv = Vp*U[1,:,:]
scatter(Vp*xq,Vp*yq,vv,zcolor=vv,camera=(0,90))
savefig("~/Desktop/tmp.png")


# Linferr = maximum(abs.(exact_rho-rho))/maximum(abs.(exact_rho)) +
#           maximum(abs.(exact_u-u))/maximum(abs.(exact_u)) +
#           maximum(abs.(exact_v-v))/maximum(abs.(exact_v)) +
#           maximum(abs.(exact_E-E))/maximum(abs.(exact_E))

# L1err = sum(J*M.*abs.(exact_rho-rho))/sum(J*M.*abs.(rho)) +
#         sum(J*M.*abs.(exact_u-u))/sum(J*M.*abs.(u)) +
#         sum(J*M.*abs.(exact_v-v))/sum(J*M.*abs.(v)) +
#         sum(J*M.*abs.(exact_E-E))/sum(J*M.*abs.(E))

# L2err = sqrt(sum(J*M.*abs.(exact_rho-rho).^2))/sqrt(sum(J*M.*abs.(rho).^2)) +
#         sqrt(sum(J*M.*abs.(exact_u-u).^2))/sqrt(sum(J*M.*abs.(u).^2)) +
#         sqrt(sum(J*M.*abs.(exact_v-v).^2))/sqrt(sum(J*M.*abs.(v).^2)) +
#         sqrt(sum(J*M.*abs.(exact_E-E).^2))/sqrt(sum(J*M.*abs.(E).^2))



# Linferr = maximum(abs.(exact_rho-rho))/maximum(abs.(exact_rho)) +
#           maximum(abs.(exact_rhou-rhou))/maximum(abs.(exact_rhou)) +
#           maximum(abs.(exact_rhov-rhov))/maximum(abs.(exact_rhov)) +
#           maximum(abs.(exact_E-E))/maximum(abs.(exact_E))

# L1err = sum(J*M.*abs.(exact_rho-rho))/sum(J*M.*abs.(exact_rho)) +
#         sum(J*M.*abs.(exact_rhou-rhou))/sum(J*M.*abs.(exact_rhou)) +
#         sum(J*M.*abs.(exact_rhov-rhov))/sum(J*M.*abs.(exact_rhov)) +
#         sum(J*M.*abs.(exact_E-E))/sum(J*M.*abs.(exact_E))

# L2err = sqrt(sum(J*M.*abs.(exact_rho-rho).^2))/sqrt(sum(J*M.*abs.(exact_rho).^2)) +
#         sqrt(sum(J*M.*abs.(exact_rhou-rhou).^2))/sqrt(sum(J*M.*abs.(exact_rhou).^2)) +
#         sqrt(sum(J*M.*abs.(exact_rhov-rhov).^2))/sqrt(sum(J*M.*abs.(exact_rhov).^2)) +
#         sqrt(sum(J*M.*abs.(exact_E-E).^2))/sqrt(sum(J*M.*abs.(exact_E).^2))



# println("N = $N, K = $K")
# println("L1 error is $L1err")
# println("L2 error is $L2err")


end #muladd