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
using Basis2DTri
using UniformTriMesh
using NodesAndModes


using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
include("../../EntropyStableEuler.jl/src/logmean.jl")
include("../SBP_quad_data.jl")
using EntropyStableEuler
using EntropyStableEuler.Fluxes2D

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

function construct_SBP(N)
    # Construct matrices on reference elements
    rd = init_reference_tri_sbp_GQ(N, "GQ")
    @unpack r,s,rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd
    @unpack VDM,V1,Vq,Vf,Dr,Ds,M,Pq,LIFT = rd

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

    return Qr_ES,Qs_ES,E_ES
end

const TOL = 5e-16
const POSTOL = 1e-14
const WALLPT = 1.0/6.0
const Nc = 4 # number of components
"Approximation parameters"
const N = 2
const K1D = 10
const T = 1.0
const dt0 = 1e-2
const XLENGTH = 2.0
const CFL = 0.5
const NUM_THREADS = Threads.nthreads()
qnode_choice = "GQ" #"GQ" "GL" "tri_diage"

# Initial condition 2D shocktube
const γ = 1.4

# Mesh related variables
Kx = Int(round(XLENGTH*K1D))
Ky = K1D
VX, VY, EToV = uniform_tri_mesh(Kx,Ky)

# @. VX = 15*(1+VX)/2
# @. VY = 5*VY
@. VX = 10*(1+VX)/2
@. VY = 5*(1+VY)/2
FToF = CommonUtils.connect_mesh(EToV,tri_face_vertices())
Nfaces,K = size(FToF)

# Construct matrices on reference elements
rd = init_reference_tri_sbp_GQ(N, qnode_choice)
@unpack r,s,rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd
@unpack VDM,V1,Vq,Vf,Dr,Ds,M,Pq,LIFT = rd

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
Qr_ID,Qs_ID,E,Br,Bs,A = build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)
if (norm(sum(Qr_ID,dims=2)) > 1e-10) | (norm(sum(Qs_ID,dims=2)) > 1e-10)
    error("Qr_ID or Qs_ID doesn't sum to zero for α = $α")
end
Qr_ID = Matrix(droptol!(sparse(Qr_ID),1e-15))
Qs_ID = Matrix(droptol!(sparse(Qs_ID),1e-15))
Qrskew_ID = .5*(Qr_ID-transpose(Qr_ID))
Qsskew_ID = .5*(Qs_ID-transpose(Qs_ID))

# "Construct global coordinates"
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

# "Connectivity maps"
xf,yf = (x->Vf*x).((x,y))
mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
mapM = reshape(mapM,length(wf),K)
global mapP = reshape(mapP,length(wf),K)

mapM = reshape(collect(1:K*length(wf)),length(wf),K)
mapP = reshape(collect(1:K*length(wf)),length(wf),K)

for k = 1:K
    Npf = div(length(wf),Nfaces)
    k1 = Int(k+2*(mod(k,2)-.5))
    # k2 = Int(mod1(k+2*3*(mod(k-1,2)-.5),2*Kx)+2*Kx*div(k-1,2*Kx))
    k2 = Int(mod1(k+2*3*(mod(k,2)-.5),2*Kx)+2*Kx*div(k-1,2*Kx))
    k3 = Int(mod1(k+2*(2*Kx-1)*(mod(k-1,2)-.5),K))
    for i = 1:Npf
        mapP[i,k]       = mapM[Npf-i+1,k1]
        mapP[i+Npf,k]   = mapM[2*Npf-i+1,k2]
        mapP[i+2*Npf,k] = mapM[3*Npf-i+1,k3]
    end
end

# "Geometric factors and surface normals"
rxJ, sxJ, ryJ, syJ, J = CommonUtils.geometric_factors(x, y, Dr, Ds)
rxJ = Matrix(droptol!(sparse(rxJ),1e-14)); sxJ = Matrix(droptol!(sparse(sxJ),1e-14))
ryJ = Matrix(droptol!(sparse(ryJ),1e-14)); syJ = Matrix(droptol!(sparse(syJ),1e-14))
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
sJ = @. sqrt(nxJ^2 + nyJ^2)
nx = nxJ./sJ; ny = nyJ./sJ;
rxJ,sxJ,ryJ,syJ,J = (x->Vq*x).((rxJ,sxJ,ryJ,syJ,J))

# TODO:
J = -J
rxJ = -rxJ
ryJ = -ryJ
sxJ = -sxJ
syJ = -syJ

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
M   = diagm(wq)
Minv   = diagm(1 ./ wq)

Fmask = zeros(Int64,size(Vf,1))
for i = 1:size(Vf,1)
    tmparr = findall(abs.(E[i,:] .- 1) .< 1e-12)
    Fmask[i] = tmparr[1]
end

xq = Vq*x
yq = Vq*y



gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)
VDM = vandermonde(Tri(),N, rq, sq)
rp,sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/VDM

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


@inline function get_infoP(mapP,Fmask,i,k)
    gP = mapP[i,k]           # exterior global face node number
    kP = fld1(gP,Nfp)        # exterior element number
    iP = Fmask[mod1(gP,Nfp)] # exterior node number
    return iP,kP
end

function rhs_ESDG!(U,rhsU,ops,geom)
    Sr,Ss,S0r,S0s,Br,Bs,Minv,E = ops
    mapP,Fmask,rxJ,ryJ,sxJ,syJ,J = geom

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


function vortex_sol(x,y,t)
    # x0 = 5
    # y0 = 0
    x0 = 4.5
    y0 = 2.5
    beta = 5
    r2 = @. (x-x0-t)^2 + (y-y0)^2

    u = @. 1 - beta*exp(1-r2)*(y-y0)/(2*pi)
    v = @. beta*exp(1-r2)*(x-x0-t)/(2*pi)
    rho = @. 1 - (1/(8*γ*pi^2))*(γ-1)/2*(beta*exp(1-r2))^2
    rho = @. rho^(1/(γ-1))
    p = @. rho^γ

    return (rho, u, v, p)
end



# const Np  = size(x,1)
const Np  = size(xq,1)
const Nfp = size(Vf,1)
U = zeros(Nc,Np,K)
for k = 1:K
    for i = 1:Np
        # rho,u,v,p = vortex_sol(x[i,k],y[i,k],0.0)
        rho,u,v,p = vortex_sol(xq[i,k],yq[i,k],0.0)
        U[1,i,k] = rho
        U[2,i,k] = rho*u
        U[3,i,k] = rho*v
        U[4,i,k] = Efun(rho,u,v,p)
    end
end

ops      = (Sr,Ss,S0r,S0s,Br,Bs,Minv,E)
geom     = (mapP,Fmask,rxJ,ryJ,sxJ,syJ,J)


# TODO: previous Time stepping
"Time integration"
t = 0.0
U = collect(U)
rhsU = zeros(size(U))
resW = zeros(size(U))

dt_hist = []
i = 1

@time while t < T
    @show t
    # SSPRK(3,3)
    # dt = min(dt0,T-t)
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


exact_U = @. vortex_sol.(x,y,T)
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
# E = U[4,:,:]

vv = Vp*U[1,:,:]
scatter(Vp*xq,Vp*yq,vv,zcolor=vv,camera=(0,90))
savefig("~/Desktop/tmptri.png")
