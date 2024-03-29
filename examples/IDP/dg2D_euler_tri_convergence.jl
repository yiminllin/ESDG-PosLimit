using Pkg
Pkg.activate("Project.toml")
using Revise # reduce recompilation time
# using Plots
# using Documenter
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

push!(LOAD_PATH, "./src")

using CommonUtils
using Basis2DTri
using UniformTriMesh
using NodesAndModes


using SetupDG

# push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
# include("../EntropyStableEuler.jl/src/logmean.jl")
# include("SBP_quad_data.jl")
# using EntropyStableEuler
# using EntropyStableEuler.Fluxes2D

@muladd begin

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

const LIMITOPT = 2 # 1 if elementwise limiting lij, 2 if elementwise limiting li
const POSDETECT = 0 # 1 if turn on detection, 0 otherwise
const LBOUNDTYPE = 0.1 # 0 if use POSTOL as lower bound, if > 0, use LBOUNDTYPE*loworder
const TOL = 5e-16
const POSTOL = 1e-13
const WALLPT = 1.0/6.0
const Nc = 4 # number of components
"Approximation parameters"
const N = parse(Int,ARGS[1])     # N = 2,3,4
const K1D = parse(Int,ARGS[2])  # K = 2,4,8,16,32
const T = 2.0
const dt0 = 1e+2
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
@. VX = 10*(1+VX)/2*2
@. VY = 5*(1+VY)/2*2
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
α = 3.5 # for N=3,4
if N == 1
    α = 4
elseif N == 2
    α = 2.5
elseif N == 3 || N == 4
    α = 3.5
end
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



# gr(aspect_ratio=:equal,legend=false,
#    markerstrokewidth=0,markersize=2)
# VDM = vandermonde(Tri(),N, rq, sq)
# rp,sp = equi_nodes_2D(10)
# Vp = vandermonde_2D(N,rp,sp)/VDM

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

function rhs_Limited!(U,rhsU,T,dt,prealloc,ops,geom,in_s1)
    f_x,f_y,rholog,betalog,U_low,U_high,F_low,F_high,F_P,L,λ_arr,λf_arr,dii_arr,L_plot,U_hat,indicator = prealloc
    Sr,Ss,S0r,S0s,Br,Bs,Minv,Extrap,Pq,Vq,wq,Vq_hat,M = ops
    mapP,Fmask,rxJ,ryJ,sxJ,syJ,J = geom

    fill!(rhsU,0.0)
    @batch for k = 1:K
        for i = 1:Np
            rho  = U[1,i,k]
            rhou = U[2,i,k]
            rhov = U[3,i,k]
            E    = U[4,i,k]
            p          = pfun(rho,rhou,rhov,E)
            if p < 0
                @show "====================="
                @show "====================="
                @show "====================="
                @show "====================="
                @show "====================="
                @show i,k,p,rho
                @show "====================="
                @show "====================="
                @show "====================="
                @show "====================="
                @show "====================="
            end
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
            iM = Fmask[i]
            nij_x = (rxJ_k*Br[i,i]+sxJ_k*Bs[i,i])*.5 
            nij_y = (ryJ_k*Br[i,i]+syJ_k*Bs[i,i])*.5
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
                dt = min(dt,1.0/(Minv[i,i]/J_k)/2.0/dii_arr[i,k])
            end
        end
        dt = min(CFL*dt,T-t)
    end

    #=
    # # Troubled cell indicator
    for c = 1:Nc
        U_hat[c,:,:] = Vq_hat*U[c,:,:]
    end

    @batch for k = 1:K
        J_k = J[1,k]
        indicator[k] = sum(J_k*(M*abs.(U_hat[1,:,k]-U[1,:,k]).^2))/sum(J_k*(M*abs.(U[1,:,k]).^2)) +
                       #sum(J_k*(M*abs.(U_hat[2,:,k]-U[2,:,k]).^2))/sum(J_k*(M*abs.(U[2,:,k]).^2)) +
                       # TODO: ignore third zero component
                       #sum(J_k*(M*abs.(U_hat[3,:,k]-U[3,:,k]).^2))/sum(J_k*(M*abs.(U[3,:,k]).^2)) +
                       sum(J_k*(M*abs.(U_hat[4,:,k]-U[4,:,k]).^2))/sum(J_k*(M*abs.(U[4,:,k]).^2))

        if sum(J_k*(M*abs.(U[3,:,k]).^2)) > TOL
            indicator[k] = indicator[k] + sum(J_k*(M*abs.(U_hat[3,:,k]-U[3,:,k]).^2))/sum(J_k*(M*abs.(U[3,:,k]).^2))
        end
        if sum(J_k*(M*abs.(U[2,:,k]).^2)) > TOL
            indicator[k] = indicator[k] + sum(J_k*(M*abs.(U_hat[2,:,k]-U[2,:,k]).^2))/sum(J_k*(M*abs.(U[2,:,k]).^2))
        end
    end
    # @show indicator[1]
    # @show indicator[541]
    indicator = log.(10,indicator)
    # @show indicator[1]
    # @show indicator[541]
    s0 = log(10,1/N^4)
    @show s0
    @show maximum(indicator)
    @show minimum(indicator)
    # indicator = indicator .> s0 + .5 # N = 3
    # indicator = indicator .> s0 - .5# N = 4
    indicator = indicator .> s0 + 1
    @show sum(indicator)
    # @show indicator[1]
    =#

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
                U_high[c,i,tid] = 0.0
            end
        end

        for j = 2:Np
            for i = 1:j-1
                SxJ_ij_db = 2*(rxJ_k*Sr[i,j] + sxJ_k*Ss[i,j])
                SyJ_ij_db = 2*(ryJ_k*Sr[i,j] + syJ_k*Ss[i,j])
                update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,SyJ_ij_db,U,rholog,betalog)
            end
        end

        for j = 2:Np
            for i = 1:j-1
                λ = λ_arr[i,j,k]
                S0xJ_ij = rxJ_k*S0r[i,j] + sxJ_k*S0s[i,j]
                S0yJ_ij = ryJ_k*S0r[i,j] + syJ_k*S0s[i,j]
                update_F_low!(F_low,k,tid,i,j,λ,S0xJ_ij,S0yJ_ij,U,f_x,f_y)
            end
        end

        for i = 1:Nfp
            iM    = Fmask[i]
            BxJ_ii_halved = (rxJ_k*Br[i,i]+sxJ_k*Bs[i,i])*.5
            ByJ_ii_halved = (ryJ_k*Br[i,i]+syJ_k*Bs[i,i])*.5
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
                U_low[c,i,tid]  = U[c,i,k] - dt*Minv[i,i]/J_k*U_low[c,i,tid]
                U_high[c,i,tid] = U[c,i,k] - dt*Minv[i,i]/J_k*U_high[c,i,tid]
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
       
       if !is_H_positive
            if LIMITOPT == 1 #&& indicator[k]
                # Calculate limiting parameters
                for j = 2:Np
                    for i = 1:j-1
                        coeff_i = dt*(Np-1)*Minv[i,i]/J_k
                        coeff_j = dt*(Np-1)*Minv[j,j]/J_k
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
                        Lrho  = POSTOL
                        Lrhoe = POSTOL
                        l = min(limiting_param(rhoi,rhoui,rhovi,Ei,rhoP_i,rhouP_i,rhovP_i,EP_i,Lrho,Lrhoe),
                                limiting_param(rhoj,rhouj,rhovj,Ej,rhoP_j,rhouP_j,rhovP_j,EP_j,Lrho,Lrhoe))
                        L[i,j,tid] = l
                        L[j,i,tid] = l
                    end
                end
            elseif LIMITOPT == 2 #&& indicator[k]
                li_min = 1.0
                for i = 1:Np
                    rhoi  = U_low[1,i,tid]
                    rhoui = U_low[2,i,tid]
                    rhovi = U_low[3,i,tid]
                    Ei    = U_low[4,i,tid]
                    rhoP_i  = 0.0
                    rhouP_i = 0.0
                    rhovP_i = 0.0
                    EP_i    = 0.0
                    coeff   = dt*Minv[i,i]/J_k
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
                    elseif (LBOUNDTYPE > 0.0)
                        Lrho  = LBOUNDTYPE*rhoi
                        Lrhoe = LBOUNDTYPE*pfun(rhoi,rhoui,rhovi,Ei)/(γ-1)
                    end
                    l = limiting_param(rhoi,rhoui,rhovi,Ei,rhoP_i,rhouP_i,rhovP_i,EP_i,Lrho,Lrhoe)
                    li_min = min(li_min,l)
                end
                for i = 1:Np
                    for j = 1:Np
                        L[i,j,tid] = li_min
                    end
                end
            end
       end

        # Elementwise limiting
        l_e = 1.0
        for j = 1:Np
            for i = 1:Np
                l_e = min(l_e,L[i,j,tid])
            end
        end
        l_em1 = l_e-1.0
        
        if is_H_positive
            l_e = 1.0
            l_em1 = 0.0
        end

        # if !indicator[k]
        #     l_e = 1.0
        #     l_em1 = 0.0
        # end

        if in_s1
            L_plot[k] = l_e
        end

        for j = 2:Np
            for i = 1:j-1
                for c = 1:Nc
                    FL_ij = F_low[c,i,j,tid]
                    FH_ij = F_high[c,i,j,tid]
                    rhsU[c,i,k] = rhsU[c,i,k] + l_em1*FL_ij - l_e*FH_ij
                    rhsU[c,j,k] = rhsU[c,j,k] - l_em1*FL_ij + l_e*FH_ij
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
                rhsU[c,i,k] = Minv[i,i]/J_k*rhsU[c,i,k]
            end
        end
    end

    @show minimum(L_plot)
    @show sum(L_plot)/K

    return dt
end



function rhs_IDP!(U,rhsU,dt,prealloc,ops,geom,in_s1)
    f_x,f_y,rholog,betalog,U_low,F_low,F_high,F_P,L,λ_arr,λf_arr,dii_arr = prealloc
    Sr,Ss,S0r,S0s,Br,Bs,Minv,Extrap = ops
    mapP,Fmask,rxJ,ryJ,sxJ,syJ,J = geom

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
            iM = Fmask[i]
            nij_x = (rxJ_k*Br[i,i]+sxJ_k*Bs[i,i])*.5 
            nij_y = (ryJ_k*Br[i,i]+syJ_k*Bs[i,i])*.5
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
                dt = min(dt,1.0/(Minv[i,i]/J_k)/2.0/dii_arr[i,k])
            end
        end
    end

    @batch for k = 1:K
        tid = Threads.threadid()
        rxJ_k = rxJ[1,k]
        ryJ_k = ryJ[1,k]
        sxJ_k = sxJ[1,k]
        syJ_k = syJ[1,k]

        for i = 1:Np
            for c = 1:Nc
                U_low[c,i,tid] = 0.0
            end
        end

        for j = 2:Np
            for i = 1:j-1
                λ = λ_arr[i,j,k]
                S0xJ_ij = rxJ_k*S0r[i,j] + sxJ_k*S0s[i,j]
                S0yJ_ij = ryJ_k*S0r[i,j] + syJ_k*S0s[i,j]
                update_F_low!(F_low,k,tid,i,j,λ,S0xJ_ij,S0yJ_ij,U,f_x,f_y)
            end
        end

        for i = 1:Nfp
            iM    = Fmask[i]
            BxJ_ii_halved = (rxJ_k*Br[i,i]+sxJ_k*Bs[i,i])*.5
            ByJ_ii_halved = (ryJ_k*Br[i,i]+syJ_k*Bs[i,i])*.5
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

            F_P[1,i,tid] = (BxJ_ii_halved*(fx_1_M+fx_1_P) + ByJ_ii_halved*(fy_1_M+fy_1_P)
                            -λ*(rhoP-rhoM) )
            F_P[2,i,tid] = (BxJ_ii_halved*(fx_2_M+fx_2_P) + ByJ_ii_halved*(fy_2_M+fy_2_P)
                            -λ*(rhouP-rhouM) )
            F_P[3,i,tid] = (BxJ_ii_halved*(fx_3_M+fx_3_P) + ByJ_ii_halved*(fy_3_M+fy_3_P)
                            -λ*(rhovP-rhovM) )
            F_P[4,i,tid] = (BxJ_ii_halved*(fx_4_M+fx_4_P) + ByJ_ii_halved*(fy_4_M+fy_4_P)
                            -λ*(EP-EM) )
        end

        for j = 2:Np
            for i = 1:j-1
                for c = 1:Nc
                    FL_ij = F_low[c,i,j,tid]
                    rhsU[c,i,k] = rhsU[c,i,k] - FL_ij 
                    rhsU[c,j,k] = rhsU[c,j,k] + FL_ij 
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
                rhsU[c,i,k] = Minv[i,i]/J_k*rhsU[c,i,k]
            end
        end
    end

    return dt
end

function rhs_ESDG!(U,rhsU,prealloc,ops,geom)

    f_x,f_y,rholog,betalog,U_low,F_low,F_high,F_P,L,λ_arr,λf_arr,dii_arr = prealloc
    Sr,Ss,S0r,S0s,Br,Bs,Minv,Extrap = ops
    mapP,Fmask,rxJ,ryJ,sxJ,syJ,J = geom

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

    @batch for k = 1:K
        tid = Threads.threadid()
        rxJ_k = rxJ[1,k]
        ryJ_k = ryJ[1,k]
        sxJ_k = sxJ[1,k]
        syJ_k = syJ[1,k]

        for j = 2:Np
            for i = 1:j-1
                SxJ_ij_db = 2*(rxJ_k*Sr[i,j] + sxJ_k*Ss[i,j])
                SyJ_ij_db = 2*(ryJ_k*Sr[i,j] + syJ_k*Ss[i,j])
                update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,SyJ_ij_db,U,rholog,betalog)
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
            
            F_P[1,i,tid] = (rxJ_k*Br[i,i]+sxJ_k*Bs[i,i])*Fx1 + (ryJ_k*Br[i,i]+syJ_k*Bs[i,i])*Fy1
            F_P[2,i,tid] = (rxJ_k*Br[i,i]+sxJ_k*Bs[i,i])*Fx2 + (ryJ_k*Br[i,i]+syJ_k*Bs[i,i])*Fy2
            F_P[3,i,tid] = (rxJ_k*Br[i,i]+sxJ_k*Bs[i,i])*Fx3 + (ryJ_k*Br[i,i]+syJ_k*Bs[i,i])*Fy3
            F_P[4,i,tid] = (rxJ_k*Br[i,i]+sxJ_k*Bs[i,i])*Fx4 + (ryJ_k*Br[i,i]+syJ_k*Bs[i,i])*Fy4
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
                rhsU[c,i,k] = Minv[i,i]/J_k*rhsU[c,i,k]
            end
        end
    end
end


function vortex_sol(x,y,t)
    # x0 = 5
    # y0 = 0
    x0 = 9.0
    y0 = 5.0
    beta = 8.5
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

# Preallocation
rhsU   = zeros(Float64,size(U))
f_x    = zeros(Float64,size(U))
f_y    = zeros(Float64,size(U))
rholog  = zeros(Float64,Np,K)
betalog = zeros(Float64,Np,K)
U_low   = zeros(Float64,Nc,Np,NUM_THREADS)
U_high  = zeros(Float64,Nc,Np,NUM_THREADS)
F_low   = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_high  = zeros(Float64,Nc,Np,Np,NUM_THREADS)
F_P     = zeros(Float64,Nc,Nfp,NUM_THREADS)
L       =  ones(Float64,Np,Np,NUM_THREADS)
λ_arr    = zeros(Float64,Np,Np,K)
λf_arr   = zeros(Float64,Nfp,K)
dii_arr  = zeros(Float64,Np,K)
L_plot   = zeros(Float64,K)
U_hat    = zeros(Float64,size(U))
indicator = zeros(Float64,K)

VDM_hat = copy(VDM)
cnt = 0
for i = 0:N
    for j = 0:N-i
        global cnt = cnt+1
        if (j == N-i)
            VDM_hat[:,cnt] .= 0.0 
        end
    end
end
Vq_hat = VDM_hat/VDM
prealloc = (f_x,f_y,rholog,betalog,U_low,U_high,F_low,F_high,F_P,L,λ_arr,λf_arr,dii_arr,L_plot,U_hat,indicator)
ops      = (Sr,Ss,S0r,S0s,Br,Bs,Minv,E,Pq,Vq,wq,Vq_hat,M)
geom     = (mapP,Fmask,rxJ,ryJ,sxJ,syJ,J)


# TODO: previous Time stepping
"Time integration"
t = 0.0
U = collect(U)
resW = zeros(size(U))

dt_hist = []
i = 1

# open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,x,convtri.txt","w") do io
#     writedlm(io,xq)
# end
# open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,y,convtri.txt","w") do io
#     writedlm(io,yq)
# end

@time while t < T
    # SSPRK(3,3)
    fill!(dii_arr,0.0)
    # dt = min(dt0,T-t)
    dt = dt0
    dt = rhs_Limited!(U,rhsU,T,dt,prealloc,ops,geom,true)
    # dt = rhs_IDP!(U,rhsU,dt,prealloc,ops,geom,true)
    # rhs_ESDG!(U,rhsU,prealloc,ops,geom)
    # dt = min(CFL*dt,T-t)
    @. resW = U + dt*rhsU
    rhs_Limited!(resW,rhsU,T,dt,prealloc,ops,geom,false)
    # rhs_IDP!(resW,rhsU,dt,prealloc,ops,geom,false)
    # rhs_ESDG!(resW,rhsU,prealloc,ops,geom)
    @. resW = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resW
    rhs_Limited!(resW,rhsU,T,dt,prealloc,ops,geom,false)
    # rhs_IDP!(resW,rhsU,dt,prealloc,ops,geom,false)
    # rhs_ESDG!(resW,rhsU,prealloc,ops,geom)
    @. resW = resW+dt*rhsU
    @. U = 1/3*U+2/3*resW

    push!(dt_hist,dt)
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T, at step $i")
    flush(stdout)
    global i = i + 1
    if (mod(i,100) == 1)
        # open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,rho,convtri.txt","w") do io
        #     writedlm(io,U[1,:,:])
        # end
        # open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,rhou,convtri.txt","w") do io
        #     writedlm(io,U[2,:,:])
        # end
        # open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,rhov,convtri.txt","w") do io
        #     writedlm(io,U[3,:,:])
        # end
        # open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,E,convtri.txt","w") do io
        #     writedlm(io,U[4,:,:])
        # end
        # open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,L_plot,convtri.txt","w") do io
        #     writedlm(io,L_plot)
        # end
    end
end

# open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,rho,convtri.txt","w") do io
#     writedlm(io,U[1,:,:])
# end
# open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,rhou,convtri.txt","w") do io
#     writedlm(io,U[2,:,:])
# end
# open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,rhov,convtri.txt","w") do io
#     writedlm(io,U[3,:,:])
# end
# open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,E,convtri.txt","w") do io
#     writedlm(io,U[4,:,:])
# end
# open("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,L_plot,convtri.txt","w") do io
#     writedlm(io,L_plot)
# end


exact_U = @. vortex_sol.(xq,yq,T)
exact_rho = [x[1] for x in exact_U]
exact_u = [x[2] for x in exact_U]
exact_v = [x[3] for x in exact_U]
exact_p = [x[4] for x in exact_U]
exact_rhou = exact_rho .* exact_u
exact_rhov = exact_rho .* exact_v
exact_E = Efun.(exact_rho,exact_u,exact_v,exact_p)

rho = U[1,:,:]
rhou = U[2,:,:]
rhov = U[3,:,:]
u = U[2,:,:]./U[1,:,:]
v = U[3,:,:]./U[1,:,:]
E = U[4,:,:]

M = diag(M)
J = J[1,1]

Linferr = maximum(abs.(exact_rho-rho))/maximum(abs.(exact_rho)) +
          maximum(abs.(exact_rhou-rhou))/maximum(abs.(exact_rhou)) +
          maximum(abs.(exact_rhov-rhov))/maximum(abs.(exact_rhov)) +
          maximum(abs.(exact_E-E))/maximum(abs.(exact_E))

L1err = sum(J*M.*abs.(exact_rho-rho))/sum(J*M.*abs.(exact_rho)) +
        sum(J*M.*abs.(exact_rhou-rhou))/sum(J*M.*abs.(exact_rhou)) +
        sum(J*M.*abs.(exact_rhov-rhov))/sum(J*M.*abs.(exact_rhov)) +
        sum(J*M.*abs.(exact_E-E))/sum(J*M.*abs.(exact_E))

L2err = sqrt(sum(J*M.*abs.(exact_rho-rho).^2))/sqrt(sum(J*M.*abs.(exact_rho).^2)) +
        sqrt(sum(J*M.*abs.(exact_rhou-rhou).^2))/sqrt(sum(J*M.*abs.(exact_rhou).^2)) +
        sqrt(sum(J*M.*abs.(exact_rhov-rhov).^2))/sqrt(sum(J*M.*abs.(exact_rhov).^2)) +
        sqrt(sum(J*M.*abs.(exact_E-E).^2))/sqrt(sum(J*M.*abs.(exact_E).^2))

println("convtri, N = $N, K = $K")
println("L1 error is $L1err")
println("L2 error is $L2err")
println("Linf error is $Linferr")

# df = DataFrame(N = Int64[], K = Int64[], T = Float64[], CFL = Float64[], LIMITOPT = Int64[], POSDETECT = Int64[], LBOUNDTYPE = Float64[], L1err = Float64[], L2err = Float64[], Linferr = Float64[])
df = load("dg2D_euler_tri_convergence.jld2","convergence_data")
push!(df,(N,K,T,CFL,LIMITOPT,POSDETECT,LBOUNDTYPE,L1err,L2err,Linferr))
save("dg2D_euler_tri_convergence.jld2","convergence_data",df)

end