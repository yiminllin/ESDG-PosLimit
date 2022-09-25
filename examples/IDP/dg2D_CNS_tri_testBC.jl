using Pkg
Pkg.activate("Project.toml")
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
using DataFrames
using JLD2
using FileIO

push!(LOAD_PATH, "./src")

using CommonUtils
using Basis2DTri
using UniformTriMesh
using NodesAndModes
using SetupDG

include("CompressibleNavierStokes.jl")
using .CompressibleNavierStokes



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

const LIMITOPT = 2 # 1 if elementwise limiting lij, 2 if elementwise limiting li
const POSDETECT = 0 # 1 if turn on detection, 0 otherwise
const LBOUNDTYPE = 0.1 # 0 if use POSTOL as lower bound, if > 0, use LBOUNDTYPE*loworder
const BCFLUXTYPE = 2 # 0 - Central, 1 - Nondissipative, 2 - dissipative
const VISCPENTYPE = 0.0 # \in [0,1]: alpha = VISCPENTYPE, -1: -1/Re/v_4
const TOL = 5e-16
const POSTOL = 1e-13
const WALLPT = 1.0/6.0
const Nc = 4 # number of components
"Approximation parameters"
const N = 3     # N = 2,3,4
const K1D = 20  # K = 2,4,8,16,32
const T = 1.0
const dt0 = 1e+2
const XLENGTH = 2.0
const CFL = 0.75
const NUM_THREADS = Threads.nthreads()
qnode_choice = "GQ" #"GQ" "GL" "tri_diage"

const ISSMOOTH = true

# Becker viscous shocktube
if ISSMOOTH
    const M_0 = 3.0  # Smooth
    const mu = 0.01
    const Re = 100
else
    const M_0 = 20.0   # Sharp
    const mu = 0.001
    const Re = 1000
end
const lambda = 2/3*mu
const Pr = 3/4
const cp = γ/(γ-1)
const cv = 1/(γ-1)
const kappa = mu*cp/Pr

const v_inf = 0.2
const rho_0 = 1.0
const v_0 = 1.0
const m_0 = rho_0*v_0
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

# Mesh related variables
# Kx = Int(round(XLENGTH*K1D))
Kx = Int(round(2*K1D))
Ky = K1D
VX, VY, EToV = uniform_tri_mesh(Kx,Ky)

@. VX = VX*2
@. VY = VY

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

@inline function get_infoP(mapP,Fmask,i,k)
    gP = mapP[i,k]           # exterior global face node number
    kP = fld1(gP,Nfp)        # exterior element number
    iP = Fmask[mod1(gP,Nfp)] # exterior node number
    return iP,kP
end

@inline function is_left_wall(i,xM)
    return (abs(xM+2.0) < TOL)
end

@inline function is_right_wall(i,xM)
    return (abs(xM-2.0) < TOL)
end

@inline function is_top_wall(i,yM)
    return (abs(yM-1.0) < TOL)
end

@inline function is_bottom_wall(i,yM)
    return (abs(yM+1.0) < TOL)
end

@inline function is_horizontal_wall(i,yM)
    return is_top_wall(i,yM) || is_bottom_wall(i,yM)
end

@inline function is_vertical_wall(i,xM)
    return is_left_wall(i,xM) || is_right_wall(i,xM)
end

@inline function is_wall(i,xM,yM)
    return (is_horizontal_wall(i,yM) || is_vertical_wall(i,xM))
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

@inline function get_consP(UP,U,mapP,Fmask,i,iM,xM,yM,uM,vM,k,tid)
    iP,kP = get_infoP(mapP,Fmask,i,k)

    UP[1,tid] = U[1,iP,kP]
    UP[2,tid] = U[2,iP,kP]
    UP[3,tid] = U[3,iP,kP]
    UP[4,tid] = U[4,iP,kP]

    if is_wall(i,xM,yM)              # If reflective wall
        if is_vertical_wall(i,xM)
            uP        = -uM
            vP        = vM
            rhoP      = U[1,iM,k]
            UP[1,tid] = rhoP
            UP[2,tid] = rhoP*uP
            UP[3,tid] = rhoP*vP
            UP[4,tid] = U[4,iM,k]
        end
        if is_horizontal_wall(i,yM)
            uP        = uM
            vP        = -vM
            rhoP      = U[1,iM,k]
            UP[1,tid] = rhoP
            UP[2,tid] = rhoP*uP
            UP[3,tid] = rhoP*vP
            UP[4,tid] = U[4,iM,k]
        end
    end
end

@inline function get_fluxP(fP,f_x,f_y,mapP,Fmask,i,xM,yM,k,UP,tid)
    iP,kP = get_infoP(mapP,Fmask,i,k)

    fP[1,1,tid] = f_x[1,iP,kP]
    fP[2,1,tid] = f_x[2,iP,kP]
    fP[3,1,tid] = f_x[3,iP,kP]
    fP[4,1,tid] = f_x[4,iP,kP]
    fP[1,2,tid] = f_y[1,iP,kP]
    fP[2,2,tid] = f_y[2,iP,kP]
    fP[3,2,tid] = f_y[3,iP,kP]
    fP[4,2,tid] = f_y[4,iP,kP]

    if is_wall(i,xM,yM)              # If reflective wall
        rhoP  = UP[1,tid]
        rhouP = UP[2,tid]
        rhovP = UP[3,tid]
        EP    = UP[4,tid]
        uP = rhouP/rhoP
        vP = rhovP/rhoP
        pP = pfun(rhoP,rhouP,rhovP,EP)
        fP[1,1,tid],fP[2,1,tid],fP[3,1,tid],fP[4,1,tid],
        fP[1,2,tid],fP[2,2,tid],fP[3,2,tid],fP[4,2,tid] = inviscid_flux_prim(rhoP,uP,vP,pP)
    end
end

@inline function get_vP(VUP,VU,mapP,Fmask,i,iM,xM,yM,k,tid)  # TODO: preallocation
    iP,kP = get_infoP(mapP,Fmask,i,k)

    VUP[1,tid] = VU[1,iP,kP]
    VUP[2,tid] = VU[2,iP,kP]
    VUP[3,tid] = VU[3,iP,kP]
    VUP[4,tid] = VU[4,iP,kP]

    if is_wall(i,xM,yM)
        if is_vertical_wall(i,xM)
            VUP[1,tid] =  VU[1,iM,k]
            VUP[2,tid] = -VU[2,iM,k]
            VUP[3,tid] =  VU[3,iM,k]
            VUP[4,tid] =  VU[4,iM,k]
        end
        if is_horizontal_wall(i,yM)
            if is_top_wall(i,yM)    # Top reflective
                VUP[1,tid] =  VU[1,iM,k]
                VUP[2,tid] =  VU[2,iM,k]
                VUP[3,tid] = -VU[3,iM,k]
                VUP[4,tid] =  VU[4,iM,k]
            end
            if is_bottom_wall(i,yM)
                VUP[1,tid] =  VU[1,iM,k]
                VUP[2,tid] = -VU[2,iM,k]
                VUP[3,tid] = -VU[3,iM,k]
                VUP[4,tid] =  VU[4,iM,k]
            end
        end
    end
end

@inline function get_sigmaP(sigmaP,sigma_x,sigma_y,i,iM,xM,yM,k,mapP,Fmask,tid)
    iP,kP = get_infoP(mapP,Fmask,i,k)
    sigmaP[1,1,tid] = 0.0
    sigmaP[2,1,tid] = sigma_x[2,iP,kP]
    sigmaP[3,1,tid] = sigma_x[3,iP,kP]
    sigmaP[4,1,tid] = sigma_x[4,iP,kP]
    sigmaP[1,2,tid] = 0.0
    sigmaP[2,2,tid] = sigma_y[2,iP,kP]
    sigmaP[3,2,tid] = sigma_y[3,iP,kP]
    sigmaP[4,2,tid] = sigma_y[4,iP,kP]

    if is_wall(i,xM,yM)
        if is_vertical_wall(i,xM)
            sigmaP[1,1,tid] =  0.0
            sigmaP[2,1,tid] =  sigma_x[2,iM,k]
            sigmaP[3,1,tid] = -sigma_x[3,iM,k]
            sigmaP[4,1,tid] = -sigma_x[4,iM,k]
            sigmaP[1,2,tid] =  0.0
            sigmaP[2,2,tid] =  sigma_y[2,iM,k]
            sigmaP[3,2,tid] = -sigma_y[3,iM,k]
            sigmaP[4,2,tid] = -sigma_y[4,iM,k]
        end
        if is_horizontal_wall(i,yM)
            if is_top_wall(i,yM)    # Top reflective
                sigmaP[1,1,tid] =  0.0
                sigmaP[2,1,tid] = -sigma_x[2,iM,k]
                sigmaP[3,1,tid] =  sigma_x[3,iM,k]
                sigmaP[4,1,tid] = -sigma_x[4,iM,k]
                sigmaP[1,2,tid] =  0.0
                sigmaP[2,2,tid] = -sigma_y[2,iM,k]
                sigmaP[3,2,tid] =  sigma_y[3,iM,k]
                sigmaP[4,2,tid] = -sigma_y[4,iM,k]
            end
            if is_bottom_wall(i,yM)
                sigmaP[1,1,tid] =  0.0
                sigmaP[2,1,tid] =  sigma_x[2,iM,k]
                sigmaP[3,1,tid] =  sigma_x[3,iM,k]
                sigmaP[4,1,tid] = -sigma_x[4,iM,k]
                sigmaP[1,2,tid] =  0.0
                sigmaP[2,2,tid] =  sigma_y[2,iM,k]
                sigmaP[3,2,tid] =  sigma_y[3,iM,k]
                sigmaP[4,2,tid] = -sigma_y[4,iM,k]
            end
        end
    end
end

@inline function get_valP(UP,fP,sigmaP,U,f_x,f_y,sigma_x,sigma_y,mapP,Fmask,i,iM,xM,yM,uM,vM,k,tid)
    get_consP(UP,U,mapP,Fmask,i,iM,xM,yM,uM,vM,k,tid)
    get_fluxP(fP,f_x,f_y,mapP,Fmask,i,xM,yM,k,UP,tid)
    get_sigmaP(sigmaP,sigma_x,sigma_y,i,iM,xM,yM,k,mapP,Fmask,tid)
end

@inline function update_F_low!(F_low,k,tid,i,j,λ,S0xJ_ij,S0yJ_ij,U,fx,fy,sigmax,sigmay)
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
    sigmax1_j = sigmax[1,j,k]
    sigmax2_j = sigmax[2,j,k]
    sigmax3_j = sigmax[3,j,k]
    sigmax4_j = sigmax[4,j,k]
    sigmay1_j = sigmay[1,j,k]
    sigmay2_j = sigmay[2,j,k]
    sigmay3_j = sigmay[3,j,k]
    sigmay4_j = sigmay[4,j,k]
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
    sigmax1_i = sigmax[1,i,k]
    sigmax2_i = sigmax[2,i,k]
    sigmax3_i = sigmax[3,i,k]
    sigmax4_i = sigmax[4,i,k]
    sigmay1_i = sigmay[1,i,k]
    sigmay2_i = sigmay[2,i,k]
    sigmay3_i = sigmay[3,i,k]
    sigmay4_i = sigmay[4,i,k]

    FL1 = (S0xJ_ij*(fx_1_i+fx_1_j-sigmax1_i-sigmax1_j) + S0yJ_ij*(fy_1_i+fy_1_j-sigmay1_i-sigmay1_j) - λ*(rho_j-rho_i))
    FL2 = (S0xJ_ij*(fx_2_i+fx_2_j-sigmax2_i-sigmax2_j) + S0yJ_ij*(fy_2_i+fy_2_j-sigmay2_i-sigmay2_j) - λ*(rhou_j-rhou_i))
    FL3 = (S0xJ_ij*(fx_3_i+fx_3_j-sigmax3_i-sigmax3_j) + S0yJ_ij*(fy_3_i+fy_3_j-sigmay3_i-sigmay3_j) - λ*(rhov_j-rhov_i))
    FL4 = (S0xJ_ij*(fx_4_i+fx_4_j-sigmax4_i-sigmax4_j) + S0yJ_ij*(fy_4_i+fy_4_j-sigmay4_i-sigmay4_j) - λ*(E_j-E_i))

    F_low[1,i,j,tid] = FL1
    F_low[2,i,j,tid] = FL2
    F_low[3,i,j,tid] = FL3
    F_low[4,i,j,tid] = FL4

    F_low[1,j,i,tid] = -FL1
    F_low[2,j,i,tid] = -FL2
    F_low[3,j,i,tid] = -FL3
    F_low[4,j,i,tid] = -FL4
end

@inline function update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,SyJ_ij_db,U,rholog,betalog,sigmax,sigmay)
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
    sigmax1_j = sigmax[1,j,k]
    sigmax2_j = sigmax[2,j,k]
    sigmax3_j = sigmax[3,j,k]
    sigmax4_j = sigmax[4,j,k]
    sigmay1_j = sigmay[1,j,k]
    sigmay2_j = sigmay[2,j,k]
    sigmay3_j = sigmay[3,j,k]
    sigmay4_j = sigmay[4,j,k]
    sigmax1_i = sigmax[1,i,k]
    sigmax2_i = sigmax[2,i,k]
    sigmax3_i = sigmax[3,i,k]
    sigmax4_i = sigmax[4,i,k]
    sigmay1_i = sigmay[1,i,k]
    sigmay2_i = sigmay[2,i,k]
    sigmay3_i = sigmay[3,i,k]
    sigmay4_i = sigmay[4,i,k]

    Fx1,Fx2,Fx3,Fx4,Fy1,Fy2,Fy3,Fy4 = euler_fluxes_2D(rho_i,u_i,v_i,beta_i,rholog_i,betalog_i,
                                                      rho_j,u_j,v_j,beta_j,rholog_j,betalog_j)

    SxJ_ij = SxJ_ij_db/2
    SyJ_ij = SyJ_ij_db/2
    FH1 = SxJ_ij_db*Fx1 + SyJ_ij_db*Fy1 - SxJ_ij*(sigmax1_i+sigmax1_j) - SyJ_ij*(sigmay1_i+sigmay1_j)
    FH2 = SxJ_ij_db*Fx2 + SyJ_ij_db*Fy2 - SxJ_ij*(sigmax2_i+sigmax2_j) - SyJ_ij*(sigmay2_i+sigmay2_j)
    FH3 = SxJ_ij_db*Fx3 + SyJ_ij_db*Fy3 - SxJ_ij*(sigmax3_i+sigmax3_j) - SyJ_ij*(sigmay3_i+sigmay3_j)
    FH4 = SxJ_ij_db*Fx4 + SyJ_ij_db*Fy4 - SxJ_ij*(sigmax4_i+sigmax4_j) - SyJ_ij*(sigmay4_i+sigmay4_j)

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
    f_x,f_y,theta_x,theta_y,sigma_x,sigma_y,VU,rholog,betalog,U_low,U_high,F_low,F_high,F_P,L,λ_arr,λf_arr,dii_arr,L_plot,Kxx,Kyy,Kxy,UP,VUP,fP,sigmaP,viscpen = prealloc
    Sr,Ss,S0r,S0s,Br,Bs,Minv,Extrap,Pq,Vq,wq,wf,M = ops
    mapP,Fmask,xq,yq,rxJ,ryJ,sxJ,syJ,J = geom

    @batch for k = 1:K
        tid = Threads.threadid()
        J_k   = J[1,k]
        rxJ_k = rxJ[1,k]
        ryJ_k = ryJ[1,k]
        sxJ_k = sxJ[1,k]
        syJ_k = syJ[1,k]

        for i = 1:Np
            for c = 1:Nc
                theta_x[c,i,k] = 0.0
                theta_y[c,i,k] = 0.0
                sigma_x[c,i,k] = 0.0
                sigma_y[c,i,k] = 0.0
            end
        end

        for c = 1:Nc
            # TODO: why doesn't work?
            # @views theta_x[c,:,k] .= (rxJ_k*Sr+sxJ_k*Ss)*VU[c,:,k]
            # @views theta_y[c,:,k] .= (ryJ_k*Sr+syJ_k*Ss)*VU[c,:,k]
            theta_x[c,:,k] .= (rxJ_k*Sr+sxJ_k*Ss)*VU[c,:,k]
            theta_y[c,:,k] .= (ryJ_k*Sr+syJ_k*Ss)*VU[c,:,k]
        end

        for i = 1:Nfp
            iM = Fmask[i]
            xM = xq[iM,k]
            yM = yq[iM,k]
            get_vP(VUP,VU,mapP,Fmask,i,iM,xM,yM,k,tid)
            for c = 1:Nc
                theta_x[c,iM,k] += .5*(rxJ_k*Br[i,i]+sxJ_k*Bs[i,i])*VUP[c,tid]
                theta_y[c,iM,k] += .5*(ryJ_k*Br[i,i]+syJ_k*Bs[i,i])*VUP[c,tid]
            end
        end

        for i = 1:Np
            mJ_inv_ii = Minv[i,i]/J_k
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

function rhs_Limited!(U,rhsU,T,dt,prealloc,ops,geom,in_s1)
    f_x,f_y,theta_x,theta_y,sigma_x,sigma_y,VU,rholog,betalog,U_low,U_high,F_low,F_high,F_P,L,λ_arr,λf_arr,dii_arr,L_plot,Kxx,Kyy,Kxy,UP,VUP,fP,sigmaP,viscpen = prealloc
    Sr,Ss,S0r,S0s,Br,Bs,Minv,Extrap,Pq,Vq,wq,wf,M = ops
    mapP,Fmask,xq,yq,rxJ,ryJ,sxJ,syJ,J = geom

    fill!(rhsU,0.0)
    @batch for k = 1:K
        for i = 1:Np
            rho  = U[1,i,k]
            rhou = U[2,i,k]
            rhov = U[3,i,k]
            E    = U[4,i,k]
            p          = pfun(rho,rhou,rhov,E)
            v1,v2,v3,v4 = entropyvar(rho,rhou,rhov,E,p)
            f_x[1,i,k] = rhou
            f_x[2,i,k] = rhou^2/rho+p
            f_x[3,i,k] = rhou*rhov/rho
            f_x[4,i,k] = E*rhou/rho+p*rhou/rho
            f_y[1,i,k] = rhov
            f_y[2,i,k] = rhou*rhov/rho
            f_y[3,i,k] = rhov^2/rho+p
            f_y[4,i,k] = E*rhov/rho+p*rhov/rho
            VU[1,i,k]  = v1
            VU[2,i,k]  = v2
            VU[3,i,k]  = v3
            VU[4,i,k]  = v4
            rholog[i,k]  = log(rho)
            betalog[i,k] = log(rho/(2*p))
        end
    end

    compute_sigma(prealloc,ops,geom)

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
                                     wavespeed_1D(rho_j,nij_x/nij_norm*rhou_j+nij_y/nij_norm*rhov_j,E_j),
                                     zhang_wavespd(rho_i,rhou_i,rhov_i,E_i,
                                                   sigma_x[2,i,k],sigma_x[3,i,k],sigma_x[4,i,k],
                                                   sigma_y[2,i,k],sigma_y[3,i,k],sigma_y[4,i,k],
                                                   nij_x/nij_norm, nij_y/nij_norm),
                                     zhang_wavespd(rho_j,rhou_j,rhov_j,E_j,
                                                   sigma_x[2,j,k],sigma_x[3,j,k],sigma_x[4,j,k],
                                                   sigma_y[2,j,k],sigma_y[3,j,k],sigma_y[4,j,k],
                                                   -nij_x/nij_norm, -nij_y/nij_norm))
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
        tid = Threads.threadid()
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
            xM    = xq[iM,k]
            yM    = yq[iM,k]
            uM    = rhouM/rhoM
            vM    = rhovM/rhoM
            rhoP  = U[1,iP,kP]
            rhouP = U[2,iP,kP]
            rhovP = U[3,iP,kP]
            EP    = U[4,iP,kP]

            get_valP(UP,fP,sigmaP,U,f_x,f_y,sigma_x,sigma_y,mapP,Fmask,i,iM,xM,yM,uM,vM,k,tid)

            λ = nij_norm*max(wavespeed_1D(rhoM,nij_x/nij_norm*rhouM+nij_y/nij_norm*rhovM,EM),
                             wavespeed_1D(rhoP,nij_x/nij_norm*rhouP+nij_y/nij_norm*rhovP,EP),
                             zhang_wavespd(rhoM,rhouM,rhovM,EM,
                                           sigma_x[2,iM,k],sigma_x[3,iM,k],sigma_x[4,iM,k],
                                           sigma_y[2,iM,k],sigma_y[3,iM,k],sigma_y[4,iM,k],
                                           nij_x/nij_norm,nij_y/nij_norm),
                             zhang_wavespd(UP[1,tid],UP[2,tid],UP[3,tid],UP[4,tid],
                                           sigmaP[2,1,tid],sigmaP[3,1,tid],sigmaP[4,1,tid],
                                           sigmaP[2,2,tid],sigmaP[3,2,tid],sigmaP[4,2,tid],
                                           -nij_x/nij_norm,-nij_y/nij_norm))
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
                update_F_high!(F_high,k,tid,i,j,SxJ_ij_db,SyJ_ij_db,U,rholog,betalog,sigma_x,sigma_y)
            end
        end

        for j = 2:Np
            for i = 1:j-1
                λ = λ_arr[i,j,k]
                S0xJ_ij = rxJ_k*S0r[i,j] + sxJ_k*S0s[i,j]
                S0yJ_ij = ryJ_k*S0r[i,j] + syJ_k*S0s[i,j]
                update_F_low!(F_low,k,tid,i,j,λ,S0xJ_ij,S0yJ_ij,U,f_x,f_y,sigma_x,sigma_y)
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

            get_valP(UP,fP,sigmaP,U,f_x,f_y,sigma_x,sigma_y,mapP,Fmask,i,iM,xM,yM,uM,vM,k,tid)
            λ = λf_arr[i,k]

            for c = 1:Nc
                F_P[c,i,tid] = (BxJ_ii_halved*(f_x[c,iM,k]+fP[c,1,tid]-sigma_x[c,iM,k]-sigmaP[c,1,tid])
                              + ByJ_ii_halved*(f_y[c,iM,k]+fP[c,2,tid]-sigma_y[c,iM,k]-sigmaP[c,2,tid])
                              - λ*(UP[c,tid]-U[c,iM,k]))
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
            if LIMITOPT == 1
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
            elseif LIMITOPT == 2
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

function vortex_sol(x,y,t)
    # x0 = 5
    # y0 = 0
    x0 = 9.0
    y0 = 5.0
    beta = 5.0
    r2 = @. (x-x0-t)^2 + (y-y0)^2

    u = @. 1 - beta*exp(1-r2)*(y-y0)/(2*pi)
    v = @. beta*exp(1-r2)*(x-x0-t)/(2*pi)
    rho = @. 1 - (1/(8*γ*pi^2))*(γ-1)/2*(beta*exp(1-r2))^2
    rho = @. rho^(1/(γ-1))
    p = @. rho^γ

    return (rho, u, v, p)
end

function initial_cond(xi,yi,t)

    Ma = 1.5
    if (xi < 0.0)
        rho0 = 5.0
        u0   = 0.0
        v0   = 0.0
        p0   = rho0/Ma^2/γ
    else
        rho0 = 1.0
        u0   = 0.0
        v0   = 0.0
        p0   = rho0/Ma^2/γ
    end

    return (rho0, u0, v0, p0)
end




# const Np  = size(x,1)
const Np  = size(xq,1)
const Nfp = size(Vf,1)
U = zeros(Nc,Np,K)
for k = 1:K
    for i = 1:Np
        # rho,u,v,p = vortex_sol(xq[i,k],yq[i,k],0.0)
        rho,u,v,p = initial_cond(xq[i,k],yq[i,k],0.0)
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
L       =  ones(Float64,Np,Np,NUM_THREADS)
λ_arr    = zeros(Float64,Np,Np,K) # Assume S0r and S0s has same number of nonzero entries
λf_arr   = zeros(Float64,Nfp,K)
dii_arr  = zeros(Float64,Np,K)
L_plot   = zeros(Float64,K)
Kxx      = [zeros(MMatrix{Nc,Nc,Float64}) for _ in 1:NUM_THREADS]
Kyy      = [zeros(MMatrix{Nc,Nc,Float64}) for _ in 1:NUM_THREADS]
Kxy      = [zeros(MMatrix{Nc,Nc,Float64}) for _ in 1:NUM_THREADS]
viscpen  = zeros(Float64,Nc,NUM_THREADS)

prealloc = (f_x,f_y,theta_x,theta_y,sigma_x,sigma_y,VU,rholog,betalog,U_low,U_high,F_low,F_high,F_P,L,λ_arr,λf_arr,dii_arr,L_plot,Kxx,Kyy,Kxy,UP,VUP,fP,sigmaP,viscpen)
ops      = (Sr,Ss,S0r,S0s,Br,Bs,Minv,E,Pq,Vq,wq,wf,M)
geom     = (mapP,Fmask,xq,yq,rxJ,ryJ,sxJ,syJ,J)

# TODO: previous Time stepping
"Time integration"
t = 0.0
U = collect(U)
resW = zeros(size(U))

dt_hist = []
i = 1

@time while t < T
    # SSPRK(3,3)
    fill!(dii_arr,0.0)
    # dt = min(dt0,T-t)
    dt = dt0
    dt = rhs_Limited!(U,rhsU,T,dt,prealloc,ops,geom,true)
    @. resW = U + dt*rhsU
    rhs_Limited!(resW,rhsU,T,dt,prealloc,ops,geom,false)
    @. resW = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resW
    rhs_Limited!(resW,rhsU,T,dt,prealloc,ops,geom,false)
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

@unpack Vp = rd
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)

vv = Vp*Pq*U[1,:,:]
up = Vp*Pq*(U[2,:,:]./U[1,:,:])
vp = Vp*Pq*(U[3,:,:]./U[1,:,:])
vmag = @. up^2 + vp^2

scatter(Vp*x,Vp*y,vmag,zcolor=vmag,camera=(0,90))
savefig("/home/yiminlin/Desktop/dg2D_CNS_tri_convergence_dev.png")

end