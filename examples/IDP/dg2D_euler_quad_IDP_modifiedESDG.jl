using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack


push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh 

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
include("../EntropyStableEuler.jl/src/logmean.jl")
#using EntropyStableEuler
#using EntropyStableEuler.Fluxes1D

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

function is_zero(a)
    return abs(a) < TOL
end

function Riemann_2D(x,y)
    # if x >= 0 && y >= 0
    #     return .5313, 0.0, 0.0, 0.4
    # elseif x < 0 && y >= 0
    #     return 1.0, .7276, 0.0, 1.0
    # elseif x < 0 && y < 0
    #     return .8, 0.0, 0.0, 1.0
    # else
    #     return 1.0, 0.0, .7276, 1.0
    # end

    # if x >= 0 && y >= 0
    #     return .5313, 0.0, 0.0, 0.004#0.4
    # elseif x < 0 && y >= 0
    #     return 1.0, .7276, 0.0, 1.0
    # elseif x < 0 && y < 0
    #     return .8, 0.0, 0.0, 1.0
    # else
    #     return 1.0, 0.0, .7276, 1.0
    # end

    if x >= 0 && y >= 0
        return 1.5, 0.0, 0.0, 1.5
    elseif x < 0 && y >= 0
        return 0.5323, 1.206, 0.0, 0.3
    elseif x < 0 && y < 0
        return .138, 1.206, 1.206, 0.029
    else
        return 0.5323, 0.0, 1.206, 0.3
    end
end

function vortex(x,y,t,γ=1.4)
    x0 = 0
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

const γ = 1.4
const TOL = 1e-14
const Nc = 4 # number of components

"Approximation parameters"
# N = 3 # The order of approximation
# K1D = 20
N = 4
K1D = 10
# N = 3
# K1D = 64
#T = 0.25
T = 0.3

"Mesh related variables"
Kx = K1D
Ky = K1D
VX, VY, EToV = uniform_quad_mesh(Kx, Ky)

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

# Make domain periodic, TODO: hardcoded
Jf = 1/K1D # Assume uniform quad mesh
K = K1D^2
Nfaces = 4
Nfp = Nfaces*N

# physical normal, assume standard uniform quad mesh
#=
nxJ = zeros(Nfp,K)
nyJ = zeros(Nfp,K)
for k = 1:K
    nxJ[1,k] = -Jf
    nxJ[2:N,k] .= zeros(N-1)
    nxJ[N+1,k] = Jf
    for i = 0:N-2
        nxJ[N+2+i*2,k] = -Jf
        nxJ[N+2+i*2+1,k] = Jf
    end
    nxJ[end-N,k] = -Jf
    nxJ[end-(N-1):end-1,k] .= zeros(N-1)
    nxJ[end,k] = Jf

    nyJ[1:N+1,k] .= -Jf*ones(N+1)
    nyJ[end-N:end,k] .= Jf*ones(N+1)
end
=#
nxJ = zeros((N+1)*(N+1),K)
nyJ = zeros((N+1)*(N+1),K)
for k = 1:K
    for i = 0:N
        nxJ[i*(N+1)+1,k] = -Jf
        nxJ[i*(N+1)+N+1,k] = Jf
    end
    nyJ[1:N+1,k] .= -Jf*ones(N+1)
    nyJ[end-N:end,k] .= Jf*ones(N+1)
end

# indices of nodal points on faces
face_idx = zeros(Nfp)
face_idx[1:N+1] .= 1:N+1
face_idx[N+2:2*N+2] .= N*(N+1)+1:(N+1)*(N+1)
for i = 1:N-1
    face_idx[2*N+3+2*(i-1)] = i*(N+1)+1
    face_idx[2*N+3+2*(i-1)+1] = i*(N+1)+N+1
end
sort!(face_idx)
face_idx = Int64.(face_idx)

# face connectivity via global node number
mapM = zeros(Int64,Nfp,K)
mapP = zeros(Int64,Nfp,K)
mapP_x = zeros(Int64,Nfp,K)
mapP_y = zeros(Int64,Nfp,K)
for kj = 1:K1D
    for ki = 1:K1D
        k = (kj-1)*K1D+ki
        mapM[:,k] .= (k-1)*(N+1)*(N+1).+face_idx
        mapP[1:N+1,k] = mod(k-1-K1D,K)*(N+1)*(N+1).+face_idx[end-N:end]
        mapP[end-N:end,k] = mod(k-1+K1D,K)*(N+1)*(N+1).+face_idx[1:N+1]
        mapP[N+2:2:end-N-2,k] = ((kj-1)*K1D+mod(ki-2,K1D))*(N+1)*(N+1).+face_idx[N+3:2:end-N-1] # left interface
        mapP[N+3:2:end-N-1,k] = ((kj-1)*K1D+mod(ki,K1D))*(N+1)*(N+1).+face_idx[N+2:2:end-N-2] # right interface
        mapP_x[[1;N+2:2:end-N],k] = ((kj-1)*K1D+mod(ki-2,K1D))*(N+1)*(N+1).+face_idx[[N+1:2:end-N-1;end]] 
        mapP_x[[N+1:2:end-N-1;end],k] = ((kj-1)*K1D+mod(ki,K1D))*(N+1)*(N+1).+face_idx[[1;N+2:2:end-N]]
        mapP_y[1:N+1,k] = mod(k-1-K1D,K)*(N+1)*(N+1).+face_idx[end-N:end]
        mapP_y[end-N:end,k] = mod(k-1+K1D,K)*(N+1)*(N+1).+face_idx[1:N+1]
    end
end

rd = init_reference_quad(N,gauss_lobatto_quad(0,0,N))
md = init_mesh((VX,VY),EToV,rd)
@unpack x,y,rxJ,ryJ,sxJ,syJ = md

function translate_bump(x,y)
    rho = exp(1/10/(x^2+y^2+0.1))
    u = 1.0
    v = 0.0
    p = 1.0
    return rho,u,v,p
end
#U = @. translate_bump(x,y)
U = @. Riemann_2D(x,y)
#U = @. vortex(x,y,0.0)
rho,u,v,p = [(x->x[i]).(U) for i = 1:Nc]
U = @. primitive_to_conservative_hardcode(rho,u,v,p)
rho,rhou,rhov,E = [(x->x[i]).(U) for i = 1:Nc]
U = (rho,rhou,rhov,E)


function limiting_param(U_low, P_ij)
    l = 1.0
    # Limit density
    if U_low[1] + P_ij[1] < -TOL
        l = min(abs(U_low[1])/(abs(P_ij[1])+1e-14), 1.0)
    end

    # limiting internal energy (via quadratic function)
    # a = P_ij[1]*P_ij[3]-1.0/2.0*P_ij[2]^2
    # b = U_low[3]*P_ij[1]+U_low[1]*P_ij[3]-U_low[2]*P_ij[2]
    # c = U_low[3]*U_low[1]-1.0/2.0*U_low[2]^2
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

function rhs_IDP(U,K1D,N,Nfaces,nxJ,nyJ,Minv,Sr,Ss,S0r,S0s,Br,Bs,face_idx,mapM,mapP_x,mapP_y,step)
    p = pfun_nd.(U[1],U[2],U[3],U[4])
    flux_x = zero.(U)
    flux_y = zero.(U)
    @. flux_x[1] = U[2]
    @. flux_x[2] = U[2]^2/U[1]+p
    @. flux_x[3] = U[2]*U[3]/U[1]
    @. flux_x[4] = U[4]*U[2]/U[1]+p*U[2]/U[1]
    @. flux_y[1] = U[3]
    @. flux_y[2] = U[2]*U[3]/U[1]
    @. flux_y[3] = U[3]^2/U[1]+p
    @. flux_y[4] = U[4]*U[3]/U[1]+p*U[3]/U[1]

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[3]/U[1]
    @. Ub[4] = U[1]/(2*p)

    K = K1D^2
    Nfp = Nfaces*N
    Np = (N+1)*(N+1)
    J = (1/K1D)^2
    Jf = 1/K1D
    rxJ = 1/K1D
    syJ = 1/K1D
    sJ = 1/K1D

    F_low  = [zeros(Np,Np) for i = 1:Nc]
    F_high = [zeros(Np,Np) for i = 1:Nc]
    F_P    = [zeros(Nfp) for i = 1:Nc]
    rhsU   = [zeros(Np,K) for i = 1:Nc]
    U_low  = [zeros(Np) for i = 1:Nc]
    L = zeros(Np,Np)

    d_ii_arr = zeros(Np,K)
    # TODO: precompute
    for k = 1:K
        for i = 1:Np
            for j = 1:Np
                c_ij_norm = sqrt(S0r[i,j]^2+S0s[i,j]^2)
                if abs(c_ij_norm) >= TOL
                    n_ij = [S0r[i,j]; S0s[i,j]]./c_ij_norm
                    wavespd_i = wavespeed_1D(U[1][i,k],(n_ij[1]*U[2][i,k]+n_ij[2]*U[3][i,k]),U[4][i,k])
                    wavespd_j = wavespeed_1D(U[1][j,k],(n_ij[1]*U[2][j,k]+n_ij[2]*U[3][j,k]),U[4][j,k])
                    wavespd = max(wavespd_i,wavespd_j)
                    d_ij = wavespd*c_ij_norm
                    d_ii_arr[i,k] -= d_ij
                end
            end
        end

        S0r1 = sum(S0r,dims=2)
        S0s1 = sum(S0s,dims=2)
        for i = 1:Nfp
            fidM = mapM[i,k]
            S0rP = -S0r1[face_idx[i]]
            S0sP = -S0s1[face_idx[i]]

            # flux in x direction
            fidP = mapP_x[i,k]
            if fidP != 0
                wavespd_M = wavespeed_1D(U[1][fidM],U[2][fidM],U[4][fidM])
                wavespd_P = wavespeed_1D(U[1][fidP],U[2][fidP],U[4][fidP])
                wavespd = max(wavespd_M,wavespd_P)
                d_ij = wavespd*abs(S0rP)
                d_ii_arr[face_idx[i],k] -= d_ij
            end

            # flux in y direction
            fidP = mapP_y[i,k]
            if fidP != 0
                wavespd_M = wavespeed_1D(U[1][fidM],U[3][fidM],U[4][fidM])
                wavespd_P = wavespeed_1D(U[1][fidP],U[3][fidP],U[4][fidP])
                wavespd = max(wavespd_M,wavespd_P)
                d_ij = wavespd*abs(S0sP)
                d_ii_arr[face_idx[i],k] -= d_ij
            end
        end
    end

    dt = minimum(-J/2*M*(1 ./d_ii_arr))
    dt = dt

    for k = 1:K
        for c = 1:Nc
            for i = 1:Nfp
                F_P[c][i] = 0.0
            end
            for i = 1:Np
                U_low[c][i] = 0.0
            end
        end

        # Calculate interior low and high order flux
        for i = 1:Np
            for j = 1:Np
                c_ij_norm = sqrt(S0r[i,j]^2+S0s[i,j]^2)
                if abs(c_ij_norm) >= TOL
                    n_ij = [S0r[i,j]; S0s[i,j]]./c_ij_norm
                    wavespd_i = wavespeed_1D(U[1][i,k],(n_ij[1]*U[2][i,k]+n_ij[2]*U[3][i,k]),U[4][i,k])
                    wavespd_j = wavespeed_1D(U[1][j,k],(n_ij[1]*U[2][j,k]+n_ij[2]*U[3][j,k]),U[4][j,k])
                    wavespd = max(wavespd_i,wavespd_j)
                    d_ij = wavespd*c_ij_norm

                    for c = 1:Nc
                        F_low[c][i,j]  = (rxJ*S0r[i,j]*(flux_x[c][i,k]+flux_x[c][j,k]) 
                                        + syJ*S0s[i,j]*(flux_y[c][i,k]+flux_y[c][j,k])
                                        - d_ij*(U[c][j,k]-U[c][i,k]))
                    end
                end
            end
        end

        # High order flux
        for i = 1:Np-1
            for j = i+1:Np
                # TODO: can preallocate nonzero entries
                if Sr[i,j] != 0 || Ss[i,j] != 0
                    F1,F2 = euler_fluxes(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[4][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],Ub[4][j,k])
                    for c = 1:Nc
                        val = 2*rxJ*Sr[i,j]*F1[c]+2*syJ*Ss[i,j]*F2[c]
                        F_high[c][i,j] = val
                        F_high[c][j,i] = -val
                    end
                end
            end
        end

        # Calculate interface flux
        S0r1 = sum(S0r,dims=2)
        S0s1 = sum(S0s,dims=2)
        for i = 1:Nfp
            fidM = mapM[i,k]
            S0rP = -S0r1[face_idx[i]]
            S0sP = -S0s1[face_idx[i]]
            
            # flux in x direction
            fidP = mapP_x[i,k]
            if fidP != 0
                wavespd_M = wavespeed_1D(U[1][fidM],U[2][fidM],U[4][fidM])
                wavespd_P = wavespeed_1D(U[1][fidP],U[2][fidP],U[4][fidP])
                wavespd = max(wavespd_M,wavespd_P)
                d_ij = wavespd*abs(S0rP)
                for c = 1:Nc
                    F_P[c][i] += (rxJ*S0rP*(flux_x[c][fidM]+flux_x[c][fidP])
                                - d_ij*(U[c][fidP]-U[c][fidM]))
                end
            end

            # flux in y direction
            fidP = mapP_y[i,k]
            if fidP != 0
                wavespd_M = wavespeed_1D(U[1][fidM],U[3][fidM],U[4][fidM])
                wavespd_P = wavespeed_1D(U[1][fidP],U[3][fidP],U[4][fidP])
                wavespd = max(wavespd_M,wavespd_P)
                d_ij = wavespd*abs(S0sP)
                for c = 1:Nc
                    F_P[c][i] += (syJ*S0sP*(flux_y[c][fidM]+flux_y[c][fidP])
                                - d_ij*(U[c][fidP]-U[c][fidM]))
                end
            end
        end

        # Calculate low order solution
        for i = 1:Np
            for c = 1:Nc
                U_low[c][i] = sum(F_low[c][i,:])
            end
        end
        # TODO: redundant
        for i = 1:Nfp
            for c = 1:Nc
                U_low[c][face_idx[i]] += F_P[c][i]
            end
        end
        for c = 1:Nc
            for i = 1:Np
                U_low[c][i] = U[c][i,k]-dt/J/M[i,i]*U_low[c][i]
            end
        end

        # Calculate limiting parameters
        P_ij = [0.0;0.0;0.0;0.0]
        for i = 1:Np
            m_i = J*M[i,i]
            lambda_j = 1/(Np-1)
            U_low_i = [U_low[c][i] for c = 1:Nc]
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
            for j = 1:Np
                if i != j 
                    l = min(L[i,j],L[j,i])
                    L[i,j] = l
                    L[j,i] = l
                end
            end
        end

        #L = ones(size(L))
        # if sum(L)-Np*Np+Np < 0
        #     @show sum(L)-Np*Np+Np
        # end

        for c = 1:Nc
            rhsU[c][:,k] = sum((L.-1).*F_low[c] - L.*F_high[c],dims=2)
            #rhsU[c][:,k] = sum((-1).*F_high[c],dims=2)
            #rhsU[c][:,k] = sum((-1).*F_low[c],dims=2)
            for i = 1:Nfp
                rhsU[c][face_idx[i],k] -= F_P[c][i]
            end
        end
    end

    for c = 1:Nc
        rhsU[c] = 1/J*Minv*rhsU[c]
    end

    return rhsU,dt
end





function rhs_IDPlow(U,K1D,N,Nfaces,nxJ,nyJ,Minv,Sr,Ss,S0r,S0s,Br,Bs,face_idx,mapM,mapP_x,mapP_y)
    p = pfun_nd.(U[1],U[2],U[3],U[4])
    flux_x = zero.(U)
    flux_y = zero.(U)
    @. flux_x[1] = U[2]
    @. flux_x[2] = U[2]^2/U[1]+p
    @. flux_x[3] = U[2]*U[3]/U[1]
    @. flux_x[4] = U[4]*U[2]/U[1]+p*U[2]/U[1]
    @. flux_y[1] = U[3]
    @. flux_y[2] = U[2]*U[3]/U[1]
    @. flux_y[3] = U[3]^2/U[1]+p
    @. flux_y[4] = U[4]*U[3]/U[1]+p*U[3]/U[1]

    K = K1D^2
    Nfp = Nfaces*N
    Np = (N+1)*(N+1)
    J = (1/K1D)^2
    Jf = 1/K1D
    rxJ = 1/K1D
    syJ = 1/K1D
    sJ = 1/K1D

    # Initialize sparse matrices to store numerical fluxes across interfaces
    I_arr = Int64[]
    J_arr = Int64[]
    for k = 1:K
        for n = 1:Nfp
            i = mapM[n,k]
            jx = mapP_x[n,k]
            jy = mapP_y[n,k]
            if jx != 0
                push!(I_arr,i)
                push!(J_arr,jx)
            end
            if jy != 0
                push!(I_arr,i)
                push!(J_arr,jy)
            end
        end
    end
    F_P_low = [sparse(I_arr,J_arr,zeros(size(I_arr))) for i = 1:Nc]
    # Low order and high order algebraic fluxes
    F_low   = [zeros(Np,Np) for i = 1:Nc]
    rhsU    = [zeros(Np,K) for i = 1:Nc]
    
    for k = 1:K
        # TODO: redundant iterations
        for i = 1:Np
            for j = 1:Np
                c_ij_norm = sqrt(S0r[i,j]^2 + S0s[i,j]^2)
                if abs(c_ij_norm) >= TOL
                    n_ij = [S0r[i,j]; S0s[i,j]]./c_ij_norm
                    #c_ji_norm = sqrt(S0r[j,i]^2 + S0s[j,i]^2)
                    #n_ji  = [S0r[j,i]; S0s[j,i]]./c_ji_norm
                    wavespd_i = wavespeed_1D(U[1][i,k],(n_ij[1]*U[2][i,k]+n_ij[2]*U[3][i,k]),U[4][i,k])
                    wavespd_j = wavespeed_1D(U[1][j,k],(n_ij[1]*U[2][j,k]+n_ij[2]*U[3][j,k]),U[4][j,k])
                    wavespd = max(wavespd_i,wavespd_j)
                    d_ij = wavespd*c_ij_norm

                    for c = 1:Nc
                        F_low[c][i,j] = (rxJ*S0r[i,j]*(flux_x[c][i,k]+flux_x[c][j,k]) 
                                       + syJ*S0s[i,j]*(flux_y[c][i,k]+flux_y[c][j,k])
                                       - d_ij*(U[c][j,k]-U[c][i,k]))
                    end
                end
            end
        end

        S0r1 = sum(S0r,dims=2) 
        S0s1 = sum(S0s,dims=2)
        # Interface low order flux
        for i in 1:Nfp
            fidM = mapM[i,k]
            
            S0rP = -S0r1[face_idx[i]]
            S0sP = -S0s1[face_idx[i]] 

            # numerical flux in x direction
            fidP = mapP_x[i,k]
            if fidP != 0
                c_ij_norm = abs(S0rP)
                if abs(c_ij_norm) >= TOL
                    n_ij_x = S0rP/c_ij_norm
                    wavespd_M = wavespeed_1D(U[1][fidM],n_ij_x*U[2][fidM],U[4][fidM])
                    wavespd_P = wavespeed_1D(U[1][fidP],n_ij_x*U[2][fidP],U[4][fidP])
                    wavespd = max(wavespd_M,wavespd_P)
                    d_ij = wavespd*c_ij_norm
                    for c = 1:Nc
                        F_P_low[c][fidM,fidP] += (rxJ*S0rP*(flux_x[c][fidM]+flux_x[c][fidP])
                                                - d_ij*(U[c][fidP]-U[c][fidM]))
                    end
                end
            end

            # numerical flux in y direction
            fidP = mapP_y[i,k]
            if fidP != 0
                c_ij_norm = abs(S0sP)
                if abs(c_ij_norm) >= TOL
                    n_ij_y = S0sP/c_ij_norm
                    wavespd_M = wavespeed_1D(U[1][fidM],n_ij_y*U[3][fidM],U[4][fidM])
                    wavespd_P = wavespeed_1D(U[1][fidP],n_ij_y*U[3][fidP],U[4][fidP])
                    wavespd = max(wavespd_M,wavespd_P)
                    d_ij = wavespd*c_ij_norm
                    for c = 1:Nc
                        F_P_low[c][fidM,fidP] += (syJ*S0sP*(flux_y[c][fidM]+flux_y[c][fidP])
                                                - d_ij*(U[c][fidP]-U[c][fidM]))
                    end
                end               
            end
        end

        for i = 1:Np
            for c = 1:Nc
                rhsU[c][i,k] += sum(F_low[c][i,:])
                rhsU[c][i,k] += sum(F_P_low[c][i+(k-1)*Np,:])
            end
        end

    end

    for c = 1:Nc
        rhsU[c] = -1/J*Minv*rhsU[c]
    end

    return rhsU
end

function rhs_ESDG(U,K1D,N,Nfaces,nxJ,nyJ,Minv,Sr,Ss,S0r,S0s,Br,Bs,face_idx,mapM,mapP_x,mapP_y)
    p = pfun_nd.(U[1],U[2],U[3],U[4])

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[3]/U[1]
    @. Ub[4] = U[1]/(2*p)

    K = K1D^2
    Nfp = Nfaces*N
    Np = (N+1)*(N+1)
    J = (1/K1D)^2
    Jf = 1/K1D

    # Low order and high order algebraic fluxes
    # Initialize sparse matrices to store numerical fluxes across interfaces
    I_arr = Int64[]
    J_arr = Int64[]
    for k = 1:K
        for n = 1:Nfp
            i = mapM[n,k]
            jx = mapP_x[n,k]
            jy = mapP_y[n,k]
            if jx != 0
                push!(I_arr,i)
                push!(J_arr,jx)
            end
            if jy != 0
                push!(I_arr,i)
                push!(J_arr,jy)
            end
        end
    end
    F_P_high = [sparse(I_arr,J_arr,zeros(size(I_arr))) for i = 1:Nc]
    F_high   = [zeros(Np,Np) for i = 1:Nc]
    rhsU     = [zeros(Np,K) for i = 1:Nc]

    # TODO: assume mesh is uniform, so geometric factors constant
    #       and mesh is not rotated, i.e. ryJ = sxJ = 0.0
    rxJ = 1/K1D
    syJ = 1/K1D
    sJ = 1/K1D

    for k = 1:K
        for i = 1:Np-1
            for j = i+1:Np
                # TODO: can preallocate nonzero entries
                if Sr[i,j] != 0 || Ss[i,j] != 0
                    F1,F2 = euler_fluxes(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[4][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],Ub[4][j,k])
                    for c = 1:Nc
                        val = 2*rxJ*Sr[i,j]*F1[c]+2*syJ*Ss[i,j]*F2[c]
                        F_high[c][i,j] = val
                        F_high[c][j,i] = -val
                    end
                end
            end
        end

        # Interface high order flux
        for i in 1:Nfp
            fidM = mapM[i,k]
            nxJM = nxJ[fidM]
            nyJM = nyJ[fidM]
            rhou_nM_x = (U[2][fidM]*nxJM)/sJ
            rhou_nM_y = (U[3][fidM]*nyJM)/sJ
            wavespdM_x = wavespeed_1D(U[1][fidM],rhou_nM_x,U[4][fidM])
            wavespdM_y = wavespeed_1D(U[1][fidM],rhou_nM_y,U[4][fidM])

            # numerical flux in x direction
            fidP = mapP_x[i,k]
            if fidP != 0
                nxJP = nxJ[fidP]
                nyJP = nyJ[fidP]
                #rhou_nP = (U[2][fidP]*nxJP+U[3][fidP]*nyJP)/sJ
                rhou_nP = (U[2][fidP]*nxJP)/sJ
                wavespdP = wavespeed_1D(U[1][fidP],rhou_nP,U[4][fidP])
                wavespd = max(wavespdM_x,wavespdP)
                F1,_ = euler_fluxes(Ub[1][fidM],Ub[2][fidM],Ub[3][fidM],Ub[4][fidM],Ub[1][fidP],Ub[2][fidP],Ub[3][fidP],Ub[4][fidP])
                for c = 1:Nc
                    F_P_high[c][fidM,fidP] += Jf*Br[face_idx[i],face_idx[i]]*F1[c]-sJ*wavespd/2*(U[c][fidP]-U[c][fidM])#nxJM*Br[face_idx[i],face_idx[i]]*F1[c]#-wavespd/2*(U[c][fidP]-U[c][fidM])
                end
            end

            # numerical flux in y direction
            fidP = mapP_y[i,k]
            if fidP != 0
                nxJP = nxJ[fidP]
                nyJP = nyJ[fidP]
                #rhou_nP = (U[2][fidP]*nxJP+U[3][fidP]*nyJP)/sJ
                rhou_nP = (U[3][fidP]*nyJP)/sJ
                wavespdP = wavespeed_1D(U[1][fidP],rhou_nP,U[4][fidP])
                wavespd = max(wavespdM_y,wavespdP)
                _,F2 = euler_fluxes(Ub[1][fidM],Ub[2][fidM],Ub[3][fidM],Ub[4][fidM],Ub[1][fidP],Ub[2][fidP],Ub[3][fidP],Ub[4][fidP])
                for c = 1:Nc
                    F_P_high[c][fidM,fidP] += Jf*Bs[face_idx[i],face_idx[i]]*F2[c]-sJ*wavespd/2*(U[c][fidP]-U[c][fidM])#nyJM*Bs[face_idx[i],face_idx[i]]*F2[c]#-wavespd/2*(U[c][fidP]-U[c][fidM])
                end               
            end
        end

        for i = 1:Np
            for c = 1:Nc
                rhsU[c][i,k] += sum(F_high[c][i,:])
                rhsU[c][i,k] += sum(F_P_high[c][i+(k-1)*Np,:])
            end
        end
    end

    
    for c = 1:Nc
        rhsU[c] = -1/J*Minv*rhsU[c]
    end

    return rhsU

end



# Time stepping
"Time integration"
t = 0.0
U = collect(U)

@unpack VDM = rd
rp,sp = equi_nodes_2D(30)
Vp = vandermonde_2D(N,rp,sp)/VDM
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2,axis=nothing)
const GIFINTERVAL = 100
#plot()


dt_hist = []
anim = Animation()
i = 1

while t < T
    rhsU,dt = rhs_IDP(U,K1D,N,Nfaces,nxJ,nyJ,Minv,Sr,Ss,S0r,S0s,Br,Bs,face_idx,mapM,mapP_x,mapP_y,i)
    @. U = U + dt*rhsU
    #rhsU,dt,U_low_k = rhs_IDP(U,K1D,N,Nfaces,nxJ,nyJ,Minv,Sr,Ss,S0r,S0s,Br,Bs,face_idx,mapM,mapP_x,mapP_y)
    push!(dt_hist,dt)
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T, at step $i")  
    global i = i + 1
    # if mod(i,50) == 1
    #     scatter(Vp*x,Vp*y,Vp*U[1],zcolor=Vp*U[1],camera=(0,90))
    #     frame(anim)
    # end
end

xp = Vp*x
yp = Vp*y
rhop = Vp*U[1]
p = pfun_nd.(U[1],U[2],U[3],U[4])
pp = Vp*p
#plt = scatter(Vp*x,Vp*y,Vp*U[1],zcolor=Vp*U[1],camera=(0,90))
plt = scatter(xp,yp,pp,zcolor=pp,camera=(0,90))
#display(plt)
savefig(plt,"~/Desktop/tmp.png")
#gif(anim,"~/Desktop/tmp.gif",fps=15)
