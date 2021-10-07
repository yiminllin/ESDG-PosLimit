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

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
using EntropyStableEuler
using EntropyStableEuler.Fluxes1D

function wavespeed_1D(rho,rhou,E)
    p = pfun_nd(rho,(rhou,),E)
    cvel = @. sqrt(γ*p/rho)
    return @. abs(rhou/rho) + cvel
end
unorm(U) = sum(map((x->x.^2),U))
function pfun_nd(rho, rhoU, E)
    rhoUnorm2 = unorm(rhoU)./rho
    return @. (γ-1)*(E - .5*rhoUnorm2)
end

function primitive_to_conservative_hardcode(rho,U,p)
    rhoU = rho.*U
    Unorm = unorm(U)
    E = @. p/(γ-1) + .5*rho*Unorm
    return (rho,rhoU,E)
end

function euler_fluxes_1D(rhoL,uL,betaL,rhologL,betalogL,
    rhoR,uR,betaR,rhologR,betalogR)

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = (@. .5*(rhoL+rhoR))
    uavg   = (@. .5*(uL+uR))

    unorm = (@. uL*uR)
    pa    = (@. rhoavg/(betaL+betaR))
    f4aux = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)

    FxS1 = (@. rholog*uavg)
    FxS2 = (@. FxS1*uavg + pa)
    FxS3 = (@. f4aux*uavg)

    return (FxS1,FxS2,FxS3)
end

function euler_fluxes(rhoL,uL,betaL,rhoR,uR,betaR)
    rhologL,betalogL,rhologR,betalogR = map(x->log.(x),(rhoL,betaL,rhoR,betaR))
    return euler_fluxes_1D(rhoL,uL,betaL,rhologL,betalogL,
                           rhoR,uR,betaR,rhologR,betalogR)
end

const TOL = 1e-16
"Approximation parameters"
N = 2 # The order of approximation
K = 100
T = 0.5#3.0

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
const v_0 = 1.0
const m_0 = rho_0*v_0
const v_1 = (γ-1+2/M_0^2)/(γ+1)
const v_01 = sqrt(v_0*v_1)

const uL = v_0+v_inf
const uR = v_1+v_inf
const rhoL = m_0/v_0
const rhoR = m_0/v_1
const eL = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_0^2)
const eR = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_1^2)
const pL = (γ-1)*rhoL*eL
const pR = (γ-1)*rhoR*eR
const EL = pL/(γ-1)+0.5*rhoL*uL^2
const ER = pR/(γ-1)+0.5*rhoR*uR^2

const Bl = -1.0
const Br = 1.5
gr(size=(300,300),ylims=(0,5.0),legend=false,markerstrokewidth=1,markersize=2)
plot()

# # Sod tube
# const γ = 1.4
# const rhoL = 1.0
# const rhoR = 0.125
# const uL = 0.0
# const uR = 0.0
# const pL = 1.0
# const pR = 0.1
# const EL = pL/(γ-1)+0.5*rhoL*uL^2
# const ER = pR/(γ-1)+0.5*rhoR*uR^2
# const Bl = -0.5
# const Br = 0.5

# const γ = 1.4
# const M_0 = 3.0
# const mu = 0.01
# const lambda = 0.0
# const Pr = 3/4
# const cp = γ/(γ-1)
# const cv = 1/(γ-1)
# const kappa = mu*cp/Pr

# T = 0.2
# gr(size=(300,300),ylims=(0,1.2),legend=false,markerstrokewidth=1,markersize=2)
# plot()

# # shock translation
# const γ = 1.4
# const rhoL = 1.0
# const rhoR = 0.125
# const uL = 1.0
# const uR = 1.0
# const pL = 1.0
# const pR = 1.0
# const EL = pL/(γ-1)+0.5*rhoL*uL^2
# const ER = pR/(γ-1)+0.5*rhoR*uR^2
# const Bl = -0.5
# const Br = 0.5

# const γ = 1.4
# const M_0 = 3.0
# const mu = 0.01
# const lambda = 0.0
# const Pr = 3/4
# const cp = γ/(γ-1)
# const cv = 1/(γ-1)
# const kappa = mu*cp/Pr

# T = 0.1
# gr(size=(300,300),ylims=(0,1.2),legend=false,markerstrokewidth=1,markersize=2)
# plot()


"Mesh related variables"
VX = LinRange(Bl,Br,K+1)
EToV = transpose(reshape(sort([1:K; 2:K+1]),2,K))

"Initialize reference element"
r,_ = gauss_lobatto_quad(0,0,N)   # Reference nodes
VDM = vandermonde_1D(N,r)         # modal to nodal
Dr = grad_vandermonde_1D(N,r)/VDM # nodal differentiation
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1]) # nodal linear interpolation

Nq = N
rq,wq = gauss_lobatto_quad(0,0,Nq)
Vq = vandermonde_1D(N,rq)/VDM
M = Vq'*diagm(wq)*Vq
Mlump = zeros(size(M))
Mlump_inv = zeros(size(M))
for i = 1:Nq+1
    Mlump[i,i] = sum(M[i,:])
    Mlump_inv[i,i] = 1.0/Mlump[i,i]
end
rf = [-1.0;1.0]
nrJ = [-1.0;1.0]
Vf = vandermonde_1D(N,rf)/VDM
LIFT = Mlump_inv*Vf'

# operators
Qr = M*Dr
B = zeros(N+1,N+1)
B[1,1] = -1
B[end,end] = 1
L = Array(spdiagm(0=>-2*ones(N+1), 1=>ones(N), -1=>ones(N)))
L[1,1] = -1
L[end,end] = -1
psi = pinv(L)*-1/2*B*ones(N+1)
S0 = zeros(N+1,N+1)
for i = 1:N+1
    for j = 1:N+1
        if L[i,j] != 0
            S0[i,j] = psi[j] - psi[i]
        end
    end
end
Qr0 = S0+1/2*B

# Drop zeros
Qr = Matrix(droptol!(sparse(Qr),TOL))
Qr0 = Matrix(droptol!(sparse(Qr0),TOL))
B = Matrix(droptol!(sparse(B),TOL))
L = Matrix(droptol!(sparse(L),TOL))
S0 = Matrix(droptol!(sparse(S0),TOL))
S = Matrix(droptol!(sparse((Qr-Qr')/2),TOL))


"""High order mesh"""
x = V1*VX[transpose(EToV)]
xf = Vf*x
mapM = reshape(1:2*K,2,K)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]

# """Periodic"""
# mapP[1] = mapM[end]
# mapP[end] = mapP[1]

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1.0

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1.0

"""Initial condition"""
function bisection_solve_velocity(x,max_iter,tol)
    v_L = v_1
    v_R = v_0
    num_iter = 0

    L_k = kappa/m_0/cv
    #f(v) = -x+2*L_k/(γ+1)*(v_0/(v_0-v_1)*log((v_0-v)/(v_0-v_01))-v_1/(v_0-v_1)*log((v-v_1)/(v_01-v_1)))
    f(v) = -x + 2*L_k/(γ+1)*log((v_0-v)^((v_0/(v_0-v_1)))*(v-v_1)^(-(v_1)/(v_0-v_1)))

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

const max_iter = 10000
const tol = 1e-16

function exact_sol_viscous_shocktube(x,t)
    u   = bisection_solve_velocity(x-v_inf*t,max_iter,tol)
    rho = m_0/u
    e   = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-u^2)
    return rho, rho*(v_inf+u), rho*(e+1/2*(v_inf+u)^2)
end


U = exact_sol_viscous_shocktube.(x,0.0)
U = ([x[1] for x in U], [x[2] for x in U], [x[3] for x in U])
# rho_init(x) = (x <= 0.0) ? rhoL : rhoR
# u_init(x) = (x <= 0.0) ? uL : uR
# p_init(x) = (x <= 0.0) ? pL : pR
# U = (rho_init.(x), u_init.(x), p_init.(x))
# U = primitive_to_conservative_hardcode.(U[1],U[2],U[3])
# U = ([x[1] for x in U], [x[2] for x in U], [x[3] for x in U])

function rhs_inviscid(U,K,N,Mlump_inv,S)
    J = (Br-Bl)/K/2 # assume uniform interval
    Nc = 3
    rhsU        = [zeros(N+1,K) for i = 1:Nc]

    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[1]/(2*p)

    VU = v_ufun(U...)

    wavespd_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
        end
    end

    for k = 1:K
        # Volume term (flux differencing)
        for j = 1:N+1
            for i = 1:N+1
                if i != j
                    F = euler_fluxes(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k])
                    for c = 1:Nc
                        rhsU[c][i,k] += 2*S[i,j]*F[c]
                    end
                end
            end
        end

        # Surface term (numerical fluxes)
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        F_l = euler_fluxes(Ub[1][1,k],Ub[2][1,k],Ub[3][1,k],Ub_left[1],Ub_left[2],Ub_left[3])
        F_r = euler_fluxes(Ub[1][end,k],Ub[2][end,k],Ub[3][end,k],Ub_right[1],Ub_right[2],Ub_right[3])
        for c = 1:Nc
            rhsU[c][1,k] += -F_l[c]-wavespd_l/2*(U_left[c]-U[c][1,k])
            rhsU[c][end,k] += F_r[c]-wavespd_r/2*(U_right[c]-U[c][end,k])
        end

        for c = 1:Nc
            rhsU[c][:,k] = -1/J*Mlump_inv*rhsU[c][:,k]
        end
    end

    return rhsU
end

function rhs_viscous(U,K,N,Mlump_inv,S)
    J = (Br-Bl)/K/2 # assume uniform interval
    Nc = 3
    rhsU  = [zeros(N+1,K) for i = 1:Nc]
    theta = [zeros(N+1,K) for i = 1:Nc]
    sigma = [zeros(N+1,K) for i = 1:Nc]

    p = pfun_nd.(U[1],U[2],U[3])

    VU = v_ufun(U...)
    VUf = (x->Vf*x).(VU)
    VUP = (x->x[mapP]).(VUf)
    VL = v_ufun(rhoL,rhoL*uL,EL)
    VUP[1][1] = VL[1]
    VUP[2][1] = VL[2]
    VUP[3][1] = VL[3]
    VUP[1][end] = VUf[1][end]
    VUP[2][end] = VUf[2][end]
    VUP[3][end] = VUf[3][end]

    surfx(uP,uf) = LIFT*(@. .5*(uP-uf)*nxJ)
    # Strong form, dv/dx
    VUx = (x->Dr*x).(VU)
    VUx = VUx .+ surfx.(VUP,VUf)
    VUx = VUx./J

    # σ = K dv/dx
    for k = 1:K
        for i = 1:N+1
            Kx = zeros(3,3)
            v1 = VU[1][i,k]
            v2 = VU[2][i,k]
            v4 = VU[3][i,k]
            Kx[2,2] = -(2*mu-lambda)/v4
            Kx[2,3] = (2*mu-lambda)*v2/v4^2
            Kx[3,2] = Kx[2,3]
            Kx[3,3] = -(2*mu-lambda)*v2^2/v4^3+kappa/cv/v4^2

            sigma[2][i,k] += Kx[2,2]*VUx[2][i,k] + Kx[2,3]*VUx[3][i,k]
            sigma[3][i,k] += Kx[3,2]*VUx[2][i,k] + Kx[3,3]*VUx[3][i,k]
        end
    end

    sxf = (x->Vf*x).(sigma)
    sxP = (x->x[mapP]).(sxf)
    sxP[1][1] = sxf[1][1]
    sxP[2][1] = sxf[2][1]
    sxP[3][1] = sxf[3][1]
    sxP[1][end] = sxf[1][end]
    sxP[2][end] = sxf[2][end]
    sxP[3][end] = sxf[3][end]
    # strong form, dσ/dx
    penalization(uP,uf) = LIFT*(@. -.5*(uP-uf))
    sigmax = (x->Dr*x).(sigma)
    sigmax = sigmax .+ surfx.(sxP,sxf) #.+ penalization.(VUP,VUf)
    sigmax = sigmax./J

    return sigmax
end

function rhs_ESDG(U,K,N,Mlump_inv,S)
    rhsI = rhs_inviscid(U,K,N,Mlump_inv,S)
    rhsV = rhs_viscous(U,K,N,Mlump_inv,S)
    return rhsI .+ rhsV
end




# Time stepping
"Time integration"
t = 0.0
U = collect(U)
resU = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
resW = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
resZ = [zeros(size(x)),zeros(size(x)),zeros(size(x))]

Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM


anim = Animation()
const ptL = Bl+(Br-Bl)/K/(N+1)/2
const ptR = Br-(Br-Bl)/K/(N+1)/2
const hplot = (Br-Bl)/K/(N+1)
i = 1

while t < T

    # SSPRK(3,3)
    dt = min(1e-4,T-t)
    rhsU = rhs_ESDG(U,K,N,Mlump_inv,S)
    @. resW = U + dt*rhsU
    rhsU = rhs_ESDG(resW,K,N,Mlump_inv,S)
    @. resZ = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resZ
    rhsU = rhs_ESDG(resW,K,N,Mlump_inv,S)
    @. resZ = resW+dt*rhsU
    @. U = 1/3*U+2/3*resZ

    global t = t + dt
    global i = i + 1
    println("Current time $t with time step size $dt, and final time $T")
    if mod(i,1000) == 1
        plot(x,U[1])
        # for k = 1:K
        #     plot!(ptL+(k-1)*hplot*(N+1):hplot:ptL+k*hplot*(N+1), 1 .-L_plot[:,k],st=:bar,alpha=0.2)
        # end
        frame(anim)
    end

    # if i == 10
    #     break
    # end
end

plot(x,U[1])
gif(anim,"~/Desktop/tmp.gif",fps=15)


exact_U = @. exact_sol_viscous_shocktube.(x,T)
exact_rho = [x[1] for x in exact_U]
exact_u = [x[2] for x in exact_U]
exact_u = exact_u./exact_rho
exact_E = [x[3] for x in exact_U]

rho = U[1]
u = U[2]./U[1]
E = U[3]
p = pfun_nd.(U[1],U[2],U[3])
J = (Br-Bl)/K/2

Linferr = maximum(abs.(exact_rho-rho))/maximum(abs.(exact_rho)) +
          maximum(abs.(exact_u-u))/maximum(abs.(exact_u)) +
          maximum(abs.(exact_E-E))/maximum(abs.(exact_E))

L1err = sum(J*wq.*abs.(exact_rho-rho))/sum(J*wq.*abs.(rho)) +
        sum(J*wq.*abs.(exact_u-u))/sum(J*wq.*abs.(u)) +
        sum(J*wq.*abs.(exact_E-E))/sum(J*wq.*abs.(E))

println("N = $N, K = $K")
println("L1 error is $L1err")
println("Linf error is $Linferr")
