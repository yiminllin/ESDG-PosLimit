using Revise # reduce recompilation time
# using CairoMakie
using Plots
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using DelimitedFiles
using CSV
using DataFrames


push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
using EntropyStableEuler
#using EntropyStableEuler.Fluxes1D



const POSTOL = 1e-14 
const BOUNDTOL = 1e-7
const TOL = 5e-14
const CFL = 0.5

const γ = 1.4

"Approximation parameters"
N = 3 # The order of approximation
K = 50#200
T = 0.2
is_low_order = true 
const CASENUM = 2                 # 1 - standard case, 2 - smooth sine convergence, 3 - smooth sine convergence small density
const PLOTGIF = true
const ADDTODF = false
# 0 - no limit
# 1 - Zalesak limiter local bounds wo smoothness indicator
# 2 - Zalesak limiter local bounds with smoothness indicator
# 3 - integrated version of elementwise limiter 
# 4 - Zalesak global bound
# 5 - nodewise local bound (original IDP)
# 6 - nodewise global bound 
# 7 - Subcell limiting local bound
# 8 - Subcell limiting global bound
# 9 - Subcell limiting relaxed global bound
# 10 - Zalesak relaxed glbal bound
# 11 - Subcell limiting relaxed global bound - my version (more relaxed version) of the limiting parameter
const RHO_LIMIT_TYPE = 0
const S_LIMIT_TYPE = 0
const IS_ELEMENTWISE_LIMIT = false # false - nodewise limit
const IS_SUBCELL_LIMIT = false
const HIGH_ORDER_FLUX_TYPE = 1 # 1 - ES flux, 2 - central flux

"Mesh related variables"
VX = LinRange(-1.0,1.0,K+1)
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

rf = [-1.0;1.0]
nrJ = [-1.0;1.0]
Vf = vandermonde_1D(N,rf)/VDM

"""High order mesh"""
x = V1*VX[transpose(EToV)]
xf = Vf*x
mapM = reshape(1:2*K,2,K)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]
mapP[1,1] = mapM[end,end]
mapP[end,end] = mapM[1,1]

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1.0

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1.0

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

function sfun(rho, rhou, E)
    p = pfun_nd(rho,rhou,E)
    return log(p/rho^γ)
end

rho = @. 1.0 + 2.0 * (x > 0)
u = @. -.1 + .5 * (x > 0)
p = @. rho^γ
E = @. p / (γ - 1) + 0.5 * rho * u^2
rhou = @. rho*u
s = sfun.(rho,rhou,E)
U = (rho,rhou,E)

function exact_sol_sine_wave(x,t)
    rho = 2.0+sin(pi*(x-t))
    u = 1.0
    p = 1.0
    E = p / (γ - 1) + 0.5 * rho * u^2
    return rho,rho*u,E
end

if (CASENUM == 2)
    rho = @. 2.0 + sin(pi*x)
    u = ones(size(x))
    p = ones(size(x))
    E = @. p / (γ - 1) + 0.5 * rho * u^2
    rhou = @. rho*u
    s = sfun.(rho,rhou,E)
    U = (rho,rhou,E)
end

const RHOMIN_GLOBAL = minimum(rho)
const RHOMAX_GLOBAL = maximum(rho)
const SMIN_GLOBAL   = minimum(s)

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

function limiting_param(U_low, P_ij)
    l = 1.0
    # Limit density
    if U_low[1] + P_ij[1] < -TOL
        l = min(abs(U_low[1])/(abs(P_ij[1])+POSTOL), 1.0)
    end

    # limiting internal energy (via quadratic function)
    a = P_ij[1]*P_ij[3]-1.0/2.0*P_ij[2]^2
    b = U_low[3]*P_ij[1]+U_low[1]*P_ij[3]-U_low[2]*P_ij[2]
    c = U_low[3]*U_low[1]-1.0/2.0*U_low[2]^2

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

function limiting_param_rhobound(rhoL,rhoP,Lrho,Urho)
    if ((rhoL + rhoP >= Lrho + TOL) && (rhoL + rhoP <= Urho - TOL))
        return 1.0
    else
        if ((rhoL < Lrho - TOL) || (rhoL > Urho + TOL))
            @show "Rho not in bound",rhoL,rhoP,Lrho,Urho
            if (abs(rhoL-Lrho) > 1E-8 && abs(rhoL-Urho) > 1E-8)
                @show "Difference large:",abs(rhoL-Lrho),abs(rhoL-Urho)
            end
        end
        if (rhoL + rhoP < Lrho - TOL)
            return (Lrho-rhoL)/rhoP
        elseif (rhoL + rhoP > Urho + TOL)
            return (Urho-rhoL)/rhoP
        else
            return 1.0
        end
    end
end

function limiting_param_sbound(U1,U2,U3,P1,P2,P3,smin)
    if sfun(U1+P1,U2+P2,U3+P3) >= smin - TOL
        return 1.0
    else
        if (sfun(U1,U2,U3) < smin - TOL)
            @show "s not in bound",sfun(U1,U2,U3),smin
        end
        lpt = 0.0    # Assume At l = 0, s(U) - smin > 0
        rpt = 1.0
        while sfun(U1+rpt*P1,U2+rpt*P2,U3+rpt*P3) - smin >= 0 
            rpt -= 0.0001
        end
        ls = sfun(U1,U2,U3) - smin
        rs = sfun(U1+rpt*P1,U2+rpt*P2,U3+rpt*P3) - smin

        # Start bisection
        niter = 0
        while niter < 1000
            mpt = (lpt+rpt)/2
            ms = sfun(U1+mpt*P1,U2+mpt*P2,U3+mpt*P3) - smin
            if abs(ms) < TOL
                return mpt
            end
            if ms > -TOL
                lpt = mpt
                ls = ms
            end
            if ms < TOL
                rpt = mpt
                rs = ms
            end
            niter = niter + 1
        end

        @show "Bisection fails to converge"
    end
end

function flux_lowIDP(U_i,U_j,f_i,f_j,c_ij,wavespd)
    # nij = abs(c_ij) = abs(S0[i,j])
    return c_ij*(f_i+f_j)-abs(c_ij)*wavespd*(U_j-U_i)
end

function flux_high(f_i,f_j,c_ij)
    return c_ij*(f_i+f_j)
end

function flux_ES(rho_i,u_i,beta_i,rho_j,u_j,beta_j,c_ij)
    return 2*c_ij.*euler_fluxes(rho_i,u_i,beta_i,rho_j,u_j,beta_j)
end

function get_bar_state(rhoL,rhouL,EL,f1L,f2L,f3L,rhoR,rhouR,ER,f1R,f2R,f3R,n,alpha)
    bar1 = .5*(rhoL+rhoR)  -1/(2*alpha)*n*(f1R-f1L)
    bar2 = .5*(rhouL+rhouR)-1/(2*alpha)*n*(f2R-f2L)
    bar3 = .5*(EL+ER)      -1/(2*alpha)*n*(f3R-f3L)
    return bar1,bar2,bar3
end

function rhs_IDP(U,K,N,wq,S,S0,Mlump_inv,T,dt,in_s1,is_low_order)
    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[1]/(2*p)

    sk = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            sk[i,k] = sfun(U[1][i,k],U[2][i,k],U[3][i,k])
        end
    end

    J = 1/K # assume uniform interval, and domain [-1,1]

    Umodal = collect((VDM\U[1],VDM\U[2],VDM\U[3]))
    Umodaltruncated = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]
    for c = 1:3
        for k = 1:K
            for i = 1:N
                Umodaltruncated[c][i,k] = Umodal[c][i,k]
            end
        end
    end
    Utruncated = [VDM*Umodaltruncated[1],VDM*Umodaltruncated[2],VDM*Umodaltruncated[3]]

    smoothness_indicator = zeros(K)
    kappa = 1
    s0 = log(10,N^-4)
    for k = 1:K
        normtrunc = 0.0
        normdenom = 0.0
        for i = 1:N+1
            for c = 1:3
                normtrunc += J*wq[i]*(U[c][i,k]-Utruncated[c][i,k])^2
                normdenom += J*wq[i]*U[c][i,k]^2
            end
        end
        skk = log(10,normtrunc/normdenom)
        if (skk < s0 - kappa) 
            smoothness_indicator[k] = 0.0
        elseif ((skk <= s0 + kappa) && (skk >= s0-kappa))
            smoothness_indicator[k] = .5-.5*sin(pi*(skk-s0)/(2*kappa))
        else
            smoothness_indicator[k] = 1.0
        end
    end

    rhoavg  = zeros(K)
    rhouavg = zeros(K)
    Eavg    = zeros(K)
    savg    = zeros(K)
    sUavg   = zeros(K)
    for k = 1:K
        rhoavgk  = 0.0
        rhouavgk = 0.0
        Eavgk    = 0.0
        savgk   = 0.0
        for i = 1:N+1
            rhoavgk  += J*wq[i]*U[1][i,k]
            rhouavgk += J*wq[i]*U[2][i,k]
            Eavgk    += J*wq[i]*U[3][i,k]
            savgk   += J*wq[i]*sfun(U[1][i,k],U[2][i,k],U[3][i,k])
        end
        rhoavg[k]  = rhoavgk/(2/K)
        rhouavg[k] = rhouavgk/(2/K)
        Eavg[k]    = Eavgk/(2/K)
        savg[k]    = savgk/(2/K)
        sUavg[k]   = sfun(rhoavg[k],rhouavg[k],Eavg[k])
    end

    # Low order and high order algebraic fluxes
    F_low       = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_high      = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    rhobar      = zeros(N+1,N+1,K)
    rhobarb     = zeros(2,K)
    F_P         = [zeros(2),zeros(2),zeros(2)] # 1: left boundary, 2: right boundary
    L           = zeros(N+1,N+1) # Array of limiting params
    Limrho      = zeros(N+1,N+1)
    Lplot       = zeros(N+1,K)
    Lims        = zeros(N+1,N+1)
    rhsU        = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]
    dii_arr     = zeros(N+1,K)

    wavespd_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
        end
    end

    for k = 1:K
        # Interior dissp coeff
        for i = 1:N+1
            for j = 1:N+1
                if i != j 
                    dij = abs(S0[i,j])*max(wavespd_arr[i,k],wavespd_arr[j,k])
                    dii_arr[i,k] = dii_arr[i,k] + dij
                end
            end
        end

        # Interface dissipation
        U_left   = [U[1][end,mod1(k-1,K)]; U[2][end,mod1(k-1,K)]; U[3][end,mod1(k-1,K)]]
        U_right  = [U[1][1  ,mod1(k+1,K)]; U[2][1  ,mod1(k+1,K)]; U[3][1  ,mod1(k+1,K)]]
        wavespd_l = max(wavespd_arr[1,k]  ,wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))
        
        dii_arr[1,k] = dii_arr[1,k] + 0.5*wavespd_l 
        dii_arr[N+1,k] = dii_arr[N+1,k] + 0.5*wavespd_r 
    end

    if in_s1
        for k = 1:K 
            for i = 1:N+1
                dt = min(dt,1.0*wq[i]*J/2.0/dii_arr[i,k])
            end
        end
        dt = min(CFL*dt,T-t)
        #dt = CFL*dt
    end

    # Compute bar states for bound limiters
    for k = 1:K
        for j = 1:N+1
            for i = 1:N+1
                rhoL  = U[1][i,k]
                rhouL = U[2][i,k]
                EL    = U[3][i,k]
                f1L   = flux[1][i,k]
                f2L   = flux[2][i,k]
                f3L   = flux[3][i,k]
                rhoR  = U[1][j,k]
                rhouR = U[2][j,k]
                ER    = U[3][j,k]
                f1R   = flux[1][j,k]
                f2R   = flux[2][j,k]
                f3R   = flux[3][j,k]
                n     = S0[i,j]
                alpha = max(wavespd_arr[i,k],wavespd_arr[j,k])
                bar1,_,_ = get_bar_state(rhoL,rhouL,EL,f1L,f2L,f3L,rhoR,rhouR,ER,f1R,f2R,f3R,sign(n),alpha)
                rhobar[i,j,k] = bar1
            end
        end
    end
    for k = 1:K
        rhoL  = U[1][1,k]
        rhouL = U[2][1,k]
        EL    = U[3][1,k]
        f1L   = flux[1][1,k]
        f2L   = flux[2][1,k]
        f3L   = flux[3][1,k]
        rhoR  = U[1][end,mod1(k-1,K)]
        rhouR = U[2][end,mod1(k-1,K)]
        ER    = U[3][end,mod1(k-1,K)]
        f1R   = flux[1][end,mod1(k-1,K)]
        f2R   = flux[2][end,mod1(k-1,K)]
        f3R   = flux[3][end,mod1(k-1,K)]
        n     = -.5
        U_l   = [U[1][end,mod1(k-1,K)]; U[2][end,mod1(k-1,K)]; U[3][end,mod1(k-1,K)]]
        alpha = max(wavespd_arr[1,k],wavespeed_1D(U_l[1],U_l[2],U_l[3]))
        bar1,_,_ = get_bar_state(rhoL,rhouL,EL,f1L,f2L,f3L,rhoR,rhouR,ER,f1R,f2R,f3R,sign(n),alpha)
        rhobarb[1,k] = bar1

        rhoL  = U[1][end,k]
        rhouL = U[2][end,k]
        EL    = U[3][end,k]
        f1L   = flux[1][end,k]
        f2L   = flux[2][end,k]
        f3L   = flux[3][end,k]
        rhoR  = U[1][1,mod1(k+1,K)]
        rhouR = U[2][1,mod1(k+1,K)]
        ER    = U[3][1,mod1(k+1,K)]
        f1R   = flux[1][1,mod1(k+1,K)]
        f2R   = flux[2][1,mod1(k+1,K)]
        f3R   = flux[3][1,mod1(k+1,K)]
        n     = .5
        U_r  = [U[1][1  ,mod1(k+1,K)]; U[2][1  ,mod1(k+1,K)]; U[3][1  ,mod1(k+1,K)]]
        alpha = max(wavespd_arr[end,k],wavespeed_1D(U_r[1],U_r[2],U_r[3]))
        bar1,_,_ = get_bar_state(rhoL,rhouL,EL,f1L,f2L,f3L,rhoR,rhouR,ER,f1R,f2R,f3R,sign(n),alpha)
        rhobarb[end,k] = bar1
    end

    rhoLavg = zeros(K)
    rhouLavg = zeros(K)
    ELavg = zeros(K)
    sLavg = zeros(K)          # avg(s(u^L))
    sULavg = zeros(K)         # s(avg(u^L))
    ULn,_ = get_U_low(U,K,N,wq,S0,Mlump_inv,T,dt,false)
    for k = 1:K
        rhoavgk = 0.0
        rhouavgk = 0.0
        Eavgk = 0.0
        savgk = 0.0        # This is \overline s(u)
        srhoavgk = 0.0     # This is \overline \rho s(u)
        for i = 1:N+1
            rhoavgk += J*wq[i]*ULn[1][i,k]
            rhouavgk += J*wq[i]*ULn[2][i,k]
            Eavgk += J*wq[i]*ULn[3][i,k]
            savgk   += J*wq[i]*sfun(ULn[1][i,k],ULn[2][i,k],ULn[3][i,k])
            srhoavgk   += J*wq[i]*ULn[1][i,k]*sfun(ULn[1][i,k],ULn[2][i,k],ULn[3][i,k])
        end
        rhoLavg[k] = rhoavgk/(2/K)
        rhouLavg[k] = rhouavgk/(2/K)
        ELavg[k] = Eavgk/(2/K)
        sLavg[k] = savgk/(2/K)
        srhoavgk = srhoavgk/(2/K)
        sULavg[k] = sfun(rhoLavg[k],rhouLavg[k],ELavg[k])
        if !((rhoLavg[k] <= RHOMAX_GLOBAL + BOUNDTOL) && (rhoLavg[k] >= RHOMIN_GLOBAL - BOUNDTOL))
            @show "rhoLavg doesn't satisfy global max/min principle",rhoLavg[k]
        end

        sbarLk = sfun(rhoavgk,rhouavgk,Eavgk)   # This is s (\overline u)
        if (srhoavgk >= rhoavgk*sbarLk)    # If \overline{rho*s(u)} >= rhos(\overline{u})
            @show "overline{rho*s(u)} >= rhos(overline{u})"
        end
        # if (savgk >= sbarLk)               # If \overline{s(u)} >= s(\overline{u})
        #     @show "overline{s(u)} >= s(overline{u})"
        # end
    end

    # Try out integrated bounds on specific entropy
    for k = 1:K
        l1 = sULavg[k]
        l2 = sLavg[k]
        b1 = min(minimum(sk[:,k]),sk[end,mod1(k-1,K)],sk[1,mod1(k+1,K)])
        b2 = min(sUavg[mod1(k-1,K)],sUavg[k],sUavg[mod1(k+1,K)])
        # b2 = minimum(sUavg)
        b3 = min(savg[mod1(k-1,K)],savg[k],savg[mod1(k+1,K)])
        # b3 = minimum(savg)
        b4 = min(sULavg[mod1(k-1,K)],sULavg[k],sULavg[mod1(k+1,K)])
        b5 = min(sLavg[mod1(k-1,K)],sLavg[k],sLavg[mod1(k+1,K)])
        # if !(l1 >= b1)
        #     @show "inequality 1 fails! Difference",abs(l1-b1)
        # end
        # if !(l1 >= b2)
        #     @show "inequality 2 fails! Difference",abs(l1-b2)
        # end
        # if !(l1 >= b3)
        #     @show "inequality 3 fails! Difference",abs(l1-b3)
        # end
        # if !(l1 >= b4)
        #     @show "inequality 4 fails! Difference",abs(l1-b4)
        # end
        # if !(l1 >= b5)
        #     @show "inequality 5 fails! Difference",abs(l1-b5)
        # end

        # if !(l2 >= b1)
        #     @show "inequality 6 fails! Difference",abs(l2-b1)
        # end
        # if !(l2 >= b2)
        #     @show "inequality 7 fails! Difference",abs(l2-b2)
        # end
        # if !(l2 >= b3)
        #     @show "inequality 8 fails! Difference",abs(l2-b3)
        # end
        # if !(l2 >= b4)
        #     @show "inequality 9 fails! Difference",abs(l2-b4)
        # end
        # if !(l2 >= b5)
        #     @show "inequality 10 fails! Difference",abs(l2-b5)
        # end
    end

    # Ulnbar = zeros(N+1,K)
    # for k = 1:K
    #     for i = 1:N+1
    #         Ulnbar[i,k] = (1-2*dt*dii_arr[i,k]/(J*wq[i]))*U[1][i,k]
    #         for j = 1:N+1
    #             dij = abs(S0[i,j])*max(wavespd_arr[i,k],wavespd_arr[j,k])
    #             Ulnbar[i,k] += 2*dt*dij/(J*wq[i])*rhobar[i,j,k]
    #         end   
    #     end
    #     # Interface dissipation
    #     U_left   = [U[1][end,mod1(k-1,K)]; U[2][end,mod1(k-1,K)]; U[3][end,mod1(k-1,K)]]
    #     U_right  = [U[1][1  ,mod1(k+1,K)]; U[2][1  ,mod1(k+1,K)]; U[3][1  ,mod1(k+1,K)]]
    #     wavespd_l = .5*max(wavespd_arr[1,k]  ,wavespeed_1D(U_left[1],U_left[2],U_left[3]))
    #     wavespd_r = .5*max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))
    #     Ulnbar[1,k]   += 2*dt*wavespd_l/(J*wq[1])*rhobarb[1,k]
    #     Ulnbar[end,k] += 2*dt*wavespd_r/(J*wq[end])*rhobarb[2,k]
    # end

    # @show norm(Ulnbar-ULn[1])
    # for k = 1:K
    #     for i = 1:N+1
    #         if i == 1
    #             Lrho = min(rhobar[i,2,k],rhobarb[1,k],U[1][i,k])
    #             Urho = max(rhobar[i,2,k],rhobarb[1,k],U[1][i,k])
    #         elseif i == N+1
    #             Lrho = min(rhobar[i,N,k],rhobarb[end,k],U[1][i,k])
    #             Urho = max(rhobar[i,N,k],rhobarb[end,k],U[1][i,k])
    #         else
    #             Lrho = min(rhobar[i,i-1,k],rhobar[i,i+1,k],U[1][i,k])
    #             Urho = max(rhobar[i,i-1,k],rhobar[i,i+1,k],U[1][i,k])
    #         end
    #         if ((ULn[1][i,k] < Lrho - TOL))
    #             @show "Error", i,k,ULn[1][i,k]-Ulnbar[i,k]

    #             tmp = 0.0
    #             tmp2 = 0.0
    #             @show (1-2*dt*dii_arr[i,k]/(J*wq[i]))*U[1][i,k]
    #             tmp += (1-2*dt*dii_arr[i,k]/(J*wq[i]))*U[1][i,k]
    #             tmp2 += (1-2*dt*dii_arr[i,k]/(J*wq[i]))
    #             for j = 1:N+1
    #                 dij = abs(S0[i,j])*max(wavespd_arr[i,k],wavespd_arr[j,k])
    #                 @show 2*dt*dij/(J*wq[i])*rhobar[i,j,k]
    #                 tmp += 2*dt*dij/(J*wq[i])*rhobar[i,j,k]
    #                 tmp2 += 2*dt*dij/(J*wq[i])
    #             end   
    #             @show tmp
    #             @show tmp2
    #             @show ULn[1][i,k],Ulnbar[i,k]
    #         end
    #     end
    # end

    #####################
    #     Main Loop     #
    #####################
    for k = 1:K
        L = ones(N+1,N+1)
        Limrho = ones(N+1,N+1)
        Lims = ones(N+1,N+1)

        # Assemble matrix of low and high order algebraic fluxes
        # interior of the element
        for i = 1:N+1
            for j = 1:N+1
                if i != j # skip diagonal
                    wavespd = max(wavespd_arr[i,k],wavespd_arr[j,k])
                    fluxS = flux_ES(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],S[i,j])
                    for c = 1:3
                        F_low[c][i,j]  = flux_lowIDP(U[c][i,k],U[c][j,k],flux[c][i,k],flux[c][j,k],S0[i,j],wavespd)
                        if (HIGH_ORDER_FLUX_TYPE == 1)
                            F_high[c][i,j] = fluxS[c]
                        elseif (HIGH_ORDER_FLUX_TYPE == 2)
                            F_high[c][i,j] = flux_high(flux[c][i,k],flux[c][j,k],S[i,j])
                        end
                    end
                end
            end
        end

        # Assemble matrix of low and high order algebraic fluxes
        # interface of the element
        U_left   = [U[1][end,mod1(k-1,K)]; U[2][end,mod1(k-1,K)]; U[3][end,mod1(k-1,K)]]
        U_right  = [U[1][1  ,mod1(k+1,K)]; U[2][1  ,mod1(k+1,K)]; U[3][1  ,mod1(k+1,K)]]
        f_left   = [flux[1][end,mod1(k-1,K)]; flux[2][end,mod1(k-1,K)]; flux[3][end,mod1(k-1,K)]]
        f_right  = [flux[1][1  ,mod1(k+1,K)]; flux[2][1  ,mod1(k+1,K)]; flux[3][1  ,mod1(k+1,K)]]
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        for c = 1:3
            F_P[c][1] = flux_lowIDP(U[c][1,k],U_left[c],flux[c][1,k],f_left[c],-0.5,wavespd_l)
            F_P[c][2] = flux_lowIDP(U[c][end,k],U_right[c],flux[c][end,k],f_right[c],0.5,wavespd_r)
        end

        # Calculate limiting parameters over interior of the element
        P_ij = zeros(3,1)
        # TODO: redundant
        U_low =  [zeros(N+1,1),zeros(N+1,1),zeros(N+1,1)]
        for c = 1:3
            U_low[c] .= sum(-F_low[c],dims=2)
            U_low[c][1] -= F_P[c][1]
            U_low[c][end] -= F_P[c][2]
            U_low[c] .= U[c][:,k]+dt/J*Mlump_inv*U_low[c]
        end
        U_high =  [zeros(N+1,1),zeros(N+1,1),zeros(N+1,1)]
        for c = 1:3
            U_high[c] .= sum(-F_high[c],dims=2)
            U_high[c][1] -= F_P[c][1]
            U_high[c][end] -= F_P[c][2]
            U_high[c] .= U[c][:,k]+dt/J*Mlump_inv*U_high[c]
        end

        # Positivity Limiting
        is_H_positive = true
        for i = 1:N+1
            rhoH_i  = U_high[1][i]
            rhouH_i = U_high[2][i]
            EH_i    = U_high[3][i]
            pH_i    = pfun_nd(rhoH_i,rhouH_i,EH_i)
            if pH_i < POSTOL || rhoH_i < POSTOL
                is_H_positive = false
            end
        end

        for i = 1:N+1
            lambda_j = 1/N
            m_i = J*wq[i]
            for j = 1:N+1
                if i != j
                    for c = 1:3
                        P_ij[c] = dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
                    end
                    if !is_H_positive
                        L[i,j] = limiting_param([U_low[1][i]; U_low[2][i]; U_low[3][i]],P_ij)
                    else
                        L[i,j] = 1.0
                    end
                end
            end
        end

        # sum of residual states in subcell limiting
        rbararr = zeros(3,N+1)
        for i = 1:N+1
            for j = 1:N+1
                for c = 1:3
                    rbararr[c,i] = rbararr[c,i] + (F_low[c][i,j]-F_high[c][i,j])
                end
            end
            if (i != N+1)
                for c = 1:3
                    rbararr[c,i+1] = rbararr[c,i]
                end
            end
        end

        # Bound Limiting 
        if (RHO_LIMIT_TYPE == 0)
            Limrho .= 1.0
        elseif ((RHO_LIMIT_TYPE == 1) || (RHO_LIMIT_TYPE == 2) || (RHO_LIMIT_TYPE == 4) || (RHO_LIMIT_TYPE == 10))
            # Bound limiting (nodewise)
            Pi = zeros(3)
            for i = 1:N+1
                Pi = zeros(3)
                m_i = J*wq[i]
                for j = 1:N+1
                    if i != j
                        for c = 1:3
                            Pi[c] += dt/(m_i)*(F_low[c][i,j]-F_high[c][i,j])
                        end 
                    end
                end
                if (RHO_LIMIT_TYPE == 1) # Strongest nodewise bound
                    if i == 1
                        Lrho = min(rhobar[i,2,k],rhobarb[1,k],U[1][i,k])
                        Urho = max(rhobar[i,2,k],rhobarb[1,k],U[1][i,k])
                    elseif i == N+1
                        Lrho = min(rhobar[i,N,k],rhobarb[end,k],U[1][i,k])
                        Urho = max(rhobar[i,N,k],rhobarb[end,k],U[1][i,k])
                    else
                        Lrho = min(rhobar[i,i-1,k],rhobar[i,i+1,k],U[1][i,k])
                        Urho = max(rhobar[i,i-1,k],rhobar[i,i+1,k],U[1][i,k])
                    end
                elseif (RHO_LIMIT_TYPE == 4)   # Weakest nodewise bound
                    Lrho = RHOMIN_GLOBAL
                    Urho = RHOMAX_GLOBAL
                elseif (RHO_LIMIT_TYPE == 10)  # Relaxed global bound
                    Lrho = 0.99*RHOMIN_GLOBAL
                    Urho = 1.01*RHOMAX_GLOBAL
                else                     # Nodewise bound with smoothness indicator
                    epsk = smoothness_indicator[k]
                    if i == 1
                        Lrho = epsk*min(rhobar[i,2,k],rhobarb[1,k],U[1][i,k])+(1-epsk)*RHOMIN_GLOBAL
                        Urho = epsk*max(rhobar[i,2,k],rhobarb[1,k],U[1][i,k])+(1-epsk)*RHOMAX_GLOBAL
                    elseif i == N+1
                        Lrho = epsk*min(rhobar[i,N,k],rhobarb[end,k],U[1][i,k])+(1-epsk)*RHOMIN_GLOBAL
                        Urho = epsk*max(rhobar[i,N,k],rhobarb[end,k],U[1][i,k])+(1-epsk)*RHOMAX_GLOBAL
                    else
                        Lrho = epsk*min(rhobar[i,i-1,k],rhobar[i,i+1,k],U[1][i,k])+(1-epsk)*RHOMIN_GLOBAL
                        Urho = epsk*max(rhobar[i,i-1,k],rhobar[i,i+1,k],U[1][i,k])+(1-epsk)*RHOMAX_GLOBAL
                    end
                end
                li = limiting_param_rhobound(U_low[1][i],Pi[1],Lrho,Urho)
                for j = 1:N+1
                    Limrho[i,j] = li
                end
            end
        elseif (RHO_LIMIT_TYPE == 3)
            # Bound limiting (integrated)
            P = zeros(3)
            for i = 1:N+1
                for j = 1:N+1
                    if i != j
                        for c = 1:3
                            P[c] += dt/(2/K)*(F_low[c][i,j]-F_high[c][i,j])
                        end
                    end
                end
            end
            Lrho = min(rhoLavg[mod1(k-1,K)],rhoLavg[k],rhoLavg[mod1(k+1,K)])
            Urho = max(rhoLavg[mod1(k-1,K)],rhoLavg[k],rhoLavg[mod1(k+1,K)])
            l = limiting_param_rhobound(rhoLavg[k],P[1],Lrho,Urho)
            for i = 1:N+1
                for j = 1:N+1
                    Limrho[i,j] = l
                end
            end
        elseif ((RHO_LIMIT_TYPE == 5) || (RHO_LIMIT_TYPE == 6))
            Pij = zeros(3)
            for i = 1:N+1
                lambda_j = 1/N
                m_i = J*wq[i]
                for j = 1:N+1
                    if i != j
                        for c = 1:3
                            Pij[c] = dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
                        end
                        if (RHO_LIMIT_TYPE == 5)
                            if i == 1
                                Lrho = min(rhobar[i,2,k],rhobarb[1,k],U[1][i,k])
                                Urho = max(rhobar[i,2,k],rhobarb[1,k],U[1][i,k])
                            elseif i == N+1
                                Lrho = min(rhobar[i,N,k],rhobarb[end,k],U[1][i,k])
                                Urho = max(rhobar[i,N,k],rhobarb[end,k],U[1][i,k])
                            else
                                Lrho = min(rhobar[i,i-1,k],rhobar[i,i+1,k],U[1][i,k])
                                Urho = max(rhobar[i,i-1,k],rhobar[i,i+1,k],U[1][i,k])
                            end
                        elseif (RHO_LIMIT_TYPE == 6)
                            Lrho = RHOMIN_GLOBAL
                            Urho = RHOMAX_GLOBAL
                        end
                        Limrho[i,j] = limiting_param_rhobound(U_low[1][i],Pij[1],Lrho,Urho)
                    end
                end
            end
            # Lrho = RHOMIN_GLOBAL
            # Urho = RHOMAX_GLOBAL
            # P1 = zeros(3)
            # P2 = zeros(3)
            # for i = 1:N+1
            #     lambda_j = 1/2
            #     m_i = J*wq[i]
            #     for j = 1:N+1
            #         if j < i
            #             for c = 1:3
            #                 P1[c] += dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
            #             end
            #         end
            #         if j > i
            #             for c = 1:3
            #                 P2[c] += dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
            #             end
            #         end
            #     end
            #     l1 = limiting_param_rhobound(U_low[1][i],P1[1],Lrho,Urho)
            #     l2 = limiting_param_rhobound(U_low[1][i],P2[1],Lrho,Urho)
            #     for j = 1:N+1
            #         Limrho[i,j] = min(l1,l2)
            #     end
            # end
        elseif ((RHO_LIMIT_TYPE == 7) || (RHO_LIMIT_TYPE == 8) || (RHO_LIMIT_TYPE == 9) || (RHO_LIMIT_TYPE == 11))
            lambda = 1/2
            for i = 1:N+1
                alphai = Inf
                m_i = J*wq[i]
                coeff = dt/(m_i*lambda)
                
                if (RHO_LIMIT_TYPE == 7)
                    if i == 1
                        Lrho = min(rhobar[i,2,k],rhobarb[1,k],U[1][i,k])
                        Urho = max(rhobar[i,2,k],rhobarb[1,k],U[1][i,k])
                    elseif i == N+1
                        Lrho = min(rhobar[i,N,k],rhobarb[end,k],U[1][i,k])
                        Urho = max(rhobar[i,N,k],rhobarb[end,k],U[1][i,k])
                    else
                        Lrho = min(rhobar[i,i-1,k],rhobar[i,i+1,k],U[1][i,k])
                        Urho = max(rhobar[i,i-1,k],rhobar[i,i+1,k],U[1][i,k])
                    end
                elseif (RHO_LIMIT_TYPE == 8)
                    Lrho = RHOMIN_GLOBAL
                    Urho = RHOMAX_GLOBAL
                elseif ((RHO_LIMIT_TYPE == 9) || (RHO_LIMIT_TYPE == 11))
                    Lrho = 0.99*RHOMIN_GLOBAL
                    Urho = 1.01*RHOMAX_GLOBAL
                end

                if i > 1
                    alphai = min(alphai,limiting_param_rhobound(U_low[1][i], coeff*rbararr[1,i-1],Lrho,Urho))
                    alphai = min(alphai,limiting_param_rhobound(U_low[1][i],-coeff*rbararr[1,i-1],Lrho,Urho))
                end
                alphai = min(alphai,limiting_param_rhobound(U_low[1][i], coeff*rbararr[1,i],Lrho,Urho))
                alphai = min(alphai,limiting_param_rhobound(U_low[1][i],-coeff*rbararr[1,i],Lrho,Urho))

                if (RHO_LIMIT_TYPE == 11)
                    alphai = limiting_param_rhobound(U_low[1][i], coeff*rbararr[1,i],Lrho,Urho)
                    if i < N+1
                        m_ip1 = J*wq[i+1]
                        coeffip1 = dt/(m_ip1*lambda)
                        alphai = min(alphai,limiting_param_rhobound(U_low[1][i+1],-coeffip1*rbararr[1,i],Lrho,Urho))
                    end
                end
                for j = 1:N+1
                    Limrho[i,j] = alphai
                end
            end
        end

        if (S_LIMIT_TYPE == 0)
            Lims .= 1.0
        elseif ((S_LIMIT_TYPE == 1) || (S_LIMIT_TYPE == 2) || (S_LIMIT_TYPE == 4) || (S_LIMIT_TYPE == 10))
            # Minimum specific entropy principle
            Pi = zeros(3)
            for i = 1:N+1
                Pi = zeros(3)
                m_i = J*wq[i]
                for j = 1:N+1
                    if i != j
                        for c = 1:3
                            Pi[c] += dt/(m_i)*(F_low[c][i,j]-F_high[c][i,j])
                        end 
                    end
                end
                if (S_LIMIT_TYPE == 1) # Strongest nodewise bound
                    if i == 1
                        Ls = min(sk[2,k],sk[end,mod1(k-1,K)],sk[i,k])
                    elseif i == N+1
                        Ls = min(sk[N,k],sk[1,mod1(k+1,K)],sk[i,k])
                    else
                        Ls = min(sk[i-1,k],sk[i+1,K],sk[i,k])
                    end
                elseif (S_LIMIT_TYPE == 4) # Weakest nodewise bound
                    Ls = SMIN_GLOBAL
                elseif (S_LIMIT_TYPE == 10)
                    if (SMIN_GLOBAL > 0)
                        Ls = 0.99*SMIN_GLOBAL
                    elseif (SMIN_GLOBAL < 0)
                        Ls = 1.01*SMIN_GLOBAL
                    else
                        Ls = SMIN_GLOBAL - 1e-6
                    end
                else                     # Nodewise bound with smoothness indicator
                    epsk = smoothness_indicator[k]
                    if i == 1
                        Ls = epsk*min(sk[2,k],sk[end,mod1(k-1,K)],sk[i,k])+(1-epsk)*SMIN_GLOBAL
                    elseif i == N+1
                        Ls = epsk*min(sk[N,k],sk[1,mod1(k+1,K)],sk[i,k])+(1-epsk)*SMIN_GLOBAL
                    else
                        Ls = epsk*min(sk[i-1,k],sk[i+1,K],sk[i,k])+(1-epsk)*SMIN_GLOBAL
                    end
                end
                li = limiting_param_sbound(U_low[1][i],U_low[2][i],U_low[3][i],Pi[1],Pi[2],Pi[3],Ls)
                for j = 1:N+1
                    Lims[i,j] = li
                end
            end
        elseif (S_LIMIT_TYPE == 3)
            # Minimum specific entropy principle (integrated)
            P = zeros(3)
            for i = 1:N+1
                for j = 1:N+1
                    if i != j
                        for c = 1:3
                            P[c] += dt/(2/K)*(F_low[c][i,j]-F_high[c][i,j])
                        end
                    end
                end
            end
            Ls = min(sULavg[mod1(k-1,K)],sULavg[k],sULavg[mod1(k+1,K)])
            l = limiting_param_sbound(rhoLavg[k],rhouLavg[k],ELavg[k],P[1],P[2],P[3],Ls)
            for i = 1:N+1
                for j = 1:N+1
                    Lims[i,j] = l
                end
            end
        elseif ((S_LIMIT_TYPE == 5) || (S_LIMIT_TYPE == 6))
            Pij = zeros(3)
            for i = 1:N+1
                lambda_j = 1/N
                m_i = J*wq[i]
                for j = 1:N+1
                    if i != j
                        for c = 1:3
                            Pij[c] = dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
                        end
                        if (S_LIMIT_TYPE== 5)
                            if i == 1
                                Ls = min(sk[2,k],sk[end,mod1(k-1,K)],sk[i,k])
                            elseif i == N+1
                                Ls = min(sk[N,k],sk[1,mod1(k+1,K)],sk[i,k])
                            else
                                Ls = min(sk[i-1,k],sk[i+1,K],sk[i,k])
                            end
                        elseif (S_LIMIT_TYPE== 6)
                            Ls = SMIN_GLOBAL
                        end
                        Lims[i,j] = limiting_param_sbound(U_low[1][i],U_low[2][i],U_low[3][i],Pij[1],Pij[2],Pij[3],Ls)
                    end
                end
            end
            # Ls = SMIN_GLOBAL
            # P1 = zeros(3)
            # P2 = zeros(3)
            # for i = 1:N+1
            #     lambda_j = 1/2
            #     m_i = J*wq[i]
            #     for j = 1:N+1
            #         if j < i
            #             for c = 1:3
            #                 P1[c] += dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
            #             end
            #         end
            #         if j > i
            #             for c = 1:3
            #                 P2[c] += dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
            #             end
            #         end
            #     end
            #     l1 = limiting_param_sbound(U_low[1][i],U_low[2][i],U_low[3][i],P1[1],P1[2],P1[3],Ls)
            #     l2 = limiting_param_sbound(U_low[1][i],U_low[2][i],U_low[3][i],P2[1],P2[2],P2[3],Ls)
            #     for j = 1:N+1
            #         Lims[i,j] = min(l1,l2)
            #     end
            # end
        elseif ((S_LIMIT_TYPE == 7) || (S_LIMIT_TYPE == 8) || (S_LIMIT_TYPE == 9) || (S_LIMIT_TYPE == 11))
            lambda = 1/2

            for i = 1:N+1
                alphai = Inf
                m_i = J*wq[i]
                coeff = dt/(m_i*lambda)

                if (S_LIMIT_TYPE == 7)
                    if i == 1
                        Ls = min(sk[2,k],sk[end,mod1(k-1,K)],sk[i,k])
                    elseif i == N+1
                        Ls = min(sk[N,k],sk[1,mod1(k+1,K)],sk[i,k])
                    else
                        Ls = min(sk[i-1,k],sk[i+1,K],sk[i,k])
                    end
                elseif (S_LIMIT_TYPE == 8)
                    Ls = SMIN_GLOBAL
                elseif ((S_LIMIT_TYPE == 9) || (S_LIMIT_TYPE == 11))
                    if (SMIN_GLOBAL > 0)
                        Ls = 0.99*SMIN_GLOBAL
                    elseif (SMIN_GLOBAL < 0)
                        Ls = 1.01*SMIN_GLOBAL
                    else
                        Ls = SMIN_GLOBAL - 1e-6
                    end
                end

                if i > 1
                    alphai = min(alphai,limiting_param_sbound(U_low[1][i],U_low[2][i],U_low[3][i], coeff*rbararr[1,i-1], coeff*rbararr[2,i-1], coeff*rbararr[3,i-1],Ls))
                    alphai = min(alphai,limiting_param_sbound(U_low[1][i],U_low[2][i],U_low[3][i],-coeff*rbararr[1,i-1],-coeff*rbararr[2,i-1],-coeff*rbararr[3,i-1],Ls))
                end
                alphai = min(alphai,limiting_param_sbound(U_low[1][i],U_low[2][i],U_low[3][i], coeff*rbararr[1,i], coeff*rbararr[2,i], coeff*rbararr[3,i],Ls))
                alphai = min(alphai,limiting_param_sbound(U_low[1][i],U_low[2][i],U_low[3][i],-coeff*rbararr[1,i],-coeff*rbararr[2,i],-coeff*rbararr[3,i],Ls))

                if (S_LIMIT_TYPE == 11)
                    alphai = limiting_param_sbound(U_low[1][i],U_low[2][i],U_low[3][i], coeff*rbararr[1,i], coeff*rbararr[2,i], coeff*rbararr[3,i],Ls)
                    if i < N+1
                        m_ip1 = J*wq[i+1]
                        coeffip1 = dt/(m_ip1*lambda)
                        alphai = min(alphai,limiting_param_sbound(U_low[1][i+1],U_low[2][i+1],U_low[3][i+1],-coeffip1*rbararr[1,i],-coeffip1*rbararr[2,i],-coeffip1*rbararr[3,i],Ls))
                    end
                end
                for j = 1:N+1
                    Lims[i,j] = alphai
                end
            end

        end

        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    L[i,j] = min(L[i,j],Limrho[i,j],Lims[i,j])
                    if is_low_order
                        L[i,j] = 0.0
                    end
                end
            end
        end

        if (IS_SUBCELL_LIMIT)
            alphaarr = minimum(L,dims=2)
            for i = 1:N+1
                Lplot[i,k] = alphaarr[i]
            end

            for c = 1:3
                # With limiting
                rhsU[c][:,k] = sum(-F_low[c],dims=2)
                rhsU[c][1,k] += -F_P[c][1]
                rhsU[c][N+1,k] += -F_P[c][2]
            end

            for i = 1:N+1
                for c = 1:3
                    if ((RHO_LIMIT_TYPE == 11) && (S_LIMIT_TYPE == 11))
                        if (i == 1)
                            rhsU[c][i,k] += alphaarr[1]*rbararr[c,1]
                        elseif (i == N+1)
                            rhsU[c][i,k] += alphaarr[i]*rbararr[c,i] - alphaarr[i-1]*rbararr[c,i-1]
                        else
                            rhsU[c][i,k] += alphaarr[i]*rbararr[c,i] - alphaarr[i-1]*rbararr[c,i-1]
                        end
                    else
                        if (i == 1)
                            rhsU[c][i,k] += min(alphaarr[1],alphaarr[2])*rbararr[c,1]
                        elseif (i == N+1)
                            rhsU[c][i,k] += alphaarr[i]*rbararr[c,i] - min(alphaarr[i-1],alphaarr[i])*rbararr[c,i-1]
                        else
                            rhsU[c][i,k] += min(alphaarr[i],alphaarr[i+1])*rbararr[c,i] - min(alphaarr[i-1],alphaarr[i])*rbararr[c,i-1]
                        end
                    end
                end
            end

            for c = 1:3
                rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
            end
        else
            # Symmetrize limiting parameters
            for i = 1:N+1
                for j = 1:N+1
                    if i != j
                        l_ij = min(L[i,j],L[j,i])
                        L[i,j] = l_ij
                        L[j,i] = l_ij
                    end
                end
            end

            if (IS_ELEMENTWISE_LIMIT)
                # elementwise limiting
                l = 1.0
                for i = 1:N+1
                    for j = 1:N+1
                        if i != j
                            l = min(l,L[i,j])
                        end
                    end
                end

                for i = 1:N+1
                    for j = 1:N+1
                        if i != j
                            L[i,j] = l
                            L[j,i] = l
                        end
                    end
                end
            end

            for i = 1:N+1
                Lplot[i,k] = minimum(L[i,:])
            end

            # construct rhs
            for c = 1:3
                # With limiting
                rhsU[c][:,k] = sum((L.-1).*F_low[c] - L.*F_high[c],dims=2)
                rhsU[c][1,k] += -F_P[c][1]
                rhsU[c][N+1,k] += -F_P[c][2]
                rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
            end
        end
    end

    return rhsU,dt,Lplot
end


function get_U_low(U,K,N,wq,S0,Mlump_inv,T,dt,in_s1)
    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[1]/(2*p)

    J = 1/K # assume uniform interval, and domain [-1,1]

    # Low order and high order algebraic fluxes
    F_low       = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_P         = [zeros(2),zeros(2),zeros(2)] # 1: left boundary, 2: right boundary
    U_low       = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]
    dii_arr     = zeros(N+1,K)

    wavespd_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
        end
    end

    for k = 1:K
        # Interior dissp coeff
        for i = 1:N+1
            for j = 1:N+1
                if i != j 
                    dij = abs(S0[i,j])*max(wavespd_arr[i,k],wavespd_arr[j,k])
                    if in_s1
                        dii_arr[i,k] = dii_arr[i,k] + dij
                    end
                end
            end
        end

        # Interface dissipation
        U_left   = [U[1][end,mod1(k-1,K)]; U[2][end,mod1(k-1,K)]; U[3][end,mod1(k-1,K)]]
        U_right  = [U[1][1  ,mod1(k+1,K)]; U[2][1  ,mod1(k+1,K)]; U[3][1  ,mod1(k+1,K)]]
        wavespd_l = max(wavespd_arr[1,k]  ,wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))
        
        if in_s1
            dii_arr[1,k] = dii_arr[1,k] + 0.5*wavespd_l 
            dii_arr[N+1,k] = dii_arr[N+1,k] + 0.5*wavespd_r 
        end
    end

    if in_s1
        for k = 1:K 
            for i = 1:N+1
                dt = min(dt,1.0*wq[i]*J/2.0/dii_arr[i,k])
            end
        end
        dt = min(CFL*dt,T-t)
    end

    # Calculate low order averages
    for k = 1:K
        for i = 1:N+1
            for j = 1:N+1
                if i != j # skip diagonal
                    wavespd = max(wavespd_arr[i,k],wavespd_arr[j,k])
                    for c = 1:3
                        F_low[c][i,j]  = flux_lowIDP(U[c][i,k],U[c][j,k],flux[c][i,k],flux[c][j,k],S0[i,j],wavespd)
                    end
                end
            end
        end

        # Assemble matrix of low and high order algebraic fluxes
        # interface of the element
        U_left   = [U[1][end,mod1(k-1,K)]; U[2][end,mod1(k-1,K)]; U[3][end,mod1(k-1,K)]]
        U_right  = [U[1][1  ,mod1(k+1,K)]; U[2][1  ,mod1(k+1,K)]; U[3][1  ,mod1(k+1,K)]]
        f_left   = [flux[1][end,mod1(k-1,K)]; flux[2][end,mod1(k-1,K)]; flux[3][end,mod1(k-1,K)]]
        f_right  = [flux[1][1  ,mod1(k+1,K)]; flux[2][1  ,mod1(k+1,K)]; flux[3][1  ,mod1(k+1,K)]]
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        for c = 1:3
            F_P[c][1] = flux_lowIDP(U[c][1,k],U_left[c],flux[c][1,k],f_left[c],-0.5,wavespd_l)
            F_P[c][2] = flux_lowIDP(U[c][end,k],U_right[c],flux[c][end,k],f_right[c],0.5,wavespd_r)
        end

        # Calculate limiting parameters over interior of the element
        for c = 1:3
            tmp = sum(-F_low[c],dims=2)
            for i = 1:N+1
                U_low[c][i,k] = tmp[i]
            end
            U_low[c][1,k] -= F_P[c][1]
            U_low[c][end,k] -= F_P[c][2]
            U_low[c][:,k] .= 
            tmp = U[c][:,k]+dt/J*Mlump_inv*U_low[c][:,k]
            for i = 1:N+1
                U_low[c][i,k] = tmp[i]
            end
        end
    end

    return U_low,dt
end



# Time stepping
"Time integration"
t = 0.0
i = 0
U = collect(U)

Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM
resW = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
resZ = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
barL = zeros(size(x))

gr(size=(800,800),ylims=(0,3.2),legend=false,markerstrokewidth=1,markersize=2)

anim = Animation()
while t < T

    # if abs(T-t) < dt
    #     global dt = T - t
    # end

    dt = Inf
    # SSPRK(3,3)
    rhsU,dt,Lplot = rhs_IDP(U,K,N,wq,S,S0,Mlump_inv,T,dt,true,is_low_order)
    @. resW = U + dt*rhsU
    rhsU,_,_ = rhs_IDP(resW,K,N,wq,S,S0,Mlump_inv,T,dt,false,is_low_order)
    @. resZ = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resZ
    rhsU,_,_ = rhs_IDP(resW,K,N,wq,S,S0,Mlump_inv,T,dt,false,is_low_order)
    @. resZ = resW+dt*rhsU
    @. U = 1/3*U+2/3*resZ

    barL .= Lplot

    # resW,dt = get_U_low(U,K,N,wq,S0,Mlump_inv,T,dt,true)
    # resZ,_ = get_U_low(resW,K,N,wq,S0,Mlump_inv,T,dt,false)
    # @. resW = 3/4*U+1/4*resZ
    # resZ,_ = get_U_low(resW,K,N,wq,S0,Mlump_inv,T,dt,false)
    # @. U = 1/3*U+2/3*resZ

    global t = t + dt
    global i = i + 1
    println("Current time $t with time step size $dt, and final time $T")

    if ((i % 10 == 0) && PLOTGIF)
        #plot(Vp*x,Vp*U[1])
        plot(Vp*x,Vp*U[1])
        Bl = -1
        Br = 1
        ptL = Bl+(Br-Bl)/K/(N+1)/2
        ptR = Br-(Br-Bl)/K/(N+1)/2
        hplot = (Br-Bl)/K/(N+1)
        for k = 1:K
            plot!(ptL+(k-1)*hplot*(N+1):hplot:ptL+k*hplot*(N+1), 1 .-barL[:,k],st=:bar,alpha=0.2)
        end
        frame(anim)
    end
end


# rho = U[1]
# rhou = U[2]
# E = U[3]

# f1 = Figure()
# axis = Axis(f1[1,1])
# lw = 3

# xp = Vp*x
# rhop = Vp*U[1]
# l1 = lines!(xp[:],rhop[:],linestyle=nothing,linewidth=lw,color=:royalblue1)
# barplot!(x[:],1 .- barL[:],color=(:darkorange1,0.5),strokecolor=(:black,0.5),strokewidth=1)

# save("N=$N,K=$K,T=$T,is_low_order=$is_low_order,RHO_LIMIT_TYPE=$RHO_LIMIT_TYPE,S_LIMIT_TYPE=$S_LIMIT_TYPE,HIGH_ORDER_FLUX_TYPE=$HIGH_ORDER_FLUX_TYPE,IS_ELEMENTWISE_LIMIT=$IS_ELEMENTWISE_LIMIT,bounds.png",f1)

# plot(Vp*x,Vp*U[1])
# savefig("N=$N,K=$K,T=$T,is_low_order=$is_low_order,RHO_LIMIT_TYPE=$RHO_LIMIT_TYPE,S_LIMIT_TYPE=$S_LIMIT_TYPE,HIGH_ORDER_FLUX_TYPE=$HIGH_ORDER_FLUX_TYPE,IS_ELEMENTWISE_LIMIT=$IS_ELEMENTWISE_LIMIT,bounds.png")
if (PLOTGIF)
    gif(anim,"N=$N,K=$K,T=$T,is_low_order=$is_low_order,RHO_LIMIT_TYPE=$RHO_LIMIT_TYPE,S_LIMIT_TYPE=$S_LIMIT_TYPE,HIGH_ORDER_FLUX_TYPE=$HIGH_ORDER_FLUX_TYPE,IS_ELEMENTWISE_LIMIT=$IS_ELEMENTWISE_LIMIT,IS_SUBCELL_LIMIT=$IS_SUBCELL_LIMIT,bounds.gif",fps=30)
end


rho = U[1]
rhou = U[2]
E = U[3]
exact_rho = zeros(N+1,K)
exact_rhou = zeros(N+1,K)
exact_E = zeros(N+1,K)
for k = 1:K
    for i = 1:N+1
        rho,rhou,E = exact_sol_sine_wave(x[i,k],T)
        exact_rho[i,k] = rho
        exact_rhou[i,k] = rhou
        exact_E[i,k] = E
    end
end
J = 1/K  # TODO: hardcoded jacobian

Linferr = maximum(abs.(exact_rho-rho))/maximum(abs.(exact_rho)) +
          maximum(abs.(exact_rhou-rhou))/maximum(abs.(exact_rhou)) +
          maximum(abs.(exact_E-E))/maximum(abs.(exact_E))

L1err = sum(J*wq.*abs.(exact_rho-rho))/sum(J*wq.*abs.(exact_rho)) +
        sum(J*wq.*abs.(exact_rhou-rhou))/sum(J*wq.*abs.(exact_rhou)) +
        sum(J*wq.*abs.(exact_E-E))/sum(J*wq.*abs.(exact_E))

L2err = sqrt(sum(J*wq.*abs.(exact_rho-rho).^2))/sqrt(sum(J*wq.*abs.(exact_rho).^2)) +
        sqrt(sum(J*wq.*abs.(exact_rhou-rhou).^2))/sqrt(sum(J*wq.*abs.(exact_rhou).^2)) +
        sqrt(sum(J*wq.*abs.(exact_E-E).^2))/sqrt(sum(J*wq.*abs.(exact_E).^2))

println("N = $N, K = $K")
println("L1 error is $L1err")
println("L2 error is $L2err")
println("Linf error is $Linferr")

# df = DataFrame(N = Int64[], K = Int64[], T = Float64[], ISLOWORDER = Bool[], CASENUM = Int64[], RHO_LIMIT_TYPE = Int64[], S_LIMIT_TYPE = Int64[], IS_ELEMENTWISE_LIMIT = Bool[], IS_SUBCELL_LIMIT = Bool[], HIGH_ORDER_FLUX_TYPE = Int64[], L1ERR = Float64[], L2ERR = Float64[], LINFERR = Float64[])
# CSV.write("dg1D_euler_bounds_convergence.csv",df)
if ADDTODF
    df = CSV.read("dg1D_euler_bounds_convergence.csv", DataFrame)
    push!(df,(N,K,T,is_low_order,CASENUM,RHO_LIMIT_TYPE,S_LIMIT_TYPE,IS_ELEMENTWISE_LIMIT,IS_SUBCELL_LIMIT,HIGH_ORDER_FLUX_TYPE,L1err,L2err,Linferr))
    CSV.write("dg1D_euler_bounds_convergence.csv",df)
    @show df
end

#=
# Plot a single time step of low order update
dt = Inf
rhsU,dt = rhs_IDP(U,K,N,wq,S,S0,Mlump_inv,T,dt,true,true)
@. U = U + dt*rhsU

# Ulow = get_U_low(U,K,N,wq,S0,Mlump_inv,T,dt,true)

xp = Vp*x
rhop = Vp*U[1]
l3 = lines!(xp[:],rhop[:],linestyle=:dash,linewidth=lw,color=:seagreen)
rhoavg2 = zeros(N+1,K)
J = 1/K
for k = 1:K
    rhoavgk = 0.0
    for i = 1:N+1
        rhoavgk += J*wq[i]*Ulow[1][i,k]
    end
    for i = 1:N+1
        rhoavg2[i,k] = rhoavgk/(2/K)
    end
end
xp = Vp*x
rhoavgp = Vp*rhoavg
l4 = lines!(xp[:],rhoavgp[:],linestyle=:dot,linewidth=lw,color=:goldenrod1)

averagedmin = zeros(N+1,K)
averagedmax = zeros(N+1,K)
for k = 1:K
    for i = 1:N+1
        averagedmin[i,k] = min(rhoavg1[1,mod1(k-1,K)],rhoavg1[1,k],rhoavg1[1,mod1(k+1,K)])
        averagedmax[i,k] = max(rhoavg1[1,mod1(k-1,K)],rhoavg1[1,k],rhoavg1[1,mod1(k+1,K)])
    end
end
=#
