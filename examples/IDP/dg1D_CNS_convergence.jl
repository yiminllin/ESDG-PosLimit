using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using DelimitedFiles
using DataFrames
using JLD2
using FileIO
using CairoMakie

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
using EntropyStableEuler
using EntropyStableEuler.Fluxes1D

"""Initial condition"""
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

const max_iter = 10000
const tol = 1e-14

function exact_sol_viscous_shocktube(x,t)
    u   = bisection_solve_velocity(x-v_inf*t,max_iter,tol)
    rho = m_0/u
    e   = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-u^2)
    return rho, rho*(v_inf+u), rho*(e+1/2*(v_inf+u)^2)
end

"Approximation parameters"
# const N = 4 # The order of approximation
# const K = 50#800
const T = 1.0#3.0
# const is_low_order = false

const LIMITOPT = 2     # 1 if elementwise limiting lij, 2 if elementwise limiting li
const POSDETECT = 0    # 1 if turn on detection, 0 otherwise
const LBOUNDTYPE = 0.5   # 0 if use POSTOL as lower bound, if > 0, use LBOUNDTYPE*loworder
const POSTOL = 1e-14
const TOL = 5e-16
const CFL = 0.1

const ISSMOOTH = true

# Becker viscous shocktube
const γ = 1.4
if ISSMOOTH
    const M_0 = 3.0  # Smooth
    const mu = 0.01
else
    const M_0 = 20.0   # Sharp
    const mu = 0.001
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

# const uL = v_0+v_inf
# const uR = v_1+v_inf
# const rhoL = m_0/v_0
# const rhoR = m_0/v_1
# const eL = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_0^2)
# const eR = 1/(2*γ)*((γ+1)/(γ-1)*v_01^2-v_1^2)
# const pL = (γ-1)*rhoL*eL
# const pR = (γ-1)*rhoR*eR
# const EL = pL/(γ-1)+0.5*rhoL*uL^2
# const ER = pR/(γ-1)+0.5*rhoR*uR^2

const Bl = -1.0
const Br = 1.5

const rhoL,rhouL,EL = exact_sol_viscous_shocktube(Bl,T)
const rhoR,rhouR,ER = exact_sol_viscous_shocktube(Br,T)
const uL = rhouL/rhoL
const uR = rhouR/rhoR
const pL = pfun_nd(rhoL,rhouL,EL)
const pR = pfun_nd(rhoR,rhouR,ER)

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

function zhang_wavespd(rhoL,rhouL,EL,sigma2L,sigma3L,pL,rhoR,rhouR,ER,sigma2R,sigma3R,pR)
    uL   = rhouL/rhoL
    eL   = (EL - .5*rhoL*uL^2)/rhoL
    tauL = sigma2L
    qL   = uL*tauL-sigma3L
    uR   = rhouR/rhoR
    eR   = (ER - .5*rhoR*uR^2)/rhoR
    tauR = sigma2R
    qR   = uR*tauR-sigma3R
    wavespdL = POSTOL+abs(uL)+1/(2*rhoL^2*eL)*(sqrt(rhoL^2*qL^2+2*rhoL^2*eL*abs(tauL-pL)^2)+rhoL*abs(qL))
    wavespdR = POSTOL+abs(uR)+1/(2*rhoR^2*eR)*(sqrt(rhoR^2*qR^2+2*rhoR^2*eR*abs(tauR-pR)^2)+rhoR*abs(qR))
    return max(wavespdL,wavespdR)
end

function zhang_wavespd(rhoL,rhouL,EL,sigma2L,sigma3L,pL)
    uL   = rhouL/rhoL
    eL   = (EL - .5*rhoL*uL^2)/rhoL
    tauL = sigma2L
    qL   = uL*tauL-sigma3L
    wavespdL = abs(uL)+1/(2*rhoL^2*eL)*(sqrt(rhoL^2*qL^2+2*rhoL^2*eL*abs(tauL-pL)^2)+rhoL*abs(qL))
    return wavespdL
end

function limiting_param(U_low, P_ij, Lrho, Lrhoe)
    l = 1.0
    # Limit density
    if U_low[1] + P_ij[1] < Lrho
        l = max((Lrho-U_low[1])/P_ij[1],0.0)
    end

    # limiting internal energy (via quadratic function)
    a = P_ij[1]*P_ij[3]-1.0/2.0*P_ij[2]^2
    b = U_low[3]*P_ij[1]+U_low[1]*P_ij[3]-U_low[2]*P_ij[2]-P_ij[1]*Lrhoe
    c = U_low[3]*U_low[1]-1.0/2.0*U_low[2]^2-U_low[1]*Lrhoe

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

function flux_lowIDP(U_i,U_j,f_i,f_j,c_ij,wavespd)
    return c_ij*(f_i+f_j)-abs(c_ij)*wavespd*(U_j-U_i)
end

function flux_ES(rho_i,u_i,beta_i,rho_j,u_j,beta_j,c_ij)
    return 2*c_ij.*euler_fluxes(rho_i,u_i,beta_i,rho_j,u_j,beta_j)
end

function flux_viscous(sigma_i,sigma_j,c_ij)
    return c_ij*(sigma_i+sigma_j)
end

function rhs_viscous(U,K,N,Mlump_inv,Vf,mapP,nxJ,S,Dr,LIFT)
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

    return sigmax, sigma
end

function rhs_IDP(U,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,T,dt,t,in_s1,is_low_order)
    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[1]/(2*p)

    J = (Br-Bl)/K/2 # assume uniform interval

    # Low order and high order algebraic fluxes
    F_low       = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_high      = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_low_P     = [zeros(2),zeros(2),zeros(2)] # 1: left boundary, 2: right boundary
    F_high_P    = [zeros(2),zeros(2),zeros(2)]
    L           = zeros(N+1,N+1) # Array of limiting params
    rhsU        = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]

    # TODO: redundant!
    _,sigma = rhs_viscous(U,K,N,Mlump_inv,Vf,mapP,nxJ,S,Dr,LIFT)
    wavespd_arr = zeros(N+1,K)
    beta_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
            beta_arr[i,k] = zhang_wavespd(U[1][i,k],U[2][i,k],U[3][i,k],sigma[2][i,k],sigma[3][i,k],p[i,k])
        end
    end

    d_ii_arr = zeros(N+1,K)
    for k = 1:K 
        for i = 1:N+1
            for j = 1:N+1
                if i != j 
                    d_ij = abs(S0[i,j])*max(beta_arr[i,k],beta_arr[j,k],wavespd_arr[i,k],wavespd_arr[j,k])
                    if in_s1
                        d_ii_arr[i,k] = d_ii_arr[i,k] + d_ij
                    end
                end
            end
        end

        EL = pL/(γ-1)+0.5*rhoL*uL^2
        ER = pR/(γ-1)+0.5*rhoR*uR^2
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        p_left   = (k == 1) ? pL : p[end,k-1]
        p_right  = (k == K) ? pR : p[1,k+1]
        # Assume velocity doesn't change at boundary, isothermal
        TL = pL/rhoL/(γ-1)/cv
        TR = pR/rhoR/(γ-1)/cv
        sigma_left  = (k == 1) ? [sigma[1][1,1];sigma[2][1,1];sigma[3][1,1]]       : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [sigma[1][end,k];sigma[2][end,k];sigma[3][end,k]] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]]
        wavespd_l = max(beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left),
                        wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right),
                        wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        if in_s1
            d_ii_arr[1,k] = d_ii_arr[1,k] + 0.5*wavespd_l 
            d_ii_arr[N+1,k] = d_ii_arr[N+1,k] + 0.5*wavespd_r 
        end
    end

    if in_s1
        for k = 1:K 
            for i = 1:N+1
                dt = min(dt,1.0*wq[i]*J/2.0/d_ii_arr[i,k])
            end
        end
        dt = min(CFL*dt,T-t)
        dt = min(dt, CFL*(N+1)*(N+2)/2/K/K)
    end

    Lplot = zeros(K)

    # TODO: hardcoded
    for k = 1:K
        L = zeros(N+1,N+1)
        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    wavespd = max(beta_arr[i,k],beta_arr[j,k],wavespd_arr[i,k],wavespd_arr[j,k])
                    fluxS = flux_ES(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],S[i,j])
                    for c = 1:3
                        F_low[c][i,j]  = -flux_viscous(sigma[c][i,k],sigma[c][j,k],S0[i,j])+flux_lowIDP(U[c][i,k],U[c][j,k],flux[c][i,k],flux[c][j,k],S0[i,j],wavespd)
                        F_high[c][i,j] = -flux_viscous(sigma[c][i,k],sigma[c][j,k],S[i,j])+fluxS[c]
                    end
                end
            end
        end

        EL = pL/(γ-1)+0.5*rhoL*uL^2
        ER = pR/(γ-1)+0.5*rhoR*uR^2
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        p_left   = (k == 1) ? pL : p[end,k-1]
        p_right  = (k == K) ? pR : p[1,k+1]
        # Assume velocity doesn't change at boundary, isothermal
        TL = pL/rhoL/(γ-1)/cv
        TR = pR/rhoR/(γ-1)/cv
        sigma_left  = (k == 1) ? [sigma[1][1,1];sigma[2][1,1];sigma[3][1,1]]       : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [sigma[1][end,k];sigma[2][end,k];sigma[3][end,k]] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]]
        # wavespd_l = max(beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left))
        # wavespd_r = max(beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right))
        wavespd_l = max(beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left),
                        wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right),
                        wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        for c = 1:3
            F_low_P[c][1] = -flux_viscous(sigma[c][1,k],sigma_left[c],-1/2)+flux_lowIDP(U[c][1,k],U_left[c],flux[c][1,k],f_left[c],-0.5,wavespd_l)
            F_low_P[c][2] = -flux_viscous(sigma[c][end,k],sigma_right[c],1/2)+flux_lowIDP(U[c][end,k],U_right[c],flux[c][end,k],f_right[c],0.5,wavespd_r)

            if k == 1
                F_low_P[c][1] = -flux_viscous(sigma[c][1,k],sigma_left[c],-1/2)-0.5*(flux[c][1,k]+f_left[c])
            end

            if k == K
                F_low_P[c][2] = -flux_viscous(sigma[c][end,k],sigma_right[c],1/2)+0.5*(flux[c][end,k]+f_right[c])
            end

            F_high_P[c][1] = F_low_P[c][1]
            F_high_P[c][2] = F_low_P[c][2]
        end

        P_ij = zeros(3,1)
        # TODO: redundant
        U_low =  [zeros(N+1,1),zeros(N+1,1),zeros(N+1,1)]
        for c = 1:3
            U_low[c] .= sum(-F_low[c],dims=2)
            U_low[c][1] -= F_low_P[c][1]
            U_low[c][end] -= F_low_P[c][2]
            U_low[c] .= U[c][:,k]+dt/J*Mlump_inv*U_low[c]
        end
        U_high =  [zeros(N+1,1),zeros(N+1,1),zeros(N+1,1)]
        for c = 1:3
            U_high[c] .= sum(-F_high[c],dims=2)
            U_high[c][1] -= F_high_P[c][1]
            U_high[c][end] -= F_high_P[c][2]
            U_high[c] .= U[c][:,k]+dt/J*Mlump_inv*U_high[c]
        end
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

        if (POSDETECT == 0)
            is_H_positive = false
        end

        if (LIMITOPT == 1)
            for i = 1:N+1
                lambda_j = 1/N
                m_i = J*wq[i]
                for j = 1:N+1
                    if i != j
                        for c = 1:3
                            P_ij[c] = dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
                        end
                        if !is_H_positive
                            Lrho  = POSTOL
                            Lrhoe = POSTOL
                            L[i,j] = limiting_param([U_low[1][i]; U_low[2][i]; U_low[3][i]],P_ij,Lrho,Lrhoe)
                        else
                            L[i,j] = 1.0
                        end
                    end
                end
            end
        elseif (LIMITOPT == 2)
            li_min = 1.0
            for i = 1:N+1
                lambda_j = 1/N
                m_i = J*wq[i]
                for c = 1:3
                    P_ij[c] = 0.0
                end
                for j = 1:N+1
                    for c = 1:3
                        P_ij[c] += dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
                    end
                end
                if (LBOUNDTYPE == 0)
                    Lrho  = POSTOL
                    Lrhoe = POSTOL
                elseif (LBOUNDTYPE > 0)
                    Lrho  = LBOUNDTYPE*U_low[1][i]
                    Lrhoe = LBOUNDTYPE*pfun_nd(U_low[1][i],U_low[2][i],U_low[3][i])/(γ-1)
                end
                l = limiting_param([U_low[1][i]; U_low[2][i]; U_low[3][i]],P_ij,Lrho,Lrhoe)
                li_min = min(li_min,l)
            end

            for i = 1:N+1
                for j = 1:N+1
                    if i != j
                        L[i,j] = li_min
                        L[j,i] = li_min
                    end
                end
            end
        end

        # # Symmetrize limiting parameters
        # for i = 1:N+1
        #     for j = 1:N+1
        #         if i != j
        #             l_ij = min(L[i,j],L[j,i])
        #             L[i,j] = l_ij
        #             L[j,i] = l_ij
        #         end
        #     end
        # end

        # elementwise limiting
        l = 1.0
        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    l = min(l,L[i,j])
                end
            end
        end
        if is_low_order
            l = 0.0
        end

        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    L[i,j] = l
                    L[j,i] = l
                end
            end
        end

        Lplot[k] = l

        # L = ones(size(L))

        for c = 1:3
            # With limiting
            rhsU[c][:,k] = sum((L.-1).*F_low[c] - L.*F_high[c],dims=2)
            # rhsU[c][:,k] = sum((-1).*F_low[c],dims=2)

            rhsU[c][1,k] += -F_high_P[c][1]
            rhsU[c][N+1,k] += -F_high_P[c][2]

            rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
        end
    end

    @show minimum(Lplot)

    return rhsU,dt
end

function rhs_IDPlow(U,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,dt,in_s1)
    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[1]/(2*p)

    J = (Br-Bl)/K/2 # assume uniform interval

    # Low order and high order algebraic fluxes
    F_low       = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_low_P     = [zeros(2),zeros(2),zeros(2)] # 1: left boundary, 2: right boundary
    rhsU        = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]

    # TODO: redundant!
    _,sigma = rhs_viscous(U,K,N,Mlump_inv,Vf,mapP,nxJ,S0,Dr,LIFT)
    wavespd_arr = zeros(N+1,K)
    beta_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
            beta_arr[i,k] = zhang_wavespd(U[1][i,k],U[2][i,k],U[3][i,k],sigma[2][i,k],sigma[3][i,k],p[i,k])
        end
    end

    d_ii_arr = zeros(N+1,K)
    for k = 1:K 
        for i = 1:N+1
            for j = 1:N+1
                if i != j 
                    d_ij = abs(S0[i,j])*max(beta_arr[i,k],beta_arr[j,k])
                    if in_s1
                        d_ii_arr[i,k] = d_ii_arr[i,k] + d_ij
                    end
                end
            end
        end

        EL = pL/(γ-1)+0.5*rhoL*uL^2
        ER = pR/(γ-1)+0.5*rhoR*uR^2
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        p_left   = (k == 1) ? pL : p[end,k-1]
        p_right  = (k == K) ? pR : p[1,k+1]
        # Assume velocity doesn't change at boundary, isothermal
        TL = pL/rhoL/(γ-1)/cv
        TR = pR/rhoR/(γ-1)/cv
        sigma_left  = (k == 1) ? [sigma[1][1,1];sigma[2][1,1];sigma[3][1,1]]       : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [sigma[1][end,k];sigma[2][end,k];sigma[3][end,k]] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]]
        wavespd_l = max(beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left))
        wavespd_r = max(beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right))

        if in_s1
            d_ii_arr[1,k] = d_ii_arr[1,k] + 0.5*wavespd_l 
            d_ii_arr[N+1,k] = d_ii_arr[N+1,k] + 0.5*wavespd_r 
        end
    end

    if in_s1
        for k = 1:K 
            for i = 1:N+1
                dt = min(dt,1.0*wq[i]*J/2.0/d_ii_arr[i,k])
            end
        end
    end

    # TODO: hardcoded
    for k = 1:K
        L = zeros(N+1,N+1)
        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    wavespd = max(beta_arr[i,k],beta_arr[j,k])
                    for c = 1:3
                        F_low[c][i,j]  = -flux_viscous(sigma[c][i,k],sigma[c][j,k],S0[i,j])+flux_lowIDP(U[c][i,k],U[c][j,k],flux[c][i,k],flux[c][j,k],S0[i,j],wavespd)
                    end
                end
            end
        end

        EL = pL/(γ-1)+0.5*rhoL*uL^2
        ER = pR/(γ-1)+0.5*rhoR*uR^2
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        p_left   = (k == 1) ? pL : p[end,k-1]
        p_right  = (k == K) ? pR : p[1,k+1]
        # Assume velocity doesn't change at boundary, isothermal
        TL = pL/rhoL/(γ-1)/cv
        TR = pR/rhoR/(γ-1)/cv
        sigma_left  = (k == 1) ? [sigma[1][1,1];sigma[2][1,1];sigma[3][1,1]]       : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [sigma[1][end,k];sigma[2][end,k];sigma[3][end,k]] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]]
        wavespd_l = max(beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left))
        wavespd_r = max(beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right))


        for c = 1:3
            F_low_P[c][1] = -flux_viscous(sigma[c][1,k],sigma_left[c],-1/2)+flux_lowIDP(U[c][1,k],U_left[c],flux[c][1,k],f_left[c],-0.5,wavespd_l)
            F_low_P[c][2] = -flux_viscous(sigma[c][end,k],sigma_right[c],1/2)+flux_lowIDP(U[c][end,k],U_right[c],flux[c][end,k],f_right[c],0.5,wavespd_r)

            if k == 1
                F_low_P[c][1] = -flux_viscous(sigma[c][1,k],sigma_left[c],-1/2)-0.5*(flux[c][1,k]+f_left[c])
            end

            if k == K
                F_low_P[c][2] = -flux_viscous(sigma[c][end,k],sigma_right[c],1/2)+0.5*(flux[c][end,k]+f_right[c])
            end
        end

        for c = 1:3
            # With limiting
            rhsU[c][:,k] = sum((-1).*F_low[c],dims=2)

            rhsU[c][1,k] += -F_low_P[c][1]
            rhsU[c][N+1,k] += -F_low_P[c][2]

            rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
        end
    end

    return rhsU,dt
end

function rhs_ESDG(U,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,T,dt,t,in_s1,is_low_order)
    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    Ub = zero.(U)
    @. Ub[1] = U[1]
    @. Ub[2] = U[2]/U[1]
    @. Ub[3] = U[1]/(2*p)

    J = (Br-Bl)/K/2 # assume uniform interval

    # Low order and high order algebraic fluxes
    F_low       = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_high      = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_low_P     = [zeros(2),zeros(2),zeros(2)] # 1: left boundary, 2: right boundary
    F_high_P    = [zeros(2),zeros(2),zeros(2)]
    L           = zeros(N+1,N+1) # Array of limiting params
    rhsU        = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]

    # TODO: redundant!
    _,sigma = rhs_viscous(U,K,N,Mlump_inv,Vf,mapP,nxJ,S,Dr,LIFT)
    wavespd_arr = zeros(N+1,K)
    beta_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
            beta_arr[i,k] = zhang_wavespd(U[1][i,k],U[2][i,k],U[3][i,k],sigma[2][i,k],sigma[3][i,k],p[i,k])
        end
    end

    d_ii_arr = zeros(N+1,K)
    for k = 1:K 
        for i = 1:N+1
            for j = 1:N+1
                if i != j 
                    d_ij = abs(S0[i,j])*max(beta_arr[i,k],beta_arr[j,k],wavespd_arr[i,k],wavespd_arr[j,k])
                    if in_s1
                        d_ii_arr[i,k] = d_ii_arr[i,k] + d_ij
                    end
                end
            end
        end

        EL = pL/(γ-1)+0.5*rhoL*uL^2
        ER = pR/(γ-1)+0.5*rhoR*uR^2
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        p_left   = (k == 1) ? pL : p[end,k-1]
        p_right  = (k == K) ? pR : p[1,k+1]
        # Assume velocity doesn't change at boundary, isothermal
        TL = pL/rhoL/(γ-1)/cv
        TR = pR/rhoR/(γ-1)/cv
        sigma_left  = (k == 1) ? [sigma[1][1,1];sigma[2][1,1];sigma[3][1,1]]       : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [sigma[1][end,k];sigma[2][end,k];sigma[3][end,k]] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]]
        wavespd_l = max(beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left),
                        wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right),
                        wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        if in_s1
            d_ii_arr[1,k] = d_ii_arr[1,k] + 0.5*wavespd_l 
            d_ii_arr[N+1,k] = d_ii_arr[N+1,k] + 0.5*wavespd_r 
        end
    end

    dt = min(1e-6,T-t)

    # TODO: hardcoded
    for k = 1:K
        L = zeros(N+1,N+1)
        for i = 1:N+1
            for j = 1:N+1
                if i != j
                    fluxS = flux_ES(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],S[i,j])
                    for c = 1:3
                        F_high[c][i,j] = -flux_viscous(sigma[c][i,k],sigma[c][j,k],S[i,j])+fluxS[c]
                    end
                end
            end
        end

        EL = pL/(γ-1)+0.5*rhoL*uL^2
        ER = pR/(γ-1)+0.5*rhoR*uR^2
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        p_left   = (k == 1) ? pL : p[end,k-1]
        p_right  = (k == K) ? pR : p[1,k+1]
        # Assume velocity doesn't change at boundary, isothermal
        TL = pL/rhoL/(γ-1)/cv
        TR = pR/rhoR/(γ-1)/cv
        sigma_left  = (k == 1) ? [sigma[1][1,1];sigma[2][1,1];sigma[3][1,1]]       : [sigma[1][end,k-1]; sigma[2][end,k-1]; sigma[3][end,k-1]]
        sigma_right = (k == K) ? [sigma[1][end,k];sigma[2][end,k];sigma[3][end,k]] : [sigma[1][1,k+1]; sigma[2][1,k+1]; sigma[3][1,k+1]]
        # wavespd_l = max(beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left))
        # wavespd_r = max(beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right))
        wavespd_l = max(beta_arr[1,k],zhang_wavespd(U_left[1],U_left[2],U_left[3],sigma_left[2],sigma_left[3],p_left),
                        wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(beta_arr[end,k],zhang_wavespd(U_right[1],U_right[2],U_right[3],sigma_right[2],sigma_right[3],p_right),
                        wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        for c = 1:3
            F_low_P[c][1] = -flux_viscous(sigma[c][1,k],sigma_left[c],-1/2)+flux_lowIDP(U[c][1,k],U_left[c],flux[c][1,k],f_left[c],-0.5,wavespd_l)
            F_low_P[c][2] = -flux_viscous(sigma[c][end,k],sigma_right[c],1/2)+flux_lowIDP(U[c][end,k],U_right[c],flux[c][end,k],f_right[c],0.5,wavespd_r)

            if k == 1
                F_low_P[c][1] = -flux_viscous(sigma[c][1,k],sigma_left[c],-1/2)-0.5*(flux[c][1,k]+f_left[c])
            end

            if k == K
                F_low_P[c][2] = -flux_viscous(sigma[c][end,k],sigma_right[c],1/2)+0.5*(flux[c][end,k]+f_right[c])
            end

            F_high_P[c][1] = F_low_P[c][1]
            F_high_P[c][2] = F_low_P[c][2]
        end

        for c = 1:3
            # With limiting
            rhsU[c][:,k] = sum((-1).*F_high[c],dims=2)

            rhsU[c][1,k] += -F_high_P[c][1]
            rhsU[c][N+1,k] += -F_high_P[c][2]

            rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
        end
    end

    return rhsU,dt
end



Narr = [2;3;4;5]
Karr = [10;20;40;80;160]
# Narr = [2]#[5]
# Karr = [400]#[200]#[1600]#[50;100;200;400;800]
low_order_arr = [false]#[true;false]

for N in Narr
for K in Karr
for is_low_order in low_order_arr

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

U = exact_sol_viscous_shocktube.(x,0.0)
U = ([x[1] for x in U], [x[2] for x in U], [x[3] for x in U])

# Time stepping
"Time integration"
t = 0.0
U = collect(U)
resU = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
resW = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
resZ = [zeros(size(x)),zeros(size(x)),zeros(size(x))]

Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM

i = 1

while t < T

    # SSPRK(3,3)
    dt = Inf
    rhsU,dt = rhs_IDP(U,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,T,dt,t,true,is_low_order)
    # rhsU,dt = rhs_IDPlow(U,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,dt,true)
    # rhsU,dt = rhs_ESDG(U,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,T,dt,t,true,is_low_order)
    @. resW = U + dt*rhsU
    rhsU,_ = rhs_IDP(resW,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,T,dt,t,false,is_low_order)
    # rhsU,_ = rhs_IDPlow(resW,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,dt,false)
    # rhsU,_ = rhs_ESDG(resW,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,T,dt,t,false,is_low_order)
    @. resZ = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resZ
    rhsU,_ = rhs_IDP(resW,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,T,dt,t,false,is_low_order)
    # rhsU,_ = rhs_IDPlow(resW,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,dt,false)
    # rhsU,_ = rhs_ESDG(resW,K,N,Mlump_inv,Vf,mapP,nxJ,S0,S,Dr,LIFT,wq,T,dt,t,false,is_low_order)
    @. resZ = resW+dt*rhsU
    @. U = 1/3*U+2/3*resZ

    U[1][end,end] = rhoR
    U[2][end,end] = rhoR*uR
    U[3][end,end] = ER

    U[1][1,1] = rhoL
    U[2][1,1] = rhoL*uL
    U[3][1,1] = EL

    t = t + dt
    i = i + 1
    println("Current time $t with time step size $dt, and final time $T")
    # if mod(i,1000) == 1
    #     plot(x,U[1])
    #     # for k = 1:K
    #     #     plot!(ptL+(k-1)*hplot*(N+1):hplot:ptL+k*hplot*(N+1), 1 .-L_plot[:,k],st=:bar,alpha=0.2)
    #     # end
    #     frame(anim)
    # end
end

# plot(x,U[1])
# gif(anim,"~/Desktop/tmp.gif",fps=15)


exact_U = @. exact_sol_viscous_shocktube.(x,t)
exact_rho = [x[1] for x in exact_U]
exact_rhou = [x[2] for x in exact_U]
exact_u = exact_rhou./exact_rho
exact_E = [x[3] for x in exact_U]

rho = U[1]
u = U[2]./U[1]
p = pfun_nd.(U[1],U[2],U[3])
rhou = U[2]
E = U[3]
J = (Br-Bl)/K/2

# rho = rho[:,5:end-5]
# rhou = rhou[:,5:end-5]
# E = E[:,5:end-5]
# exact_rho = exact_rho[:,5:end-5]
# exact_rhou = exact_rhou[:,5:end-5]
# exact_E = exact_E[:,5:end-5]

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

# open("/data/yl184/1D-CNS-conv/N=$N,K=$K,CFL=$CFL,LOW=$is_low_order,issmooth=$ISSMOOTH,x,1D-CNS-conv.txt","w") do io
#     writedlm(io,x)
# end
# open("/data/yl184/1D-CNS-conv/N=$N,K=$K,CFL=$CFL,LOW=$is_low_order,issmooth=$ISSMOOTH,rho,1D-CNS-conv.txt","w") do io
#     writedlm(io,U[1])
# end

# df = DataFrame(N = Int64[], K = Int64[], T = Float64[], CFL = Float64[], LIMITOPT = Int64[], POSDETECT = Int64[], LBOUNDTYPE = Float64[], L1err = Float64[], L2err = Float64[], Linferr = Float64[])
df = load("dg1D_CNS_convergence.jld2","convergence_data")
push!(df,(N,K,T,CFL,LIMITOPT,POSDETECT,LBOUNDTYPE,L1err,L2err,Linferr))
save("dg1D_CNS_convergence.jld2","convergence_data",df)

f1 = Figure()
axis = Axis(f1[1,1])

lw = 3
is_low = [true;false]

rho = U[1]
l3 = lines!(x[:],rho[:],linestyle=:dot,linewidth=lw,color=:darkorange1,label="")

rho = [x[1] for x in exact_sol_viscous_shocktube.(x,T)]
l2 = lines!(x[:],rho[:],linestyle=:dash,linewidth=lw,color=:gray,label="Exact solution")

axislegend(labelsize=20,position = :rb)

save("dg1D-CNS-limited-smooth.png",f1)


end
end
end # for k in Karr
# println("Linf error is $Linferr")

#=
f1 = Figure()
axis = Axis(f1[1,1])

lw = 3
is_low = [true;false]

N = 2
k = 400
il = false
rho2 = U2[1]
l1 = lines!(x2[:],rho2[:],linestyle=nothing,linewidth=lw,color=:royalblue1,label="N=2,K=400")

N = 5
k = 200
il = false
rho5 = U5[1]
l3 = lines!(x5[:],rho5[:],linestyle=:dot,linewidth=lw,color=:darkorange1,label="N=5,K=200")

rho = [x[1] for x in exact_sol_viscous_shocktube.(x,T)]
l2 = lines!(x[:],rho[:],linestyle=:dash,linewidth=lw,color=:gray,label="Exact solution")


axislegend(labelsize=20,position = :rb)

save("dg1D-CNS-limited.png",f1)



f3 = Figure()
axis = Axis(f3[1,1])
lptl = 180
rptl = 240
lpt = 90
rpt = 120

N = 2
k = 400
il = false
l1 = lines!(x2[:,lptl:rptl][:],rho2[:,lptl:rptl][:],linestyle=nothing,linewidth=lw,color=:royalblue1,label="N=2,K=400")

N = 5
k = 200
il = false
l3 = lines!(x5[:,lpt:rpt][:],rho5[:,lpt:rpt][:],linestyle=:dot,linewidth=lw,color=:darkorange1,label="N=5,K=200")

rho = [x[1] for x in exact_sol_viscous_shocktube.(x,T)]
l2 = lines!(x[:,lpt:rpt][:],rho[:,lpt:rpt][:],linestyle=:dash,linewidth=lw,color=:gray,label="Exact solution")
axislegend(labelsize=20,position = :rb)

save("dg1D-CNS-limited-zoom.png",f3)

=#