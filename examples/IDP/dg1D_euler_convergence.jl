using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using DelimitedFiles


push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
using EntropyStableEuler
#using EntropyStableEuler.Fluxes1D



const POSTOL = 1e-14 
const TOL = 5e-16
const CFL = 0.5

# Ignacio - Leblanc shocktube 2
const γ = 5/3
const Bl = 0.0
const Br = 1.0
const rhoL = 1.0
const rhoR = 1e-3
const uL = 0.0
const uR = 0.0
const pL = (γ-1)*1e-1
const pR = (γ-1)*1e-10
const xC = 0.33
T = 2/3

# const GIFINTERVAL = Int(floor((T-t0)/dt/60))



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

function exact_sol_Leblanc(x,t)
    xi = (x-0.33)/t
    rhoLstar = 5.4079335349316249*1e-2
    rhoRstar = 3.9999980604299963*1e-3
    vstar    = 0.62183867139173454
    pstar    = 0.51557792765096996*1e-3
    lambda1  = 0.49578489518897934
    lambda3  = 0.82911836253346982
    if xi <= -1/3
        return rhoL, 0.0, pL
    elseif xi <= lambda1
        return (0.75-0.75*xi)^3, 0.75*(1/3+xi), 1/15*(0.75-0.75*xi)^5
    elseif xi <= vstar
        return rhoLstar, vstar, pstar
    elseif xi <= lambda3
        return rhoRstar, vstar, pstar
    else
        return rhoR, 0.0, pR
    end
end

function exact_sol_Leblanc(x,t)
    xi = (x-0.33)/t
    rhoLstar = 5.4079335349316249*1e-2
    rhoRstar = 3.9999980604299963*1e-3
    vstar    = 0.62183867139173454
    pstar    = 0.51557792765096996*1e-3
    lambda1  = 0.49578489518897934
    lambda3  = 0.82911836253346982
    if xi <= -1/3
        return rhoL, 0.0, pL
    elseif xi <= lambda1
        return (0.75-0.75*xi)^3, 0.75*(1/3+xi), 1/15*(0.75-0.75*xi)^5
    elseif xi <= vstar
        return rhoLstar, vstar, pstar
    elseif xi <= lambda3
        return rhoRstar, vstar, pstar
    else
        return rhoR, 0.0, pR
    end
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

function flux_lowIDP(U_i,U_j,f_i,f_j,c_ij,wavespd)
    return c_ij*(f_i+f_j)-abs(c_ij)*wavespd*(U_j-U_i)
end

function flux_high(f_i,f_j,c_ij)
    return c_ij*(f_i+f_j)
end

function flux_ES(rho_i,u_i,beta_i,rho_j,u_j,beta_j,c_ij)
    return 2*c_ij.*euler_fluxes(rho_i,u_i,beta_i,rho_j,u_j,beta_j)
end

function flux_central(c_ij,f_i,f_j)
    return c_ij*(f_i+f_j)
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

    J = (Br-Bl)/K/2 # assume uniform interval

    # Low order and high order algebraic fluxes
    F_low       = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_high      = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_low_P     = [zeros(2),zeros(2),zeros(2)] # 1: left boundary, 2: right boundary
    F_high_P    = [zeros(2),zeros(2),zeros(2)]
    L           = zeros(N+1,N+1) # Array of limiting params
    L_P         = zeros(K-1) # limiting params at left/right boundary
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
                    if in_s1
                        dii_arr[i,k] = dii_arr[i,k] + dij
                    end
                end
            end
        end

        # Interface dissipation
        EL = pL/(γ-1)+0.5*rhoL*uL^2
        ER = pR/(γ-1)+0.5*rhoR*uR^2
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
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

    for k = 1:K
        L = zeros(N+1,N+1)
        # Assemble matrix of low and high order algebraic fluxes
        # interior of the element
        for i = 1:N+1
            for j = 1:N+1
                if i != j # skip diagonal
                    wavespd = max(wavespd_arr[i,k],wavespd_arr[j,k])
                    fluxS = flux_ES(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],S[i,j])
                    for c = 1:3
                        F_low[c][i,j]  = flux_lowIDP(U[c][i,k],U[c][j,k],flux[c][i,k],flux[c][j,k],S0[i,j],wavespd)
                        F_high[c][i,j] = fluxS[c]
                    end
                end
            end
        end

        # Assemble matrix of low and high order algebraic fluxes
        # interface of the element
        EL = pL/(γ-1)+0.5*rhoL*uL^2
        ER = pR/(γ-1)+0.5*rhoR*uR^2
        U_left   = (k == 1) ? [rhoL; rhoL*uL; EL]                 : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; uL; rhoL/(2*pL)]             : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [rhoL*uL; rhoL*uL^2+pL; uL*(pL+EL)] : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; rhoR*uR; ER]                 : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; uR; rhoR/(2*pR)]             : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [rhoR*uR; rhoR*uR^2+pR; uR*(pR+ER)] : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        for c = 1:3
            F_low_P[c][1] = flux_lowIDP(U[c][1,k],U_left[c],flux[c][1,k],f_left[c],-0.5,wavespd_l)
            F_low_P[c][2] = flux_lowIDP(U[c][end,k],U_right[c],flux[c][end,k],f_right[c],0.5,wavespd_r)

            F_high_P[c][1] = F_low_P[c][1]
            F_high_P[c][2] = F_low_P[c][2]
        end

        # Calculate limiting parameters over interior of the element
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

        for i = 1:N+1
            #lambda_j = (i >= 2 && i <= N) ? 1/N : 1/(N+1)
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

        # # li limiting
        # for i = 1:N+1
        #     #lambda_j = (i >= 2 && i <= N) ? 1/N : 1/(N+1)
        #     lambda_j = 1/N
        #     m_i = J*wq[i]
        #     for j = 1:N+1
        #         if i != j
        #             for c = 1:3
        #                 P_ij[c] = dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
        #             end
        #             L[i,j] = limiting_param([U_low[1][i]; U_low[2][i]; U_low[3][i]],P_ij)
        #         end
        #     end
        # end

        # li_min = 1.0
        # for i = 1:N+1
        #     lambda_j = 1/N
        #     m_i = J*wq[i]
        #     for c = 1:3
        #         P_ij[c] = 0.0
        #     end
        #     for j = 1:N+1
        #         for c = 1:3
        #             P_ij[c] += dt/(m_i*lambda_j)*(F_low[c][i,j]-F_high[c][i,j])
        #         end
        #     end
        #     l = limiting_param([U_low[1][i]; U_low[2][i]; U_low[3][i]],P_ij)
        #     li_min = min(li_min,l)
        # end

        # for i = 1:N+1
        #     for j = 1:N+1
        #         if i != j
        #             L[i,j] = li_min
        #             L[j,i] = li_min
        #         end
        #     end
        # end

        # construct rhs
        for c = 1:3
            # With limiting
            rhsU[c][:,k] = sum((L.-1).*F_low[c] - L.*F_high[c],dims=2)

            if k > 1
                rhsU[c][1,k] += (L_P[k-1]-1)*F_low_P[c][1] - L_P[k-1]*F_high_P[c][1]
            else
                rhsU[c][1,k] += -F_high_P[c][1]
            end
            if k < K
                rhsU[c][N+1,k] += (L_P[k]-1)*F_low_P[c][2] - L_P[k]*F_high_P[c][2]
            else
                rhsU[c][N+1,k] += -F_high_P[c][2]
            end

            rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
        end
    end

    return rhsU,dt
end

function rhs_IDPlow(U,K,N,Mlump_inv,wq)
    p = pfun_nd.(U[1],U[2],U[3])
    flux = zero.(U)
    @. flux[1] = U[2]
    @. flux[2] = U[2]^2/U[1]+p
    @. flux[3] = U[3]*U[2]/U[1]+p*U[2]/U[1]

    J = (Br-Bl)/K/2

    dt = Inf
    d_ii_arr = zeros(N+1,K)
    dfdx = (zeros(N+1,K),zeros(N+1,K),zeros(N+1,K))
    for i = 2:K*(N+1)-1
        for c = 1:3
            dfdx[c][i] = 1/2*(flux[c][mod1(i+1,K*(N+1))] - flux[c][mod1(i-1,K*(N+1))])
        end
    end
    dfdx[1][1] = 1/2*(flux[1][2] - 0.0)
    dfdx[2][1] = 1/2*(flux[2][2] - pL)
    dfdx[3][1] = 1/2*(flux[3][2] - 0.0)
    dfdx[1][end] = 1/2*(0.0 - flux[1][end-1])
    dfdx[2][end] = 1/2*(pR - flux[2][end-1])
    dfdx[3][end] = 1/2*(0.0 - flux[3][end-1])

    visc = (zeros(N+1,K),zeros(N+1,K),zeros(N+1,K))
    for i = 2:K*(N+1)-1
        wavespd_curr = wavespeed_1D(U[1][i],U[2][i],U[3][i])
        wavespd_R = wavespeed_1D(U[1][mod1(i+1,K*(N+1))],U[2][mod1(i+1,K*(N+1))],U[3][mod1(i+1,K*(N+1))])
        wavespd_L = wavespeed_1D(U[1][mod1(i-1,K*(N+1))],U[2][mod1(i-1,K*(N+1))],U[3][mod1(i-1,K*(N+1))])
        dL = 1/2*max(wavespd_curr,wavespd_L)
        dR = 1/2*max(wavespd_curr,wavespd_R)
        for c = 1:3
            visc[c][i] = dL*(U[c][mod1(i-1,K*(N+1))]-U[c][i]) + dR*(U[c][mod1(i+1,K*(N+1))]-U[c][i])
        end
        d_ii_arr[i] += .5*max(wavespd_curr,wavespd_L) + .5*max(wavespd_curr,wavespd_R)
    end

    # i = 1
    wavespd_curr = wavespeed_1D(U[1][1],U[2][1],U[3][1])
    wavespd_R = wavespeed_1D(U[1][2],U[2][2],U[3][2])
    wavespd_L = wavespeed_1D(rhoL,0.0,pL/(γ-1))
    dL = 1/2*max(wavespd_curr,wavespd_L)
    dR = 1/2*max(wavespd_curr,wavespd_R)
    visc[1][1] = dL*(rhoL-U[1][1]) + dR*(U[1][2]-U[1][1])
    visc[2][1] = dL*(0.0-U[2][1]) + dR*(U[2][2]-U[2][1])
    visc[3][1] = dL*(pL/(γ-1)-U[3][1]) + dR*(U[3][2]-U[3][1])

    d_ii_arr[1] += (dL+dR)

    # i = end
    wavespd_curr = wavespeed_1D(U[1][end],U[2][end],U[3][end])
    wavespd_R = wavespeed_1D(rhoR,0.0,pR/(γ-1))
    wavespd_L = wavespeed_1D(U[1][end-1],U[2][end-1],U[3][end-1])
    dL = 1/2*max(wavespd_curr,wavespd_L)
    dR = 1/2*max(wavespd_curr,wavespd_R)
    visc[1][end] = dL*(U[1][end-1]-U[1][end]) + dR*(rhoR-U[1][end])
    visc[2][end] = dL*(U[2][end-1]-U[2][end]) + dR*(0.0-U[2][end])
    visc[3][end] = dL*(U[3][end-1]-U[3][end]) + dR*(pR/(γ-1)-U[3][end])

    d_ii_arr[end] += (dL+dR)
    dt = minimum(J*wq./d_ii_arr./2.0)

    rhsU = (x->1/J*Mlump_inv*x).(.-dfdx.+visc)
    return rhsU,dt
end

function rhs_high(U,K,N,Mlump_inv,S)
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
    F_high      = [zeros(N+1,N+1),zeros(N+1,N+1),zeros(N+1,N+1)]
    F_high_P    = [zeros(2),zeros(2),zeros(2)]
    rhsU        = [zeros(N+1,K),zeros(N+1,K),zeros(N+1,K)]

    # Determine dt
    wavespd_arr = zeros(N+1,K)
    for k = 1:K
        for i = 1:N+1
            wavespd_arr[i,k] = wavespeed_1D(U[1][i,k],U[2][i,k],U[3][i,k])
        end
    end

    for k = 1:K
        # Assemble matrix of low and high order algebraic fluxes
        # interior of the element
        for i = 1:N+1
            for j = 1:N+1
                if i != j # skip diagonal
                    # wavespd = max(wavespd_arr[i,k],wavespd_arr[j,k])
                    # for c = 1:3
                    #     F_high[c][i,j] = flux_ES(Ub[c][i,k],Ub[c][j,k],S[i,j])
                    # end
                    tmp = flux_ES(Ub[1][i,k],Ub[2][i,k],Ub[3][i,k],Ub[1][j,k],Ub[2][j,k],Ub[3][j,k],S[i,j])
                    for c = 1:3
                        F_high[c][i,j] = tmp[c]
                    end
                end
            end
        end

        # Assemble matrix of low and high order algebraic fluxes
        # interface of the element
        U_left   = (k == 1) ? [rhoL; 0.0; pL/(γ-1)]    : [U[1][end,k-1]; U[2][end,k-1]; U[3][end,k-1]]
        Ub_left  = (k == 1) ? [rhoL; 0.0; rhoL/(2*pL)] : [Ub[1][end,k-1]; Ub[2][end,k-1]; Ub[3][end,k-1]]
        f_left   = (k == 1) ? [0.0; pL; 0.0]           : [flux[1][end,k-1]; flux[2][end,k-1]; flux[3][end,k-1]]
        U_right  = (k == K) ? [rhoR; 0.0; pR/(γ-1)]    : [U[1][1,k+1]; U[2][1,k+1]; U[3][1,k+1]]
        Ub_right = (k == K) ? [rhoR; 0.0; rhoR/(2*pR)] : [Ub[1][1,k+1]; Ub[2][1,k+1]; Ub[3][1,k+1]]
        f_right  = (k == K) ? [0.0; pR; 0.0]           : [flux[1][1,k+1]; flux[2][1,k+1]; flux[3][1,k+1]]
        wavespd_l = max(wavespd_arr[1,k],wavespeed_1D(U_left[1],U_left[2],U_left[3]))
        wavespd_r = max(wavespd_arr[end,k],wavespeed_1D(U_right[1],U_right[2],U_right[3]))

        tmp = .-euler_fluxes(Ub[1][1,k],Ub[2][1,k],Ub[3][1,k],Ub_left[1],Ub_left[2],Ub_left[3])
        tmp2 = euler_fluxes(Ub[1][end,k],Ub[2][end,k],Ub[3][end,k],Ub_right[1],Ub_right[2],Ub_right[3])
        for c = 1:3
            # F_high_P[c][1] = -1/2*(flux[c][1,k]+f_left[c])-wavespd_l/2*(U_left[c]-U[c][1,k])#tmp[c]-wavespd_l/2*(U_left[c]-U[c][1,k])#-euler_fluxes(Ub[c][1,k],Ub_left[c])#flux_high(Ub[c][1,k],Ub_left[c],-0.5)-wavespd_l/2*(Ub_left[c]-Ub[c][1,k])
            # F_high_P[c][2] = 1/2*(flux[c][end,k]+f_right[c])-wavespd_r/2*(U_right[c]-U[c][end,k])#tmp2[c]-wavespd_r/2*(U_right[c]-U[c][end,k])#euler_fluxes(Ub[c][end,k],Ub_right[c])#flux_high(Ub[c][end,k],Ub_right[c],0.5)-wavespd_r/2*(Ub_right[c]-Ub[c][end,k])
            F_high_P[c][1] = tmp[c]-wavespd_l/2*(U_left[c]-U[c][1,k])#-euler_fluxes(Ub[c][1,k],Ub_left[c])#flux_high(Ub[c][1,k],Ub_left[c],-0.5)-wavespd_l/2*(Ub_left[c]-Ub[c][1,k])
            F_high_P[c][2] = tmp2[c]-wavespd_r/2*(U_right[c]-U[c][end,k])#euler_fluxes(Ub[c][end,k],Ub_right[c])#flux_high(Ub[c][end,k],Ub_right[c],0.5)-wavespd_r/2*(Ub_right[c]-Ub[c][end,k])
        end

        # construct rhs
        for c = 1:3
            # With limiting
            rhsU[c][:,k] = -sum(F_high[c],dims=2)
            rhsU[c][1,k] += -F_high_P[c][1]
            rhsU[c][N+1,k] += -F_high_P[c][2]
            rhsU[c][:,k] .= 1/J*Mlump_inv*rhsU[c][:,k]
        end

    end

    return rhsU
end


exact_sol = exact_sol_Leblanc

Narr = [2;5]
Karr = [50;100;200;400;800]
low_order_arr = [false]#[true;false]

# Start N, K loop
for N in Narr
for K in Karr
for is_low_order in low_order_arr

"Approximation parameters"
# N = 2 # The order of approximation
# K = 800
t0 = 0.01
# dt = 0.0003

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

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1.0

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1.0

"""Initial condition"""
rho_x(x) = (x <= xC) ? rhoL : rhoR
u_x(x) = 0.0
p_x(x) = (x <= xC) ? pL : pR

U_init = @. exact_sol(x,t0)
rho_init = [x[1] for x in U_init]
u_init = [x[2] for x in U_init]
p_init = [x[3] for x in U_init]
U = primitive_to_conservative_hardcode.(rho_init,u_init,p_init)
rho = [x[1] for x in U]
rhou = [x[2] for x in U]
E = [x[3] for x in U]
U = (rho,rhou,E)


# Time stepping
"Time integration"
t = t0
U = collect(U)


Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM
# Nsteps = Int(ceil((T-t0)/dt))
resW = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
resZ = [zeros(size(x)),zeros(size(x)),zeros(size(x))]
#for i = 1:Nsteps
#@gif for i = 1:Nsteps
while t < T

    # if abs(T-t) < dt
    #     global dt = T - t
    # end

    dt = Inf
    # SSPRK(3,3)
    rhsU,dt = rhs_IDP(U,K,N,wq,S,S0,Mlump_inv,T,dt,true,is_low_order)
    #rhsU = rhs_high(U,K,N,Mlump_inv,S)
    # rhsU,dt = rhs_IDPlow(U,K,N,Mlump_inv,wq)
    @. resW = U + dt*rhsU
    rhsU,_ = rhs_IDP(resW,K,N,wq,S,S0,Mlump_inv,T,dt,false,is_low_order)
    #rhsU = rhs_high(resW,K,N,Mlump_inv,S)
    # rhsU,_ = rhs_IDPlow(resW,K,N,Mlump_inv,wq)
    @. resZ = resW+dt*rhsU
    @. resW = 3/4*U+1/4*resZ
    rhsU,_ = rhs_IDP(resW,K,N,wq,S,S0,Mlump_inv,T,dt,false,is_low_order)
    #rhsU = rhs_high(resW,K,N,Mlump_inv,S)
    # rhsU,_ = rhs_IDPlow(resW,K,N,Mlump_inv,wq)
    @. resZ = resW+dt*rhsU
    @. U = 1/3*U+2/3*resZ

    t = t + dt
    println("Current time $t with time step size $dt, and final time $T")
    if t == T
        break
    end

    # if i % GIFINTERVAL == 0
    #     #plot(Vp*x,Vp*U[1])
    #     plot(Vp*x,Vp*U[1])
    #     # ptL = Bl+(Br-Bl)/K/(N+1)/2
    #     # ptR = Br-(Br-Bl)/K/(N+1)/2
    #     # hplot = (Br-Bl)/K/(N+1)
    #     # for k = 1:K
    #     #     plot!(ptL+(k-1)*hplot*(N+1):hplot:ptL+k*hplot*(N+1), 1 .-L_plot2[:,k],st=:bar,alpha=0.2)
    #     # end
    # end
end
#end every GIFINTERVAL
#end





@inline function Efun(rho,u,p)
    return p/(γ-1) + .5*rho*(u^2)
end

exact_U = @. exact_sol(x,T)
exact_rho = [x[1] for x in exact_U]
exact_u = [x[2] for x in exact_U]
exact_p = [x[3] for x in exact_U]
exact_rhou = exact_rho .* exact_u
exact_E = Efun.(exact_rho,exact_u,exact_p)

rho = U[1]
u = U[2]./U[1]
p = pfun_nd.(U[1],U[2],U[3])
rhou = U[2]
E = U[3]
J = (Br-Bl)/K/2

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

open("/data/yl184/1D-euler-conv/N=$N,K=$K,CFL=$CFL,LOW=$is_low_order,x,1D-euler-conv.txt","w") do io
    writedlm(io,x)
end
open("/data/yl184/1D-euler-conv/N=$N,K=$K,CFL=$CFL,LOW=$is_low_order,rho,1D-euler-conv.txt","w") do io
    writedlm(io,U[1])
end

end
end
end # End N, K loop

# println("Linf error is $Linferr")

# open("convergence.txt", "a") do io
#     write(io, "N = $N, K = $K\n")
#     write(io, "L1 error is $L1err\n")
#     write(io, "Linf error is $Linferr\n")
# end

# plot(Vp*x,Vp*U[1])
# savefig("~/Desktop/N=$N,K=$K,dt=$dt,modifiedESDG.png")

