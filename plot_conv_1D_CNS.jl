using DelimitedFiles
using CairoMakie

f1 = Figure()
axis = Axis(f1[1,1])

lw = 3
is_low = [true;false]

N = 2
k = 400
il = false
x   = readdlm("1D-CNS-conv/N=$N,K=$k,CFL=0.5,LOW=$il,issmooth=false,x,1D-CNS-conv.txt")
rho = readdlm("1D-CNS-conv/N=$N,K=$k,CFL=0.5,LOW=$il,issmooth=false,rho,1D-CNS-conv.txt")
l1 = lines!(x[:],rho[:],linestyle=nothing,linewidth=lw,color=:royalblue1,label="N=2,K=400")

N = 3
k = 300
il = false
x   = readdlm("1D-CNS-conv/N=$N,K=$k,CFL=0.5,LOW=$il,issmooth=false,x,1D-CNS-conv.txt")
rho = readdlm("1D-CNS-conv/N=$N,K=$k,CFL=0.5,LOW=$il,issmooth=false,rho,1D-CNS-conv.txt")
l3 = lines!(x[:],rho[:],linestyle=:dot,linewidth=lw,color=:darkorange1,label="N=3,K=300")

# Becker viscous shocktube
T = 1.0
const γ = 1.4
const M_0 = 20.0   # Sharp
const mu = 0.001
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

rho = [x[1] for x in exact_sol_viscous_shocktube.(x,T)]
l2 = lines!(x[:],rho[:],linestyle=:dash,linewidth=lw,color=:gray,label="Exact solution")


axislegend(labelsize=20,position=:lt)

save("dg1D-CNS-limited.png",f1)

f2 = Figure()
axis = Axis(f2[1,1])

lw = 3
is_low = [true;false]

N = 2
k = 400
il = true
x   = readdlm("1D-CNS-conv/N=$N,K=$k,CFL=0.5,LOW=$il,issmooth=false,x,1D-CNS-conv.txt")
rho = readdlm("1D-CNS-conv/N=$N,K=$k,CFL=0.5,LOW=$il,issmooth=false,rho,1D-CNS-conv.txt")
l1 = lines!(x[:],rho[:],linestyle=nothing,linewidth=lw,color=:royalblue1,label="N=2,K=400")


N = 3
k = 300
il = true
x   = readdlm("1D-CNS-conv/N=$N,K=$k,CFL=0.5,LOW=$il,issmooth=false,x,1D-CNS-conv.txt")
rho = readdlm("1D-CNS-conv/N=$N,K=$k,CFL=0.5,LOW=$il,issmooth=false,rho,1D-CNS-conv.txt")
l3 = lines!(x[:],rho[:],linestyle=:dot,linewidth=lw,color=:darkorange1,label="N=3,K=300")


T = 1.0
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
rho = [x[1] for x in exact_sol_viscous_shocktube.(x,T)]
l2 = lines!(x[:],rho[:],linestyle=:dash,linewidth=lw,color=:gray,label="Exact solution")



axislegend(labelsize=20,position=:lt)

save("dg1D-CNS-low.png",f2)

