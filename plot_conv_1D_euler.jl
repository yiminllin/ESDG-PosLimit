using DelimitedFiles
using CairoMakie

f1 = Figure()
axis = Axis(f1[1,1])

lw = 3
is_low = [true;false]

N = 2
k = 200
il = false
x   = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,x,1D-euler-conv.txt")
rho = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,rho,1D-euler-conv.txt")
l1 = lines!(x[:],rho[:],linestyle=nothing,linewidth=lw,color=:royalblue1,label="N=2,K=200")

N = 5
k = 100
il = false
x   = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,x,1D-euler-conv.txt")
rho = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,rho,1D-euler-conv.txt")
l3 = lines!(x[:],rho[:],linestyle=:dot,linewidth=lw,color=:darkorange1,label="N=5,K=100")


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
rho = [x[1] for x in exact_sol_Leblanc.(x,T)]
l2 = lines!(x[:],rho[:],linestyle=:dash,linewidth=lw,color=:gray,label="Exact solution")


axislegend(labelsize=20)

save("dg1D-euler-limited.png",f2)



f3 = Figure()
axis = Axis(f3[1,1])
lptl = 140
rptl = 180
lpt = 70
rpt = 90

N = 2
k = 200
il = false
x   = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,x,1D-euler-conv.txt")
rho = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,rho,1D-euler-conv.txt")
l1 = lines!(x[:,lptl:rptl][:],rho[:,lptl:rptl][:],linestyle=nothing,linewidth=lw,color=:royalblue1,label="N=2,K=200")

N = 5
k = 100
il = false
x   = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,x,1D-euler-conv.txt")
rho = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,rho,1D-euler-conv.txt")
l3 = lines!(x[:,lpt:rpt][:],rho[:,lpt:rpt][:],linestyle=:dot,linewidth=lw,color=:darkorange1,label="N=5,K=100")

rho = [x[1] for x in exact_sol_Leblanc.(x,T)]
l2 = lines!(x[:,lpt:rpt][:],rho[:,lpt:rpt][:],linestyle=:dash,linewidth=lw,color=:gray,label="Exact solution")
axislegend(labelsize=20)

save("dg1D-euler-limited-zoom.png",f3)





f2 = Figure()
axis = Axis(f2[1,1])

lw = 3
is_low = [true;false]

N = 2
k = 200
il = true
x   = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,x,1D-euler-conv.txt")
rho = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,rho,1D-euler-conv.txt")
l1 = lines!(x[:],rho[:],linestyle=nothing,linewidth=lw,color=:royalblue1,label="N=2,K=200")


N = 5
k = 100
il = true
x   = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,x,1D-euler-conv.txt")
rho = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,rho,1D-euler-conv.txt")
l3 = lines!(x[:],rho[:],linestyle=:dot,linewidth=lw,color=:darkorange1,label="N=5,K=100")


T = 2/3
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
rho = [x[1] for x in exact_sol_Leblanc.(x,T)]
l2 = lines!(x[:],rho[:],linestyle=:dash,linewidth=lw,color=:gray,label="Exact solution")



axislegend(labelsize=20)

save("dg1D-euler-low.png",f2)





f3 = Figure()
axis = Axis(f3[1,1])
lptl = 140
rptl = 180
lpt = 70
rpt = 90

N = 2
k = 200
il = true 
x   = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,x,1D-euler-conv.txt")
rho = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,rho,1D-euler-conv.txt")
l1 = lines!(x[:,lptl:rptl][:],rho[:,lptl:rptl][:],linestyle=nothing,linewidth=lw,color=:royalblue1,label="N=2,K=200")

N = 5
k = 100
il = true
x   = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,x,1D-euler-conv.txt")
rho = readdlm("1D-euler-conv/N=$N,K=$k,CFL=0.5,LOW=$il,rho,1D-euler-conv.txt")
l3 = lines!(x[:,lpt:rpt][:],rho[:,lpt:rpt][:],linestyle=:dot,linewidth=lw,color=:darkorange1,label="N=5,K=100")

rho = [x[1] for x in exact_sol_Leblanc.(x,T)]
l2 = lines!(x[:,lpt:rpt][:],rho[:,lpt:rpt][:],linestyle=:dash,linewidth=lw,color=:gray,label="Exact solution")
axislegend(labelsize=20)

save("dg1D-euler-low-zoom.png",f3)


