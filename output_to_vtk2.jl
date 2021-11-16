push!(LOAD_PATH, "./src")
using DelimitedFiles

using WriteVTK

using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh
using UnPack

using SetupDG



N = 2
K1D = 20
# t=0.027652618923773267
# t=0.0008610198724528106
#t=0.004593014339084991
#t=0.009755907564170784
# t=0.021238249430540698
# t=0.01107131919723909
# t=0.016685099657334425
#t=0.038174565986393724
# t=0.09433024491945503
# t=0.09655400705355822
# t=0.19575285632508105
# t=0.07550828108140312
# t=0.07590867790313385
# t=0.024393258642572562
# t=0.025518477789014574
# t=0.024393255476498787
# t=0.025518475001162794
# t=0.10674597963531857
# t = 0.2
# t =0.02549578894035154 
# t=0.013226118481316584
# t=0.013037857428159683
t=0.014477691777086832
t=0.009145127272670283
t=0.013037781513546847
t=0.19060893512806573
t=0.2
t=0.06000064065592031
t=0.0783551163019341
t=0.09945653897728977
t=0.2
t=0.17009267106494133
t=0.0
flname = "dmr"
CFL = 0.9
XLENGTH = 2.0
Np = (N+1)*(N+1)

# Initial condition 2D shocktube
const γ = 1.4
const rhoL  = 8.0
const uL    = 8.25*cos(pi/6)
const vL    = -8.25*sin(pi/6)
const pL    = 116.5
const rhouL = rhoL*uL
const rhovL = rhoL*vL
const EL    = pL/(γ-1)+.5*rhoL*(uL^2+vL^2)
const betaL = rhoL/(2*pL)
const rhoR  = 1.4
const uR    = 0.0
const vR    = 0.0
const pR    = 1.0
const rhouR = rhoR*uR
const rhovR = rhoR*vR
const ER    = pR/(γ-1)+.5*rhoR*(uR^2+vR^2)
const betaR = rhoR/(2*pR)



"Mesh related variables"
# VX, VY, EToV = uniform_quad_mesh(Int(round(XLENGTH*K1D)),K1D)
# @. VX = (VX+1)/2*XLENGTH
# @. VY = (VY+1)/2
VX, VY, EToV = uniform_quad_mesh(Int(round(XLENGTH*K1D)),K1D)
# @. VX = (VX+1)/2
# @. VY = (VY+1)/4
@. VX = (VX+1)/2*XLENGTH*5
@. VY = (VY+1)/2*5


rd = init_reference_quad(N,gauss_lobatto_quad(0,0,N))
md = init_mesh((VX,VY),EToV,rd)
@unpack x,y = md;

@unpack VDM = rd
rp,sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/VDM

r_vtk = x[1:4]
x_vtk = []
y_vtk = []
for k = 1:Int(round(XLENGTH*K1D))
    push!(x_vtk,r_vtk[1]+(k-1)*5/K1D)
    push!(x_vtk,r_vtk[2]+(k-1)*5/K1D)
    push!(x_vtk,r_vtk[3]+(k-1)*5/K1D)
    # push!(x_vtk,r_vtk[4]+(k-1)*1/K1D)
end
for k = 1:K1D
    push!(y_vtk,r_vtk[1]+(k-1)*5/K1D)
    push!(y_vtk,r_vtk[2]+(k-1)*5/K1D)
    push!(y_vtk,r_vtk[3]+(k-1)*5/K1D)
    # push!(y_vtk,r_vtk[4]+(k-1)*1/K1D)
end

x_vtk = x_vtk .* ones((N+1)*K1D,1)'
y_vtk = y_vtk' .* ones((N+1)*Int(round(XLENGTH*K1D)),1)
# y_vtk = y_vtk' .* ones((N+1)*Int(round(2*K1D)),1)
rho_vtk,rhou_vtk,rhov_vtk,E_vtk,sigmax_vtk,sigmay_vtk = (zeros(size(x_vtk)) for i = 1:6)


# vtk_grid("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH",x,y) do vtk
#     vtk["rho"] = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,xt=0.014317166411179642=$XLENGTH,rho,dmr.txt")
#     vtk["rhou"] = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rhou,dmr.txt")
#     vtk["rhov"] = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rhov,dmr.txt")
#     vtk["E"] = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,E,dmr.txt")
# end

vtk_grid("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,$flname",x_vtk,y_vtk) do vtk
    rho = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rho,$flname.txt")
    rhou = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rhou,$flname.txt")
    rhov = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rhov,$flname.txt")
    E = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,E,$flname.txt")
    # rho = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,rho,dmr.txt")
    # rhou = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,rhou,dmr.txt")
    # rhov = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,rhov,dmr.txt")
    # E = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,E,dmr.txt")
    # sigmax = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,sigmax,dmr.txt")
    # sigmay = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,sigmay,dmr.txt")

    for k = 1:Int(round(XLENGTH*K1D))*K1D
        kx = mod1(k,Int(round(XLENGTH*K1D)))
        ky = div(k-1,Int(round(XLENGTH*K1D)))+1

        # if kx > 3.3*K1D
        #     rho[:,k] .= rhoR
        #     rhou[:,k] .= rhouR
        #     rhov[:,k] .= rhovR
        #     E[:,k] .= ER
        # end
    
        rho_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rho[:,k]
        rhou_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rhou[:,k]
        rhov_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rhov[:,k]
        E_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = E[:,k]
        # sigmax_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = sigmax[:,k]
        # sigmay_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = sigmay[:,k]
    end

    vtk["rho"]  = rho_vtk
    vtk["rhou"] = rhou_vtk
    vtk["rhov"] = rhov_vtk
    vtk["E"]    = E_vtk
    # vtk["sigmax"]    = sigmax_vtk
    # vtk["sigmay"]    = sigmay_vtk
end 

# t_arr = []
# for file in readdir()
#     if length(file) >= length("N=$N,K1D=$K1D,") && file[1:length("N=$N,K1D=$K1D,")] == "N=$N,K1D=$K1D," && file[end-3:end] == ".txt"
#         idx = findnext(",",file,length("N=$N,K1D=$K1D,t="))
#         t = parse(Float64,file[length("N=$N,K1D=$K1D,t=")+1:idx[1]-1])
#         if !(t in t_arr)
#             push!(t_arr,t)
#         end
#     end
# end

# pvd = paraview_collection("N=$N,K1D=$K1D,CFL=$CFL,x=$XLENGTH")

# for t in t_arr
#     vtk_grid("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH",Vp*x,Vp*y) do vtk
#         vtk["rho"] = Vp*readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rho,dmr.txt")
#         vtk["rhou"] = Vp*readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rhou,dmr.txt")
#         vtk["rhov"] = Vp*readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rhov,dmr.txt")
#         vtk["E"] = Vp*readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,E,dmr.txt")
#         pvd[t] = vtk
#     end
# end

# vtk_save(pvd)