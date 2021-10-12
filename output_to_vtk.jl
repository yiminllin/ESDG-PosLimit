push!(LOAD_PATH, "./src")
using DelimitedFiles

using WriteVTK

using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh
using UnPack

using SetupDG


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


N = 3
K1D = 250
# t=0.027652618923773267
# t=0.0008610198724528106
#t=0.004593014339084991
#t=0.009755907564170784
# t=0.021238249430540698
# t=0.01107131919723909
# t=0.016685099657334425
#t=0.038174565986393724
t=0.09433024491945503
CFL = 0.75
XLENGTH = 7/2
Np = (N+1)*(N+1)


"Mesh related variables"
VX, VY, EToV = uniform_quad_mesh(Int(round(XLENGTH*K1D)),K1D)
@. VX = (VX+1)/2*XLENGTH
@. VY = (VY+1)/2

rd = init_reference_quad(N,gauss_lobatto_quad(0,0,N))
md = init_mesh((VX,VY),EToV,rd)
@unpack x,y = md;

@unpack VDM = rd
rp,sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/VDM

r_vtk = x[1:4]
x_vtk = []
y_vtk = []
for k = 1:Int(round(3.5*K1D))
    push!(x_vtk,r_vtk[1]+(k-1)*1/K1D)
    push!(x_vtk,r_vtk[2]+(k-1)*1/K1D)
    push!(x_vtk,r_vtk[3]+(k-1)*1/K1D)
    push!(x_vtk,r_vtk[4]+(k-1)*1/K1D)
end
for k = 1:K1D
    push!(y_vtk,r_vtk[1]+(k-1)*1/K1D)
    push!(y_vtk,r_vtk[2]+(k-1)*1/K1D)
    push!(y_vtk,r_vtk[3]+(k-1)*1/K1D)
    push!(y_vtk,r_vtk[4]+(k-1)*1/K1D)
end

x_vtk = x_vtk .* ones((N+1)*K1D,1)'
y_vtk = y_vtk' .* ones((N+1)*Int(round(3.5*K1D)),1)
rho_vtk,rhou_vtk,rhov_vtk,E_vtk = (zeros(size(x_vtk)) for i = 1:4)

tarr = []
filenames = readdir("/data/yl184")
for fn in filenames
    for str in split(fn,",")
        if length(str) < 2
            continue
        end
        if "t=" == str[1:2]
            append!(tarr,parse(Float64,str[3:end]))
        end
    end
end
unique!(tarr)

pvd = paraview_collection("/data/yl184/N=$N,K1D=$K1D,CFL=$CFL,x=$XLENGTH,element-wise", append=true)

for t in tarr
    if t > 0.0
        println("Current time $t")
        flush(stdout)
        vtk_grid("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH",x_vtk,y_vtk) do vtk
            rho = readdlm("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rho,dmr.txt")
            rhou = readdlm("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rhou,dmr.txt")
            rhov = readdlm("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rhov,dmr.txt")
            E = readdlm("/data/yl184/N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,E,dmr.txt")

            for k = 1:Int(round(3.5*K1D))*K1D
                kx = mod1(k,Int(round(3.5*K1D)))
                ky = div(k-1,Int(round(3.5*K1D)))+1
                rho_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rho[:,k]
                rhou_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rhou[:,k]
                rhov_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rhov[:,k]
                E_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = E[:,k]
            end

            vtk["rho"]  = rho_vtk
            vtk["rhou"] = rhou_vtk
            vtk["rhov"] = rhov_vtk
            vtk["E"]    = E_vtk
            pvd[t] = vtk
        end 
    end
end

vtk_save(pvd)