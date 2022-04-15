push!(LOAD_PATH, "./src")
using DelimitedFiles

using WriteVTK
using FileIO
using DataFrames
using JLD2

using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh
using UnPack
using Polyester
using SparseArrays

using SetupDG

N = 2
K1D = 40
CFL = 0.9
Np = (N+1)*(N+1)


"Mesh related variables"
VX, VY, EToV = uniform_quad_mesh(2*K1D,K1D)
# @. VX = sin(.5*pi*(VX+1)/2)
# @. VY = ((VY+1)/2)^2/2
@. VX = (VX+1)/2
@. VY = (VY+1)/4
# @. VX = ((VX+1)/2)^(.25)
# @. VY = ((VY+1)/2)^(2)/2

rd = init_reference_quad(N,gauss_lobatto_quad(0,0,N))
md = init_mesh((VX,VY),EToV,rd)
@unpack x,y,mapP,rxJ,syJ,J,sJ,nxJ,nyJ = md;

@unpack VDM,Vf,LIFT,M,Dr,Ds = rd
droptol!(sparse(M),1e-16)
rp,sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/VDM

x_vtk = []
y_vtk = []
for k = 1:Int(round(2*K1D))
    push!(x_vtk,x[1,k])
    push!(x_vtk,x[2,k])
    push!(x_vtk,x[3,k])
    # push!(x_vtk,x[4,k])
    # push!(x_vtk,x[5,k])
end
for k = 1:K1D
    # push!(y_vtk,y[1,1+(k-1)*2*K1D])
    # push!(y_vtk,y[5,1+(k-1)*2*K1D])
    # push!(y_vtk,y[9,1+(k-1)*2*K1D])
    # push!(y_vtk,y[13,1+(k-1)*2*K1D])
    # push!(y_vtk,y[1,1+(k-1)*2*K1D])
    # push!(y_vtk,y[6,1+(k-1)*2*K1D])
    # push!(y_vtk,y[11,1+(k-1)*2*K1D])
    # push!(y_vtk,y[16,1+(k-1)*2*K1D])
    # push!(y_vtk,y[21,1+(k-1)*2*K1D])
    push!(y_vtk,y[1,1+(k-1)*2*K1D])
    push!(y_vtk,y[4,1+(k-1)*2*K1D])
    push!(y_vtk,y[7,1+(k-1)*2*K1D])
end
x_vtk = x_vtk .* ones((N+1)*K1D,1)'
y_vtk = y_vtk' .* ones((N+1)*Int(round(2*K1D)),1)

rho_vtk,rhou_vtk,rhov_vtk,E_vtk,gradg_vtk = (zeros(size(x_vtk)) for i = 1:5)

df = load("dg2D_CNS_quad_shocktube.jld2","data")
Uhist = df[end,:Uhist]
thist = df[end,:thist]
# TODO: add savethist to df

pvd = paraview_collection("data/shocktube/N=$N,K1D=$K1D,CFL=$CFL,shocktube", append=true)

for i in 1:length(Uhist)
    t = thist[100*(i-1)+1]
    println("Current time $t")
    flush(stdout)
    vtk_grid("data/shocktube/N=$N,K1D=$K1D,t=$t,CFL=$CFL",x_vtk,y_vtk) do vtk
        rho  = Uhist[i][1,:,:]
        rhou = Uhist[i][2,:,:]
        rhov = Uhist[i][3,:,:]
        E    = Uhist[i][4,:,:]

        # TODO: hardcode rx,ry, put in dataframe
        u     = rhou./rho
        v     = rhov./rho
        vmag  = @. u^2 + v^2
        g     = sqrt.(((rxJ./J).*(Dr*rho)).^2 .+ ((syJ./J).*(Ds*rho)).^2)
        g_min = minimum(g)
        g_max = maximum(g)
        vv    = @. exp(-10*(g-g_min)/(g_max-g_min))

        for k = 1:Int(round(2*K1D))*K1D
            kx = mod1(k,Int(round(2*K1D)))
            ky = div(k-1,Int(round(2*K1D)))+1
        
            rho_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rho[:,k]
            rhou_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rhou[:,k]
            rhov_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rhov[:,k]
            E_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = E[:,k]
            gradg_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = vv[:,k]
        end


        vtk["rho"]   = rho_vtk
        vtk["rhou"]  = rhou_vtk
        vtk["rhov"]  = rhov_vtk
        vtk["E"]     = E_vtk
        vtk["gradg"] = gradg_vtk
        pvd[t] = vtk
    end
end

vtk_save(pvd)