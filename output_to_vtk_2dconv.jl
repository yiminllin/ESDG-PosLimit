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
t=2.0
CFL=0.9
flname = "euler2Dconv"
Np = (N+1)*(N+1)

"Mesh related variables"
VX, VY, EToV = uniform_quad_mesh(Int(round(3/2*K1D)),K1D)
@. VX = VX*7.5+2.5
@. VY = VY*5

rd = init_reference_quad(N,gauss_lobatto_quad(0,0,N))
md = init_mesh((VX,VY),EToV,rd)
@unpack x,y = md;

@unpack VDM = rd
rp,sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/VDM

x_vtk = []
y_vtk = []
for k = 1:Int(round(3/2*K1D))
    push!(x_vtk,x[1,k])
    push!(x_vtk,x[2,k])
    push!(x_vtk,x[3,k])
end
for k = 1:K1D
    push!(y_vtk,y[1,1+Int(round((k-1)*3/2*K1D))])
    push!(y_vtk,y[4,1+Int(round((k-1)*3/2*K1D))])
    push!(y_vtk,y[7,1+Int(round((k-1)*3/2*K1D))])
end
x_vtk = x_vtk .* ones((N+1)*K1D,1)'
y_vtk = y_vtk' .* ones((N+1)*Int(round(3/2*K1D)),1)
rho_vtk,rhou_vtk,rhov_vtk,E_vtk,sigmax_vtk,sigmay_vtk = (zeros(size(x_vtk)) for i = 1:6)


# vtk_grid("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH",x,y) do vtk
#     vtk["rho"] = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,xt=0.014317166411179642=$XLENGTH,rho,dmr.txt")
#     vtk["rhou"] = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rhou,dmr.txt")
#     vtk["rhov"] = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,rhov,dmr.txt")
#     vtk["E"] = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,x=$XLENGTH,E,dmr.txt")
# end

vtk_grid("N=$N,K1D=$K1D,t=$t,CFL=$CFL,$flname",x_vtk,y_vtk) do vtk
    rho = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,rho,$flname.txt")
    rhou = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,rhou,$flname.txt")
    rhov = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,rhov,$flname.txt")
    E = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,E,$flname.txt")

    for k = 1:Int(round(3/2*K1D))*K1D
        kx = mod1(k,Int(round(3/2*K1D)))
        ky = div(k-1,Int(round(3/2*K1D)))+1
   
        rho_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rho[:,k]
        rhou_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rhou[:,k]
        rhov_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = rhov[:,k]
        E_vtk[(kx-1)*(N+1)+1:(kx-1)*(N+1)+N+1, (ky-1)*(N+1)+1:(ky-1)*(N+1)+N+1] = E[:,k]
    end

    vtk["rho"]  = rho_vtk
    vtk["rhou"] = rhou_vtk
    vtk["rhov"] = rhov_vtk
    vtk["E"]    = E_vtk
end 