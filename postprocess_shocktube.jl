using Pkg
Pkg.activate("Project.toml")
using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using StaticArrays
using DelimitedFiles
using Polyester
using MuladdMacro
using DataFrames
using JLD2
using FileIO
using WriteVTK

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh

using SetupDG

function construct_vtk_mesh!(x_vtk,y_vtk,K,K1D,N,x,y)
    # TODO: assume sturctured Mesh
    for k = 1:K
        iK = mod1(k,2*K1D)
        jK = div(k-1,2*K1D)+1
        xk = reshape(x[:,k],N+1,N+1)
        yk = reshape(y[:,k],N+1,N+1)
        x_vtk[(iK-1)*(N+1)+1:iK*(N+1),(jK-1)*(N+1)+1:jK*(N+1)] = xk
        y_vtk[(iK-1)*(N+1)+1:iK*(N+1),(jK-1)*(N+1)+1:jK*(N+1)] = yk
    end
end

function construct_vtk_file!(U,sch,Uvtk,schlvtk,pvdfile,xvtk,yvtk,t,PATH,K,K1D,N,x,y)
    vtk_grid("$(PATH)/dg2D_CNS_quad_shocktube_t=$t",xvtk,yvtk) do vtk
        for k = 1:K
            iK = mod1(k,2*K1D)
            jK = div(k-1,2*K1D)+1
            for c = 1:Nc
                Uvtk[c,(iK-1)*(N+1)+1:iK*(N+1),(jK-1)*(N+1)+1:jK*(N+1)] = U[c,:,k]
            end
            schlvtk[(iK-1)*(N+1)+1:iK*(N+1),(jK-1)*(N+1)+1:jK*(N+1)] = sch[:,k]
        end    
        vtk["rho"]  = Uvtk[1,:,:] 
        vtk["rhou"] = Uvtk[2,:,:] 
        vtk["rhov"] = Uvtk[3,:,:] 
        vtk["E"]    = Uvtk[4,:,:] 
        vtk["schl"] = schlvtk

        pvdfile[t]  = vtk
    end
end


const Nc  = 4
# const OUTPUTPATH = "/home/yiminlin/Desktop/dg2D_CNS_quad_shocktube_output"
const OUTPUTPATH = "/home/yiminlin/Desktop/dg2D_CNS_quad_shocktube_output/"
df = load("$(OUTPUTPATH)/dg2D_CNS_quad_shocktube.jld2","data")

# for ROWNUM in 1:size(df,1)
for ROWNUM in size(df,1)-3:size(df,1)
    OUTPUTPATHrow = "$(OUTPUTPATH)/ROW-$(ROWNUM)"
    run(`mkdir -p $(OUTPUTPATHrow)`)
    SAVEINT = df[ROWNUM,:SAVEINT]
    N       = df[ROWNUM,:N]
    K       = df[ROWNUM,:K]
    T       = df[ROWNUM,:T]
    K1D     = Int(sqrt(K/2))
    VX   = df[ROWNUM,:VX]
    VY   = df[ROWNUM,:VY]
    EToV = df[ROWNUM,:EToV]
    rd = init_reference_quad(N,gauss_lobatto_quad(0,0,N))
    md = init_mesh((VX,VY),EToV,rd)
    Uhist    = df[ROWNUM,:Uhist]
    schlhist = df[ROWNUM,:schlhist]
    schlhist = df[ROWNUM,:schlhist]
    thist    = df[ROWNUM,:thist]
    @unpack x,y,xf,yf,mapM,mapP,mapB,J,rxJ,syJ,sJ = md
    @unpack Vf,Dr,Ds,LIFT = rd
    rx = rxJ./J
    sy = syJ./J

    x_vtk    = zeros(Float64,2*(N+1)*K1D,(N+1)*K1D)    # TODO: hardcoded domain size
    y_vtk    = zeros(Float64,2*(N+1)*K1D,(N+1)*K1D)
    U_vtk    = zeros(Float64,Nc,2*(N+1)*K1D,(N+1)*K1D)
    schl_vtk = zeros(Float64,2*(N+1)*K1D,(N+1)*K1D)
    pvd = paraview_collection("$(OUTPUTPATHrow)/dg2D_CNS_quad_shocktube.pvd")

    construct_vtk_mesh!(x_vtk,y_vtk,K,K1D,N,x,y)

    count = 1
    for i = 1:length(thist)
        if (i % SAVEINT == 0)
            construct_vtk_file!(Uhist[count],schlhist[count],U_vtk,schl_vtk,pvd,x_vtk,y_vtk,thist[i],OUTPUTPATHrow,K,K1D,N,x,y)
            count = count + 1
        end
    end
    construct_vtk_file!(Uhist[count],schlhist[count],U_vtk,schl_vtk,pvd,x_vtk,y_vtk,T,OUTPUTPATHrow,K,K1D,N,x,y)

    vtk_save(pvd)
end

print(df[:,[1:5; 8; 13:21]])