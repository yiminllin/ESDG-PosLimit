using Plots
using DelimitedFiles

gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=3.5)#,ticks=nothing,border=nothing,axis=nothing)

N = 4
K1D = 16
CFL = 0.5
t=0.25#1.0
t=0.20754679528405856
x   = readdlm("N=$N,K1D=$K1D,t=0.0,CFL=$CFL,x,convtri.txt")
y   = readdlm("N=$N,K1D=$K1D,t=0.0,CFL=$CFL,y,convtri.txt")
lim = ones(22,1)*readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,L_plot,convtri.txt")'
Plots.scatter(x[:],y[:],lim[:],zcolor=lim[:],camera=(0,90),c=:haline,legend=false,colorbar=true,axis=nothing,ticks=nothing,border=nothing,showaxis=false)
savefig("2D-euler-tri-conv-lim.png")
rho = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,rho,convtri.txt")
Plots.scatter(x[:],y[:],rho[:],zcolor=rho[:],camera=(0,90),c=:haline,legend=false)
savefig("2D-euler-tri-conv.png")