using Plots
using DelimitedFiles

# x   = readdlm("N=4,K1D=80,t=0.0,CFL=0.9,x,convquad.txt")
# y   = readdlm("N=4,K1D=80,t=0.0,CFL=0.9,y,convquad.txt")
# rho = readdlm("N=4,K1D=80,t=2.0,CFL=0.9,rho,convquad.txt")

# gr(aspect_ratio=:equal,legend=false,
#    markerstrokewidth=0,markersize=2)
# Plots.scatter(x[:],y[:],rho[:],zcolor=rho[:],camera=(0,90))
# savefig("~/Desktop/tmp.png")

N = 4
K1D = 16
CFL = 0.9
t = 0.243660151583256#0.4
gr(aspect_ratio=:equal,
   markerstrokewidth=0,markersize=5,ticks=nothing,border=nothing,axis=nothing)
x   = readdlm("N=$N,K1D=$K1D,t=0.0,CFL=$CFL,x,convquad.txt")
y   = readdlm("N=$N,K1D=$K1D,t=0.0,CFL=$CFL,y,convquad.txt")
lim = ones(25,1)*readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,L_plot,convquad.txt")'
Plots.scatter(x[:],y[:],lim[:],zcolor=lim[:],camera=(0,90),c=:haline,legend=false,colorbar=true,axis=nothing,ticks=nothing,border=nothing,showaxis=false)
savefig("2D-euler-quad-conv-lim.png")
rho = readdlm("N=$N,K1D=$K1D,t=$t,CFL=$CFL,rho,convquad.txt")
Plots.scatter(x[:],y[:],rho[:],zcolor=rho[:],camera=(0,90),c=:haline,legend=false)
savefig("2D-euler-quad-conv.png")