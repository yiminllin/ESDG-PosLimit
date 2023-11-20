using DelimitedFiles
using CairoMakie
using Formatting

file = readdlm("logdmr_lieuler")

num_arr = Int64[]
dt_arr = Float64[]

dtsum = 0.0
finalstep = size(file,1)-1
for l in 1:size(file,1)-1
    if l % 1000 == 1
        append!(num_arr,l)
        append!(dt_arr,parse(BigFloat,file[l,8][1:end-1]))
    end
    global dtsum += parse(BigFloat,file[l,8][1:end-1])
end
append!(num_arr,size(file,1)-1)
append!(dt_arr,parse(BigFloat,file[size(file,1)-1,8][1:end-1]))

avgdt = dtsum/size(file,1)
f1 = Figure()
axis = Axis(f1[1,1])



lw = 3
l1 = lines!(num_arr[:],dt_arr[:],linestyle=nothing,linewidth=lw,color=:royalblue1,label="Elementwise (Zhang-Shu type) limiting")
CairoMakie.ylims!(axis,minimum(dt_arr)/10,maximum(dt_arr)*1.02)
# axis.xticks = [1; 50000:50000:150000; finalstep]
axis.xticks = [1; 150000; 300000; finalstep]
axis.xtickformat = "{:d}"
axis.yticks = minimum(dt_arr):(maximum(dt_arr)-minimum(dt_arr))/4:maximum(dt_arr)
axis.ytickformat = "{:.2e}"
# l1 = lines!(1:10,1:10,linestyle=nothing,linewidth=lw,color=:royalblue1)

file = readdlm("logdmr_lijpost")

num_arr = Int64[]
dt_arr = Float64[]

dtsum = 0.0
finalstep = size(file,1)-1
for l in 1:size(file,1)-1
    if l % 1000 == 1
        append!(num_arr,l)
        append!(dt_arr,parse(BigFloat,file[l,8][1:end-1]))
    end
    global dtsum += parse(BigFloat,file[l,8][1:end-1])
end
append!(num_arr,size(file,1)-1)
append!(dt_arr,parse(BigFloat,file[size(file,1)-1,8][1:end-1]))

avgdt = dtsum/size(file,1)

lw = 3
l1 = lines!(num_arr[:],dt_arr[:],linestyle=nothing,linewidth=lw,color=:darkorange1,label="Elementwise (Zalesak-type) limiting")
# CairoMakie.ylims!(axis,minimum(dt_arr)/10,maximum(dt_arr)*1.02)
# axis.xticks = [1; 50000:50000:150000; finalstep]
# axis.xticks = [1; 150000; 300000; finalstep]
# axis.xtickformat = "{:d}"
# axis.yticks = minimum(dt_arr):(maximum(dt_arr)-minimum(dt_arr))/4:maximum(dt_arr)
# axis.ytickformat = "{:.2e}"
# l1 = lines!(1:10,1:10,linestyle=nothing,linewidth=lw,color=:royalblue1)


axislegend(labelsize=20,position=:rt)

save("euler-li-lij-timestep.png",f1)



