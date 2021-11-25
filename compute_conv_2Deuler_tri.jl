using DelimitedFiles

filename = "./log-conv-euler-2D-tri-Limited_CFL.9"
file = readdlm(filename)

num_scheme = 1
Narr = [1;2;3;4]
Karr = [2^2*4;4^2*4;8^2*4;16^2*4;32^2*4]
block_size = 4
num_lines = size(file,1)
num_block = div(num_lines,block_size)
L1_err_dict = Dict()
L2_err_dict = Dict()
for b in 1:num_block
    l = 1+block_size*(b-1)
    N = parse(Int,file[l,3][1])
    K = file[l,6]
    L1_err = file[l+1,4]
    L2_err = file[l+2,4]
    if !haskey(L1_err_dict,(N,K))
        L1_err_dict[(N,K)] = [L1_err]
    else
        push!(L1_err_dict[(N,K)],L1_err)
    end

    if !haskey(L2_err_dict,(N,K))
        L2_err_dict[(N,K)] = [L2_err]
    else
        push!(L2_err_dict[(N,K)],L2_err)
    end
end

num_row = length(Karr)-1
num_col = length(Narr)
conv_table_L1 = zeros(num_row,num_col)
conv_table_L2 = zeros(num_row,num_col)
for i = 1:num_row
    for j = 1:num_col
        K = Karr[i]
        N = Narr[j]
        conv_table_L1[i,j] = log2(L1_err_dict[(N,K)][1]/L1_err_dict[(N,4*K)][1])
        conv_table_L2[i,j] = log2(L2_err_dict[(N,K)][1]/L2_err_dict[(N,4*K)][1])
    end
end

err_table = zeros(length(Karr),2*length(Narr))
for i = 1:length(Karr)
    for j = 1:2*length(Narr)
        K = Karr[i]
        N = Narr[div(j-1,2)+1]
        is_L1 = mod1(j,2) == 1
        if is_L1
            err_table[i,j] = L1_err_dict[(N,K)][1]
        else
            err_table[i,j] = L2_err_dict[(N,K)][1]
        end
    end
end

conv_table = [conv_table_L1[:,1] conv_table_L2[:,1] conv_table_L1[:,2] conv_table_L2[:,2] conv_table_L1[:,3] conv_table_L2[:,3] conv_table_L1[:,4] conv_table_L2[:,4]]