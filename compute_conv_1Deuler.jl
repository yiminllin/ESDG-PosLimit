using DelimitedFiles

filename = "./log-conv-euler-1D-CFL.5"
file = readdlm(filename)

num_scheme = 2
Narr = [2;5]
Karr = [50;100;200;400;800]
block_size = 4
num_lines = size(file,1)
num_block = div(num_lines,block_size)
L1_err_dict = Dict()
for b in 1:num_block
    l = 1+block_size*(b-1)
    N = parse(Int,file[l,3][1])
    K = file[l,6]
    L1_err = file[l+1,4]
    @show N,K
    @show L1_err
    if !haskey(L1_err_dict,(N,K))
        L1_err_dict[(N,K)] = [L1_err]
    else
        push!(L1_err_dict[(N,K)],L1_err)
    end
end

num_row = length(Karr)-1
num_col = 2*length(Narr)
conv_table = zeros(num_row,num_col)
for i = 1:num_row
    for j = 1:num_col
        K = Karr[i]
        N = Narr[div(j-1,num_scheme)+1]
        s = mod1(j,num_scheme)
        conv_table[i,j] = log2(L1_err_dict[(N,K)][s]/L1_err_dict[(N,2*K)][s])
    end
end