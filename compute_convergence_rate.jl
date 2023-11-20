using JLD2
using DataFrames
using FileIO

df1DCNS   = load("dg1D_CNS_convergence.jld2","convergence_data")
df1Deuler = load("dg1D_euler_convergence.jld2","convergence_data")
df2Deulertri  = load("dg2D_euler_tri_convergence.jld2","convergence_data")
df2Deulerquad = load("dg2D_euler_quad_convergence.jld2","convergence_data")

Narr_1DCNS     = [2;3]
Narr_1Deuler   = [2;5]
Narr_2D        = [1;2;3;4]
Karr_1DCNS     = [50;100;200;400;800;1600]
Karr_1Deuler   = [50;100;200;400;800]
Karr_2Dtri     = [16;64;256;1024;4096]
Karr_2Dquad    = [8;32;128;512;2048]
LBOUNDTYPE_arr = [0.1; 0.5]

function update_conv_table!(df,conv_table,Narr,Karr,LBOUNDTYPEarr)
    for Ni in 1:length(Narr)
        for Ki in 1:length(Karr)
            for LBOUNDTYPEi in 1:length(LBOUNDTYPEarr)
                N = Narr[Ni]
                K = Karr[Ki]
                LBOUNDTYPE = LBOUNDTYPE_arr[LBOUNDTYPEi] 
                row = filter([:N,:K,:LBOUNDTYPE] => ((x,y,z)->(x == N && y == K && z == LBOUNDTYPE)), df)
                L1err   = row[1,:L1err]
                L2err   = row[1,:L2err]
                Linferr = row[1,:Linferr]

                i = Ki
                j = 2*(Ni-1)+1
                conv_table[i,j,LBOUNDTYPEi,1] = L1err
                conv_table[i,j,LBOUNDTYPEi,2] = L2err
                conv_table[i,j,LBOUNDTYPEi,3] = Linferr

                # Compute rate
                if (Ki > 1)
                    for c = 1:3
                        conv_table[i,j+1,LBOUNDTYPEi,c] = log2(conv_table[i-1,j,LBOUNDTYPEi,c]/conv_table[i,j,LBOUNDTYPEi,c])
                    end
                end
            end
        end
    end
end

conv_table_1DCNS   = zeros(length(Karr_1DCNS),2*length(Narr_1DCNS),length(LBOUNDTYPE_arr),3)
conv_table_1Deuler = zeros(length(Karr_1Deuler),2*length(Narr_1Deuler),length(LBOUNDTYPE_arr),3)
conv_table_2Deulerquad = zeros(length(Karr_2D),2*length(Narr_2D),length(LBOUNDTYPE_arr),3)
conv_table_2Deulertri  = zeros(length(Karr_2D),2*length(Narr_2D),length(LBOUNDTYPE_arr),3)

update_conv_table!(df1DCNS,conv_table_1DCNS,Narr_1DCNS,Karr_1DCNS,LBOUNDTYPE_arr)
update_conv_table!(df1Deuler,conv_table_1Deuler,Narr_1Deuler,Karr_1Deuler,LBOUNDTYPE_arr)
update_conv_table!(df2Deulerquad,conv_table_2Deulerquad,Narr_2D,Karr_2Dquad,LBOUNDTYPE_arr)
update_conv_table!(df2Deulertri,conv_table_2Deulertri,Narr_2D,Karr_2Dtri,LBOUNDTYPE_arr)