
module BlockSparseMatrices

using LinearAlgebra
using SparseArrays
# import Base: Order.Forward

# include Block type for indexing
import BlockArrays.Block
export Block # for indexing

export SparseMatrixBSC, SparseMatrixCSC!, getCSCordering
export block_lrmul
# export blocksize, getCSCindex
# export nnzblocks, nblocks, nzblockrange

include("SparseMatrixBSC.jl")

end # module
