//
// Created by Congcong Liu on 24-10-22.
//

#ifndef MYMATH_WARP_H
#define MYMATH_WARP_H

#include <curand_kernel.h>
#include <cooperative_groups.h>


/**
 * @brief The tile of a one-warp block whose sync() is the block barrier. Every kernel that runs the warp routines
 * launches blocks of exactly one tile (TILE_SIZE threads), so the two barriers are equivalent; bar.sync is aligned (the
 * whole warp executes it), so ptxas treats the warp as converged after it and inlines the shuffles / votes that follow
 * instead of calling the divergence-safe __cuda_sm70_* helpers. The routines take the tile type as a template parameter;
 * only mc_kernel's screening instantiation passes this one (mc_kernel<true,12> came out slower and not bit-identical
 * with it).
 */
struct AlignedTile : cooperative_groups::thread_block_tile<TILE_SIZE>{
    __device__ AlignedTile(const cooperative_groups::thread_block_tile<TILE_SIZE>& t)
        : cooperative_groups::thread_block_tile<TILE_SIZE>(t){}
    __device__ void sync() const{ __syncthreads(); }
};

/** @brief The plain tile behind a tile type, for cg::reduce (which accepts only cooperative-groups types). */
template<class Tile>
__device__ __forceinline__ const cooperative_groups::thread_block_tile<TILE_SIZE>& cg_tile(const Tile& tile){
    return tile;
}



template<class Tile>
__device__ __forceinline__ void init_tri_mat_warp(const Tile& tile,
                                                 Real *m, int dim, Real fill_data) {
    // no padding, since the size is variable for each ligand
    for (int i = tile.thread_rank(); i < dim * (dim + 1) / 2; i += tile.num_threads()) {
        m[i] = fill_data;
    }
    tile.sync();
}


template<class Tile>
__device__ __forceinline__ void set_tri_mat_diagonal_warp(const Tile& tile,
                                                    Real *m, int dim, Real fill_data) {
    for (int i = tile.thread_rank(); i < dim; i += tile.num_threads()) {
        m[i + i * (i + 1) / 2] = fill_data;
    }
    tile.sync();
}











#endif //MYMATH_WARP_H
