//
// Host-side FlexTopo construction from a parsed UDFlexMol.
// Shared by the CPU backend (alloc_device) and by host-only stages of the
// CUDA backend (FF minimization), which cannot read device-resident topologies.
//

#ifndef HOST_TOPO_H
#define HOST_TOPO_H

#include "model/model.h"
#include <vector>

/**
 * @brief Count the int / Real buffer sizes fill_flex_topo_host will consume for one ligand.
 */
void count_flex_topo_host(const UDFlexMol& m, int* out_n_int, int* out_n_real);

/**
 * @brief Fill topo from m, carving storage out of p_int / p_real and advancing both.
 * The caller owns the buffers and must size them via count_flex_topo_host.
 */
void fill_flex_topo_host(const UDFlexMol& m, FlexTopo& topo, int*& p_int, Real*& p_real);

/**
 * @brief Self-contained host FlexTopo whose pointer fields live in the owned vectors.
 */
struct HostFlexTopo {
    FlexTopo topo;
    std::vector<int> ints;
    std::vector<Real> reals;
};

void build_host_flex_topo(const UDFlexMol& m, HostFlexTopo& out);

#endif // HOST_TOPO_H
