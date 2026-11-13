// Internal: top-level BFGS minimizers exposed for direct unit testing.
#pragma once

#include "bfgs_cpu.h"
#include "score_ff_cpu.h"
#include <vector>


void bfgs_ff_cpu(FlexPose* out_x, const FlexTopo& flex_topo,
                 const FixMol& fix_mol, const FFMinData& ff,
                 int max_steps, BFGSThreadMemory& mem);


void bfgs_ff_cartesian_cpu(FlexPose* out_x, const FlexTopo& flex_topo,
                           const FixMol& fix_mol, const FFMinData& ff,
                           int max_steps, BFGSThreadMemory& mem,
                           const std::vector<bool>* constrained_atom_mask = nullptr);
