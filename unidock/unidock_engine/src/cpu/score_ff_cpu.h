//
// CPU Full Force Field Scoring (GAFF2 + ff14SB)
// Computes energy and Cartesian forces for FF minimization.
// All receptor atoms are fixed; only ligand is optimized.
//

#ifndef SCORE_FF_CPU_H
#define SCORE_FF_CPU_H

#include "common_cpu.h"
#include "score_cpu.h"
#include "model/model.h"
#include "format/dms_reader.h"
#include "geometry/quaternion.h"
#include "geometry/rotation.h"
#include "score/gaff2.h"

/// Nonbonded cutoff for inter-molecular interactions (Angstrom)
static constexpr Real FF_CUTOFF = 12.0;
static constexpr Real FF_CUTOFF_SQ = FF_CUTOFF * FF_CUTOFF;

/// Minimum distance to avoid division by zero (Angstrom)
static constexpr Real FF_MIN_DIST = 0.5;

/**
 * @brief Compute dihedral angle from 4 atom positions.
 *
 * Uses the atan2 formulation for numerical stability.
 *
 * @param p0,p1,p2,p3  Atom positions (each Real[3])
 * @return Dihedral angle in radians [-pi, pi]
 */
Real compute_dihedral_angle(const Real* p0, const Real* p1, const Real* p2, const Real* p3);

/**
 * @brief Distribute dihedral torsion force to 4 atoms in Cartesian coordinates.
 *
 * Given dE/dphi, computes forces on atoms p0, p1, p2, p3 such that
 * sum of forces = 0 and sum of torques matches dE/dphi.
 *
 * Forces are ACCUMULATED (+=) into f0, f1, f2, f3.
 *
 * @param p0..p3   Atom positions (each Real[3])
 * @param dE_dphi  Energy derivative with respect to dihedral angle
 * @param f0..f3   [in/out] Force arrays (each Real[3])
 */
void distribute_dihedral_forces(const Real* p0, const Real* p1, const Real* p2, const Real* p3,
                                Real dE_dphi,
                                Real* f0, Real* f1, Real* f2, Real* f3);

/**
 * @brief Distribute angle bend force to 3 atoms in Cartesian coordinates.
 *
 * Given dE/dtheta, computes gradients on atoms pi, pj, pk (j is the apex)
 * such that sum of gradients = 0.
 *
 * Gradients are ACCUMULATED (+=) into fi, fj, fk with sign convention
 * fi = +dE/dx_i (matches aux_f convention).
 */
void distribute_angle_forces(const Real* pi, const Real* pj, const Real* pk,
                             Real dE_dtheta,
                             Real* fi, Real* fj, Real* fk);

/**
 * @brief Calculate energy and Cartesian forces using full force field.
 *
 * Includes:
 * 1. Intra-molecular LJ + Coulomb (non-excluded, non-1-4 pairs)
 * 2. 1-4 pair interactions (pre-scaled from DMS)
 * 3. Torsion energy (all dihedrals from DMS)
 * 4. Inter-molecular LJ + Coulomb (ligand-receptor, within cutoff)
 * 5. Box penalty
 *
 * @param pose       Current ligand pose
 * @param flex_topo  Ligand topology (for box penalty atom types)
 * @param fix_mol    Fixed receptor molecule (for coordinates)
 * @param ff         Force field parameters from DMS
 * @param aux_f      [out] Energy gradient ∂E/∂x on ligand atoms (size: natom*3)
 * @return Total energy (kcal/mol)
 */
Real cal_e_f_ff_cpu(const FlexPose* pose, const FlexTopo& flex_topo,
                    const FixMol& fix_mol, const FFMinData& ff,
                    Real* aux_f);

/**
 * @brief Project Cartesian gradients into internal coordinates (FF version, includes H atoms).
 *
 * Same logic as cal_grad_cpu but does NOT skip hydrogen atoms.
 *
 * @param aux_f      Energy gradient ∂E/∂x on ligand atoms
 * @param flex_topo  Ligand topology
 * @param pose       Current ligand pose
 * @param out_g      [out] Gradient (force direction, same convention as Vina)
 */
void cal_grad_ff_cpu(const Real* aux_f, const FlexTopo& flex_topo,
                     const FlexPose* pose, FlexPoseGradient* out_g);

/**
 * @brief Combined energy + gradient calculation for FF minimization.
 *
 * @return Total energy (kcal/mol)
 */
Real cal_e_grad_ff_cpu(const FlexPose* pose, FlexPoseGradient* out_g,
                       const FlexTopo& flex_topo,
                       const FixMol& fix_mol, const FFMinData& ff,
                       Real* aux_f);

/**
 * @brief Combined energy + flat-Cartesian-gradient calculation for full-DOF BFGS.
 *
 * Writes g[k] = +dE/dx_k = aux_f[k] (the existing aux_f convention matches dE/dx).
 * Returns total energy.
 */
Real cal_e_grad_cartesian_cpu(const FlexPose* pose, const FlexTopo& flex_topo,
                              const FixMol& fix_mol, const FFMinData& ff,
                              Real* grad, Real* aux_f);

/**
 * @brief Per-atom FF energy decomposition (analogous to Vina's score_decomp).
 *
 * For each ligand atom, decomposes energy into:
 *   inter_lj, inter_coulomb, intra_lj, intra_coulomb, torsion, total
 *
 * Ligand coordinates come from pose->coords; receptor coordinates and all FF
 * parameters come from FFMinData (loaded from DMS). This avoids dependency
 * on FixMol which may reside in GPU memory.
 *
 * @param pose  Current ligand pose (provides coords)
 * @param ff    Force field parameters from DMS (includes receptor coords)
 * @return vector of FFAtomTerms, one per ligand atom
 */
std::vector<FFAtomTerms> score_decomp_ff(const FlexPose* pose, const FFMinData& ff);

#endif // SCORE_FF_CPU_H
