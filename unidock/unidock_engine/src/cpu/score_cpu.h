//
// CPU Energy and Force Calculation
// Created for Uni-Dock2 CPU backend
//

#ifndef SCORE_CPU_H
#define SCORE_CPU_H

#include "common_cpu.h"
#include "model/model.h"
#include "score/vina.h"
#include "constants/constants.h"
#include "geometry/quaternion.h"
#include "geometry/rotation.h"

/**
 * @brief Calculate box penalty and clamp coordinate
 * CPU version of cal_box_penalty_clamp_coord
 */
inline Real cal_box_penalty_clamp_coord_cpu(Real* out_x, Real* out_y, Real* out_z, Real* out_f) {
    Real penalty = 0.0;
    const auto& c = g_cpu_constants;
    
    if (*out_x < c.BOX_X_LO) {
        penalty += c.BOX_X_LO - *out_x;
        *out_x = c.BOX_X_LO;
        out_f[0] += -c.PENALTY_SLOPE;
    } else if (*out_x > c.BOX_X_HI) {
        penalty += *out_x - c.BOX_X_HI;
        *out_x = c.BOX_X_HI;
        out_f[0] -= -c.PENALTY_SLOPE;
    }
    
    if (*out_y < c.BOX_Y_LO) {
        penalty += c.BOX_Y_LO - *out_y;
        *out_y = c.BOX_Y_LO;
        out_f[1] += -c.PENALTY_SLOPE;
    } else if (*out_y > c.BOX_Y_HI) {
        penalty += *out_y - c.BOX_Y_HI;
        *out_y = c.BOX_Y_HI;
        out_f[1] -= -c.PENALTY_SLOPE;
    }
    
    if (*out_z < c.BOX_Z_LO) {
        penalty += c.BOX_Z_LO - *out_z;
        *out_z = c.BOX_Z_LO;
        out_f[2] += -c.PENALTY_SLOPE;
    } else if (*out_z > c.BOX_Z_HI) {
        penalty += *out_z - c.BOX_Z_HI;
        *out_z = c.BOX_Z_HI;
        out_f[2] -= -c.PENALTY_SLOPE;
    }
    
    return penalty * c.PENALTY_SLOPE;
}

/**
 * @brief Calculate energy and forces for a pose
 * CPU version of cal_e_f_warp
 * 
 * @param pose Current pose
 * @param flex_topo Flex topology
 * @param fix_mol Fixed molecule (receptor)
 * @param flex_param Flex parameters (atom pairs)
 * @param fix_param Fix parameters
 * @param aux_f Output forces array (size: natom * 3)
 * @return Total energy
 */
Real cal_e_f_cpu(const FlexPose* pose, const FlexTopo& flex_topo,
                 const FixMol& fix_mol, const FlexParamVina& flex_param,
                 const FixParamVina& fix_param, Real* aux_f);

/**
 * @brief Calculate gradient from forces
 * CPU version of cal_grad_warp
 * 
 * @param aux_f Input forces
 * @param flex_topo Flex topology  
 * @param pose Current pose
 * @param out_g Output gradient
 */
void cal_grad_cpu(const Real* aux_f, const FlexTopo& flex_topo,
                  const FlexPose* pose, FlexPoseGradient* out_g);

/**
 * @brief Calculate energy and gradient
 * CPU version of cal_e_grad_warp
 */
Real cal_e_grad_cpu(const FlexPose* pose, FlexPoseGradient* out_g,
                    const FlexTopo& flex_topo,
                    const FixMol& fix_mol, const FlexParamVina& flex_param,
                    const FixParamVina& fix_param, Real* aux_f);


// ==================== Bonded-H topology ====================

/**
 * @brief Build bonded-H topology for a ligand (CSR format).
 * For each atom, finds H atoms within 1.3 A (same cutoff as hbond_angle_correction_cpu).
 * Built once per ligand at init time (in alloc_device) and stored in FlexParamVina.
 */
void build_bonded_h_topo(BondedHTopo& topo, const Real* coords,
                         const int* vn_types, int natom);

/**
 * @brief Free bonded-H topology memory.
 */
void free_bonded_h_topo(BondedHTopo& topo);

/**
 * @brief Free receptor donor-H topology memory.
 */
void free_rec_donor_h_topo(RecDonorHTopo& topo);

// ==================== Cone-donor hydrogen placement ====================

/**
 * @brief A free-rotor donor whose hydrogens place_cone_donor_h may turn about its X-D bond.
 */
struct PlaceHDonor {
    int d = -1;              // donor atom
    int x = -1;              // its heavy neighbour (antecedent); the hydrogens turn about the X-D axis
    bool flip_only = false;  // hydroxyl on an aromatic sp2 atom (phenol): only the 180 deg in-plane flip
    std::vector<int> h;      // its hydrogens
};

/**
 * @brief Cone donors of one ligand whose hydrogens can be placed after the search, decided once on
 * the input conformer (mol.coords) with the bonded-H topology the final scoring builds. A donor
 * qualifies when it is a free-rotor donor with a heavy neighbour X and at least one H, has no second
 * heavy atom within 2.0 A (R2NH2+), is the only heavy atom listing each of its H, and it and its H
 * are below n_write (the atoms written to the SDF). A hydroxyl on a planar X qualifies only when X
 * is aromatic, and then only for the flip.
 */
std::vector<PlaceHDonor> find_placeable_cone_donors(const UDFlexMol& mol, int n_write, const AtomTypeProps& atp);

/**
 * @brief Turn the hydrogens of each placeable donor rigidly about its X-D axis toward the receptor
 * acceptors in hydrogen-bond range of the pose (vina_hbond ramp > 0), maximising
 * sum over acceptors of ramp * best explicit-H angular weight. An acceptor that already gets full
 * credit as a donor toward D (direction 2 of the angular rule) is left out. The turns tried are the
 * ones putting an H on the cone toward each acceptor, with two or more acceptors also every 1 deg,
 * and for a phenol only the flip. A turn is taken only when it improves the objective by more than
 * 1e-6 and keeps every turned H at least 2.0 A from the other ligand hydrogens; the current H stay
 * even when they clash. Donors are placed in order, each against the H already placed. Only H
 * coordinates are written.
 * @param coords        pose coordinates, natom * 3; donor hydrogens rewritten in place
 * @param natom         number of ligand atoms in coords
 * @param rec_acceptors indices of the receptor acceptor atoms in fix
 * @param radii         vdW radii by Vina type, as in prepare_vina
 * @param lig_types     Vina type of each ligand atom
 * @param n_h_moved     set to the number of hydrogens moved
 * @return number of donors whose hydrogens moved
 */
int place_cone_donor_h(Real* coords, int natom, const std::vector<PlaceHDonor>& donors, const UDFixMol& fix,
                       const std::vector<int>& rec_acceptors, const Real* radii, const AtomTypeProps& atp,
                       const int* lig_types, int& n_h_moved);

// ==================== Hbond angle correction ====================

/**
 * @brief Post-BFGS hbond angle correction for MC acceptance.
 *
 * Identifies ligand donor–receptor acceptor pairs where Vina awarded hbond
 * energy but the D-H...A geometry is poor (angle < 100°).  Returns a
 * positive correction that, when added to the pose energy, removes the
 * favorable hbond contribution for those pairs.
 *
 * Only corrects pairs where the ligand is a donor with bonded H and the
 * receptor is an acceptor.  Pairs that could also be valid in the reverse
 * direction (ligand is also acceptor AND receptor is also donor) are left
 * untouched because receptor H positions are unavailable.
 */
Real hbond_angle_correction_cpu(const FlexPose* pose, const FlexTopo& flex_topo,
                                 const FixMol& fix_mol);

// ==================== Pose manipulation functions ====================

/**
 * @brief Duplicate gradient
 */
void duplicate_grad_cpu(FlexPoseGradient* out_g_new, const FlexPoseGradient* g_old, int dim);

/**
 * @brief Duplicate pose
 */
void duplicate_pose_cpu(FlexPose* out_pose_new, const FlexPose* pose_old, int dim, int natom);

/**
 * @brief Apply dihedral update to pose
 * @param skip_range_clamp  If true, skip torsion library range clamping (for FF minimization)
 */
void apply_grad_update_dihe_cpu(FlexPose* out_x, const FlexTopo* flex_topo,
                                 int i_tor, Real dihe_incre_raw,
                                 bool skip_range_clamp = false);

/**
 * @brief Apply gradient update to pose (center, orientation, dihedrals)
 * @param skip_range_clamp  If true, skip torsion library range clamping (for FF minimization)
 */
void apply_grad_update_pose_cpu(FlexPose* out_x, const FlexPoseGradient* g,
                                 const FlexTopo& flex_topo, Real alpha,
                                 bool skip_range_clamp = false);

/**
 * @brief Clamp value to center range
 */
inline Real clamp_to_center_cpu(Real val, Real hi, Real lo) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

/**
 * @brief Normalize angle to [-PI, PI]
 */
inline Real normalize_angle_cpu(Real angle) {
    if (!std::isfinite(angle)) return 0;
    const Real PI_VAL = Real(3.1415927);
    angle = std::fmod(angle, 2 * PI_VAL);
    if (angle < -PI_VAL) angle += 2 * PI_VAL;
    if (angle > PI_VAL) angle -= 2 * PI_VAL;
    return angle;
}

/**
 * @brief Calculate distance to nearest allowed torsion range
 * @param angle Current dihedral angle (radians)
 * @param range_list Array of [lo, hi] pairs
 * @param nrange Number of ranges
 * @param out_grad_sign Output: sign of gradient direction (-1, 0, or +1)
 * @return Distance to nearest range (0 if within any range)
 */
Real distance_to_nearest_range_cpu(Real angle, const Real* range_list, int nrange, Real* out_grad_sign);

/**
 * @brief Clamp angle by torsion library ranges (hard constraint)
 */
Real clamp_by_ranges_cpu(Real angle, const Real* range_list, int nrange);


#endif // SCORE_CPU_H
