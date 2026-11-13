//
// Created by lccdp on 24-8-15.
// Models defined here are logical-physical models, rather than computation-efficiency models.
// They can be obtained from input files.
//

#ifndef MOLECULE_H
#define MOLECULE_H

#include <vector>
#include <string>
#include <set>
#include <map>
#include <spdlog/spdlog.h>

#include "myutils/common.h"
#include "myutils/errors.h"
// ==================  For a single flexible molecule ==================
struct FlexPose{
    Real energy = 999;
    Real center[3]={0.};
    Real rot_vec[4]={0.}; // # record the rot_vec of last rotation todo: move to FlexPoseGradient
    Real* coords; // size: natom * 3
    Real* dihedrals; // size: ntorsion. Unit: radian (not degree)
};
FlexPose* alloccp_FlexPose_gpu(const FlexPose& flex_pose, int natom, int ntorsion);
void free_FlexPose_gpu(FlexPose* flex_pose_cu);
void freecp_FlexPose_gpu(FlexPose* flex_pose_cu, FlexPose* out_flex_pose, int natom, int ntorsion);


/**
 * @brief Gradient.
 */
struct FlexPoseGradient{
    Real center_g[3] = {0}; // translation gradient (dx, dy, dz)
    Real orientation_g[4] = {0.}; // Only the top 3 are used (-torque) todo: size from 4 to 3
    Real* dihedrals_g; // size: ntorsion. dihedrals_g
};
FlexPoseGradient* alloccp_FlexPoseGradient_gpu(const FlexPoseGradient& flex_pose_gradient, int ntorsion);
void free_FlexPoseGradient_gpu(FlexPoseGradient* flex_pose_gradient);
void freecp_FlexPoseGradient_gpu(FlexPoseGradient* flex_pose_gradient_cu, FlexPoseGradient* out_flex_pose_gradient, int ntorsion);

# define dof_g  6 // DOF of grad: 3 + 3(-torque)
# define dof_x  7 // DOF of pose: 3 + 4(quaternion)

SCOPE_INLINE void grad_index_write(FlexPoseGradient* out_g, int index, Real value){
    if (index < 3){
        out_g->center_g[index] = value;
    }
    else if (index < dof_g){
        out_g->orientation_g[index - 3] = value;
    }
    else{
        out_g->dihedrals_g[index - dof_g] = value;
    }
}


SCOPE_INLINE Real grad_index_read(const FlexPoseGradient* g, int index){
    if (index < 3){
        return g->center_g[index];
    }
    if (index < dof_g){
        return g->orientation_g[index - 3];
    }

    return g->dihedrals_g[index - dof_g];
}


/**
 * @brief Hessian matrix
 */
struct FlexPoseHessian{
    Real* matrix; // size: dim * (dim + 1) / 2. dim = 3(center) + 4(orientation) + ntorsion
};
FlexPoseHessian* alloc_FlexPoseHessian_gpu(int ntorsion);
FlexPoseHessian* alloccp_FlexPoseHessian_gpu(const FlexPoseHessian& flex_pose_hessian, int dim);
void free_FlexPoseHessian_gpu(FlexPoseHessian* flex_pose_hessian);


/**
 * @brief Atomic force vectors
 */
struct FlexForce{
    Real* f; // size: natom * 3
};
FlexForce* alloc_FlexForce_gpu(int natom);
FlexForce* alloccp_FlexForce_gpu(const FlexForce& flex_force, int natom);
void free_FlexForce_gpu(FlexForce* flex_force);


/**
 * @brief Common topology. Docking logic
 */
struct FlexTopo{
    int natom = 0;
    int ntorsion = 0;

    int* vn_types; // size: natom. namely saving all elements

    int* axis_atoms; // size: ntorsion * 2. num_torsion rotation axis, each two is (from, to)

    int* range_inds; // size: ntorsion * 2. [start_index, num_range] for each torsion
    Real* range_list; // size: range_inds[ntorsion * 2] + range_inds[ntorsion * 2 + 1] + 1. from torsion library [-0.8pi,-0.7pi], [0.3pi, 0.9pi] ...

    int* rotated_inds; // size: ntorsion * 2. [start_index, end_index] for each torsion, saving index to find rotated_atoms
    int* rotated_atoms; // size: rotated_inds[ntorsion * 2 + 1] + 1. rotated atoms list
};


FlexTopo* alloccp_FlexTopo_gpu(const FlexTopo& flex_topo);
void free_FlexTopo_gpu(FlexTopo* flex_topo);


/**
 * @brief FixMol has no pose, since its pose is itself.
 */
struct FixMol{
    int natom = 0;
    Real* coords;
};
// only serves for ONE object
FixMol* alloccp_FixMol_gpu(const FixMol& fix_mol);
void free_FixMol_gpu(FixMol* fix_mol_cu);


/**
 * @brief Pre-computed bonded-H topology for O(1) lookup during angular HBond scoring.
 * CSR format: for atom i, bonded H indices are h_indices[h_start[i] .. h_start[i+1]).
 * Built once per ligand at initialization from distance-based detection (1.3 A cutoff).
 */
struct BondedHTopo {
    int* h_start = nullptr;     // size: natom + 1 (CSR row pointers)
    int* h_indices = nullptr;   // size: total_bonded_h (flat array of bonded H atom indices)
    int* antecedent = nullptr;  // size: natom. bonded heavy neighbour of each atom (-1: none); cone hydrogens of rotatable donors
    int* donor_rotatable = nullptr; // size: natom. 1: free-rotor donor (-OH, pyramidal -NH2, -NH3+), scored with the cone hydrogen
    int  natom = 0;
    int  total_h = 0;           // total number of bonded-H entries
    int  n_shared_cone_antecedent = 0; // pairs of free-rotor donors on one antecedent (gem-diol C(OH)2, B(OH)2, P(OH)2):
                                       // their cone gradients land on the same atom from two GPU lanes (bfgs.cuh cand_pass)
};

/**
 * @brief Term weights of the Vina-type scoring function. Defaults are AutoDock Vina's; the
 * search, the host rescoring and the GPU kernels all read the same set (DockParam::weights).
 * intra scales the whole intramolecular pair energy (1 = Vina).
 */
/**
 * @brief Raw, unweighted accumulators of the rescoring terms, for offline weight fitting.
 *
 * The weighted energy alone cannot be taken apart again: polar_env and arom_contact enter
 * `center[1]` summed with everything else. Writing the raw sums lets a different functional form
 * be swept in Python against a single docking run.
 */
struct ScoreDiag {
    Real polar_env = 0;      ///< sum over class atoms of 1 - exp(-n_a / POLAR_ENV_CAP)
    Real arom_contact = 0;   ///< sum of the hydrophobic ramp over aromatic C x ring C pairs
    Real torsion_strain = 0; ///< sum of the per-quadruple torsional penalties
    Real burial = 0;         ///< sum over class atoms of the raw soft count n_a
    Real hbond_sat = 0;      ///< sum over class atoms of the angular hbond magnitude they achieved
    int  n_class = 0;        ///< number of weakly polar (class X) ligand heavy atoms
    int  n_heavy = 0;        ///< number of ligand heavy atoms
    // Per-ligand-heavy-atom arrays, in ligand atom order. The class-only view was not enough:
    // fill_polar_env_class excludes every donor, so a buried-unsatisfied-polar term could not be
    // computed from the output at all.
    std::vector<Real> heavy_type;    ///< vina type
    std::vector<Real> heavy_class;   ///< 1 if the atom is in the polar_env class, else 0
    std::vector<Real> heavy_apolar;  ///< soft count of receptor C_H within POLAR_ENV_RADIUS
    std::vector<Real> heavy_polar;   ///< same kernel over receptor donors and acceptors
    std::vector<Real> heavy_hbond;   ///< angular-weighted hbond magnitude this atom achieved
    std::vector<int>  heavy_atom;    ///< flex atom index of each entry above (not written)
    // Screened Coulomb split by whether the LIGAND atom sits in a formally charged group, so
    // that E = elec * (elec_neutral + elec_charged_scale * elec_charged) reproduces the term
    // for any (elec, elec_charged_scale) without re-docking. Charges are unscaled here.
    Real elec_neutral = 0;
    Real elec_charged = 0;
    // Aromatic-ring bias energy (sum of ring_bias_frame over the ligand's rows); written only when it has rows
    Real ring_bias = 0;
    int  n_ring_bias = 0;
    // Bias energy of the pose (atom + ring terms, score/terms.h HostTermSet), the part of the pose-pass key that is
    // not vina_pose_score; set (and written as ud2_bias_energy) only in a biased run (bias type not BT_NO)
    Real bias_energy = 0;
    bool has_bias_energy = false;
};


struct VinaWeights {
    Real gauss1 = -0.035579;
    Real gauss2 = -0.005156;
    Real repulsion = 0.840245;
    Real hydrophobic = -0.035069;
    Real hbond = -0.587439;
    // Scale of the intramolecular pair energy. The pair terms carry this set's gauss1..hbond in the
    // search kernels and Vina's default weights in the host scoring (score/score.h, SF_INTRA).
    Real intra = 1.0;
    // Screened Coulomb between united-atom charges (heavy atom + its hydrogens), eps = 4 r,
    // switched off between 6 and 8 A: E = elec * 332 q_i q_j / (4 r^2) * sw(r).
    // Default off: on the Uni-Dock-Benchmarks set its held-out gain is within seed noise.
    Real elec = 0.0;
    // Scale applied to the charge of ligand atoms in formally charged groups (carboxylate,
    // ammonium, guanidinium, phosphate, ...) before the Coulomb term. 1 = no scaling.
    Real elec_charged_scale = 1.0;
    // Apolar-environment credit for weakly polar ligand atoms (C_P, N_A/N_P, S_P, halogens):
    // polar_env * sum over those atoms of (1 - exp(-n_a / cap)), n_a = soft count of receptor
    // C_H atoms within polar_env_radius. Rescoring term (per-atom, two passes). 0 = off.
    Real polar_env = 0.0;
    // Extra hydrophobic credit for ligand aromatic carbon x receptor ring carbon pairs, added on
    // top of the plain hydrophobic ramp for those pairs. 0 = off.
    Real arom_contact = 0.0;
    // Ligand torsional strain: penalty per matched rotatable-bond class (amide, ester, aryl
    // conjugation, eclipsed sp3), from the ud2_torsion_penalties table. 0 = off.
    Real torsion_strain = 0.0;
};

/// The three weight sets a docking run uses by default: the search set (Monte Carlo + local optimisation),
/// the rescore set (the pose pass that orders the output poses, vina_pose_score) and the affinity set
/// (the reported cross-ligand number, vina_affinity_score). These are the sets fitted on the re-docking
/// benchmarks (search: refit of the Vina terms plus the torsional strain; rescore: learning-to-rank fit that
/// adds polar_env, arom_contact and a lighter strain; affinity: Vina unchanged, no refit beat it for ranking
/// ligands). A plain VinaWeights{} is the Vina 1.1.2 set. Settings.score_weights / rescore_weights /
/// affinity_weights in the YAML override single keys of the respective set.
inline VinaWeights default_search_weights(){
    VinaWeights w;
    w.gauss1 = -0.0534; w.gauss2 = -0.0034; w.hydrophobic = -0.0265; w.hbond = -0.767;
    w.intra = 0.5; w.torsion_strain = 0.5;
    return w;
}
inline VinaWeights default_rescore_weights(){
    VinaWeights w;
    w.gauss1 = -0.041677; w.gauss2 = -0.002722; w.hydrophobic = -0.042773; w.hbond = -0.791096;
    w.intra = 0.60294; w.polar_env = -0.398679; w.arom_contact = 0.03834; w.torsion_strain = 0.216366;
    return w;
}
inline VinaWeights default_affinity_weights(){
    return VinaWeights{};
}

/**
 * @brief Receptor donor-H side-table for angular HBond scoring (receptor-donor direction).
 * CSR format: for compacted heavy-atom i, bonded H coords are
 *   h_coords[(h_start[i]*3) .. (h_start[i+1]*3)).
 * Stores coordinates directly because receptor H atoms are stripped from fix_mol.
 * Built once from DMS bond table at receptor read time.
 */
struct RecDonorHTopo {
    int*  h_start  = nullptr;  // size: natom + 1 (CSR row pointers, keyed by compacted heavy index)
    Real* h_coords = nullptr;  // size: total_h * 3 (x, y, z of each bonded H)
    int*  antecedent = nullptr; // size: natom. compacted index of a bonded heavy neighbour (-1: none, e.g. water O)
    int*  donor_rotatable = nullptr; // size: natom. 1: free-rotor donor (Ser/Thr/Tyr O-H, Lys N-H3, water), 0: fixed (backbone, Arg, Asn, Gln, Trp, His)
    int   natom    = 0;        // = fix_mol.natom (heavy atoms in box)
    int   total_h  = 0;
};


enum ScoreFunc{vina, gaff2};
const std::array<std::string, 2> SCOREFUNC_NAMES = {"vina", "gaff2"};

enum BiasType{
    BT_NO,
    BT_POS,
    BT_ALIGN
};

enum FFMinimizeMode {
    FFMIN_INTERNAL = 0,
    FFMIN_CARTESIAN = 1
};
const std::array<std::string, 2> FFMIN_MODE_NAMES = {"internal", "cartesian"};


struct Box{ 
    // Box is the docking area.
    // Ligand atoms are not allowed to move out of the box.
    Real x_lo = 0; // low bound of x.
    Real x_hi = 0; // high bound of x.
    Real y_lo = 0; // low bound of y
    Real y_hi = 0; // high bound of y
    Real z_lo = 0; // low bound of z
    Real z_hi = 0; // high bound of z
    SCOPE_INLINE bool is_inside(Real x, Real y, Real z) const{
        return (x_lo <= x && x <= x_hi) && (y_lo <= y && y <= y_hi) && (z_lo <= z && z <= z_hi);
    }
};

// Aromatic-ring bias (ext/bias/ring_term.h ring_bias_frame): depth per unit similarity and bias_k (the heavy-atom k_a
// of the Z-align potential), and the steepness (1/A) and midpoint (A) of the centroid-distance sigmoid. The defaults of
// RingBiasParams, i.e. of the YAML block `bias: {ring: {k, s, d0}}` (config/parse.cpp).
#ifndef RING_BIAS_K
#define RING_BIAS_K 30.0
#endif
#ifndef RING_BIAS_S
#define RING_BIAS_S 3.0
#endif
#ifndef RING_BIAS_D0
#define RING_BIAS_D0 1.0
#endif

/**
 * @brief The run-wide parameters of the aromatic-ring bias term (ext/bias/ring_term.h), DockParam::ring_bias and
 * TermSettings::ring (score/terms.h): E = -v_set * bias_k * k * F(d) * (n . n_t)^2, F(d) = 1 / (1 + exp(s (d - d0))).
 */
struct RingBiasParams{
    Real k = RING_BIAS_K;    // depth per unit similarity and bias_k
    Real s = RING_BIAS_S;    // steepness of the centroid-distance sigmoid (1/A)
    Real d0 = RING_BIAS_D0;  // midpoint of the sigmoid (A)
};

/**
 * @brief What the search may move and what it keeps from the input, resolved by the config parser
 * (config/parse.cpp set_* helpers) together with the DockParam fields it mirrors.
 */
struct SearchPolicy{
    bool randomize = true;                // = DockParam::randomize (Advanced.randomize; the binary turns it off under constraint docking)
    bool static_root = false;             // = DockParam::constraint_docking: no translation / orientation DOF (Settings.constraint_docking)
    bool keep_template_hydrogens = false; // the input hydrogens are kept, no hydrogen placement: constraint docking or an active bias
    bool use_torsion_ranges = false;      // torsion-library ranges in the search (Advanced.tor_lib / use_tor_lib)
};

struct DockParam{
    int seed = 12345;
    int rng_period = 0; // > 0: pose i of input ligand k draws curand subsequence (k % rng_period) * exhaustiveness + i; 0: its index in the launch
    bool constraint_docking = false; // mirrored by policy.static_root; set both through config/parse.h set_constraint_docking
    int exhaustiveness = 128;
    bool randomize = true; // mirrored by policy.randomize; set both through config/parse.h set_randomize
    int mc_steps = 20; // MC steps
    int opt_steps = -1; // optimization steps in MC process. Zero if only pure MC search is required.
    int refine_steps = 5; // optimization steps after MC, namely a pure local refinement
    int num_pose = 10; // poses written per ligand
    // Cluster representatives (by search energy, within energy_range) that enter the re-ranking stages:
    // hydrogen placement, FF minimisation, the pose pass and the affinity pass. The pose pass sorts this
    // pool and the first num_pose of it are written, so the output count no longer limits the choice.
    // Effective pool = max(rescore_pool, num_pose).
    int rescore_pool = 10;
    Real energy_range = 10.0;
    Real rmsd_limit = 1.0; // a limit to judge whether two poses are the same during clustering
    Real tor_prec = 0.3; // todo sampling precision for position (Angstrom)
    Real box_prec = 1.0; // todo sampling precision for dihedral (radian)
    bool decomp = false; // per-atom energy decomposition in final scoring

    // Per-ligand atom indices for clustering RMSD. Empty outer vector = use default (heavy atoms).
    // Each inner vector is 0-based atom indices for that ligand; empty inner = heavy atoms for that ligand.
    std::vector<std::vector<int>> cluster_atom_indices_per_flex;

    // Per-ligand atom indices (DMS/FlexPose space) to position-constrain during
    // Cartesian FF minimization. Outer length == nflex when non-empty.
    // Empty outer vector = no constraints (legacy behavior).
    // Empty inner vector for a given ligand = no constraint for that ligand.
    std::vector<std::vector<int>> constrained_atom_indices_per_flex;

    // FF parameters: DMS paths enable FF energy scoring; ff_minimize_steps controls optimization
    int ff_minimize_steps = 0;       // 0 = FF disabled, >0 = FF minimize with N steps
    std::string ligand_dms_path;     // DMS file for ligand (GAFF2)
    std::string protein_dms_path;    // DMS file for protein (ff14SB)
    FFMinimizeMode ff_minimize_mode = FFMIN_INTERNAL;

    ScoreFunc search_score = vina;
    ScoreFunc opt_score = vina;
    Box box;
    // Inner box: the ligand heavy-atom centroid must stay inside it (MC placement and
    // translation bounds, a wall penalty in the search energy, and the shortlist filter). A site
    // prior for site-known docking, off by default.
    Box inner_box;
    bool use_inner_box = false;
    BiasType bias_type = BT_NO; // with bias_k the run's TermSettings (score/terms.h); set through config/parse.h set_bias
    Real bias_k = 0.3;          // (policy.keep_template_hydrogens follows them)
    RingBiasParams ring_bias;   // the ring term's parameters (bias: ring: {k, s, d0}); its rows are per ligand (UDFlexMol::ring)
    SearchPolicy policy;        // what the search may move / keeps; the readers of constraint docking use policy.static_root
    bool angular_hbond = true;
    // CUDA GRID path (src/cuda/maps.cuh): precomputed receptor maps for the MC kernel
    int ncpu = 0;                 // Hardware.ncpu from the YAML; 0 = all hardware threads (host-side OpenMP stages)
    bool use_grid = true;
    Real grid_spacing = 0.375; // map spacing (A)
    Real grid_margin = 8.0;    // map extent beyond the box on every side (A)
    VinaWeights weights = default_search_weights(); // scoring-function term weights used by the search
    // Second weight set for the final scoring pass that produces the reported score and the pose
    // ranking. Two-stage split: the search picks the poses, a richer function ranks them.
    // When has_rescore_weights is false the reported score uses `weights`.
    VinaWeights rescore_weights = default_rescore_weights();
    bool has_rescore_weights = true;
    // Third weight set for the REPORTED affinity. Choosing the best pose of one ligand and comparing
    // different ligands are different questions: a term that sums over atoms is invisible to the
    // first (the atom count is fixed) and dominates the second. Poses are ordered by
    // `rescore_weights` and the reported affinity comes from `affinity_weights`.
    VinaWeights affinity_weights = default_affinity_weights();
    bool has_affinity_weights = true;
    /// Weight set of the pass that orders the poses and produces vina_pose_score.
    const VinaWeights& pose_weights() const { return has_rescore_weights ? rescore_weights : weights; }
    DockParam() = default;

    void show() const{
        spdlog::info("DockParam: seed={}, search_score={}, opt_score={}, \n"
                     "box: x_lo={} Angstrom, x_hi={} Angstrom, y_lo={} Angstrom, y_hi={} Angstrom, z_lo={} Angstrom, z_hi={} Angstrom, \n"
                     "constraint_docking={}, exhaustiveness={}, \n"
                     "mc_steps={}, opt_steps={}, refine_steps={}, \n"
                     "num_pose={}, rescore_pool={}, rmsd_limit={} Angstrom, \n"
                     "tor_prec={}, box_prec={}, \n"
                     "bias_type={}, bias_k={}, ff_minimize_mode={}.",
                     seed, static_cast<int>(search_score), static_cast<int>(opt_score),
                     box.x_lo, box.x_hi, box.y_lo, box.y_hi, box.z_lo, box.z_hi,
                     policy.static_root, exhaustiveness,
                     mc_steps, opt_steps, refine_steps,
                     num_pose, rescore_pool, rmsd_limit,
                     tor_prec, box_prec,
                     static_cast<int>(bias_type), bias_k,
                     FFMIN_MODE_NAMES[static_cast<int>(ff_minimize_mode)]);
        if (!constrained_atom_indices_per_flex.empty()) {
            for (size_t i = 0; i < constrained_atom_indices_per_flex.size(); i++) {
                spdlog::info("  constrained_atom_indices_per_flex[{}] count = {}",
                             i, constrained_atom_indices_per_flex[i].size());
            }
        }
    }
};

//------------------------------  Pre- and Post-process ------------------------------
// Use UDxxx Models, which are redundant models.

/**
 * @brief Redundant Data Model for rotatable torsions.
 */
struct UDTorsion{
    int axis[2] = {-1}; // rotation axis atoms
    int atoms[4] = {-1};
    std::vector<int> rotated_atoms; // all atoms rotated by this torsion
    std::vector<Real> range_list;
    std::vector<Real> param_gaff2; // each 4 elements are one group 
};

struct Bias{
    int i;              // ligand atom index
    float param[5];     // [x, y, z, V_set, r]
};

/**
 * @brief Host-side owner of one ligand's position-bias rows: the ud2_bias table as ext/bias/bias_reader.cpp reads
 * it (rowid order, atom ids rebased per ligand). DockTask::alloc_device flattens them (flatten_bias_rows,
 * ext/bias/bias_term.h) into the ligand's FlexParamVina::terms.bias; same lifetime as the UDFlexMol that holds them.
 */
struct PositionBiasRows{
    std::vector<Bias> rows;
};

/**
 * @brief One aromatic-ring bias row from ud2align (DMS tables ud2_ring_bias + ud2_ring_bias_atoms):
 * a query ring of the ligand, in ring order, pulled onto a template ring (centroid c, unit normal nt).
 * Scored by ring_bias_frame (ext/bias/ring_term.h) wherever the atom bias is, when bias_type is BT_ALIGN.
 */
struct RingBiasRow {
    int  n = 0;                         // ring size, 5 or 6
    int  atoms[6] = {0, 0, 0, 0, 0, 0}; // ligand atom indices in ring order
    Real c[3] = {0, 0, 0};
    Real nt[3] = {0, 0, 0};
    Real v_set = 0;                     // ring-pair similarity
};

/**
 * @brief Host-side owner of one ligand's aromatic-ring bias rows: the ud2_ring_bias / ud2_ring_bias_atoms tables as
 * ext/bias/ring_reader.cpp reads them (ring_row order, atoms in pos order, rebased per ligand, template normal
 * normalised). DockTask::alloc_device points the ligand's FlexParamVina::terms.ring (ext/bias/ring_term.h) at them
 * (CPU) or uploads them (CUDA); empty without the tables.
 */
struct RingBiasRows{
    std::vector<RingBiasRow> rows;
};

/**
 * @brief One dihedral quadruple of the ligand torsional-strain term, read from the DMS table
 * ud2_torsion_penalties (written at preparation time, where SMARTS matching is available).
 * kind: 0 = flat-bottom penalty on the deviation from planarity (conjugated single bonds),
 *       1 = flat-bottom penalty on the distance to the nearest eclipse (sp3-sp3),
 *       2 = (1 - cos phi) / 2 (cis amide / E ester).
 * weight already carries 1 / (quadruples of this class and bond), so the engine only sums.
 */
struct TorsionPenaltyRow {
    int  atoms[4] = {0, 0, 0, 0};
    int  kind = 0;
    Real tolerance = 20.0;  // degrees; unused for kind 2
    Real weight = 1.0;
};

/**
 * @brief Redundant Data Model for flexible molecule, including the ligand and those flexible chains connected to it.
 * Molecule is a natural physical concept. so interactions should not be saved here.
 */
struct UDFlexMol{
    int natom = 0;
    Real center[3] = {0};
    std::string name;
    std::vector<Real> coords;
    std::vector<int> vina_types; // vina in default todo?
    std::vector<Real> charges;
    std::vector<Real> charges_ua;   // united-atom charge per atom (hydrogens 0): charge + bonded hydrogens
    std::vector<int> charged_group; // 1 if the atom belongs to a formally charged group (from ud2_formal_charges)
    std::vector<int> aromatic;      // 1 if aromatic (ud2_atom_aromatic; empty when the table is absent)
    std::vector<int> polar_env_class; // 1 if a weakly polar atom eligible for the apolar-environment credit
    std::vector<TorsionPenaltyRow> torsion_penalties; // empty when the DMS carries no ud2_torsion_penalties
    std::set<std::pair<int, int>> pairs_1213; // ignored pairs in vdw summation
    std::set<std::pair<int, int>> pairs_14; // ignored torsions in dihedral calculation
    std::vector<UDTorsion> torsions;
    std::vector<Real> dihedrals;
    std::vector<UDTorsion> ff_only_torsions; // hydrogen-only torsions removed from the search, rotated by FF minimisation only
    std::vector<Real> ff_only_dihedrals; // input dihedral of each (radian)
    std::vector<int> intra_pairs; // each in ligand x each in ligand, except 1-2, 1-3, 1-4
    std::vector<int> inter_pairs; // each in ligand x each in protein (they both count their own atoms from zero)
    std::vector<Real> r1_plus_r2_intra; // vdW radii summations of each pair, intra part
    std::vector<Real> r1_plus_r2_inter; // vdW radii summations of each pair, inter part
    PositionBiasRows bias; // position-bias data from ud2align (ud2_bias table)
    RingBiasRows ring;     // aromatic-ring bias rows from ud2align (ud2_ring_bias tables), empty without them
};

typedef std::vector<UDFlexMol> UDFlexMolList;


struct ResidueKey {
    std::string chain;
    int resid;
    std::string resname;
    bool operator<(const ResidueKey& o) const {
        if (chain != o.chain) return chain < o.chain;
        if (resid != o.resid) return resid < o.resid;
        return resname < o.resname;
    }
    bool operator==(const ResidueKey& o) const {
        return chain == o.chain && resid == o.resid && resname == o.resname;
    }
};

/**
 * @brief Redundant Data Model for fixed molecule.
 */
struct UDFixMol{
    int natom; // only records those inside box
    std::vector<Real> coords;
    std::vector<int> vina_types; // vina in default todo?
    std::vector<Real> charges;
    std::vector<Real> charges_ua; // united-atom charge per compacted heavy atom: charge + bonded hydrogens
    std::vector<int> arom_ring_c; // 1 for Phe/Tyr/Trp aromatic ring carbons (hydrophobic Vina type only)
    RecDonorHTopo rec_donor_h;  // receptor donor-H side table for angular HBond
    std::vector<int> atom_to_residue;
    std::vector<ResidueKey> residues;
};


#endif //MOLECULE_H
