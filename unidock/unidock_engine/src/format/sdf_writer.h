//
// SDF V2000 writer for docking pose output.
// Writes output SDF directly from enriched DMS molecular data + docked
// coordinates and energies, eliminating the JSON intermediate.
//

#ifndef SDF_WRITER_H
#define SDF_WRITER_H

#include <string>
#include <vector>
#include <array>
#include "dms_reader.h"    // SDFMolData, SDFBond, SDFProperty
#include "model/model.h"   // FlexPose, Real
#include "score/vina.h"    // VinaTerms
#include "score/gaff2.h"   // FFAtomTerms

/// Write docking poses to an SDF V2000 file.
///
/// One SDF record is emitted per pose, containing:
///   - Atom coordinates from the docking result
///   - Bond connectivity/orders from SDFMolData (read from DMS); the stereo column is always 0
///   - Formal charges as M  CHG lines
///   - User SDF properties (dG_exp, etc.) from DMS ud2_sdf_properties
///   - Docking energies (Vina + FF) as SDF data items
///   - Per-atom decomposition (if enabled)
///
/// Per-atom arrays (atom_*, atom_ff_*, ud2_heavy_*) cover the written molecule: the first
/// sdf_mol.natom flex atoms, in SDF order. When the flexible system is larger (PROTAC inputs
/// with sdf_natom, where the E3 ligase rides along as flex atoms), the sum over the unwritten
/// atoms of every atom_<X> array is written as decomp_unwritten_<X>, so the totals still
/// reconcile.
///
/// @param fp_sdf           Output SDF file path.
/// @param sdf_mol          Molecular structure data from DMS.
/// @param flex_names       Ligand names (one per flexible molecule).
/// @param filtered_pose_inds_list  Pose indices per ligand after filtering.
/// @param flex_pose_list_res       Pose energy/transform data.
/// @param flex_pose_list_real_res  Flat array of coords + dihedrals.
/// @param list_i_real      Index ranges into flex_pose_list_real_res.
/// @param decomp_data      Per-atom Vina decomposition (optional).
/// @param ff_decomp_data   Per-atom FF decomposition (optional).
/// @param ff_pose_energies Per-pose FF energy summary (optional).
/// @param residue_decomp_data  Per-residue Vina decomposition (optional).
/// @param residues         Residue keys for residue_labels.
/// @param pose_diag        Raw rescoring-term accumulators (optional).
/// @param pose_score_parts Per-pose {intra, inter, penalty} of the pass that produced
///                         vina_pose_score, written as vina_pose_intra / _inter / _penalty (optional).
/// @param decomp_weights   Weight set of that pass: atom_ / residue_ elec, arom_contact and
///                         polar_env are written only when their weight is nonzero.
void write_poses_to_sdf(
    const std::string& fp_sdf,
    const std::vector<SDFMolData>& sdf_mol_list,
    const std::vector<std::string>& flex_names,
    const std::vector<std::vector<int>>& filtered_pose_inds_list,
    const FlexPose* flex_pose_list_res,
    const Real* flex_pose_list_real_res,
    const std::vector<int>& list_i_real,
    const std::vector<std::vector<std::vector<VinaTerms>>>& decomp_data = {},
    const std::vector<std::vector<std::vector<FFAtomTerms>>>& ff_decomp_data = {},
    const std::vector<std::vector<std::array<Real, 3>>>& ff_pose_energies = {},
    const std::vector<std::vector<std::vector<VinaTerms>>>& residue_decomp_data = {},
    const std::vector<ResidueKey>& residues = {},
    const std::vector<std::vector<ScoreDiag>>& pose_diag = {},
    const std::vector<std::vector<std::array<Real, 3>>>& pose_score_parts = {},
    const VinaWeights& decomp_weights = VinaWeights{});

#endif // SDF_WRITER_H
