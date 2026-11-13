//
// Created by lccdp on 24-8-15.
//

#ifndef VINA_H
#define VINA_H

#include "model/model.h"
#include "myutils/common.h"
#include "myutils/mymath.h"
#include "constants/constants.h"
#include "score/terms.h"

// Angular HBond ramp: w = 0 at ANGULAR_HBOND_ZERO_DEG, w = 1 at ANGULAR_HBOND_FULL_DEG (D-H...A angle), linear in cos
#ifndef ANGULAR_HBOND_COS_ZERO
#define ANGULAR_HBOND_COS_ZERO 0.0          // cos(90 deg)
#endif
#ifndef ANGULAR_HBOND_COS_FULL
#define ANGULAR_HBOND_COS_FULL -0.76604444  // cos(140 deg)
#endif


/**
 * @brief Vina-required parameters and additional info about flex mol
 */

/**
 * @brief Ligand torsional-strain penalty for one quadruple, with its Cartesian derivative.
 *
 * The VALUE is bit-for-bit the same expression as torsion_strain_energy in score/score.h, so the
 * search energy and the host rescoring energy agree and the deployed torsion_strain weight keeps
 * its meaning. The inverse trig is kept for that reason; at ~9 penalty rows per ligand it is
 * irrelevant beside ~850 inter pairs.
 *
 * The derivative of every kind collapses to a sign, because
 *     d/dphi asin(|sin phi|)   = sign(sin phi * cos phi)
 *     d/dphi acos(cos 3phi)    = 3 * sign(sin 3phi)
 * so no 1/sqrt(1 - x^2) singularity is ever evaluated.
 *
 * Verified against central finite differences on 400 random quadruples over all three kinds:
 * max relative error 3.4e-09.
 *
 * @param out_dedx 12 Reals, OVERWRITTEN with dP/dx for the four atoms in order (a, b, c, d).
 *                 Zero in the flat-bottom regions and for degenerate geometry.
 * @return the penalty P in [0, 1], NOT multiplied by the row weight.
 */
SCOPE_INLINE Real torsion_penalty_ef(const Real* pa, const Real* pb, const Real* pc, const Real* pd,
                                     int kind, Real tol_deg, Real* out_dedx){
    for (int i = 0; i < 12; i++) out_dedx[i] = 0;

    Real b1[3], b2[3], b3[3];
    for (int i = 0; i < 3; i++){
        b1[i] = pb[i] - pa[i];
        b2[i] = pc[i] - pb[i];
        b3[i] = pd[i] - pc[i];
    }
    Real n1[3], n2[3];
    cross_product(b1, b2, n1);
    cross_product(b2, b3, n2);
    const Real b2b2 = dot_product(b2, b2);
    const Real n1n1 = dot_product(n1, n1);
    const Real n2n2 = dot_product(n2, n2);
    if (b2b2 < Real(1e-16) || n1n1 < Real(1e-16) || n2n2 < Real(1e-16)) return 0;
    const Real nb2 = sqrt(b2b2);

    Real cr[3];
    cross_product(n1, n2, cr);
    const Real phi = atan2(dot_product(cr, b2) / nb2, dot_product(n1, n2));

    constexpr Real D2R = Real(M_PI / 180.0);
    Real pen = 0, dpen = 0;
    if (kind == 0){
        const Real s = sin(phi);
        const Real dev = asin(fmin(fabs(s), Real(1))) / D2R;
        const Real raw = (dev - tol_deg) / (Real(90) - tol_deg);
        if (raw <= 0) pen = 0;
        else if (raw >= 1) pen = 1;
        else { pen = raw; dpen = ((s * cos(phi) >= 0) ? Real(1) : Real(-1)) / D2R / (Real(90) - tol_deg); }
    } else if (kind == 1){
        Real c3 = cos(Real(3) * phi);
        c3 = fmax(Real(-1), fmin(Real(1), c3));
        const Real dev = acos(c3) / D2R / Real(3);
        const Real raw = (Real(60) - tol_deg - dev) / (Real(60) - tol_deg);
        if (raw <= 0) pen = 0;
        else if (raw >= 1) pen = 1;
        else { pen = raw; dpen = -((sin(Real(3) * phi) >= 0) ? Real(1) : Real(-1)) / D2R / (Real(60) - tol_deg); }
    } else {
        pen = (Real(1) - cos(phi)) / Real(2);
        dpen = sin(phi) / Real(2);
    }
    if (dpen == Real(0)) return pen;

    // dphi/dx. The middle-atom coefficients depend on b1 pointing pb - pa; with the opposite
    // convention their signs flip, which is the classic way to get this wrong.
    Real g1[3], g4[3];
    for (int i = 0; i < 3; i++){
        g1[i] = -nb2 / n1n1 * n1[i];
        g4[i] =  nb2 / n2n2 * n2[i];
    }
    const Real t1 = dot_product(b1, b2) / b2b2;
    const Real t2 = dot_product(b3, b2) / b2b2;
    for (int i = 0; i < 3; i++){
        out_dedx[i]      = dpen * g1[i];
        out_dedx[3 + i]  = dpen * ((Real(-1) - t1) * g1[i] + t2 * g4[i]);
        out_dedx[6 + i]  = dpen * (t1 * g1[i] + (Real(-1) - t2) * g4[i]);
        out_dedx[9 + i]  = dpen * g4[i];
    }
    return pen;
}


struct FlexParamVina{
    // One copy for one ligand, poses share this
    int npair_intra; // flex-flex
    int* pairs_intra; // size: npair_intra * 2. each two is a pair. The bound is loose, so never changed after initialization.
    Real* r1_plus_r2_intra; // size: npair_intra. vdW radii summations of each pair, intra part + inter part

    int npair_inter; // flex-protein
    int* pairs_inter; // size: npair_inter * 2. each two is a pair: (index_flex, index_protein)
    Real* r1_plus_r2_inter; // size: npair_inter. vdW radii summations of each pair, intra part + inter part

    int* atom_types; // size: natom

    TermSet terms;     // extension terms (score/terms.h): terms.bias = the position-bias rows of this ligand

    BondedHTopo bonded_h_topo = {};  // pre-built per-ligand bonded-H topology for angular HBond

    // Device path only: symmetric CSR view of pairs_intra so each lane walks its own
    // atom's neighbors and accumulates forces in registers (no atomics).
    int* intra_start = nullptr;  // size: natom + 1
    int* intra_nbr = nullptr;    // size: npair_intra * 2

    Real* charges = nullptr;     // size: natom. effective united-atom charge for the Coulomb term (nullptr: term off)

    // Ligand torsional strain. ALL rows are carried, so the search energy equals the value
    // torsion_strain_energy computes for the host rescoring. 15 % of rows sit on an axis no
    // search degree of freedom can rotate (amide C-N); their Cartesian gradient is real but
    // projects to exactly zero on every DOF, so they cost an evaluation and change nothing.
    int n_tors_penalty = 0;
    const TorsionPenaltyRow* tors_penalty = nullptr;  // size: n_tors_penalty

    // Heavy atoms of the ligand, listed first so a warp can process 32 scoring atoms per iteration.
    // Hydrogens carry no inter energy, and on this benchmark they are ~43 % of the atoms, so walking
    // them wastes about half of every warp in the receptor loop.
    int* heavy_ids = nullptr;    // size: nheavy. indices into the pose coordinate array
    int  nheavy = 0;
};
FlexParamVina* alloccp_FlexParamVina_gpu(const FlexParamVina& flex_param_vina, int natom);
void free_FlexParamVina_gpu(FlexParamVina* flex_param_vina_cu);


/**
 * @brief Vina-required parameters and additional info about fixed mol
 */
struct FixParamVina{
    int* atom_types; // size: natom
    RecDonorHTopo rec_donor_h = {};  // device path only: receptor donor-H side table for angular HBond
    Real* charges = nullptr;         // size: natom. united-atom charge per receptor atom (nullptr: Coulomb term off)
};
FixParamVina* alloccp_FixParamVina_gpu(const FixParamVina& fix_param_vina, int natom);
void free_FixParamVina_gpu(FixParamVina* fix_param_vina);



struct VinaScore{
    // todo: further modification
    Real e = 0;
    Real e_inter = 0;
    Real e_intra = 0;
};

/**
 * @brief Per-atom (or per-residue) energy decomposition into the inter terms of score():
 * the 5 Vina terms plus the screened Coulomb, aromatic-contact and apolar-environment terms,
 * each already weighted. `total` is their sum.
 */
struct VinaTerms{
    Real gauss1 = 0;
    Real gauss2 = 0;
    Real repulsion = 0;
    Real hydrophobic = 0;
    Real hbond = 0;
    Real elec = 0;
    Real arom_contact = 0;
    Real polar_env = 0;
    Real total = 0;

    void operator+=(const VinaTerms& other){
        gauss1 += other.gauss1;
        gauss2 += other.gauss2;
        repulsion += other.repulsion;
        hydrophobic += other.hydrophobic;
        hbond += other.hbond;
        elec += other.elec;
        arom_contact += other.arom_contact;
        polar_env += other.polar_env;
        total += other.total;
    }
};


// Data-driven type property lookups (use AtomTypeProps from JSON)
SCOPE_INLINE bool vn_is_hydrophobic(int t, const AtomTypeProps& atp){
    return atp.is_hydrophobic[t];
}

SCOPE_INLINE bool vn_is_acceptor(int t, const AtomTypeProps& atp){
    return atp.is_acceptor[t];
}

SCOPE_INLINE bool xs_is_donor(int t, const AtomTypeProps& atp){
    return atp.is_donor[t];
}

SCOPE_INLINE bool xs_h_bond_possible(int t1, int t2, const AtomTypeProps& atp){
    return (vn_is_acceptor(t1, atp) && xs_is_donor(t2, atp)) || (xs_is_donor(t1, atp) && vn_is_acceptor(t2, atp));
}

/**
 * @brief Linear interpolation between (x_lo, 1) and (x_hi, 0)
 * @param x_lo Lower bound
 * @param x_hi Upper bound
 * @param x Input value
 * @param out_f Output derivative
 * @return Interpolated value
 */
SCOPE_INLINE Real slope_interp(Real x_lo, Real x_hi, Real x, Real* out_f){
    // assert(x_lo < x_hi)
    if (x <= x_lo){
        *out_f = 0.;
        return 1;
    }
    if (x >= x_hi){
        *out_f = 0.;
        return 0;
    }
    *out_f = 1 / (x_lo - x_hi);
    return (x_hi - x) / (x_hi - x_lo);
}


/**
 * @brief Define a class to save one copy of parameters and functions.
 * Each model should be put to Vina for calculations?
 */
class Vina{
private:
    // Use one cutoff

    static constexpr Real gauss1_offset = 0;
    static constexpr Real gauss1_width = .5;
    // const Real gauss1_cutoff = 8.0;
    Real weight_gauss1 = -0.035579;

    static constexpr Real gauss2_offset = 3;
    static constexpr Real gauss2_width = 2.0;
    // const Real gauss2_cutoff = 8.0;
    Real weight_gauss2 = -0.005156;

    static constexpr Real repulsion_offset = 0.0;
    static constexpr Real repulsion_width = 8.0;
    // const Real repulsion_cutoff = 8.0;
    Real weight_repulsion = 0.840245;

    static constexpr Real hydrophobic_low = 0.5;
    static constexpr Real hydrophobic_high = 1.5;
    // const Real hydrophobic_cutoff = 8.0;
    Real weight_hydrophobic = -0.035069;

    const Real hbond_low = -0.7;
    static constexpr Real hbond_high = 0;
    // const Real hbond_cutoff = 8.0;
    Real weight_hbond = -0.587439;

    const Real weight_conf_indep = 0.05846;

    Real weight_intra = 1.0;
    Real weight_elec = 0.0;
    Real elec_charged_scale = 1.0;
    Real weight_polar_env = 0.0;
    Real weight_arom_contact = 0.0;
    Real weight_torsion_strain = 0.0;

    void cal_pair_dists(Real* __restrict__ coords_fix, Real* __restrict__ coords_flex, int* pair_list, int npair,
                        Real* out_dists);


public:
    const Real r2_cutoff = 64.0;

    /** @brief Upper bound of the HBond ramp: vina_hbond is 0 with 0 derivative for d >= this. */
    static constexpr SCOPE_INLINE Real hbond_high_val() { return hbond_high; }
    /** @brief Accessor for HBond weight (needed by angular HBond scoring). */
    SCOPE_INLINE Real weight_hbond_val() const { return weight_hbond; }
    /** @brief Scale applied to every intramolecular pair energy. */
    SCOPE_INLINE Real weight_intra_val() const { return weight_intra; }
    /** @brief Install a weight set (host side; the device copy is refreshed by init_constants). */
    void set_weights(const VinaWeights& w){
        weight_gauss1 = w.gauss1; weight_gauss2 = w.gauss2; weight_repulsion = w.repulsion;
        weight_hydrophobic = w.hydrophobic; weight_hbond = w.hbond; weight_intra = w.intra;
        weight_elec = w.elec; elec_charged_scale = w.elec_charged_scale;
        weight_polar_env = w.polar_env; weight_arom_contact = w.arom_contact;
        weight_torsion_strain = w.torsion_strain;
    }
    VinaWeights get_weights() const {
        VinaWeights w;
        w.gauss1 = weight_gauss1; w.gauss2 = weight_gauss2; w.repulsion = weight_repulsion;
        w.hydrophobic = weight_hydrophobic; w.hbond = weight_hbond; w.intra = weight_intra;
        w.elec = weight_elec; w.elec_charged_scale = elec_charged_scale;
        w.polar_env = weight_polar_env; w.arom_contact = weight_arom_contact;
        w.torsion_strain = weight_torsion_strain;
        return w;
    }
    /** @brief Weight of the united-atom screened Coulomb term (0 = off). */
    SCOPE_INLINE Real weight_elec_val() const { return weight_elec; }
    /** @brief Charge scale for ligand atoms in formally charged groups. */
    SCOPE_INLINE Real elec_charged_scale_val() const { return elec_charged_scale; }
    SCOPE_INLINE Real weight_polar_env_val() const { return weight_polar_env; }
    SCOPE_INLINE Real weight_arom_contact_val() const { return weight_arom_contact; }
    SCOPE_INLINE Real weight_torsion_strain_val() const { return weight_torsion_strain; }

    /**
     * @brief Screened Coulomb pair energy between united-atom charges: 332 q1 q2 / (4 r^2) with r
     * floored at 1 A and a linear switch to zero between 6 and 8 A. Returns the unweighted
     * energy; *out_f = dE/dr (same convention as eval_ef, d = r - const).
     */
    SCOPE_INLINE Real elec_pair(Real q1q2, Real r, Real* out_f) const {
        const Real sw_lo = Real(6.0), sw_hi = Real(8.0);
        if (r >= sw_hi || q1q2 == Real(0)) { *out_f = 0; return 0; }
        Real rc = r < Real(1) ? Real(1) : r;
        Real inv = Real(1) / (Real(4) * rc * rc);
        Real dinv = r < Real(1) ? Real(0) : -Real(2) * inv / rc;
        Real sw = Real(1), dsw = Real(0);
        if (r > sw_lo) { sw = (sw_hi - r) / (sw_hi - sw_lo); dsw = -Real(1) / (sw_hi - sw_lo); }
        Real k = Real(332.0) * q1q2;
        *out_f = k * (dinv * sw + inv * dsw);
        return k * inv * sw;
    }

    /**
     * @brief Compute the value and derivative of the first Gaussian function
     * @snippet test/unit/score/test_vina.cpp vina_gaussian1
     * @param d surface distance
     * @param out_f Output derivative
     * @return energy
     */
    SCOPE_INLINE Real vina_gaussian1(Real d, Real* out_f){
        return gaussian(d, gauss1_offset, gauss1_width, out_f);
    }

    /**
     * @brief Compute the value and derivative of the second Gaussian function
     * @snippet test/unit/score/test_vina.cpp vina_gaussian2
     * @param d surface distance
     * @param out_f Output derivative
     * @return energy
     */
    SCOPE_INLINE Real vina_gaussian2(Real d, Real* out_f){
        return gaussian(d, gauss2_offset, gauss2_width, out_f);
    }

    /**
     * @brief Compute the value and derivative of the repulsion function
     * @snippet test/unit/score/test_vina.cpp vina_repulsion
     * @param d surface distance
     * @param out_f Output derivative
     * @return energy
     */
    SCOPE_INLINE Real vina_repulsion(Real d, Real* out_f){
        if (d >= 0.0){
            *out_f = 0.0;
            return 0.0;
        }
        *out_f = 2 * d;
        return d * d;
    }


    /**
     * @brief Compute the value and derivative of the hydrophobic function
     * @snippet test/unit/score/test_vina.cpp vina_hydrophobic
     * @param d surface distance
     * @param out_f Output derivative
     * @return energy
     */
    SCOPE_INLINE Real vina_hydrophobic(Real d, Real* out_f){
        return slope_interp(hydrophobic_low, hydrophobic_high, d, out_f);
    }


    /**
     * @brief Compute the value and derivative of the hydrogen bond function
     * @snippet test/unit/score/test_vina.cpp vina_hbond
     * @param d surface distance
     * @param out_f Output derivative
     * @return energy
     */
    SCOPE_INLINE Real vina_hbond(Real d, Real* out_f){
        return slope_interp(hbond_low, hbond_high, d, out_f);
    }


    /**
     * @brief Compute the value of the conformation-independent term
     * @snippet test/unit/score/test_vina.cpp vina_conf_indep
     * @param e Inter-molecular energy between flex and protein
     * @param num_rotator Number of rotatable dihedrals
     * @return energy
     */
    SCOPE_INLINE Real vina_conf_indep(Real e, int num_rotator){
        // printf("e=%f, num_rotator=%d, weight_conf_indep = %f, ratio = %f\n", e, num_rotator, weight_conf_indep, 1/(1 + weight_conf_indep * num_rotator));
        return e / (1 + weight_conf_indep * num_rotator);
    }


    /**
     * @brief Compute the value and derivative of the geometrical energy. Positive d means to-origin
     * @snippet test/unit/score/test_vina.cpp eval_ef
     * @param d surface distance
     * @param at1 Atom type of the first atom
     * @param at2 Atom type of the second atom
     * @param out_f Output derivative (-force)
     * @param atp Atom type properties (data-driven from JSON)
     * @return energy
     */
    SCOPE_INLINE Real eval_ef(Real d, int at1, int at2, Real* out_f, const AtomTypeProps& atp){
        // add all items of geometrical energy
        Real e = 0;
        *out_f = 0;
        Real f = 0;
        Real e_tmp = 0;

        // ignore all hydrogen atoms (data-driven)
        if (atp.is_hydrogen[at1] || atp.is_hydrogen[at2]){
            return e;
        }

        e_tmp += weight_gauss1 * vina_gaussian1(d, &f);
        e += e_tmp;
        *out_f += weight_gauss1 * f;

        e_tmp = weight_gauss2 * vina_gaussian2(d, &f);
        e += e_tmp;
        *out_f += weight_gauss2 * f;

        e_tmp = weight_repulsion * vina_repulsion(d, &f);
        e += e_tmp;
        *out_f += weight_repulsion * f;

        if (vn_is_hydrophobic(at1, atp) && vn_is_hydrophobic(at2, atp)){
            e_tmp = weight_hydrophobic * vina_hydrophobic(d, &f);
            e += e_tmp;
            *out_f += weight_hydrophobic * f;
        }
        if (xs_h_bond_possible(at1, at2, atp)){
            e_tmp = weight_hbond * vina_hbond(d, &f);
            e += e_tmp;
            *out_f += weight_hbond * f;
        }
        return e;
    }

    /**
     * @brief eval_ef without its hydrogen-bond block: gauss1 + gauss2 + repulsion + hydrophobic.
     * The receptor maps of the CUDA GRID path store sums of this term; the hydrogen bond is added
     * exactly by the polar pass, which subtracts this value for the same pair.
     * @param d surface distance
     * @param at1 Atom type of the first atom
     * @param at2 Atom type of the second atom
     * @param out_f Output derivative (-force)
     * @param atp Atom type properties (data-driven from JSON)
     * @return energy
     */
    SCOPE_INLINE Real eval_ef_nohb(Real d, int at1, int at2, Real* out_f, const AtomTypeProps& atp){
        // add all items of geometrical energy
        Real e = 0;
        *out_f = 0;
        Real f = 0;
        Real e_tmp = 0;

        // ignore all hydrogen atoms (data-driven)
        if (atp.is_hydrogen[at1] || atp.is_hydrogen[at2]){
            return e;
        }

        e_tmp += weight_gauss1 * vina_gaussian1(d, &f);
        e += e_tmp;
        *out_f += weight_gauss1 * f;

        e_tmp = weight_gauss2 * vina_gaussian2(d, &f);
        e += e_tmp;
        *out_f += weight_gauss2 * f;

        e_tmp = weight_repulsion * vina_repulsion(d, &f);
        e += e_tmp;
        *out_f += weight_repulsion * f;

        if (vn_is_hydrophobic(at1, atp) && vn_is_hydrophobic(at2, atp)){
            e_tmp = weight_hydrophobic * vina_hydrophobic(d, &f);
            e += e_tmp;
            *out_f += weight_hydrophobic * f;
        }
        return e;
    }

    /**
     * @brief Compute per-term energy decomposition (no forces needed).
     * @param d surface distance
     * @param at1 Atom type of the first atom
     * @param at2 Atom type of the second atom
     * @param atp Atom type properties
     * @return VinaTerms with individual term contributions
     */
    SCOPE_INLINE VinaTerms eval_ef_decomp(Real d, int at1, int at2, const AtomTypeProps& atp){
        VinaTerms terms;
        if (atp.is_hydrogen[at1] || atp.is_hydrogen[at2]){
            return terms;
        }
        Real f_dummy = 0;
        terms.gauss1 = weight_gauss1 * vina_gaussian1(d, &f_dummy);
        terms.gauss2 = weight_gauss2 * vina_gaussian2(d, &f_dummy);
        terms.repulsion = weight_repulsion * vina_repulsion(d, &f_dummy);
        terms.total = terms.gauss1 + terms.gauss2 + terms.repulsion;

        if (vn_is_hydrophobic(at1, atp) && vn_is_hydrophobic(at2, atp)){
            terms.hydrophobic = weight_hydrophobic * vina_hydrophobic(d, &f_dummy);
            terms.total += terms.hydrophobic;
        }
        if (xs_h_bond_possible(at1, at2, atp)){
            terms.hbond = weight_hbond * vina_hbond(d, &f_dummy);
            terms.total += terms.hbond;
        }
        return terms;
    }

    // The position-bias potentials (eval_ef_pos / eval_ef_zalign) are PositionBiasTerm::eval_pos / eval_zalign
    // in ext/bias/bias_term.h since plan step 5a.
};


// ==================== Angular HBond weight (GPU-safe) ====================

/**
 * @brief Angular weight for D-H...A hydrogen bonds, linear in cos(D-H...A).
 *
 * Uses dot products only (no trig). With cos_dha = dot(H->D, H->A) / (|H->D| * |H->A|),
 * ANGULAR_HBOND_COS_ZERO = cos(90 deg) and ANGULAR_HBOND_COS_FULL = cos(140 deg):
 *   w = clamp((ANGULAR_HBOND_COS_ZERO - cos_dha) / (ANGULAR_HBOND_COS_ZERO - ANGULAR_HBOND_COS_FULL), 0, 1)
 *
 * So w = 0 at 90 deg and below, rises linearly to w = 1 at 140 deg, and stays 1 from 140 to
 * 180 deg. The derivative is the constant -1 / (COS_ZERO - COS_FULL) inside the ramp and 0 in
 * both clamped regions. The caller multiplies the Vina distance-based hbond term by w, applied
 * in the pair loops as (w - 1) * E_hb because E_hb was already added.
 *
 * @param cos_dha  Cosine of the D-H...A angle
 * @param out_dw_dcos  Output: dw/d(cos_dha)
 * @return Angular weight w in [0, 1]
 */
SCOPE_INLINE Real angular_hbond_weight(Real cos_dha, Real* out_dw_dcos) {
    // Linear in cos between ANGULAR_HBOND_COS_ZERO (90 deg, w = 0) and ANGULAR_HBOND_COS_FULL (140 deg, w = 1).
    // Calibrated on the Uni-Dock-Benchmarks crystal poses: with 90 -> 140 deg fixed-donor
    // hydrogen bonds keep ~88 % of their Vina energy (the former 100 -> 180 deg quadratic
    // ramp kept 63 %, giving a 150 deg hydrogen bond only 70 % credit).
    const Real cos_zero = Real(ANGULAR_HBOND_COS_ZERO);
    const Real cos_full = Real(ANGULAR_HBOND_COS_FULL);
    const Real inv_range = Real(1.0) / (cos_zero - cos_full);

    Real t = (cos_zero - cos_dha) * inv_range;
    if (t <= Real(0)) {
        *out_dw_dcos = Real(0);
        return Real(0);
    }
    if (t >= Real(1)) {
        *out_dw_dcos = Real(0);
        return Real(1);
    }
    *out_dw_dcos = -inv_range;
    return t;
}


/**
 * @brief Donors whose hydrogen rotates freely about the donor bond: hydroxyl / thiol
 * (O with H), amine and ammonium (N with >= 2 H), water. Their prepared hydrogen position
 * carries no information, so the angle is evaluated with the best hydrogen on the bond
 * cone (hbond_cone_h) or taken as ideal when there is no heavy neighbour (water, metal).
 */
SCOPE_INLINE bool hbond_donor_rotatable(int vn_type, int n_bonded_h) {
    if (vn_type == VN_TYPE_O_DA || vn_type == VN_TYPE_O_D) return true;
    if ((vn_type == VN_TYPE_N_D || vn_type == VN_TYPE_N_DA) && n_bonded_h >= 2) return true;
    return false;
}

/**
 * @brief Refines hbond_donor_rotatable with the prepared geometry: an N with exactly two
 * hydrogens and one heavy neighbour X is a free rotor only when pyramidal (amine). Planar
 * -NH2 (amide, guanidinium, aniline-like: angle sum X-N-H1 + X-N-H2 + H1-N-H2 near 360 deg)
 * keeps its hydrogens where the preparation put them. Sum below 345 deg counts as pyramidal
 * (tetrahedral is 328.5 deg).
 */
SCOPE_INLINE bool hbond_donor_rotatable_geom(int vn_type, int n_bonded_h, const Real* D,
                                             const Real* X, const Real* H1, const Real* H2) {
    if (!hbond_donor_rotatable(vn_type, n_bonded_h)) return false;
    if (vn_type == VN_TYPE_O_DA || vn_type == VN_TYPE_O_D) return true;
    if (n_bonded_h != 2 || X == nullptr) return true;
    Real v[3][3];
    const Real* P[3] = {X, H1, H2};
    for (int k = 0; k < 3; k++) {
        v[k][0] = P[k][0] - D[0]; v[k][1] = P[k][1] - D[1]; v[k][2] = P[k][2] - D[2];
        Real n = sqrt(v[k][0] * v[k][0] + v[k][1] * v[k][1] + v[k][2] * v[k][2]);
        if (n < Real(1e-6)) return true;
        v[k][0] /= n; v[k][1] /= n; v[k][2] /= n;
    }
    Real sum = 0;
    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int q = 0; q < 3; q++) {
        Real c = v[pairs[q][0]][0] * v[pairs[q][1]][0] + v[pairs[q][0]][1] * v[pairs[q][1]][1]
               + v[pairs[q][0]][2] * v[pairs[q][1]][2];
        if (c > Real(1)) c = Real(1);
        if (c < Real(-1)) c = Real(-1);
        sum += acos(c);
    }
    return sum < Real(345.0 * 3.14159265358979 / 180.0);
}

/**
 * @brief Best achievable hydrogen position for a rotatable donor D with heavy neighbour X
 * toward acceptor A: 1.0 A from D, X-D-H = 109.5 deg, in the X-D-A plane on the acceptor
 * side. If A lies on the X-D axis the hydrogen is placed on an arbitrary cone direction.
 */
SCOPE_INLINE void hbond_cone_h(const Real* D, const Real* X, const Real* A, Real* out_h) {
    const Real cos_b = Real(-0.33380686);  // cos(109.47 deg)
    const Real sin_b = Real(0.94264149);
    Real u[3] = {X[0] - D[0], X[1] - D[1], X[2] - D[2]};
    Real nu = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (nu < Real(1e-6)) { out_h[0] = D[0] + 1; out_h[1] = D[1]; out_h[2] = D[2]; return; }
    u[0] /= nu; u[1] /= nu; u[2] /= nu;
    Real a[3] = {A[0] - D[0], A[1] - D[1], A[2] - D[2]};
    Real proj = a[0] * u[0] + a[1] * u[1] + a[2] * u[2];
    Real v[3] = {a[0] - proj * u[0], a[1] - proj * u[1], a[2] - proj * u[2]};
    Real nv = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (nv < Real(1e-6)) {
        // acceptor on the axis: any perpendicular direction
        Real t[3] = {1, 0, 0};
        if (fabs(u[0]) > Real(0.9)) { t[0] = 0; t[1] = 1; }
        v[0] = t[1] * u[2] - t[2] * u[1]; v[1] = t[2] * u[0] - t[0] * u[2]; v[2] = t[0] * u[1] - t[1] * u[0];
        nv = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    v[0] /= nv; v[1] /= nv; v[2] /= nv;
    out_h[0] = D[0] + cos_b * u[0] + sin_b * v[0];
    out_h[1] = D[1] + cos_b * u[1] + sin_b * v[1];
    out_h[2] = D[2] + cos_b * u[2] + sin_b * v[2];
}


/**
 * @brief cos(D-H...A) for the best hydrogen on the bond cone of rotatable donor D (heavy
 * neighbour X) toward acceptor A, with optional gradients w.r.t. D, X and A.
 * Same hydrogen as hbond_cone_h, expressed through the X-D-A angle phi and r = |DA|:
 * the D->H / D->A angle is alpha = |phi - beta| (beta = 109.47 deg), and the D-H-A triangle
 * (|DH| = 1) gives |HA|^2 = 1 + r^2 - 2 r cos(alpha), cos(theta) = (1 - r cos(alpha)) / |HA|.
 * Degenerate input (D on X or A) returns cos = 1 (no hydrogen bond credit) and zero gradients.
 * The angle gradient is skipped when A lies on the X-D axis (sin(phi) ~ 0); the weight is
 * already clamped to zero there for hydrogen-bond distances.
 */
SCOPE_INLINE Real hbond_cone_cos(const Real* D, const Real* X, const Real* A,
                                 Real* gD, Real* gX, Real* gA) {
    const Real cos_b = Real(-0.33380686);  // cos(109.47 deg)
    const Real sin_b = Real(0.94264149);
    Real u[3] = {X[0] - D[0], X[1] - D[1], X[2] - D[2]};
    Real a[3] = {A[0] - D[0], A[1] - D[1], A[2] - D[2]};
    Real nu = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    Real r = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    if (nu < Real(1e-6) || r < Real(1e-6)) {
        if (gD) { for (int k = 0; k < 3; k++) { gD[k] = 0; gX[k] = 0; gA[k] = 0; } }
        return Real(1);
    }
    for (int k = 0; k < 3; k++) { u[k] /= nu; a[k] /= r; }
    Real cos_phi = u[0] * a[0] + u[1] * a[1] + u[2] * a[2];
    if (cos_phi > Real(1)) cos_phi = Real(1);
    if (cos_phi < Real(-1)) cos_phi = Real(-1);
    Real sin_phi = sqrt(Real(1) - cos_phi * cos_phi);
    Real c = cos_phi * cos_b + sin_phi * sin_b;  // cos(alpha) = cos(phi - beta)
    Real h2 = Real(1) + r * r - Real(2) * r * c;
    if (h2 < Real(1e-12)) h2 = Real(1e-12);
    Real h = sqrt(h2);
    Real cos_t = (Real(1) - r * c) / h;
    if (gD == nullptr) return cos_t;

    Real h3 = h2 * h;
    Real dct_dr = (-c * h2 - (Real(1) - r * c) * (r - c)) / h3;
    Real dct_dc = r * r * (c - r) / h3;
    Real dc_dphi = -(sin_phi * cos_b - cos_phi * sin_b);  // -sin(phi - beta)
    Real kX = 0, kA = 0;
    if (sin_phi > Real(1e-4)) { kX = Real(1) / (nu * sin_phi); kA = Real(1) / (r * sin_phi); }
    Real s = dct_dc * dc_dphi;
    for (int k = 0; k < 3; k++) {
        Real dphi_dX = (cos_phi * u[k] - a[k]) * kX;
        Real dphi_dA = (cos_phi * a[k] - u[k]) * kA;
        gX[k] = s * dphi_dX;
        gA[k] = dct_dr * a[k] + s * dphi_dA;
        gD[k] = -(gX[k] + gA[k]);
    }
    return cos_t;
}


#endif //VINA_H
