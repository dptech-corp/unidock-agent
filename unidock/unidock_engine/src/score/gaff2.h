//
// Force field energy and gradient functions for GAFF2/AMBER scoring.
// Used during FF minimization step on CPU.
//
// All functions return energy and set *dE_dr (or *dE_dphi) for gradient computation.
// Units: energy in kcal/mol, distances in Angstrom, angles in radians.
//

#ifndef FF_GAFF2_H
#define FF_GAFF2_H

#include "myutils/common.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Coulomb constant: 332.0637 kcal*A/(mol*e^2)
static constexpr Real COULOMB_CONST = 332.0637;

// Distance-dependent dielectric scaling factor (eps = DDD_FACTOR * r)
static constexpr Real DDD_FACTOR = 4.0;


/**
 * @brief LJ 12-6 energy and force.
 *
 * E = 4*eps * [(sig/r)^12 - (sig/r)^6]
 * dE/dr = 4*eps * [-12*(sig/r)^12/r + 6*(sig/r)^6/r]
 *
 * Caller applies Lorentz-Berthelot combining rules:
 *   sig_ij = (sig_i + sig_j) / 2
 *   eps_ij = sqrt(eps_i * eps_j)
 *
 * @param r          distance (Angstrom)
 * @param sigma_ij   combined sigma (Angstrom)
 * @param eps_ij     combined epsilon (kcal/mol)
 * @param dE_dr      [out] derivative dE/dr
 * @return energy (kcal/mol)
 */
inline Real ff_lj(Real r, Real sigma_ij, Real eps_ij, Real* dE_dr) {
    Real inv_r = 1.0f / r;
    Real sig_over_r = sigma_ij * inv_r;
    Real s6 = sig_over_r * sig_over_r * sig_over_r;
    s6 = s6 * s6;  // (sig/r)^6
    Real s12 = s6 * s6;

    Real e4 = 4.0f * eps_ij;
    *dE_dr = e4 * (-12.0f * s12 + 6.0f * s6) * inv_r;
    return e4 * (s12 - s6);
}


/**
 * @brief Coulomb energy with distance-dependent dielectric (eps = 4*r).
 *
 * E = COULOMB_CONST * qi * qj / (DDD_FACTOR * r^2)
 * dE/dr = -2 * COULOMB_CONST * qi * qj / (DDD_FACTOR * r^3)
 *
 * @param r      distance (Angstrom)
 * @param qi     charge of atom i (elementary charge)
 * @param qj     charge of atom j (elementary charge)
 * @param dE_dr  [out] derivative dE/dr
 * @return energy (kcal/mol)
 */
inline Real ff_coulomb(Real r, Real qi, Real qj, Real* dE_dr) {
    Real inv_r = 1.0f / r;
    Real inv_r2 = inv_r * inv_r;
    Real prefactor = COULOMB_CONST * qi * qj / DDD_FACTOR;

    *dE_dr = -2.0f * prefactor * inv_r * inv_r2;
    return prefactor * inv_r2;
}


/**
 * @brief Pre-scaled 1-4 pair interaction from DMS pair_12_6_es table.
 *
 * LJ part in A/B form (already scaled):
 *   E_lj = aij / r^12 - bij / r^6
 *   dE_lj/dr = -12*aij/r^13 + 6*bij/r^7
 *
 * Coulomb part (qij is pre-scaled charge product):
 *   E_coul = COULOMB_CONST * qij / (DDD_FACTOR * r^2)
 *
 * @param r      distance (Angstrom)
 * @param aij    LJ A coefficient (kcal/mol * A^12)
 * @param bij    LJ B coefficient (kcal/mol * A^6)
 * @param qij    pre-scaled charge product
 * @param dE_dr  [out] derivative dE/dr
 * @return energy (kcal/mol)
 */
inline Real ff_pair14(Real r, Real aij, Real bij, Real qij, Real* dE_dr) {
    Real inv_r = 1.0f / r;
    Real inv_r2 = inv_r * inv_r;
    Real inv_r6 = inv_r2 * inv_r2 * inv_r2;
    Real inv_r12 = inv_r6 * inv_r6;

    Real e_lj = aij * inv_r12 - bij * inv_r6;
    Real de_lj = (-12.0f * aij * inv_r12 + 6.0f * bij * inv_r6) * inv_r;

    Real coul_pre = COULOMB_CONST * qij / DDD_FACTOR;
    Real e_coul = coul_pre * inv_r2;
    Real de_coul = -2.0f * coul_pre * inv_r * inv_r2;

    *dE_dr = de_lj + de_coul;
    return e_lj + e_coul;
}


/**
 * @brief Periodic torsion energy and gradient (AMBER/GAFF2 convention).
 *
 * E = fc0 + sum_{n=1..6} fc_n * (1 + cos(n*phi - phi0))
 * dE/dphi = sum_{n=1..6} -n * fc_n * sin(n*phi - phi0)
 *
 * @param phi       current torsion angle (radians)
 * @param phi0      phase angle (radians)
 * @param fc        array of 7 force constants: fc[0]..fc[6] (kcal/mol)
 * @param dE_dphi   [out] derivative dE/dphi
 * @return energy (kcal/mol)
 */
inline Real ff_torsion(Real phi, Real phi0, const Real* fc, Real* dE_dphi) {
    Real e = fc[0];  // constant term (fc0)
    Real de = 0;

    for (int n = 1; n <= 6; n++) {
        if (fc[n] == 0) continue;
        Real angle = n * phi - phi0;
        e  += fc[n] * (1.0f + cos(angle));
        de += -static_cast<Real>(n) * fc[n] * sin(angle);
    }

    *dE_dphi = de;
    return e;
}


/**
 * @brief Harmonic bond stretch energy and gradient (fc absorbs the 1/2 factor).
 *
 * E = fc * (r - r0)^2     (fc already absorbs the 1/2 prefactor)
 * dE/dr = 2 * fc * (r - r0)
 */
inline Real ff_bond(Real r, Real r0, Real fc, Real* dE_dr) {
    Real dr = r - r0;
    *dE_dr = 2.0f * fc * dr;
    return fc * dr * dr;
}


/**
 * @brief Harmonic angle bend energy and gradient (fc absorbs the 1/2 factor).
 *
 * E = fc * (theta - theta0)^2     (theta in radians, fc absorbs 1/2)
 * dE/dtheta = 2 * fc * (theta - theta0)
 */
inline Real ff_angle(Real theta, Real theta0, Real fc, Real* dE_dtheta) {
    Real dt = theta - theta0;
    *dE_dtheta = 2.0f * fc * dt;
    return fc * dt * dt;
}


/**
 * @brief Per-atom FF energy decomposition into 5 terms + total.
 * Mirrors VinaTerms for direct comparison.
 */
struct FFAtomTerms {
    Real inter_lj = 0;
    Real inter_coulomb = 0;
    Real intra_lj = 0;
    Real intra_coulomb = 0;
    Real torsion = 0;
    Real bond = 0;
    Real angle = 0;
    Real total = 0;

    void operator+=(const FFAtomTerms& other) {
        inter_lj      += other.inter_lj;
        inter_coulomb  += other.inter_coulomb;
        intra_lj      += other.intra_lj;
        intra_coulomb  += other.intra_coulomb;
        torsion       += other.torsion;
        bond          += other.bond;
        angle         += other.angle;
        total         += other.total;
    }
};


#endif // FF_GAFF2_H
