//
// Unit tests for rescore_hbond (geometric H-bond validation).
// Includes rescore_hbond.h directly (NOT score.h) to avoid duplicate symbols.
//

#include <catch2/catch_amalgamated.hpp>
#include "score/rescore_hbond.h"
#include <cmath>


/// Helper: build minimal UDFlexMol + UDFixMol for rescore_hbond tests.
/// Ligand atom 0 is the heavy atom under test, atom 1 is H (if needed).
/// Receptor atom 0 is the partner.
static void build_hbond_test(
    UDFlexMol& flex, UDFixMol& fix,
    std::vector<Real>& flex_coords_vec,
    int lig_type, Real lig_pos[3],
    int h_type, Real h_pos[3],        // set h_type = -1 to skip H atom
    int rec_type, Real rec_pos[3])
{
    flex.natom = (h_type >= 0) ? 2 : 1;
    flex.coords.clear();
    flex.vina_types.clear();

    // atom 0: heavy atom
    flex.coords.push_back(lig_pos[0]);
    flex.coords.push_back(lig_pos[1]);
    flex.coords.push_back(lig_pos[2]);
    flex.vina_types.push_back(lig_type);

    // atom 1: H (optional)
    if (h_type >= 0){
        flex.coords.push_back(h_pos[0]);
        flex.coords.push_back(h_pos[1]);
        flex.coords.push_back(h_pos[2]);
        flex.vina_types.push_back(h_type);
    }

    // receptor: one atom
    fix.natom = 1;
    fix.coords = {rec_pos[0], rec_pos[1], rec_pos[2]};
    fix.vina_types = {rec_type};

    // inter pairs: ligand atom 0 <-> receptor atom 0
    flex.inter_pairs = {0, 0};

    // r1_plus_r2_inter (not used by rescore_hbond but needed by score_decomp)
    flex.r1_plus_r2_inter = {0};

    // copy flex coords into a flat array used by the scoring path
    flex_coords_vec.assign(flex.coords.begin(), flex.coords.end());
}


TEST_CASE("rescore_hbond: donor with good geometry keeps score", "[rescore_hbond]") {
    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    UDFlexMol flex;
    UDFixMol fix;
    std::vector<Real> coords;

    // D at origin, H at (0,0,1), A at (0,0,3)
    // H...A = 2.0 A < 3.0 A, D-H...A = 180 deg > 100 deg -> valid
    Real lig_pos[3] = {0, 0, 0};
    Real h_pos[3]   = {0, 0, 1};
    Real rec_pos[3] = {0, 0, 3};
    build_hbond_test(flex, fix, coords,
                     VN_TYPE_N_D, lig_pos,
                     VN_TYPE_H,   h_pos,
                     VN_TYPE_O_A, rec_pos);

    std::vector<VinaTerms> terms(flex.natom);
    terms[0].hbond = -0.5;
    terms[0].total = -0.5;

    rescore_hbond(terms, coords.data(), fix, flex, atp);

    REQUIRE_THAT(terms[0].hbond, Catch::Matchers::WithinAbs(-0.5, 1e-6));
    REQUIRE_THAT(terms[0].total, Catch::Matchers::WithinAbs(-0.5, 1e-6));
}


TEST_CASE("rescore_hbond: donor with H pointing away zeros score", "[rescore_hbond]") {
    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    UDFlexMol flex;
    UDFixMol fix;
    std::vector<Real> coords;

    // D at origin, H at (0,0,-1) (away), A at (0,0,3)
    // H...A = 4.0 A >= 3.0 A -> INVALID (fails distance)
    Real lig_pos[3] = {0, 0, 0};
    Real h_pos[3]   = {0, 0, -1};
    Real rec_pos[3] = {0, 0, 3};
    build_hbond_test(flex, fix, coords,
                     VN_TYPE_N_D, lig_pos,
                     VN_TYPE_H,   h_pos,
                     VN_TYPE_O_A, rec_pos);

    std::vector<VinaTerms> terms(flex.natom);
    terms[0].hbond = -0.5;
    terms[0].total = -0.3;

    rescore_hbond(terms, coords.data(), fix, flex, atp);

    REQUIRE_THAT(terms[0].hbond, Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(terms[0].total, Catch::Matchers::WithinAbs(0.2, 1e-4));
}


TEST_CASE("rescore_hbond: donor angle just above 100 deg keeps score", "[rescore_hbond]") {
    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    UDFlexMol flex;
    UDFixMol fix;
    std::vector<Real> coords;

    // D at origin, H at (0,0,1).
    // D-H...A angle = 110 deg (> 100 deg threshold) -> should be valid.
    // H->D = (0,0,-1). For cos(D-H...A) = cos(110) = -0.342:
    //   dot(H->D, H->A) / |H->A| = -0.342
    //   => -H->A_z / |H->A| = -0.342 => H->A_z / |H->A| = 0.342
    // At |H->A| = 2.5: H->A_z = 0.855, H->A_x = sqrt(2.5^2 - 0.855^2) = 2.349
    // A = H + (2.349, 0, 0.855) = (2.349, 0, 1.855)
    Real lig_pos[3] = {0, 0, 0};
    Real h_pos[3]   = {0, 0, 1};
    Real ha_dist = 2.5;
    Real cos110 = (Real)cos(110.0 * M_PI / 180.0); // -0.342
    Real ha_z = -ha_dist * cos110;  // 0.855 (note: dot(H->D,H->A) = -ha_z, so cos = -ha_z/|HA|)
    Real ha_x = (Real)sqrt(ha_dist * ha_dist - ha_z * ha_z);
    Real rec_pos[3] = {ha_x, 0, (Real)(1.0 + ha_z)};
    build_hbond_test(flex, fix, coords,
                     VN_TYPE_N_D, lig_pos,
                     VN_TYPE_H,   h_pos,
                     VN_TYPE_O_A, rec_pos);

    std::vector<VinaTerms> terms(flex.natom);
    terms[0].hbond = -0.4;
    terms[0].total = -0.4;

    rescore_hbond(terms, coords.data(), fix, flex, atp);

    // 110 deg > 100 deg threshold -> kept
    REQUIRE_THAT(terms[0].hbond, Catch::Matchers::WithinAbs(-0.4, 1e-6));
}


TEST_CASE("rescore_hbond: donor angle just below 100 deg zeros score", "[rescore_hbond]") {
    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    UDFlexMol flex;
    UDFixMol fix;
    std::vector<Real> coords;

    // D at origin, H at (0,0,1).
    // Place A so D-H...A = 90 deg < 100 deg -> invalid
    // cos(90) = 0.  A direction from H: (1, 0, 0). Distance 2.5 from H.
    // A = (2.5, 0, 1)
    Real lig_pos[3] = {0, 0, 0};
    Real h_pos[3]   = {0, 0, 1};
    Real rec_pos[3] = {2.5, 0, 1};
    build_hbond_test(flex, fix, coords,
                     VN_TYPE_N_D, lig_pos,
                     VN_TYPE_H,   h_pos,
                     VN_TYPE_O_A, rec_pos);

    std::vector<VinaTerms> terms(flex.natom);
    terms[0].hbond = -0.4;
    terms[0].total = -0.4;

    rescore_hbond(terms, coords.data(), fix, flex, atp);

    // 90 deg < 100 deg threshold -> zeroed
    REQUIRE_THAT(terms[0].hbond, Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(terms[0].total, Catch::Matchers::WithinAbs(0.0, 1e-4));
}


TEST_CASE("rescore_hbond: acceptor with nearby receptor donor keeps score", "[rescore_hbond]") {
    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    UDFlexMol flex;
    UDFixMol fix;
    std::vector<Real> coords;

    // Ligand O_A acceptor at origin, receptor N_D donor at (0,0,3.0)
    // D...A = 3.0 < 3.5 -> valid
    Real lig_pos[3] = {0, 0, 0};
    Real rec_pos[3] = {0, 0, 3.0};
    build_hbond_test(flex, fix, coords,
                     VN_TYPE_O_A, lig_pos,
                     -1, nullptr,
                     VN_TYPE_N_D, rec_pos);

    std::vector<VinaTerms> terms(flex.natom);
    terms[0].hbond = -0.4;
    terms[0].total = -0.4;

    rescore_hbond(terms, coords.data(), fix, flex, atp);

    REQUIRE_THAT(terms[0].hbond, Catch::Matchers::WithinAbs(-0.4, 1e-6));
    REQUIRE_THAT(terms[0].total, Catch::Matchers::WithinAbs(-0.4, 1e-6));
}


TEST_CASE("rescore_hbond: acceptor with distant receptor donor zeros score", "[rescore_hbond]") {
    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    UDFlexMol flex;
    UDFixMol fix;
    std::vector<Real> coords;

    // Ligand O_A acceptor at origin, receptor N_D donor at (0,0,4.0)
    // D...A = 4.0 > 3.5 -> INVALID
    Real lig_pos[3] = {0, 0, 0};
    Real rec_pos[3] = {0, 0, 4.0};
    build_hbond_test(flex, fix, coords,
                     VN_TYPE_O_A, lig_pos,
                     -1, nullptr,
                     VN_TYPE_N_D, rec_pos);

    std::vector<VinaTerms> terms(flex.natom);
    terms[0].hbond = -0.4;
    terms[0].total = -0.2;

    rescore_hbond(terms, coords.data(), fix, flex, atp);

    REQUIRE_THAT(terms[0].hbond, Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(terms[0].total, Catch::Matchers::WithinAbs(0.2, 1e-4));
}


TEST_CASE("rescore_hbond: donor+acceptor (N_DA) tries both checks", "[rescore_hbond]") {
    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    UDFlexMol flex;
    UDFixMol fix;
    std::vector<Real> coords;

    // Ligand N_DA at origin, H at (0,0,-1) (pointing AWAY from receptor)
    // Receptor O_DA at (0,0,3.0)
    //
    // Donor check: H...A = 4.0 >= 3.0 -> FAILS distance
    // Acceptor check: O_DA is_donor, D...A = 3.0 < 3.5 -> PASSES
    Real lig_pos[3] = {0, 0, 0};
    Real h_pos[3]   = {0, 0, -1};
    Real rec_pos[3] = {0, 0, 3.0};
    build_hbond_test(flex, fix, coords,
                     VN_TYPE_N_DA, lig_pos,
                     VN_TYPE_H,    h_pos,
                     VN_TYPE_O_DA, rec_pos);

    std::vector<VinaTerms> terms(flex.natom);
    terms[0].hbond = -0.3;
    terms[0].total = -0.3;

    rescore_hbond(terms, coords.data(), fix, flex, atp);

    REQUIRE_THAT(terms[0].hbond, Catch::Matchers::WithinAbs(-0.3, 1e-6));
}


TEST_CASE("rescore_hbond: donor-only with no bonded H zeros score", "[rescore_hbond]") {
    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    UDFlexMol flex;
    UDFixMol fix;
    std::vector<Real> coords;

    // N_D donor at origin, NO H atom at all, receptor O_A at (0,0,2.5)
    // Vina assigned hbond=-0.5 based on distance, but rescore can't find
    // any bonded H -> donor check skipped, N_D is not acceptor -> zeroed
    Real lig_pos[3] = {0, 0, 0};
    Real rec_pos[3] = {0, 0, 2.5};
    build_hbond_test(flex, fix, coords,
                     VN_TYPE_N_D, lig_pos,
                     -1, nullptr,          // no H atom
                     VN_TYPE_O_A, rec_pos);

    std::vector<VinaTerms> terms(flex.natom);
    terms[0].hbond = -0.5;
    terms[0].total = -0.5;

    rescore_hbond(terms, coords.data(), fix, flex, atp);

    // N_D has no bonded H -> donor check skipped, not acceptor -> zeroed
    REQUIRE_THAT(terms[0].hbond, Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(terms[0].total, Catch::Matchers::WithinAbs(0.0, 1e-4));
}


TEST_CASE("rescore_hbond: donor-only with distant H zeros score", "[rescore_hbond]") {
    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    UDFlexMol flex;
    UDFixMol fix;
    std::vector<Real> coords;

    // N_D donor at origin, H at (0,0,2.0) — beyond 1.3 A detection radius
    // Receptor O_A at (0,0,3.0).  D...A = 3.0 A, H...A = 1.0 A
    // But H is NOT detected as bonded (2.0 > 1.3), so donor check skipped
    Real lig_pos[3] = {0, 0, 0};
    Real h_pos[3]   = {0, 0, 2};
    Real rec_pos[3] = {0, 0, 3};
    build_hbond_test(flex, fix, coords,
                     VN_TYPE_N_D, lig_pos,
                     VN_TYPE_H,   h_pos,
                     VN_TYPE_O_A, rec_pos);

    std::vector<VinaTerms> terms(flex.natom);
    terms[0].hbond = -0.5;
    terms[0].total = -0.5;

    rescore_hbond(terms, coords.data(), fix, flex, atp);

    // H at 2.0 A from donor > 1.3 A detection radius -> not found as bonded
    // N_D is donor-only (not acceptor) -> both cases skipped -> zeroed
    REQUIRE_THAT(terms[0].hbond, Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(terms[0].total, Catch::Matchers::WithinAbs(0.0, 1e-4));
}


TEST_CASE("rescore_hbond: zero hbond atoms are skipped", "[rescore_hbond]") {
    AtomTypeProps atp;
    init_default_atom_type_props(atp);

    UDFlexMol flex;
    UDFixMol fix;
    std::vector<Real> coords;

    Real lig_pos[3] = {0, 0, 0};
    Real rec_pos[3] = {0, 0, 5};
    build_hbond_test(flex, fix, coords,
                     VN_TYPE_N_D, lig_pos,
                     -1, nullptr,
                     VN_TYPE_O_A, rec_pos);

    std::vector<VinaTerms> terms(flex.natom);
    terms[0].hbond = 0;
    terms[0].total = -0.1;

    rescore_hbond(terms, coords.data(), fix, flex, atp);

    REQUIRE_THAT(terms[0].hbond, Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(terms[0].total, Catch::Matchers::WithinAbs(-0.1, 1e-6));
}
