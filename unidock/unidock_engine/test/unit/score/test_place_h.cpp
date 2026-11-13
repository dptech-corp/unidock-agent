//
// Post-refinement placement of cone-donor hydrogens (find_placeable_cone_donors + place_cone_donor_h):
// the hydrogens of free-rotor donors turn about the X-D bond toward the receptor acceptors in range,
// heavy atoms and every score() number stay bit-identical, and the guarded donor classes (planar
// -NH2, single N-H, R2NH2+, foreign-H lists, non-aromatic sp2 O-H, phenol beyond the flip, atoms
// beyond the written ones, acceptors a receptor donor already credits) keep their input hydrogens.
// A turn never brings an H within 2.0 A of another ligand H, and with several acceptors the result
// is at least the best 1 deg turn.
//
// score.h defines non-inline globals that DockTask.o already carries, so score() is declared here
// instead of including score.h (see test_score_decomp.cpp).
//

#include <catch2/catch_amalgamated.hpp>
#include <cmath>
#include <cstring>
#include <vector>

#include "model/model.h"
#include "score/vina.h"
#include "cpu/common_cpu.h"
#include "cpu/score_cpu.h"
#include "constants/constants.h"

using Catch::Approx;

void score(FlexPose* out_pose, const Real* flex_coords, const UDFixMol& udfix_mol, const UDFlexMol& udflex_mol,
           const Box& box, const AtomTypeProps& atp, bool angular_hbond, const BondedHTopo* h_topo,
           const RecDonorHTopo* rh_topo, ScoreDiag* out_diag);

namespace {

const Real COS_B = Real(0.33380686);  // -cos(109.47 deg): cone hydrogens lean this far away from X
const Real SIN_B = Real(0.94264149);
const Real DEG = Real(3.14159265358979 / 180.0);

const AtomTypeProps& test_atp(){
    init_default_atom_type_props(g_cpu_constants.atp);
    return g_cpu_constants.atp;
}

UDFlexMol make_mol(const std::vector<Real>& coords, const std::vector<int>& types){
    UDFlexMol m;
    m.natom = static_cast<int>(types.size());
    m.coords = coords;
    m.vina_types = types;
    return m;
}

UDFixMol make_fix(const std::vector<Real>& coords, const std::vector<int>& types){
    UDFixMol f;
    f.natom = static_cast<int>(types.size());
    f.coords = coords;
    f.vina_types = types;
    return f;
}

std::vector<int> acceptors_of(const UDFixMol& fix, const AtomTypeProps& atp){
    std::vector<int> acc;
    for (int a = 0; a < fix.natom; a++){
        if (atp.is_acceptor[fix.vina_types[a]]) acc.push_back(a);
    }
    return acc;
}

/// The engine path for one pose: donors from the input conformer, then placement on `pose`.
/// Returns the number of donors moved; `n_h` receives the number of hydrogens moved.
int place(const UDFlexMol& mol, std::vector<Real>& pose, const UDFixMol& fix, const AtomTypeProps& atp,
          int* n_h = nullptr){
    const std::vector<PlaceHDonor> donors = find_placeable_cone_donors(mol, mol.natom, atp);
    int n_h_moved = 0;
    const int n = place_cone_donor_h(pose.data(), mol.natom, donors, fix, acceptors_of(fix, atp), VN_VDW_RADII, atp,
                                     mol.vina_types.data(), n_h_moved);
    if (n_h) *n_h = n_h_moved;
    return n;
}

/// cos of the angle a-b-c at b
Real cos_angle(const Real* a, const Real* b, const Real* c){
    Real u[3] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    Real v[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
    return (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (dist(a, b) * dist(c, b));
}

/// angular weight of D-H...A at an explicit hydrogen
Real explicit_weight(const Real* H, const Real* D, const Real* A){
    Real dw = 0;
    return angular_hbond_weight(cos_angle(D, H, A), &dw);
}

/// Placement objective of donor d: sum over receptor acceptors of ramp * best explicit-H weight
Real objective(const std::vector<Real>& c, int d, const std::vector<int>& hs, const UDFixMol& fix,
               const UDFlexMol& mol, const AtomTypeProps& atp){
    Vina v;
    Real f = 0, total = 0;
    const Real* D = c.data() + d * 3;
    for (int a = 0; a < fix.natom; a++){
        const int ta = fix.vina_types[a];
        if (!atp.is_acceptor[ta]) continue;
        const Real* A = fix.coords.data() + a * 3;
        const Real ramp = v.vina_hbond(dist(D, A) - (VN_VDW_RADII[mol.vina_types[d]] + VN_VDW_RADII[ta]), &f);
        if (ramp <= 0) continue;
        Real best = 0;
        for (int h : hs) best = std::max(best, explicit_weight(c.data() + h * 3, D, A));
        total += ramp * best;
    }
    return total;
}

/// Hydrogens `hs` of `c` turned by theta about the +x axis (right-handed), the X-D axis of the donors below
std::vector<Real> turn_about_x(const std::vector<Real>& c, const std::vector<int>& hs, Real theta){
    std::vector<Real> out = c;
    for (int h : hs){
        const Real y = c[h * 3 + 1], z = c[h * 3 + 2];
        out[h * 3 + 1] = std::cos(theta) * y - std::sin(theta) * z;
        out[h * 3 + 2] = std::sin(theta) * y + std::cos(theta) * z;
    }
    return out;
}

/// Hydroxyl C-O-H: O at the origin, C on -x, H on the 109.47 deg / 1.0 A cone at -y
UDFlexMol hydroxyl(){
    return make_mol({Real(-1.43), 0, 0,
                     0, 0, 0,
                     COS_B, -SIN_B, 0},
                    {VN_TYPE_C_P, VN_TYPE_O_DA, VN_TYPE_H});
}

/// Receptor point at distance r from the origin (the donor O), `polar` deg from the C-O direction (+x)
/// and `azimuth` deg about it from +y toward +z
std::vector<Real> toward(Real r, Real polar, Real azimuth){
    return {r * std::cos(polar * DEG), r * std::sin(polar * DEG) * std::cos(azimuth * DEG),
            r * std::sin(polar * DEG) * std::sin(azimuth * DEG)};
}

}  // namespace


TEST_CASE("place_h: hydroxyl turns its H toward an acceptor in range", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    UDFlexMol mol = hydroxyl();
    // O_A 2.8 A from O, 20 deg off the C-O axis on the +y side and 30 deg out of the C-O-H plane: the
    // cone weight sits inside the 90-140 deg ramp, and the H must turn 120 deg (not the sign-blind flip)
    UDFixMol fix = make_fix(toward(Real(2.8), 20, 30), {VN_TYPE_O_A});
    std::vector<Real> pose = mol.coords;

    int n_h = 0;
    REQUIRE(place(mol, pose, fix, atp, &n_h) == 1);
    REQUIRE(n_h == 1);
    REQUIRE(std::memcmp(pose.data(), mol.coords.data(), 6 * sizeof(Real)) == 0);
    const Real* C = pose.data();
    const Real* O = pose.data() + 3;
    const Real* H = pose.data() + 6;
    REQUIRE(dist(O, H) == Approx(dist(mol.coords.data() + 3, mol.coords.data() + 6)).margin(1e-5));
    REQUIRE(cos_angle(C, O, H) == Approx(cos_angle(mol.coords.data(), mol.coords.data() + 3, mol.coords.data() + 6)).margin(1e-5));

    Real dw = 0;
    const Real w_cone = angular_hbond_weight(hbond_cone_cos(O, C, fix.coords.data(), nullptr, nullptr, nullptr), &dw);
    REQUIRE(w_cone > Real(0.1));
    REQUIRE(w_cone < Real(0.9));
    REQUIRE(explicit_weight(H, O, fix.coords.data()) == Approx(w_cone).margin(1e-5));
}


TEST_CASE("place_h: score() is bit-identical before and after placement", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    UDFlexMol mol = hydroxyl();
    UDFixMol fix = make_fix(toward(Real(2.8), 20, 30), {VN_TYPE_O_A});
    // inter pairs as main.cpp builds them: every ligand heavy atom x every receptor atom
    mol.inter_pairs = {0, 0, 1, 0};
    mol.r1_plus_r2_inter = {VN_VDW_RADII[VN_TYPE_C_P] + VN_VDW_RADII[VN_TYPE_O_A],
                            VN_VDW_RADII[VN_TYPE_O_DA] + VN_VDW_RADII[VN_TYPE_O_A]};
    Box box;
    box.x_lo = box.y_lo = box.z_lo = -50;
    box.x_hi = box.y_hi = box.z_hi = 50;
    BondedHTopo h_topo = {};
    build_bonded_h_topo(h_topo, mol.coords.data(), mol.vina_types.data(), mol.natom);

    std::vector<Real> coords = mol.coords;
    FlexPose pose;
    pose.coords = coords.data();
    pose.dihedrals = nullptr;
    score(&pose, coords.data(), fix, mol, box, atp, false, &h_topo, &fix.rec_donor_h, nullptr);
    const Real inter_plain = pose.center[1];
    score(&pose, coords.data(), fix, mol, box, atp, true, &h_topo, &fix.rec_donor_h, nullptr);
    Real before[3] = {pose.center[0], pose.center[1], pose.center[2]};
    REQUIRE(before[1] != inter_plain);  // the angular weight acts on the O...O_A hydrogen bond

    REQUIRE(place(mol, coords, fix, atp) == 1);
    score(&pose, coords.data(), fix, mol, box, atp, true, &h_topo, &fix.rec_donor_h, nullptr);
    REQUIRE(std::memcmp(before, pose.center, sizeof(before)) == 0);
    free_bonded_h_topo(h_topo);
}


TEST_CASE("place_h: acceptor out of hydrogen-bond range leaves the H", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    UDFlexMol mol = hydroxyl();
    UDFixMol fix = make_fix(toward(Real(5.0), 20, 30), {VN_TYPE_O_A});
    std::vector<Real> pose = mol.coords;
    REQUIRE(place(mol, pose, fix, atp) == 0);
    REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
}


TEST_CASE("place_h: H already at the cone optimum stays byte-identical", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    UDFlexMol mol = hydroxyl();
    UDFixMol fix = make_fix(toward(Real(2.8), 20, 30), {VN_TYPE_O_A});
    hbond_cone_h(mol.coords.data() + 3, mol.coords.data(), fix.coords.data(), mol.coords.data() + 6);
    std::vector<Real> pose = mol.coords;
    REQUIRE(place(mol, pose, fix, atp) == 0);
    REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
}


TEST_CASE("place_h: planar -NH2 and a single N-H are never placed", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    // acceptor out of the N plane, in range of N
    UDFixMol fix = make_fix({Real(1.0), 0, Real(2.7)}, {VN_TYPE_O_A});

    SECTION("planar -NH2 (angle sum 360 deg)") {
        UDFlexMol mol = make_mol({Real(-1.33), 0, 0,
                                  0, 0, 0,
                                  Real(1.01 * 0.5), Real(1.01 * 0.866), 0,
                                  Real(1.01 * 0.5), Real(-1.01 * 0.866), 0},
                                 {VN_TYPE_C_P, VN_TYPE_N_D, VN_TYPE_H, VN_TYPE_H});
        REQUIRE(find_placeable_cone_donors(mol, mol.natom, atp).empty());
        std::vector<Real> pose = mol.coords;
        REQUIRE(place(mol, pose, fix, atp) == 0);
        REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
    }

    SECTION("amide N-H with one hydrogen") {
        UDFlexMol mol = make_mol({Real(-1.33), 0, 0,
                                  0, 0, 0,
                                  Real(1.01 * 0.5), Real(1.01 * 0.866), 0,
                                  Real(1.46 * 0.5), Real(-1.46 * 0.866), 0},
                                 {VN_TYPE_C_P, VN_TYPE_N_D, VN_TYPE_H, VN_TYPE_C_H});
        REQUIRE(find_placeable_cone_donors(mol, mol.natom, atp).empty());
        std::vector<Real> pose = mol.coords;
        REQUIRE(place(mol, pose, fix, atp) == 0);
        REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
    }
}


TEST_CASE("place_h: R2NH2+ (two heavy neighbours) is not placed", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    // tetrahedral N: two carbons and two hydrogens
    const Real s = Real(1.0 / std::sqrt(3.0));
    UDFlexMol mol = make_mol({0, 0, 0,
                              Real(1.49) * s, Real(1.49) * s, Real(1.49) * s,
                              Real(1.49) * s, -Real(1.49) * s, -Real(1.49) * s,
                              -Real(1.03) * s, Real(1.03) * s, -Real(1.03) * s,
                              -Real(1.03) * s, -Real(1.03) * s, Real(1.03) * s},
                             {VN_TYPE_N_D, VN_TYPE_C_P, VN_TYPE_C_P, VN_TYPE_H, VN_TYPE_H});
    BondedHTopo h_topo = {};
    build_bonded_h_topo(h_topo, mol.coords.data(), mol.vina_types.data(), mol.natom);
    REQUIRE(h_topo.donor_rotatable[0] == 1);  // scored as a cone donor, so only the heavy-neighbour guard excludes it
    free_bonded_h_topo(h_topo);

    UDFixMol fix = make_fix({Real(-2.9), 0, 0}, {VN_TYPE_O_A});
    REQUIRE(find_placeable_cone_donors(mol, mol.natom, atp).empty());
    std::vector<Real> pose = mol.coords;
    REQUIRE(place(mol, pose, fix, atp) == 0);
    REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
}


TEST_CASE("place_h: hydroxyl whose H list holds a foreign H is not placed", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    // C-H hydrogen 1.1 A from O: listed by O and by its own carbon (2.19 A from O, beyond 2.0 A)
    UDFlexMol mol = make_mol({Real(-1.43), 0, 0,
                              0, 0, 0,
                              COS_B, -SIN_B, 0,
                              0, 0, Real(2.19),
                              0, 0, Real(1.1)},
                             {VN_TYPE_C_P, VN_TYPE_O_DA, VN_TYPE_H, VN_TYPE_C_H, VN_TYPE_H});
    BondedHTopo h_topo = {};
    build_bonded_h_topo(h_topo, mol.coords.data(), mol.vina_types.data(), mol.natom);
    REQUIRE(h_topo.h_start[2] - h_topo.h_start[1] == 2);
    REQUIRE(h_topo.donor_rotatable[1] == 1);
    free_bonded_h_topo(h_topo);

    UDFixMol fix = make_fix(toward(Real(2.8), 20, 30), {VN_TYPE_O_A});
    REQUIRE(find_placeable_cone_donors(mol, mol.natom, atp).empty());
    std::vector<Real> pose = mol.coords;
    REQUIRE(place(mol, pose, fix, atp) == 0);
    REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
}


TEST_CASE("place_h: -NH3+ turns as one rotor toward two acceptors", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    // C on -x, three H on the cone at azimuths 0 / 120 / 240 deg about the C-N axis
    std::vector<Real> coords = {Real(-1.47), 0, 0, 0, 0, 0};
    for (Real az : {Real(0), Real(120), Real(240)}){
        coords.insert(coords.end(), {Real(1.03) * COS_B, Real(1.03) * SIN_B * std::cos(az * DEG),
                                     Real(1.03) * SIN_B * std::sin(az * DEG)});
    }
    UDFlexMol mol = make_mol(coords, {VN_TYPE_C_P, VN_TYPE_N_D, VN_TYPE_H, VN_TYPE_H, VN_TYPE_H});
    // two acceptors on the cone directions at azimuths 60 and 180 deg, halfway between the H
    std::vector<Real> rec;
    for (Real az : {Real(60), Real(180)}){
        rec.insert(rec.end(), {Real(2.9) * COS_B, Real(2.9) * SIN_B * std::cos(az * DEG),
                               Real(2.9) * SIN_B * std::sin(az * DEG)});
    }
    UDFixMol fix = make_fix(rec, {VN_TYPE_O_A, VN_TYPE_O_A});

    const std::vector<int> hs = {2, 3, 4};
    std::vector<Real> pose = mol.coords;
    const Real f_before = objective(pose, 1, hs, fix, mol, atp);
    int n_h = 0;
    REQUIRE(place(mol, pose, fix, atp, &n_h) == 1);
    REQUIRE(n_h == 3);
    REQUIRE(objective(pose, 1, hs, fix, mol, atp) > f_before);
    REQUIRE(std::memcmp(pose.data(), mol.coords.data(), 6 * sizeof(Real)) == 0);
    const Real* N = pose.data() + 3;
    for (int h : hs){
        REQUIRE(dist(N, pose.data() + h * 3) == Approx(dist(mol.coords.data() + 3, mol.coords.data() + h * 3)).margin(1e-5));
    }
    for (int a = 0; a < 3; a++){
        for (int b = a + 1; b < 3; b++){
            REQUIRE(dist(pose.data() + hs[a] * 3, pose.data() + hs[b] * 3)
                    == Approx(dist(mol.coords.data() + hs[a] * 3, mol.coords.data() + hs[b] * 3)).margin(1e-5));
        }
    }
}


TEST_CASE("place_h: phenol O-H is only ever flipped in the ring plane", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    // aromatic C1 with ring neighbours C2 / C6 at 120 deg, O on +x, H in the ring plane on +y
    UDFlexMol mol = make_mol({Real(-1.36), 0, 0,
                              Real(-1.36 - 1.39 * 0.5), Real(1.39 * 0.866), 0,
                              Real(-1.36 - 1.39 * 0.5), Real(-1.39 * 0.866), 0,
                              0, 0, 0,
                              Real(0.96) * COS_B, Real(0.96) * SIN_B, 0},
                             {VN_TYPE_C_P, VN_TYPE_C_H, VN_TYPE_C_H, VN_TYPE_O_DA, VN_TYPE_H});
    mol.aromatic = {1, 1, 1, 0, 0};
    const std::vector<PlaceHDonor> donors = find_placeable_cone_donors(mol, mol.natom, atp);
    REQUIRE(donors.size() == 1);
    REQUIRE(donors[0].d == 3);
    REQUIRE(donors[0].flip_only);

    const Real flipped[3] = {Real(0.96) * COS_B, -Real(0.96) * SIN_B, 0};
    int n_flip = 0, n_kept = 0;
    for (int az = 0; az < 360; az += 30){
        // acceptor 2.8 A from O on the cone direction at azimuth az about the C-O axis (0 = the H side)
        UDFixMol fix = make_fix({Real(2.8) * COS_B, Real(2.8) * SIN_B * std::cos(az * DEG),
                                 Real(2.8) * SIN_B * std::sin(az * DEG)}, {VN_TYPE_O_A});
        std::vector<Real> pose = mol.coords;
        const int n = place(mol, pose, fix, atp);
        INFO("azimuth " << az);
        if (n == 0){
            REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
            n_kept++;
            continue;
        }
        REQUIRE(n == 1);
        REQUIRE(std::memcmp(pose.data(), mol.coords.data(), 12 * sizeof(Real)) == 0);
        for (int c = 0; c < 3; c++) REQUIRE(pose[12 + c] == Approx(flipped[c]).margin(1e-5));
        n_flip++;
    }
    REQUIRE(n_flip > 0);
    REQUIRE(n_kept > 0);

    // two acceptors at azimuths 60 / 120 deg: an out-of-plane turn would gain, but there is no 1 deg scan,
    // and the flip only ties with the input H by symmetry
    std::vector<Real> rec;
    for (Real az : {Real(60), Real(120)}){
        rec.insert(rec.end(), {Real(2.8) * COS_B, Real(2.8) * SIN_B * std::cos(az * DEG),
                               Real(2.8) * SIN_B * std::sin(az * DEG)});
    }
    UDFixMol two = make_fix(rec, {VN_TYPE_O_A, VN_TYPE_O_A});
    REQUIRE(objective(turn_about_x(mol.coords, {4}, PI / 2), 3, {4}, two, mol, atp)
            > objective(mol.coords, 3, {4}, two, mol, atp) + Real(1e-3));
    std::vector<Real> pose = mol.coords;
    REQUIRE(place(mol, pose, two, atp) == 0);
    REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
}


TEST_CASE("place_h: carboxylic acid O-H on a non-aromatic sp2 C is not placed", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    // carboxyl C on -x with =O at 123 deg and R at -118.5 deg from the C-OH bond, planar; syn H
    const Real c_x = Real(-1.31);
    UDFlexMol mol = make_mol({c_x, 0, 0,
                              c_x + Real(1.21) * std::cos(123 * DEG), Real(1.21) * std::sin(123 * DEG), 0,
                              c_x + Real(1.50) * std::cos(Real(-118.5) * DEG), Real(1.50) * std::sin(Real(-118.5) * DEG), 0,
                              0, 0, 0,
                              Real(0.97) * COS_B, Real(0.97) * SIN_B, 0},
                             {VN_TYPE_C_P, VN_TYPE_O_A, VN_TYPE_C_H, VN_TYPE_O_DA, VN_TYPE_H});
    mol.aromatic = {0, 0, 0, 0, 0};
    // acceptor on the anti side, where a turn would gain
    UDFixMol fix = make_fix(toward(Real(2.8), 20, 180), {VN_TYPE_O_A});
    REQUIRE(find_placeable_cone_donors(mol, mol.natom, atp).empty());
    std::vector<Real> pose = mol.coords;
    REQUIRE(place(mol, pose, fix, atp) == 0);
    REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
}


TEST_CASE("place_h: acceptor credited by a receptor cone donor is skipped", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    UDFlexMol mol = hydroxyl();
    // Ser-like OG 2.8 A from the ligand O with its CB placed so that the receptor cone H points
    // straight at the ligand O (CB-OG...O = 109.47 deg): direction 2 gives full credit
    const std::vector<Real> og = toward(Real(2.8), 20, 0);
    const Real ux = std::cos(20 * DEG), uy = std::sin(20 * DEG);
    std::vector<Real> rec = og;
    rec.insert(rec.end(), {og[0] + Real(1.43) * COS_B * ux, og[1] + Real(1.43) * COS_B * uy, Real(1.43) * SIN_B});
    std::vector<int> rh_start = {0, 1, 1};
    std::vector<Real> rh_coords = {og[0], og[1], Real(-0.96)};
    std::vector<int> rh_antecedent = {1, -1};
    std::vector<int> rh_rot = {1, 0};

    SECTION("receptor O_DA cone donor with an antecedent: skipped") {
        UDFixMol fix = make_fix(rec, {VN_TYPE_O_DA, VN_TYPE_C_P});
        fix.rec_donor_h.h_start = rh_start.data();
        fix.rec_donor_h.h_coords = rh_coords.data();
        fix.rec_donor_h.antecedent = rh_antecedent.data();
        fix.rec_donor_h.donor_rotatable = rh_rot.data();
        fix.rec_donor_h.natom = 2;
        fix.rec_donor_h.total_h = 1;
        Real dw = 0;
        REQUIRE(angular_hbond_weight(hbond_cone_cos(fix.coords.data(), fix.coords.data() + 3, mol.coords.data() + 3,
                                                    nullptr, nullptr, nullptr), &dw) == Real(1));
        std::vector<Real> pose = mol.coords;
        REQUIRE(place(mol, pose, fix, atp) == 0);
        REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
    }

    SECTION("the same receptor atom as a pure acceptor O_A: placed") {
        UDFixMol fix = make_fix(rec, {VN_TYPE_O_A, VN_TYPE_C_P});
        std::vector<Real> pose = mol.coords;
        REQUIRE(place(mol, pose, fix, atp) == 1);
    }
}


TEST_CASE("place_h: direction-2 credit from receptor donors without a cone", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    UDFlexMol mol = hydroxyl();
    // receptor atom where a pure O_A is placed (first test case); its H on the line to the ligand O
    // (at the origin), 1.01 A toward it (full direction-2 credit) or away from it (none)
    const Real r_da = Real(2.8);
    const std::vector<Real> rec = toward(r_da, 20, 30);
    std::vector<Real> h_toward(3), h_away(3);
    for (int c = 0; c < 3; c++){
        h_toward[c] = rec[c] * (1 - Real(1.01) / r_da);
        h_away[c] = rec[c] * (1 + Real(1.01) / r_da);
    }
    std::vector<int> h_start = {0, 1};
    std::vector<int> antecedent = {-1};
    std::vector<int> fixed = {0};
    std::vector<int> rotatable = {1};
    auto with_rec_h = [&](UDFixMol& fix, std::vector<Real>& h_coords, std::vector<int>& rot){
        fix.rec_donor_h.h_start = h_start.data();
        fix.rec_donor_h.h_coords = h_coords.data();
        fix.rec_donor_h.antecedent = antecedent.data();
        fix.rec_donor_h.donor_rotatable = rot.data();
        fix.rec_donor_h.natom = 1;
        fix.rec_donor_h.total_h = h_start[1];
    };
    std::vector<Real> pose = mol.coords;

    SECTION("fixed N_DA with its H toward the ligand O: skipped") {
        UDFixMol fix = make_fix(rec, {VN_TYPE_N_DA});
        with_rec_h(fix, h_toward, fixed);
        REQUIRE(place(mol, pose, fix, atp) == 0);
        REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
    }

    SECTION("fixed N_DA with its H away from the ligand O: placed") {
        UDFixMol fix = make_fix(rec, {VN_TYPE_N_DA});
        with_rec_h(fix, h_away, fixed);
        REQUIRE(place(mol, pose, fix, atp) == 1);
    }

    SECTION("fixed N_DA without H: skipped") {
        UDFixMol fix = make_fix(rec, {VN_TYPE_N_DA});
        h_start[1] = 0;
        with_rec_h(fix, h_away, fixed);
        REQUIRE(place(mol, pose, fix, atp) == 0);
        REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
    }

    SECTION("water O_DA (free rotor, no antecedent), H away from the ligand O: skipped") {
        UDFixMol fix = make_fix(rec, {VN_TYPE_O_DA});
        with_rec_h(fix, h_away, rotatable);
        REQUIRE(place(mol, pose, fix, atp) == 0);
        REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
    }

    SECTION("O_DA without a receptor H table (score() has no direction 2): placed") {
        UDFixMol fix = make_fix(rec, {VN_TYPE_O_DA});
        REQUIRE(fix.rec_donor_h.natom == 0);
        REQUIRE(place(mol, pose, fix, atp) == 1);
    }
}


TEST_CASE("place_h: donors and hydrogens beyond the written atoms are never placed", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    UDFlexMol mol = hydroxyl();  // C 0, O 1, H 2
    REQUIRE(find_placeable_cone_donors(mol, 1, atp).empty());
    REQUIRE(find_placeable_cone_donors(mol, 2, atp).empty());  // O written, its H not
    REQUIRE(find_placeable_cone_donors(mol, 3, atp).size() == 1);

    // H 0, C 1, O 2: the H is written, the donor is not
    UDFlexMol h_first = make_mol({COS_B, -SIN_B, 0,
                                  Real(-1.43), 0, 0,
                                  0, 0, 0},
                                 {VN_TYPE_H, VN_TYPE_C_P, VN_TYPE_O_DA});
    REQUIRE(find_placeable_cone_donors(h_first, 2, atp).empty());
    REQUIRE(find_placeable_cone_donors(h_first, 3, atp).size() == 1);
}


TEST_CASE("place_h: a second hydroxyl is not turned onto the first one's hydrogen", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    // two C-O-H in the z = 0 plane, O 2.8 A apart and 2.78 A from an acceptor at the origin with
    // C-O...A = 109.47 deg, so either cone H points straight at it; the input H sit out of the plane
    // (+z on the first, -z on the second)
    std::vector<Real> coords;
    for (Real sx : {Real(-1), Real(1)}){
        const Real O[3] = {sx * Real(1.4), Real(2.4), 0};
        const Real n = std::sqrt(O[0] * O[0] + O[1] * O[1]);
        const Real a[2] = {-O[0] / n, -O[1] / n};
        // O -> A turned by 109.47 deg in the plane, away from the other hydroxyl
        const Real dir[2] = {-COS_B * a[0] - sx * SIN_B * a[1], sx * SIN_B * a[0] - COS_B * a[1]};
        coords.insert(coords.end(), {O[0] + Real(1.43) * dir[0], O[1] + Real(1.43) * dir[1], 0,
                                     O[0], O[1], O[2],
                                     O[0] - COS_B * dir[0], O[1] - COS_B * dir[1], -sx * SIN_B});
    }
    UDFlexMol mol = make_mol(coords, {VN_TYPE_C_P, VN_TYPE_O_DA, VN_TYPE_H, VN_TYPE_C_P, VN_TYPE_O_DA, VN_TYPE_H});
    const Real* O2_in = mol.coords.data() + 12;
    const Real* H2_in = mol.coords.data() + 15;
    const std::vector<PlaceHDonor> donors = find_placeable_cone_donors(mol, mol.natom, atp);
    REQUIRE(donors.size() == 2);

    SECTION("one acceptor: the second turn would clash and is rejected") {
        UDFixMol fix = make_fix({0, 0, 0}, {VN_TYPE_O_A});
        std::vector<Real> pose = mol.coords;
        REQUIRE(place(mol, pose, fix, atp) == 1);
        const Real* H1 = pose.data() + 6;
        REQUIRE(explicit_weight(H1, pose.data() + 3, fix.coords.data()) == Approx(1).margin(1e-5));
        REQUIRE(std::memcmp(pose.data() + 15, H2_in, 3 * sizeof(Real)) == 0);
        REQUIRE(dist(H1, pose.data() + 15) >= Real(2.0));
        // the cone H of the second hydroxyl would sit within 2.0 A of the first one's placed H ...
        Real cone_h2[3];
        hbond_cone_h(O2_in, mol.coords.data() + 9, fix.coords.data(), cone_h2);
        REQUIRE(dist(cone_h2, H1) < Real(2.0));
        // ... and is taken when the second hydroxyl is placed on its own
        std::vector<Real> alone = mol.coords;
        int n_h = 0;
        REQUIRE(place_cone_donor_h(alone.data(), mol.natom, {donors[1]}, fix, acceptors_of(fix, atp), VN_VDW_RADII,
                                   atp, mol.vina_types.data(), n_h) == 1);
        REQUIRE(dist(alone.data() + 15, cone_h2) < Real(1e-4));
    }

    SECTION("a second acceptor for the second hydroxyl: placed at least 2.0 A away") {
        const Real w[3] = {Real(0.35), Real(0.3), Real(-0.887)};
        const Real nw = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        UDFixMol fix = make_fix({0, 0, 0,
                                 O2_in[0] + Real(2.8) * w[0] / nw, O2_in[1] + Real(2.8) * w[1] / nw,
                                 O2_in[2] + Real(2.8) * w[2] / nw},
                                {VN_TYPE_O_A, VN_TYPE_O_A});
        std::vector<Real> pose = mol.coords;
        REQUIRE(place(mol, pose, fix, atp) == 2);
        REQUIRE(std::memcmp(pose.data(), mol.coords.data(), 6 * sizeof(Real)) == 0);
        REQUIRE(std::memcmp(pose.data() + 9, mol.coords.data() + 9, 6 * sizeof(Real)) == 0);
        REQUIRE(dist(pose.data() + 6, pose.data() + 15) >= Real(2.0));
        REQUIRE(dist(pose.data() + 12, pose.data() + 15) == Approx(dist(O2_in, H2_in)).margin(1e-5));
    }
}


TEST_CASE("place_h: several acceptors reach the best 1 deg turn", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    UDFlexMol mol;
    std::vector<int> hs;
    std::vector<Real> rec;

    SECTION("hydroxyl, two acceptors") {
        mol = hydroxyl();
        hs = {2};
        for (Real az : {Real(30), Real(100)}){
            const std::vector<Real> a = toward(Real(2.8), 20, az);
            rec.insert(rec.end(), a.begin(), a.end());
        }
    }

    SECTION("-NH3+, three acceptors") {
        std::vector<Real> coords = {Real(-1.47), 0, 0, 0, 0, 0};
        for (Real az : {Real(0), Real(120), Real(240)}){
            coords.insert(coords.end(), {Real(1.03) * COS_B, Real(1.03) * SIN_B * std::cos(az * DEG),
                                         Real(1.03) * SIN_B * std::sin(az * DEG)});
        }
        mol = make_mol(coords, {VN_TYPE_C_P, VN_TYPE_N_D, VN_TYPE_H, VN_TYPE_H, VN_TYPE_H});
        hs = {2, 3, 4};
        const Real acc[3][3] = {{Real(2.9), 25, 30}, {Real(2.95), 30, 95}, {Real(3.0), 20, 200}};
        for (const auto& p : acc){
            const std::vector<Real> a = toward(p[0], p[1], p[2]);
            rec.insert(rec.end(), a.begin(), a.end());
        }
    }

    UDFixMol fix = make_fix(rec, std::vector<int>(rec.size() / 3, VN_TYPE_O_A));
    const int d = 1;
    // the turns that put one H on the cone direction of one acceptor: rotation about +x by the
    // azimuth difference
    Real f_cone = objective(mol.coords, d, hs, fix, mol, atp);
    for (int a = 0; a < fix.natom; a++){
        for (int h : hs){
            const Real theta = std::atan2(fix.coords[a * 3 + 2], fix.coords[a * 3 + 1])
                             - std::atan2(mol.coords[h * 3 + 2], mol.coords[h * 3 + 1]);
            f_cone = std::max(f_cone, objective(turn_about_x(mol.coords, hs, theta), d, hs, fix, mol, atp));
        }
    }
    Real f_grid = 0;
    for (int k = 0; k < 360; k++){
        f_grid = std::max(f_grid, objective(turn_about_x(mol.coords, hs, Real(k) * PI / Real(180)), d, hs, fix, mol, atp));
    }
    REQUIRE(f_grid > f_cone + Real(1e-3));  // the optimum lies between the cone directions

    std::vector<Real> pose = mol.coords;
    REQUIRE(place(mol, pose, fix, atp) == 1);
    REQUIRE(objective(pose, d, hs, fix, mol, atp) >= f_grid - Real(2e-6));
    REQUIRE(std::memcmp(pose.data(), mol.coords.data(), 6 * sizeof(Real)) == 0);
    for (int h : hs){
        REQUIRE(dist(pose.data() + 3, pose.data() + h * 3) == Approx(dist(mol.coords.data() + 3, mol.coords.data() + h * 3)).margin(1e-5));
        REQUIRE(cos_angle(pose.data(), pose.data() + 3, pose.data() + h * 3)
                == Approx(cos_angle(mol.coords.data(), mol.coords.data() + 3, mol.coords.data() + h * 3)).margin(1e-5));
    }
}


TEST_CASE("place_h: a clashing turn is rejected, a clashing input H is kept", "[place_h]") {
    const AtomTypeProps& atp = test_atp();
    // hydroxyl plus a C-H whose H lies on the C-O axis `r` A beyond O: every point of the O-H cone,
    // the input H included, is then sqrt((r - 0.334)^2 + 0.943^2) A from it (1.91 A for r = 2.0)
    auto with_axis_h = [](Real r){
        UDFlexMol m = hydroxyl();
        m.natom = 5;
        m.coords.insert(m.coords.end(), {r + Real(1.09), 0, 0, r, 0, 0});
        m.vina_types.insert(m.vina_types.end(), {VN_TYPE_C_H, VN_TYPE_H});
        return m;
    };
    // acceptor 60 deg off the C-O axis: the cone H toward it gains full weight, the input H none
    UDFixMol one = make_fix(toward(Real(2.8), 60, 30), {VN_TYPE_O_A});
    std::vector<Real> rec2 = toward(Real(2.8), 60, 30);
    const std::vector<Real> a2 = toward(Real(2.8), 60, 90);
    rec2.insert(rec2.end(), a2.begin(), a2.end());
    UDFixMol two = make_fix(rec2, {VN_TYPE_O_A, VN_TYPE_O_A});

    SECTION("H on the axis at 2.0 A: nothing moves, one or two acceptors") {
        UDFlexMol mol = with_axis_h(Real(2.0));
        REQUIRE(find_placeable_cone_donors(mol, mol.natom, atp).size() == 1);
        REQUIRE(dist(mol.coords.data() + 6, mol.coords.data() + 12) < Real(2.0));
        for (const UDFixMol* fix : {&one, &two}){
            std::vector<Real> pose = mol.coords;
            REQUIRE(objective(turn_about_x(pose, {2}, PI), 1, {2}, *fix, mol, atp) > objective(pose, 1, {2}, *fix, mol, atp));
            REQUIRE(place(mol, pose, *fix, atp) == 0);
            REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
        }
    }

    SECTION("input H clashes at the optimum while turns beyond ~32 deg are free: kept") {
        // input H at azimuth 180 deg with a foreign H 1.8 A beyond it on -y; acceptors at azimuths 170 / 190
        // both give the input H full weight
        UDFlexMol mol = hydroxyl();
        mol.natom = 5;
        mol.coords.insert(mol.coords.end(), {COS_B, -SIN_B - Real(1.8 + 1.09), 0, COS_B, -SIN_B - Real(1.8), 0});
        mol.vina_types.insert(mol.vina_types.end(), {VN_TYPE_C_H, VN_TYPE_H});
        std::vector<Real> rec = toward(Real(2.8), 60, 170);
        const std::vector<Real> a2 = toward(Real(2.8), 60, 190);
        rec.insert(rec.end(), a2.begin(), a2.end());
        UDFixMol fix = make_fix(rec, {VN_TYPE_O_A, VN_TYPE_O_A});
        REQUIRE(find_placeable_cone_donors(mol, mol.natom, atp).size() == 1);
        REQUIRE(dist(mol.coords.data() + 6, mol.coords.data() + 12) < Real(2.0));
        const std::vector<Real> free_turn = turn_about_x(mol.coords, {2}, 60 * DEG);
        REQUIRE(dist(free_turn.data() + 6, free_turn.data() + 12) >= Real(2.0));
        REQUIRE(objective(free_turn, 1, {2}, fix, mol, atp) > 0);
        std::vector<Real> pose = mol.coords;
        REQUIRE(place(mol, pose, fix, atp) == 0);
        REQUIRE(std::memcmp(pose.data(), mol.coords.data(), pose.size() * sizeof(Real)) == 0);
    }

    SECTION("H on the axis at 2.2 A: the turn is taken") {
        UDFlexMol mol = with_axis_h(Real(2.2));
        for (const UDFixMol* fix : {&one, &two}){
            std::vector<Real> pose = mol.coords;
            REQUIRE(place(mol, pose, *fix, atp) == 1);
            REQUIRE(dist(pose.data() + 6, pose.data() + 12) >= Real(2.0));
            REQUIRE(std::memcmp(pose.data() + 9, mol.coords.data() + 9, 6 * sizeof(Real)) == 0);
        }
    }
}
