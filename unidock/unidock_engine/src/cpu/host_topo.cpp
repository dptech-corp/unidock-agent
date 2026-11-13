//
// Host-side FlexTopo construction (see host_topo.h)
//

#include "host_topo.h"
#include <cstring>


void count_flex_topo_host(const UDFlexMol& m, int* out_n_int, int* out_n_real) {
    int n_int = m.natom + m.torsions.size() * 2 * 3;  // vn_types, axis_atoms, range_inds, rotated_inds
    int n_real = 0;
    for (const auto& t : m.torsions) {
        n_int += t.rotated_atoms.size();
        n_real += t.range_list.size();
    }
    *out_n_int = n_int;
    *out_n_real = n_real;
}


void fill_flex_topo_host(const UDFlexMol& m, FlexTopo& topo, int*& p_int, Real*& p_real) {
    topo.natom = m.natom;
    topo.ntorsion = m.dihedrals.size();

    // vn_types
    topo.vn_types = p_int;
    std::memcpy(topo.vn_types, m.vina_types.data(), m.natom * sizeof(int));
    p_int += m.natom;

    // axis_atoms
    topo.axis_atoms = p_int;
    for (size_t j = 0; j < m.torsions.size(); j++) {
        topo.axis_atoms[j * 2] = m.torsions[j].axis[0];
        topo.axis_atoms[j * 2 + 1] = m.torsions[j].axis[1];
    }
    p_int += m.torsions.size() * 2;

    // range_inds and range_list
    topo.range_inds = p_int;
    topo.range_list = p_real;
    int range_offset = 0;
    for (size_t j = 0; j < m.torsions.size(); j++) {
        topo.range_inds[j * 2] = range_offset;
        topo.range_inds[j * 2 + 1] = m.torsions[j].range_list.size() / 2;
        std::memcpy(topo.range_list + range_offset, m.torsions[j].range_list.data(),
                   m.torsions[j].range_list.size() * sizeof(Real));
        range_offset += m.torsions[j].range_list.size();
    }
    p_int += m.torsions.size() * 2;
    p_real += range_offset;

    // rotated_inds and rotated_atoms
    topo.rotated_inds = p_int;
    p_int += m.torsions.size() * 2;
    topo.rotated_atoms = p_int;

    int rotated_offset = 0;
    for (size_t j = 0; j < m.torsions.size(); j++) {
        topo.rotated_inds[j * 2] = rotated_offset;
        std::memcpy(topo.rotated_atoms + rotated_offset, m.torsions[j].rotated_atoms.data(),
                   m.torsions[j].rotated_atoms.size() * sizeof(int));
        rotated_offset += m.torsions[j].rotated_atoms.size() - 1;
        topo.rotated_inds[j * 2 + 1] = rotated_offset;
        rotated_offset += 1;
    }
    p_int += rotated_offset;
}


void build_host_flex_topo(const UDFlexMol& m, HostFlexTopo& out) {
    int n_int = 0, n_real = 0;
    count_flex_topo_host(m, &n_int, &n_real);
    out.ints.assign(n_int, 0);
    out.reals.assign(n_real, 0);
    int* p_int = out.ints.data();
    Real* p_real = out.reals.data();
    fill_flex_topo_host(m, out.topo, p_int, p_real);
}
