//
// Created by Congcong Liu on 24-11-13.
//

#include "common.cuh"
#include "maps.cuh"
#include "task/DockTask.h"

// Host-side helpers from the CPU backend (compiled in both backends).
void build_bonded_h_topo(BondedHTopo& topo, const Real* coords, const int* vn_types, int natom);
void free_bonded_h_topo(BondedHTopo& topo);
#include "myutils/errors.h"
#include <cstring>

// Host mirror of one device arena: put() lands at the same offset it would have on the device and
// upload() sends the whole arena with one cudaMemcpy, replacing many small synchronous copies.
template <typename T>
struct Staged{
    T* dev;
    std::vector<T> host;
    Staged(T* dev, size_t n): dev(dev), host(n){}
    void put(T* dst, const void* src, size_t bytes){
        if (bytes == 0) return;
        std::memcpy(host.data() + (dst - dev), src, bytes);
    }
    void upload(){
        if (host.empty()) return;
        checkCUDA(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

void alloc_cu_flex_pose_list(FlexPose** flex_pose_list_cu, Real** flex_pose_list_real_cu, std::vector<int>* list_i_real,
                             const UDFlexMolList& udflex_mols, int npose, int nflex, int exhaustiveness,
                             int* list_natom_flex, int n_atom_all_flex, int* list_ndihe, int n_dihe_all_flex){
    // GPU Cost: npose * sizeof(FlexPose) + exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) * sizeof(Real)))

    size_t n_real = (size_t)exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex);
    checkCUDA(cudaMalloc(flex_pose_list_cu, npose * sizeof(FlexPose)));
    checkCUDA(cudaMalloc(flex_pose_list_real_cu, n_real * sizeof(Real)));
    // host mirrors, uploaded with one cudaMemcpy each (was two cudaMemcpy per pose)
    std::vector<FlexPose> flex_pose_list(npose);
    std::vector<Real> real_host(n_real);
    Real* p_real_cu = *flex_pose_list_real_cu;
    Real* p_real = real_host.data();

    list_i_real->push_back(0);
    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];
        for (int j = 0; j < exhaustiveness; j++){
            // I prefer to randomize pose on GPU. So here is just a copy of initial pose
            FlexPose& fp = flex_pose_list[i * exhaustiveness + j];
            fp.center[0] = m.center[0];
            fp.center[1] = m.center[1];
            fp.center[2] = m.center[2];
            fp.rot_vec[0] = 0;
            fp.rot_vec[1] = 0;
            fp.rot_vec[2] = 0;
            fp.energy = 999;

            fp.coords = p_real_cu;
            list_i_real->push_back(list_i_real->back() + list_natom_flex[i] * 3);
            std::copy(m.coords.begin(), m.coords.begin() + list_natom_flex[i] * 3, p_real);
            p_real_cu += list_natom_flex[i] * 3;
            p_real += list_natom_flex[i] * 3;

            fp.dihedrals = p_real_cu;
            list_i_real->push_back(list_i_real->back() + list_ndihe[i]);
            std::copy(m.dihedrals.begin(), m.dihedrals.end(), p_real);
            p_real_cu += list_ndihe[i];
            p_real += list_ndihe[i];
        }
    }
    checkCUDA(cudaMemcpy(*flex_pose_list_real_cu, real_host.data(), n_real * sizeof(Real), cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(*flex_pose_list_cu, flex_pose_list.data(), npose * sizeof(FlexPose), cudaMemcpyHostToDevice));
}


void DockTask::cp_to_cpu(){
    checkCUDA(cudaMallocHost(&flex_pose_list_res, nflex * dock_param.exhaustiveness * sizeof(FlexPose)));
    checkCUDA(
        cudaMallocHost(&flex_pose_list_real_res, dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex) *
            sizeof(Real)));

    checkCUDA(cudaMemcpy(flex_pose_list_res, flex_pose_list_cu, nflex * dock_param.exhaustiveness * sizeof(FlexPose),
        cudaMemcpyDeviceToHost));
    checkCUDA(
        cudaMemcpy(flex_pose_list_real_res, flex_pose_list_real_cu, dock_param.exhaustiveness * (n_atom_all_flex * 3 +
                n_dihe_all_flex) * sizeof(Real),
            cudaMemcpyDeviceToHost));

    // The copied FlexPose structs still hold device pointers. Re-point them at the
    // host mirror so host-side stages (FF minimization, FF scoring) can dereference
    // pose->coords / pose->dihedrals; these are the same addresses the shared code
    // reaches through flex_pose_list_real_res + list_i_real[...].
    for (int j = 0; j < nflex * dock_param.exhaustiveness; j++){
        flex_pose_list_res[j].coords = flex_pose_list_real_res + list_i_real[j * 2];
        flex_pose_list_res[j].dihedrals = flex_pose_list_real_res + list_i_real[j * 2 + 1];
    }
}




void DockTask::alloc_device(){
    spdlog::info("Memory Allocation on GPU...");
    checkCUDA(cudaDeviceSynchronize());
    //======================= constants =======================
    init_constants(dock_param, type_props, vdw_radii);
    for (int i = 0; i < nflex; i++){
        set_kernel_max_natom(udflex_mols[i].natom);
        set_kernel_max_ntorsion(udflex_mols[i].dihedrals.size());
    }
    //======================= global memory =======================

    // get necessary sizes
    int npose = nflex * dock_param.exhaustiveness;

    n_atom_all_flex = 0;
    n_dihe_all_flex = 0;
    int n_range_all_flex = 0, n_rotated_atoms_all_flex = 0;
    int n_dim_all_flex = 0, n_dim_tri_mat_all_flex = 0;

    int list_natom_flex[nflex];
    int list_ndihe[nflex];

    int list_nrange[nflex]; // not used
    int list_ndim_trimat[nflex];
    int list_nrotated_atoms[nflex];

    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];


        list_natom_flex[i] = m.natom;
        list_ndihe[i] = m.dihedrals.size();
        n_atom_all_flex += m.natom;
        n_dihe_all_flex += m.dihedrals.size();

        int dim = 3 + 4 + m.dihedrals.size();
        n_dim_all_flex += dim;
        list_ndim_trimat[i] = dim * (dim + 1) / 2;
        n_dim_tri_mat_all_flex += list_ndim_trimat[i];
        list_nrotated_atoms[i] = 0;
        list_nrange[i] = 0;
        for (auto t : m.torsions){
            list_nrotated_atoms[i] += t.rotated_atoms.size();
            list_nrange[i] += t.range_list.size() / 2;
            // limited: its allowed set is not the full circle (same test as bfgs.cuh has_limited_torsion_warp)
            torlib_kernels = torlib_kernels || (dock_param.policy.use_torsion_ranges &&
                torsion_room(0, t.range_list.data(), t.range_list.size() / 2, 1) != TOR_ROOM_FREE);
        }
        n_rotated_atoms_all_flex += list_nrotated_atoms[i];
        n_range_all_flex += list_nrange[i];
    }
    // prepare pointers to CUDA continuous large memory
    Real* p_real_cu;
    int* p_int_cu;

    //----- flex_pose_list -----
    alloc_cu_flex_pose_list(&flex_pose_list_cu, &flex_pose_list_real_cu, &list_i_real, udflex_mols, npose, nflex,
                            dock_param.exhaustiveness,
                            list_natom_flex, n_atom_all_flex, list_ndihe, n_dihe_all_flex);


    //----- flex_topo_list -----
    // GPU cost: nflex * sizeof(FlexTopo) + (n_atom_all_flex + n_dihe_all_flex * 2 + n_dihe_all_flex * 2 +
    // n_dihe_all_flex * 2 + n_rotated_atoms_all_flex) * sizeof(int) +
    // (n_range_all_flex) * 2 * sizeof(Real)
    checkCUDA(cudaMalloc(&flex_topo_list_cu, nflex * sizeof(FlexTopo)));
    checkCUDA( // vn_types, axis_atoms, range_inds, rotated_inds, rotated_atoms
        cudaMalloc(&flex_topo_list_int_cu, (n_atom_all_flex + n_dihe_all_flex * 2 + n_dihe_all_flex * 2 +
            n_dihe_all_flex * 2 +
            n_rotated_atoms_all_flex) * sizeof(int)));
    checkCUDA(cudaMalloc(&flex_topo_list_real_cu, (n_range_all_flex) * 2 * sizeof(Real))); //range_list
    std::vector<FlexTopo> flex_topo_list(nflex);
    Staged<int> topo_int(flex_topo_list_int_cu, n_atom_all_flex + n_dihe_all_flex * 6 + n_rotated_atoms_all_flex);
    Staged<Real> topo_real(flex_topo_list_real_cu, n_range_all_flex * 2);

    p_int_cu = flex_topo_list_int_cu;
    p_real_cu = flex_topo_list_real_cu;
    for (int i = 0; i < nflex; i++){
        int ind_range = 0, ind_rotated = 0;
        auto& m = udflex_mols[i];

        // set values
        flex_topo_list[i].natom = list_natom_flex[i];
        flex_topo_list[i].ntorsion = list_ndihe[i];

        // set pointer to GPU
        flex_topo_list[i].vn_types = p_int_cu;
        p_int_cu += list_natom_flex[i];
        flex_topo_list[i].axis_atoms = p_int_cu;
        p_int_cu += list_ndihe[i] * 2;
        flex_topo_list[i].range_inds = p_int_cu;
        p_int_cu += list_ndihe[i] * 2;
        flex_topo_list[i].rotated_inds = p_int_cu;
        p_int_cu += list_ndihe[i] * 2;
        flex_topo_list[i].rotated_atoms = p_int_cu;
        p_int_cu += list_nrotated_atoms[i];

        flex_topo_list[i].range_list = p_real_cu;

        // stage values at their device offsets
        topo_int.put(flex_topo_list[i].vn_types, m.vina_types.data(), m.natom * sizeof(int));

        for (int j = 0; j < m.torsions.size(); j++){
            topo_int.put(flex_topo_list[i].axis_atoms + j * 2, m.torsions[j].axis, 2 * sizeof(int));

            // range_inds end
            topo_int.put(flex_topo_list[i].range_inds + j * 2, &ind_range, sizeof(int));
            int n = m.torsions[j].range_list.size();
            // range list
            topo_real.put(flex_topo_list[i].range_list + ind_range, m.torsions[j].range_list.data(), n * sizeof(int));
            // update pointer
            p_real_cu += n;
            // range_inds end
            ind_range += n;
            int nrange = n / 2;
            topo_int.put(flex_topo_list[i].range_inds + j * 2 + 1, &nrange, sizeof(int));

            // rotated_inds start
            topo_int.put(flex_topo_list[i].rotated_inds + j * 2, &ind_rotated, sizeof(int));
            // rotated_atoms
            topo_int.put(flex_topo_list[i].rotated_atoms + ind_rotated, m.torsions[j].rotated_atoms.data(),
                         m.torsions[j].rotated_atoms.size() * sizeof(int));

            // rotated_inds end
            ind_rotated += m.torsions[j].rotated_atoms.size() - 1;
            topo_int.put(flex_topo_list[i].rotated_inds + j * 2 + 1, &ind_rotated, sizeof(int));
            ind_rotated += 1;
        }
    }
    topo_int.upload();
    topo_real.upload();
    checkCUDA(cudaMemcpy(flex_topo_list_cu, flex_topo_list.data(), nflex * sizeof(FlexTopo), cudaMemcpyHostToDevice));


    //----- fix_mol -----
    // GPU cost: sizeof(FixMol) + udfix_mol.coords.size() * sizeof(Real)
    checkCUDA(cudaMalloc(&fix_mol_cu, sizeof(FixMol)));
    checkCUDA(cudaMalloc(&fix_mol_real_cu, udfix_mol.coords.size() * sizeof(Real)));
    FixMol fix_mol;
    fix_mol.natom = udfix_mol.natom;
    fix_mol.coords = fix_mol_real_cu;
    checkCUDA(
        cudaMemcpy(fix_mol.coords, udfix_mol.coords.data(), udfix_mol.coords.size() * sizeof(Real),
            cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(fix_mol_cu, &fix_mol, sizeof(FixMol), cudaMemcpyHostToDevice));
    const Real* rec_q = udfix_mol.charges_ua.size() == (size_t)udfix_mol.natom ? udfix_mol.charges_ua.data() : nullptr;
    upload_receptor_grid(udfix_mol.coords.data(), udfix_mol.vina_types.data(), udfix_mol.natom, type_props, vdw_radii, rec_q);

    //----- receptor maps (GRID path) -----
    // Ligand heavy types present in the batch and the largest heavy-atom count decide eligibility.
    {
        bool present[MAX_VN_TYPES] = {};
        int max_heavy = 0;
        for (int i = 0; i < nflex; i++){
            const auto& m = udflex_mols[i];
            int nheavy = 0;
            for (int a = 0; a < m.natom; a++){
                const int t = m.vina_types[a];
                if (type_props.is_hydrogen[t]) continue;
                present[t] = true;
                nheavy++;
            }
            max_heavy = std::max(max_heavy, nheavy);
        }
        std::vector<int> types_present;
        for (int t = 0; t < MAX_VN_TYPES; t++){
            if (present[t]) types_present.push_back(t);
        }
        std::string grid_summary;
        use_grid_now = build_receptor_maps(udfix_mol, type_props, vdw_radii, dock_param, types_present, max_heavy, grid_summary);
        if (use_grid_now) spdlog::info("GRID maps: {}", grid_summary);
        else spdlog::info("GRID disabled: {}", grid_summary);
    }


    //----- flex_param_list_cu -----
    // Device layout per ligand (all ints, packed into flex_param_list_int_cu):
    //   atom_types[natom] | intra_start[natom+1] | intra_nbr[npair_intra*2] | pairs_intra[npair_intra*2]
    //   | h_start[natom+1] | h_indices[max(total_h,1)] | antecedent[natom] | donor_rotatable[natom]
    // r1 + r2 is derived from atom types on device (g_vdw_radii) and the inter pair list is
    // not uploaded: the kernel loops over every receptor atom per ligand atom.
    std::vector<std::vector<int>> csr_start(nflex), csr_nbr(nflex);
    std::vector<BondedHTopo> h_topos(nflex);
    size_t size_param_int_all = 0;
    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];
        build_intra_csr(m.intra_pairs.data(), m.intra_pairs.size() / 2, m.natom, csr_start[i], csr_nbr[i]);
        build_bonded_h_topo(h_topos[i], m.coords.data(), m.vina_types.data(), m.natom);
        size_param_int_all += m.natom + csr_start[i].size() + csr_nbr[i].size() + m.intra_pairs.size()
                            + (m.natom + 1) + std::max(h_topos[i].total_h, 1) + 3 * m.natom;
    }
    checkCUDA(cudaMalloc(&flex_param_list_cu, nflex * sizeof(FlexParamVina)));
    checkCUDA(cudaMalloc(&flex_param_list_int_cu, size_param_int_all * sizeof(int)));
    // effective united-atom charges per ligand atom (Coulomb term), charged groups scaled
    size_t size_param_real_all = 0;
    for (int i = 0; i < nflex; i++) size_param_real_all += udflex_mols[i].natom;
    checkCUDA(cudaMalloc(&flex_param_list_real_cu, std::max(size_param_real_all, (size_t)1) * sizeof(Real)));
    p_real_cu = flex_param_list_real_cu;
    // torsional-strain rows, one pooled arena sliced per ligand
    size_t size_tors_all = 0;
    for (int i = 0; i < nflex; i++) size_tors_all += udflex_mols[i].torsion_penalties.size();
    checkCUDA(cudaMalloc(&flex_param_list_tors_cu,
                         std::max(size_tors_all, (size_t)1) * sizeof(TorsionPenaltyRow)));
    TorsionPenaltyRow* p_tors_cu = flex_param_list_tors_cu;
    // aromatic-ring bias rows (RingBiasTerm), one pooled arena sliced per ligand; nothing is allocated without rows
    size_t size_ring_all = 0;
    for (int i = 0; i < nflex; i++) size_ring_all += udflex_mols[i].ring.rows.size();
    flex_param_list_ring_cu = nullptr;
    if (size_ring_all > 0) checkCUDA(cudaMalloc(&flex_param_list_ring_cu, size_ring_all * sizeof(RingBiasRow)));
    RingBiasRow* p_ring_cu = flex_param_list_ring_cu;
    // position-bias rows, a separate pooled pair of arenas sliced per ligand (same layout as task_cpu.cpp)
    size_t size_bias_int_all = 0, size_bias_real_all = 0;
    for (int i = 0; i < nflex; i++){
        size_bias_int_all += udflex_mols[i].natom * 2;
        size_bias_real_all += udflex_mols[i].bias.rows.size() * 5;
    }
    checkCUDA(cudaMalloc(&flex_param_list_bias_int_cu, std::max(size_bias_int_all, (size_t)1) * sizeof(int)));
    checkCUDA(cudaMalloc(&flex_param_list_bias_real_cu, std::max(size_bias_real_all, (size_t)1) * sizeof(Real)));
    int* p_bias_int_cu = flex_param_list_bias_int_cu;
    Real* p_bias_real_cu = flex_param_list_bias_real_cu;
    std::vector<FlexParamVina> flex_param_list(nflex);
    Staged<int> param_int(flex_param_list_int_cu, size_param_int_all);
    Staged<Real> param_real(flex_param_list_real_cu, std::max(size_param_real_all, (size_t)1));
    Staged<TorsionPenaltyRow> param_tors(flex_param_list_tors_cu, std::max(size_tors_all, (size_t)1));
    Staged<RingBiasRow> param_ring(flex_param_list_ring_cu, size_ring_all);
    Staged<int> param_bias_int(flex_param_list_bias_int_cu, std::max(size_bias_int_all, (size_t)1));
    Staged<Real> param_bias_real(flex_param_list_bias_real_cu, std::max(size_bias_real_all, (size_t)1));

    p_int_cu = flex_param_list_int_cu;
    for (int i = 0; i < nflex; i++){
        auto& m = udflex_mols[i];
        FlexParamVina& fp = flex_param_list[i];
        fp = FlexParamVina{};
        fp.npair_intra = m.intra_pairs.size() / 2;
        fp.npair_inter = 0;

        fp.atom_types = p_int_cu;
        param_int.put(fp.atom_types, m.vina_types.data(), m.natom * sizeof(int));
        p_int_cu += m.natom;

        fp.n_tors_penalty = static_cast<int>(m.torsion_penalties.size());
        fp.tors_penalty = p_tors_cu;
        if (fp.n_tors_penalty > 0){
            param_tors.put(p_tors_cu, m.torsion_penalties.data(), m.torsion_penalties.size() * sizeof(TorsionPenaltyRow));
            p_tors_cu += m.torsion_penalties.size();
        }

        fp.terms.ring.n = static_cast<int>(m.ring.rows.size());
        if (fp.terms.ring.n > 0){
            fp.terms.ring.rows = p_ring_cu;
            param_ring.put(p_ring_cu, m.ring.rows.data(), m.ring.rows.size() * sizeof(RingBiasRow));
            p_ring_cu += m.ring.rows.size();
        }

        fp.intra_start = p_int_cu;
        param_int.put(fp.intra_start, csr_start[i].data(), csr_start[i].size() * sizeof(int));
        p_int_cu += csr_start[i].size();

        fp.intra_nbr = p_int_cu;
        param_int.put(fp.intra_nbr, csr_nbr[i].data(), csr_nbr[i].size() * sizeof(int));
        p_int_cu += csr_nbr[i].size();

        fp.pairs_intra = p_int_cu;
        if (!m.intra_pairs.empty()){
            std::vector<int> rr_pairs;
            reorder_pairs_round_robin(m.intra_pairs.data(), m.intra_pairs.size() / 2, m.natom, rr_pairs);
            param_int.put(fp.pairs_intra, rr_pairs.data(), rr_pairs.size() * sizeof(int));
        }
        p_int_cu += m.intra_pairs.size();

        fp.bonded_h_topo.natom = m.natom;
        fp.bonded_h_topo.total_h = h_topos[i].total_h;
        fp.bonded_h_topo.n_shared_cone_antecedent = h_topos[i].n_shared_cone_antecedent;  // gem-diol ligands keep the g_pgrid walk (maps.cu ordering argument)
        fp.bonded_h_topo.h_start = p_int_cu;
        param_int.put(fp.bonded_h_topo.h_start, h_topos[i].h_start, (m.natom + 1) * sizeof(int));
        p_int_cu += m.natom + 1;

        fp.bonded_h_topo.h_indices = p_int_cu;
        if (h_topos[i].total_h > 0){
            param_int.put(fp.bonded_h_topo.h_indices, h_topos[i].h_indices, h_topos[i].total_h * sizeof(int));
        }
        p_int_cu += std::max(h_topos[i].total_h, 1);

        fp.bonded_h_topo.antecedent = p_int_cu;
        param_int.put(fp.bonded_h_topo.antecedent, h_topos[i].antecedent, m.natom * sizeof(int));
        p_int_cu += m.natom;

        fp.bonded_h_topo.donor_rotatable = p_int_cu;
        param_int.put(fp.bonded_h_topo.donor_rotatable, h_topos[i].donor_rotatable, m.natom * sizeof(int));
        p_int_cu += m.natom;

        {
            std::vector<int> heavy;
            heavy.reserve(m.natom);
            for (int k = 0; k < m.natom; k++) if (!type_props.is_hydrogen[m.vina_types[k]]) heavy.push_back(k);
            fp.nheavy = (int)heavy.size();
            fp.heavy_ids = p_int_cu;
            if (!heavy.empty()){
                param_int.put(fp.heavy_ids, heavy.data(), heavy.size() * sizeof(int));
            }
            p_int_cu += m.natom;
        }

        {
            std::vector<Real> q_eff(m.natom, Real(0));
            for (int k = 0; k < m.natom; k++) {
                Real q = k < (int)m.charges_ua.size() ? m.charges_ua[k] : Real(0);
                if (k < (int)m.charged_group.size() && m.charged_group[k]) q *= dock_param.weights.elec_charged_scale;
                q_eff[k] = q;
            }
            fp.charges = p_real_cu;
            param_real.put(fp.charges, q_eff.data(), m.natom * sizeof(Real));
            p_real_cu += m.natom;
        }

        {
            // PositionBiasTerm (ext/bias/bias_term.h): inds[a*2], [a*2+1]: half-open row range of atom a in Real
            // slots; [0,0] means no bias. terms.settings: the run's term settings (the kernel reads their
            // __constant__ mirror g_term_settings)
            std::vector<int> inds(m.natom * 2);
            std::vector<Real> vals(m.bias.rows.size() * 5);
            flatten_bias_rows(m, inds.data(), vals.data());
            fp.terms.settings = term_settings_of(dock_param);
            fp.terms.bias.inds = p_bias_int_cu;
            fp.terms.bias.params = p_bias_real_cu;
            param_bias_int.put(p_bias_int_cu, inds.data(), inds.size() * sizeof(int));
            if (!vals.empty()){
                param_bias_real.put(p_bias_real_cu, vals.data(), vals.size() * sizeof(Real));
            }
            p_bias_int_cu += m.natom * 2;
            p_bias_real_cu += vals.size();
        }

        free_bonded_h_topo(h_topos[i]);
    }
    param_int.upload();
    param_real.upload();
    param_tors.upload();
    param_ring.upload();
    param_bias_int.upload();
    param_bias_real.upload();
    checkCUDA(cudaMemcpy(flex_param_list_cu, flex_param_list.data(), nflex * sizeof(FlexParamVina), cudaMemcpyHostToDevice));


    //----- fix_param_cu -----
    // GPU cost: sizeof(FixParamVina) + udfix_mol.natom * sizeof(int) + receptor donor-H table
    {
        FixParamVina fix_param_host;
        fix_param_host.atom_types = const_cast<int*>(udfix_mol.vina_types.data());
        fix_param_host.rec_donor_h = udfix_mol.rec_donor_h;
        fix_param_host.charges = const_cast<Real*>(rec_q);
        fix_param_cu = alloccp_FixParamVina_gpu(fix_param_host, udfix_mol.natom);
        fix_param_int_cu = nullptr;
    }


    //----- aux_pose_cu -----
    // GPU cost: STRIDE_POSE * npose * sizeof(FlexPose)) +
    // TRIDE_POSE * dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex)
    //        * sizeof(Real))
    //
    checkCUDA(cudaMalloc(&aux_poses_cu, STRIDE_POSE * npose * sizeof(FlexPose)));
    checkCUDA(
        cudaMalloc(&aux_poses_real_cu, STRIDE_POSE * dock_param.exhaustiveness * (n_atom_all_flex * 3 + n_dihe_all_flex)
            * sizeof(Real)));
    std::vector<FlexPose> aux_poses(STRIDE_POSE * npose);
    p_real_cu = aux_poses_real_cu;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            for (int k = 0; k < STRIDE_POSE; k++){
                aux_poses[i * STRIDE_POSE * dock_param.exhaustiveness + j * STRIDE_POSE + k].coords = p_real_cu;
                p_real_cu += list_natom_flex[i] * 3;
                aux_poses[i * STRIDE_POSE * dock_param.exhaustiveness + j * STRIDE_POSE + k].dihedrals = p_real_cu;
                p_real_cu += list_ndihe[i];
            }
        }
    }
    checkCUDA(cudaMemcpy(aux_poses_cu, aux_poses.data(), STRIDE_POSE * npose * sizeof(FlexPose), cudaMemcpyHostToDevice));


    //----- aux_hessians_cu: per-pose BFGS state blocks -----
    // Per pose: the Hessian (upper triangle of dim = 7 + ntorsion rows, of which BFGS uses dim - 1) followed by the
    // STRIDE_G DOF vectors of dim - 1 Reals (bfgs_state_ptrs in bfgs.cuh); a launch may keep either part in shared memory.
    // GPU cost: npose * sizeof(FlexPoseHessian) +
    // dock_param.exhaustiveness * (n_dim_tri_mat_all_flex + STRIDE_G * (n_dim_all_flex - nflex)) * sizeof(Real),
    // less than the separate gradient structs plus Hessians that predict_gpu_flex (screening.cpp) still budgets for.

    checkCUDA(cudaMalloc(&aux_hessians_cu, npose * sizeof(FlexPoseHessian)));
    checkCUDA(cudaMalloc(&aux_hessians_real_cu,
        (size_t)dock_param.exhaustiveness * (n_dim_tri_mat_all_flex + STRIDE_G * (n_dim_all_flex - nflex)) * sizeof(Real)));
    std::vector<FlexPoseHessian> aux_hessians(npose);
    p_real_cu = aux_hessians_real_cu;
    for (int i = 0; i < nflex; i++){
        const int dim_g = dof_g + list_ndihe[i];
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            aux_hessians[i * dock_param.exhaustiveness + j].matrix = p_real_cu;
            p_real_cu += list_ndim_trimat[i] + STRIDE_G * dim_g;
        }
    }
    checkCUDA(cudaMemcpy(aux_hessians_cu, aux_hessians.data(), npose * sizeof(FlexPoseHessian), cudaMemcpyHostToDevice));


    //----- aux_forces_cu -----
    // GPU cost: npose * sizeof(FlexForce) +
    // dock_param.exhaustiveness * n_atom_all_flex * 3 * sizeof(Real)

    checkCUDA(cudaMalloc(&aux_forces_cu, npose * sizeof(FlexForce)));
    checkCUDA(cudaMalloc(&aux_forces_real_cu, dock_param.exhaustiveness * n_atom_all_flex * 3 * sizeof(Real)));
    std::vector<FlexForce> aux_forces(npose);
    p_real_cu = aux_forces_real_cu;
    for (int i = 0; i < nflex; i++){
        for (int j = 0; j < dock_param.exhaustiveness; j++){
            aux_forces[i * dock_param.exhaustiveness + j].f = p_real_cu;
            p_real_cu += list_natom_flex[i] * 3;
        }
    }
    checkCUDA(cudaMemcpy(aux_forces_cu, aux_forces.data(), npose * sizeof(FlexForce), cudaMemcpyHostToDevice));


    //----- Clustering -----
    int npair = dock_param.exhaustiveness * (dock_param.exhaustiveness - 1) / 2; // tri-mat with diagonal
    checkCUDA(cudaMalloc(&aux_list_e_cu, npose * sizeof(Real)));
    checkCUDA(cudaMalloc(&aux_list_cluster_cu, nflex * dock_param.exhaustiveness * sizeof(int)));
    checkCUDA(cudaMalloc(&aux_rmsd_ij_cu, nflex * npair * 2 * sizeof(int)));
    checkCUDA(cudaMalloc(&clustered_pose_inds_cu, nflex * dock_param.exhaustiveness * sizeof(int)));

    // the (j, k) table is the same for every ligand: upload it once, then replicate on the device
    // by doubling device-to-device copies (was an nflex-times host table of nflex * npair * 2 ints)
    std::vector<int> aux_rmsd_ij(npair * 2, 0); // excluding diagonal line
    int a = 0;
    for (int j = 0; j < dock_param.exhaustiveness; j++){
        for (int k = j + 1; k < dock_param.exhaustiveness; k++){
            aux_rmsd_ij[a * 2] = j;
            aux_rmsd_ij[a * 2 + 1] = k;
            a++;
        }
    }
    size_t n_tab = (size_t)npair * 2;
    if (nflex > 0 && n_tab > 0){
        checkCUDA(cudaMemcpy(aux_rmsd_ij_cu, aux_rmsd_ij.data(), n_tab * sizeof(int), cudaMemcpyHostToDevice));
        for (size_t done = 1; done < (size_t)nflex; ){
            size_t n = std::min(done, (size_t)nflex - done);
            checkCUDA(cudaMemcpy(aux_rmsd_ij_cu + done * n_tab, aux_rmsd_ij_cu, n * n_tab * sizeof(int), cudaMemcpyDeviceToDevice));
            done += n;
        }
    }

    // set diagonal to 1 for aux_list_cluster_cu
    std::vector<int> aux_cluster_mat(nflex * dock_param.exhaustiveness, 1);
    checkCUDA(
        cudaMemcpy(aux_list_cluster_cu, aux_cluster_mat.data(), nflex * dock_param.exhaustiveness * sizeof(int),
            cudaMemcpyHostToDevice));


    //----- Wait for cudaMemcpy -----
    checkCUDA(cudaDeviceSynchronize()); // assure that memcpy is finished
    spdlog::info("Memory allocation on GPU is done.");
}


void DockTask::free_device(){
    //----- Free all GPU memory -----

    spdlog::info("Memory free on GPU...");
    if (flex_param_list_tors_cu != nullptr) checkCUDA(cudaFree(flex_param_list_tors_cu));
    if (flex_param_list_ring_cu != nullptr) checkCUDA(cudaFree(flex_param_list_ring_cu));
    checkCUDA(cudaFree(flex_pose_list_cu));
    checkCUDA(cudaFree(flex_pose_list_real_cu));

    checkCUDA(cudaFree(flex_topo_list_cu));
    checkCUDA(cudaFree(flex_topo_list_int_cu));
    checkCUDA(cudaFree(flex_topo_list_real_cu));

    checkCUDA(cudaFree(fix_mol_cu));
    checkCUDA(cudaFree(fix_mol_real_cu));
    free_receptor_grid();
    free_receptor_maps();

    checkCUDA(cudaFree(flex_param_list_cu));
    checkCUDA(cudaFree(flex_param_list_int_cu));
    if (flex_param_list_real_cu) checkCUDA(cudaFree(flex_param_list_real_cu));
    if (flex_param_list_bias_int_cu) checkCUDA(cudaFree(flex_param_list_bias_int_cu));
    if (flex_param_list_bias_real_cu) checkCUDA(cudaFree(flex_param_list_bias_real_cu));

    free_FixParamVina_gpu(fix_param_cu);

    checkCUDA(cudaFree(aux_poses_cu));
    checkCUDA(cudaFree(aux_poses_real_cu));

    checkCUDA(cudaFree(aux_hessians_cu));
    checkCUDA(cudaFree(aux_hessians_real_cu));

    checkCUDA(cudaFree(aux_forces_cu));
    checkCUDA(cudaFree(aux_forces_real_cu));

    checkCUDA(cudaFree(aux_list_e_cu));
    checkCUDA(cudaFree(aux_list_cluster_cu));
    checkCUDA(cudaFree(aux_rmsd_ij_cu));
    checkCUDA(cudaFree(clustered_pose_inds_cu));

    spdlog::info("Memory free on GPU is done.");

    spdlog::info("Memory free on CPU...");
    checkCUDA(cudaFreeHost(flex_pose_list_res));
    checkCUDA(cudaFreeHost(flex_pose_list_real_res));
    spdlog::info("Memory free on CPU is done.");
}
