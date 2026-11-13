//
// Created by Congcong Liu on 24-12-9.
//


#include "score/vina.h"
#include "common.cuh"
#include <vector>

// ==================  Memory allocation for ONE object ==================
// The device kernels derive r1 + r2 from atom types (g_vdw_radii) and walk the intra
// neighbors through the CSR view, so the per-pair arrays and the inter pair list are not
// uploaded. pairs_intra is still needed on the host to build the CSR view.
FlexParamVina* alloccp_FlexParamVina_gpu(const FlexParamVina& flex_param_vina, int natom){
    FlexParamVina* flex_param_vina_cu;
    cudaMalloc(&flex_param_vina_cu, sizeof(FlexParamVina));

    FlexParamVina flex_param_vina_tmp;
    flex_param_vina_tmp.npair_intra = flex_param_vina.npair_intra;
    cudaMalloc(&flex_param_vina_tmp.pairs_intra, sizeof(int) * std::max(flex_param_vina.npair_intra * 2, 1));
    if (flex_param_vina.npair_intra > 0){
        std::vector<int> rr_pairs;
        reorder_pairs_round_robin(flex_param_vina.pairs_intra, flex_param_vina.npair_intra, natom, rr_pairs);
        cudaMemcpy(flex_param_vina_tmp.pairs_intra, rr_pairs.data(), sizeof(int) * flex_param_vina.npair_intra * 2, cudaMemcpyHostToDevice);
    }
    flex_param_vina_tmp.r1_plus_r2_intra = nullptr;
    flex_param_vina_tmp.npair_inter = 0;
    flex_param_vina_tmp.pairs_inter = nullptr;
    flex_param_vina_tmp.r1_plus_r2_inter = nullptr;
    flex_param_vina_tmp.terms = flex_param_vina.terms;   // the term settings and ring row count; the row pointers are replaced below
    flex_param_vina_tmp.terms.bias.inds = nullptr;
    flex_param_vina_tmp.terms.bias.params = nullptr;
    if (flex_param_vina.terms.bias.inds != nullptr && flex_param_vina.terms.bias.params != nullptr){
        int nreal = 0;  // nbias is not stored; the largest end offset is the row-slot count
        for (int a = 0; a < natom; a++) nreal = std::max(nreal, flex_param_vina.terms.bias.inds[a * 2 + 1]);
        cudaMalloc(&flex_param_vina_tmp.terms.bias.inds, sizeof(int) * natom * 2);
        cudaMemcpy(flex_param_vina_tmp.terms.bias.inds, flex_param_vina.terms.bias.inds, sizeof(int) * natom * 2, cudaMemcpyHostToDevice);
        cudaMalloc(&flex_param_vina_tmp.terms.bias.params, sizeof(Real) * std::max(nreal, 1));
        if (nreal > 0) cudaMemcpy(flex_param_vina_tmp.terms.bias.params, flex_param_vina.terms.bias.params, sizeof(Real) * nreal, cudaMemcpyHostToDevice);
    }
    flex_param_vina_tmp.terms.ring.rows = nullptr;
    if (flex_param_vina.terms.ring.n > 0 && flex_param_vina.terms.ring.rows != nullptr){
        RingBiasRow* rb = nullptr;
        cudaMalloc(&rb, sizeof(RingBiasRow) * flex_param_vina.terms.ring.n);
        cudaMemcpy(rb, flex_param_vina.terms.ring.rows, sizeof(RingBiasRow) * flex_param_vina.terms.ring.n, cudaMemcpyHostToDevice);
        flex_param_vina_tmp.terms.ring.rows = rb;
    }

    cudaMalloc(&flex_param_vina_tmp.atom_types, sizeof(int) * natom);
    cudaMemcpy(flex_param_vina_tmp.atom_types, flex_param_vina.atom_types, sizeof(int) * natom, cudaMemcpyHostToDevice);

    flex_param_vina_tmp.n_tors_penalty = flex_param_vina.n_tors_penalty;
    flex_param_vina_tmp.tors_penalty = nullptr;
    if (flex_param_vina.n_tors_penalty > 0 && flex_param_vina.tors_penalty != nullptr){
        TorsionPenaltyRow* tp = nullptr;
        cudaMalloc(&tp, sizeof(TorsionPenaltyRow) * flex_param_vina.n_tors_penalty);
        cudaMemcpy(tp, flex_param_vina.tors_penalty,
                   sizeof(TorsionPenaltyRow) * flex_param_vina.n_tors_penalty, cudaMemcpyHostToDevice);
        flex_param_vina_tmp.tors_penalty = tp;
    }

    std::vector<int> csr_start, csr_nbr;
    build_intra_csr(flex_param_vina.pairs_intra, flex_param_vina.npair_intra, natom, csr_start, csr_nbr);
    cudaMalloc(&flex_param_vina_tmp.intra_start, sizeof(int) * csr_start.size());
    cudaMemcpy(flex_param_vina_tmp.intra_start, csr_start.data(), sizeof(int) * csr_start.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&flex_param_vina_tmp.intra_nbr, sizeof(int) * csr_nbr.size());
    cudaMemcpy(flex_param_vina_tmp.intra_nbr, csr_nbr.data(), sizeof(int) * csr_nbr.size(), cudaMemcpyHostToDevice);

    // Bonded-H topology: copy if the host built one, otherwise an all-zero CSR (no bonded H)
    BondedHTopo& h = flex_param_vina_tmp.bonded_h_topo;
    h.natom = natom;
    cudaMalloc(&h.h_start, sizeof(int) * (natom + 1));
    if (flex_param_vina.bonded_h_topo.h_start != nullptr && flex_param_vina.bonded_h_topo.natom == natom){
        h.total_h = flex_param_vina.bonded_h_topo.total_h;
        h.n_shared_cone_antecedent = flex_param_vina.bonded_h_topo.n_shared_cone_antecedent;
        cudaMemcpy(h.h_start, flex_param_vina.bonded_h_topo.h_start, sizeof(int) * (natom + 1), cudaMemcpyHostToDevice);
        cudaMalloc(&h.h_indices, sizeof(int) * std::max(h.total_h, 1));
        cudaMemcpy(h.h_indices, flex_param_vina.bonded_h_topo.h_indices, sizeof(int) * h.total_h, cudaMemcpyHostToDevice);
        h.antecedent = nullptr;
        if (flex_param_vina.bonded_h_topo.antecedent != nullptr){
            cudaMalloc(&h.antecedent, sizeof(int) * natom);
            cudaMemcpy(h.antecedent, flex_param_vina.bonded_h_topo.antecedent, sizeof(int) * natom, cudaMemcpyHostToDevice);
        }
        h.donor_rotatable = nullptr;
        if (flex_param_vina.bonded_h_topo.donor_rotatable != nullptr){
            cudaMalloc(&h.donor_rotatable, sizeof(int) * natom);
            cudaMemcpy(h.donor_rotatable, flex_param_vina.bonded_h_topo.donor_rotatable, sizeof(int) * natom, cudaMemcpyHostToDevice);
        }
    } else {
        h.total_h = 0;
        cudaMemset(h.h_start, 0, sizeof(int) * (natom + 1));
        cudaMalloc(&h.h_indices, sizeof(int));
        h.antecedent = nullptr;
        h.donor_rotatable = nullptr;
    }

    flex_param_vina_tmp.heavy_ids = nullptr;
    flex_param_vina_tmp.nheavy = flex_param_vina.nheavy;
    if (flex_param_vina.heavy_ids != nullptr && flex_param_vina.nheavy > 0){
        cudaMalloc(&flex_param_vina_tmp.heavy_ids, sizeof(int) * flex_param_vina.nheavy);
        cudaMemcpy(flex_param_vina_tmp.heavy_ids, flex_param_vina.heavy_ids,
                   sizeof(int) * flex_param_vina.nheavy, cudaMemcpyHostToDevice);
    }

    flex_param_vina_tmp.charges = nullptr;
    if (flex_param_vina.charges != nullptr){
        cudaMalloc(&flex_param_vina_tmp.charges, sizeof(Real) * natom);
        cudaMemcpy(flex_param_vina_tmp.charges, flex_param_vina.charges, sizeof(Real) * natom, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(flex_param_vina_cu, &flex_param_vina_tmp, sizeof(FlexParamVina), cudaMemcpyHostToDevice);
    return flex_param_vina_cu;
}
void free_FlexParamVina_gpu(FlexParamVina* flex_param_vina_cu){
    FlexParamVina flex_param_vina_tmp;
    cudaMemcpy(&flex_param_vina_tmp, flex_param_vina_cu, sizeof(FlexParamVina), cudaMemcpyDeviceToHost);
    cudaFree(flex_param_vina_tmp.atom_types);
    cudaFree(flex_param_vina_tmp.pairs_intra);
    cudaFree(flex_param_vina_tmp.intra_start);
    cudaFree(flex_param_vina_tmp.intra_nbr);
    cudaFree(flex_param_vina_tmp.bonded_h_topo.h_start);
    cudaFree(flex_param_vina_tmp.bonded_h_topo.h_indices);
    if (flex_param_vina_tmp.bonded_h_topo.antecedent != nullptr) cudaFree(flex_param_vina_tmp.bonded_h_topo.antecedent);
    if (flex_param_vina_tmp.bonded_h_topo.donor_rotatable != nullptr) cudaFree(flex_param_vina_tmp.bonded_h_topo.donor_rotatable);
    if (flex_param_vina_tmp.charges != nullptr) cudaFree(flex_param_vina_tmp.charges);
    if (flex_param_vina_tmp.heavy_ids != nullptr) cudaFree(flex_param_vina_tmp.heavy_ids);
    if (flex_param_vina_tmp.tors_penalty != nullptr) cudaFree(const_cast<TorsionPenaltyRow*>(flex_param_vina_tmp.tors_penalty));
    if (flex_param_vina_tmp.terms.bias.inds != nullptr) cudaFree(flex_param_vina_tmp.terms.bias.inds);
    if (flex_param_vina_tmp.terms.bias.params != nullptr) cudaFree(flex_param_vina_tmp.terms.bias.params);
    if (flex_param_vina_tmp.terms.ring.rows != nullptr) cudaFree(const_cast<RingBiasRow*>(flex_param_vina_tmp.terms.ring.rows));
    cudaFree(flex_param_vina_cu);
}


FixParamVina* alloccp_FixParamVina_gpu(const FixParamVina& fix_param_vina, int natom){
    FixParamVina* fix_param_vina_cu;
    cudaMalloc(&fix_param_vina_cu, sizeof(FixParamVina));

    FixParamVina fix_param_vina_tmp;
    cudaMalloc(&fix_param_vina_tmp.atom_types, sizeof(int) * natom);
    cudaMemcpy(fix_param_vina_tmp.atom_types, fix_param_vina.atom_types, sizeof(int) * natom, cudaMemcpyHostToDevice);

    // Receptor donor-H side table (angular HBond, receptor-donor direction)
    const RecDonorHTopo& src = fix_param_vina.rec_donor_h;
    RecDonorHTopo& dst = fix_param_vina_tmp.rec_donor_h;
    if (src.h_start != nullptr && src.natom == natom){
        dst.natom = src.natom;
        dst.total_h = src.total_h;
        cudaMalloc(&dst.h_start, sizeof(int) * (natom + 1));
        cudaMemcpy(dst.h_start, src.h_start, sizeof(int) * (natom + 1), cudaMemcpyHostToDevice);
        cudaMalloc(&dst.h_coords, sizeof(Real) * 3 * std::max(src.total_h, 1));
        cudaMemcpy(dst.h_coords, src.h_coords, sizeof(Real) * 3 * src.total_h, cudaMemcpyHostToDevice);
        dst.antecedent = nullptr;
        if (src.antecedent != nullptr){
            cudaMalloc(&dst.antecedent, sizeof(int) * natom);
            cudaMemcpy(dst.antecedent, src.antecedent, sizeof(int) * natom, cudaMemcpyHostToDevice);
        }
        dst.donor_rotatable = nullptr;
        if (src.donor_rotatable != nullptr){
            cudaMalloc(&dst.donor_rotatable, sizeof(int) * natom);
            cudaMemcpy(dst.donor_rotatable, src.donor_rotatable, sizeof(int) * natom, cudaMemcpyHostToDevice);
        }
    } else {
        dst = RecDonorHTopo{};  // natom = 0 disables the receptor-donor direction on device
    }

    fix_param_vina_tmp.charges = nullptr;
    if (fix_param_vina.charges != nullptr){
        cudaMalloc(&fix_param_vina_tmp.charges, sizeof(Real) * natom);
        cudaMemcpy(fix_param_vina_tmp.charges, fix_param_vina.charges, sizeof(Real) * natom, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(fix_param_vina_cu, &fix_param_vina_tmp, sizeof(FixParamVina), cudaMemcpyHostToDevice);
    return fix_param_vina_cu;
}
void free_FixParamVina_gpu(FixParamVina* fix_param_vina_cu){
    FixParamVina fix_param_vina_tmp;
    cudaMemcpy(&fix_param_vina_tmp, fix_param_vina_cu, sizeof(FixParamVina), cudaMemcpyDeviceToHost);
    cudaFree(fix_param_vina_tmp.atom_types);
    cudaFree(fix_param_vina_tmp.rec_donor_h.h_start);
    cudaFree(fix_param_vina_tmp.rec_donor_h.h_coords);
    if (fix_param_vina_tmp.rec_donor_h.antecedent != nullptr) cudaFree(fix_param_vina_tmp.rec_donor_h.antecedent);
    if (fix_param_vina_tmp.rec_donor_h.donor_rotatable != nullptr) cudaFree(fix_param_vina_tmp.rec_donor_h.donor_rotatable);
    if (fix_param_vina_tmp.charges != nullptr) cudaFree(fix_param_vina_tmp.charges);
    cudaFree(fix_param_vina_cu);
}
