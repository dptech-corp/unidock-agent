//
// PositionBiasTerm (ext/bias/bias_term.h) against the code it replaced: the two row potentials against verbatim
// copies of the former Vina::eval_ef_pos / eval_ef_zalign, and accumulate_as against the former per-atom loop of
// cpu/score_cpu.cpp "4. Compute position-bias", evaluated inline here. Equality is exact (same expressions, same
// order), not approximate.
//

#include <catch2/catch_amalgamated.hpp>
#include <cmath>
#include <vector>

#include "ext/bias/bias_term.h"
#include "score/vina.h"
#include "constants/constants.h"

namespace {

// ---- the former Vina members, copied unchanged from score/vina.h at reorg step 4 (9fdc0a5) ----
Real old_eval_ef_pos(Real* r_, Real V_set, Real r2, Real* out_f){
    Real rr = r_[0] * r_[0] + r_[1] * r_[1] + r_[2] * r_[2];
    Real e_bias = V_set * std::exp(- rr / r2);

    out_f[0] += e_bias * 2 * r_[0] / r2;
    out_f[1] += e_bias * 2 * r_[1] / r2;
    out_f[2] += e_bias * 2 * r_[2] / r2;

    return e_bias;
}

Real old_eval_ef_zalign(Real* r_, Real a_qt, int vn_type, Real* out_f){
    const Real s_a = 3.;
    const Real c_a = 1.;
    Real k_a = (vn_type == VN_TYPE_H ? Real(0.1) : Real(30.));

    Real r = cal_norm(r_);

    Real e_item = std::exp(s_a * (c_a - r));
    Real e_bias = a_qt * k_a / (1 + e_item);
    Real f = - e_bias * e_item / (1 + e_item) * s_a;

    out_f[0] += f * r_[0] / r;
    out_f[1] += f * r_[1] / r;
    out_f[2] += f * r_[2] / r;

    return e_bias;
}

// ---- the former loop of cpu/score_cpu.cpp, with the old member functions ----
void old_bias_loop(BiasType bias_type, Real bias_k, const int* inds_bias, const Real* params_bias, int natom,
                   const Real* pose_coords, const int* flex_types, Real& energy, Real* aux_f){
    if (bias_type != BT_NO) {
        for (int i = 0; i < natom; i++) {
            int i1 = inds_bias[i * 2];
            int i2 = inds_bias[i * 2 + 1];
            if (i1 >= i2) continue;

            Real f_bias[3] = {0., 0., 0.};
            const Real ax = pose_coords[i * 3];
            const Real ay = pose_coords[i * 3 + 1];
            const Real az = pose_coords[i * 3 + 2];

            for (int j = i1; j < i2; j += 5) {
                Real r_[3] = {
                    params_bias[j]     - ax,
                    params_bias[j + 1] - ay,
                    params_bias[j + 2] - az
                };

                if (bias_type == BT_POS) {
                    energy += old_eval_ef_pos(r_, params_bias[j + 3] * bias_k, params_bias[j + 4], f_bias);
                } else if (bias_type == BT_ALIGN) {
                    energy += old_eval_ef_zalign(r_, params_bias[j + 3] * bias_k, flex_types[i], f_bias);
                }
            }

            aux_f[i * 3]     += f_bias[0];
            aux_f[i * 3 + 1] += f_bias[1];
            aux_f[i * 3 + 2] += f_bias[2];
        }
    }
}

// Flatten rows the way DockTask::alloc_device does (task_cpu.cpp / task.cu)
void flatten(const std::vector<Bias>& rows, int natom, std::vector<int>& inds, std::vector<Real>& params){
    inds.assign(natom * 2, 0);
    params.clear();
    int last_i = -1;
    for (const auto& b : rows){
        if (b.i != last_i) inds[b.i * 2] = (int)params.size();
        for (int k = 0; k < 5; k++) params.push_back(static_cast<Real>(b.param[k]));
        inds[b.i * 2 + 1] = (int)params.size();
        last_i = b.i;
    }
}

Bias row(int i, float x, float y, float z, float v_set, float w){
    Bias b;
    b.i = i;
    b.param[0] = x; b.param[1] = y; b.param[2] = z; b.param[3] = v_set; b.param[4] = w;
    return b;
}

} // namespace


TEST_CASE("PositionBiasTerm::eval_pos / eval_zalign are the former Vina potentials, bit for bit", "[ext][bias_term]"){
    const Real vectors[][3] = {{0.3, -0.7, 1.1}, {-1.0, 0.0, 0.0}, {0.05, 0.02, -0.01}, {2.5, 1.5, -3.0}, {0.0, 0.0, 1.0}};
    const Real depths[] = {Real(1.0), Real(0.3), Real(2.5 * 0.05)};
    for (const auto& v : vectors){
        for (Real a : depths){
            for (Real w : {Real(2.0), Real(0.5)}){
                Real r_new[3] = {v[0], v[1], v[2]}, r_old[3] = {v[0], v[1], v[2]};
                Real f_new[3] = {0.1, -0.2, 0.3}, f_old[3] = {0.1, -0.2, 0.3};   // accumulation onto a prior force
                const Real e_new = PositionBiasTerm::eval_pos(r_new, a, w, f_new);
                const Real e_old = old_eval_ef_pos(r_old, a, w, f_old);
                REQUIRE(e_new == e_old);
                for (int d = 0; d < 3; d++) REQUIRE(f_new[d] == f_old[d]);
            }
            for (int vn_type : {VN_TYPE_C_H, VN_TYPE_N_A, VN_TYPE_H}){
                Real r_new[3] = {v[0], v[1], v[2]}, r_old[3] = {v[0], v[1], v[2]};
                Real f_new[3] = {0.1, -0.2, 0.3}, f_old[3] = {0.1, -0.2, 0.3};
                const Real e_new = PositionBiasTerm::eval_zalign(r_new, a, vn_type, f_new);
                const Real e_old = old_eval_ef_zalign(r_old, a, vn_type, f_old);
                REQUIRE(e_new == e_old);
                for (int d = 0; d < 3; d++) REQUIRE(f_new[d] == f_old[d]);
            }
        }
    }
}


TEST_CASE("TermSet::accumulate (the CPU path): one atom, one row, both types, equals the former loop", "[ext][bias_term]"){
    const std::vector<Real> coords = {1.0, -0.5, 2.0};
    const std::vector<int> types = {VN_TYPE_C_H};
    const std::vector<Bias> rows = {row(0, 1.6f, 0.3f, 1.1f, 0.8f, 2.0f)};
    std::vector<int> inds; std::vector<Real> params;
    flatten(rows, 1, inds, params);

    for (BiasType bt : {BT_POS, BT_ALIGN}){
        for (Real k : {Real(0.05), Real(0.3)}){
            TermSet terms;                                                 // as DockTask::alloc_device fills it
            terms.bias.inds = inds.data(); terms.bias.params = params.data();
            terms.settings.bias_type = bt; terms.settings.bias_k = k;

            Real e_new = Real(-3.25), e_old = Real(-3.25);                 // the pair energy already accumulated
            std::vector<Real> f_new = {0.5, 0.25, -1.0}, f_old = {0.5, 0.25, -1.0};
            terms.accumulate(1, coords.data(), types.data(), e_new, f_new.data());
            old_bias_loop(bt, k, inds.data(), params.data(), 1, coords.data(), types.data(), e_old, f_old.data());

            INFO("type " << bt << " k " << k);
            REQUIRE(e_new == e_old);
            REQUIRE(e_new != Real(-3.25));                                 // the row contributed
            for (int d = 0; d < 3; d++) REQUIRE(f_new[d] == f_old[d]);

            // the energy is the row potential, the force its derivative, as evaluated inline
            Real r_[3] = {params[0] - coords[0], params[1] - coords[1], params[2] - coords[2]};
            Real f_row[3] = {0., 0., 0.};
            const Real e_row = bt == BT_POS ? old_eval_ef_pos(r_, params[3] * k, params[4], f_row)
                                             : old_eval_ef_zalign(r_, params[3] * k, types[0], f_row);
            REQUIRE(e_new == Real(-3.25) + e_row);
            REQUIRE(f_new[0] == Real(0.5) + f_row[0]);
            REQUIRE(f_new[1] == Real(0.25) + f_row[1]);
            REQUIRE(f_new[2] == Real(-1.0) + f_row[2]);
        }
    }
}


TEST_CASE("PositionBiasTerm::accumulate_as: several atoms, gaps, many rows, lanes; equals the former loop", "[ext][bias_term]"){
    // 5 atoms: two rows on atom 0, none on 1, one on 2 (hydrogen), three on 4, none on 3
    const std::vector<Real> coords = {0, 0, 0,  3.2, 0.4, 0,  0.3, 3.1, 0.5,  -2.0, 1.0, 1.5,  1.0, -2.5, 3.0};
    const std::vector<int> types = {VN_TYPE_C_H, VN_TYPE_C_H, VN_TYPE_H, VN_TYPE_N_A, VN_TYPE_O_A};
    const std::vector<Bias> rows = {
        row(0, 0.6f, 0.8f, 0.0f, 1.0f, 2.0f), row(0, 0.0f, 0.948f, -1.264f, 2.5f, 2.0f),
        row(2, 1.3f, 2.6f, 1.4f, 4.0f, 2.0f),
        row(4, 1.7f, -2.9f, 3.9f, 3.0f, 2.0f), row(4, 0.5f, -2.0f, 3.6f, 1.5f, 1.0f), row(4, 1.0f, -3.5f, 2.0f, 0.7f, 2.0f),
    };
    std::vector<int> inds; std::vector<Real> params;
    flatten(rows, 5, inds, params);
    REQUIRE(inds[1 * 2] == 0); REQUIRE(inds[1 * 2 + 1] == 0);   // gap atoms
    REQUIRE(inds[3 * 2] == 0); REQUIRE(inds[3 * 2 + 1] == 0);

    for (BiasType bt : {BT_POS, BT_ALIGN}){
        const Real k = Real(0.3);
        TermSet terms;
        terms.bias.inds = inds.data(); terms.bias.params = params.data();
        terms.settings.bias_type = bt; terms.settings.bias_k = k;
        const PositionBiasTerm& term = terms.bias;

        // serial (the CPU path: TermSet::accumulate under the set's own settings)
        Real e_new = Real(2.0), e_old = Real(2.0);
        std::vector<Real> f_new(15, Real(0.125)), f_old(15, Real(0.125));
        terms.accumulate(5, coords.data(), types.data(), e_new, f_new.data());
        old_bias_loop(bt, k, inds.data(), params.data(), 5, coords.data(), types.data(), e_old, f_old.data());
        INFO("type " << bt);
        REQUIRE(e_new == e_old);
        for (int d = 0; d < 15; d++) REQUIRE(f_new[d] == f_old[d]);
        for (int a : {1, 3}) for (int d = 0; d < 3; d++) REQUIRE(f_new[a * 3 + d] == Real(0.125));   // untouched

        // strided (the kernel's lane / stride form): each lane's energy is its own atoms' rows, in atom order,
        // so the per-lane sums are the serial sum's partial sums; the forces are the same values
        for (int stride : {2, 3}){
            std::vector<Real> f_lanes(15, Real(0.125));
            Real e_lanes_total_check = Real(2.0);
            for (int lane = 0; lane < stride; lane++){
                Real e_lane = 0;
                term.accumulate_as(bt, k, lane, stride, 5, coords.data(), types.data(), e_lane, f_lanes.data());
                // the lane's rows, re-added serially onto the running total in the lane's atom order
                for (int i = lane; i < 5; i += stride){
                    for (int j = inds[i * 2]; j < inds[i * 2 + 1]; j += 5){
                        Real r_[3] = {params[j] - coords[i * 3], params[j + 1] - coords[i * 3 + 1], params[j + 2] - coords[i * 3 + 2]};
                        Real f_dummy[3] = {0., 0., 0.};
                        e_lanes_total_check += bt == BT_POS ? old_eval_ef_pos(r_, params[j + 3] * k, params[j + 4], f_dummy)
                                                            : old_eval_ef_zalign(r_, params[j + 3] * k, types[i], f_dummy);
                    }
                }
            }
            for (int d = 0; d < 15; d++) REQUIRE(f_lanes[d] == f_old[d]);
            REQUIRE(e_lanes_total_check == Catch::Approx(e_old).epsilon(1e-6));
        }
    }
}


TEST_CASE("TermSet: the dispatch both kernels call; every term off adds nothing even with rows present", "[ext][bias_term][terms]"){
    PositionBiasTerm fresh;
    REQUIRE(fresh.inds == nullptr);
    REQUIRE(fresh.params == nullptr);
    TermSettings defaults;
    REQUIRE(defaults.bias_type == BT_NO);

    const std::vector<Real> coords = {1.0, -0.5, 2.0};
    const std::vector<int> types = {VN_TYPE_C_H};
    std::vector<int> inds; std::vector<Real> params;
    flatten({row(0, 1.6f, 0.3f, 1.1f, 0.8f, 2.0f)}, 1, inds, params);
    TermSet terms;
    terms.bias.inds = inds.data(); terms.bias.params = params.data();

    // every term off (the default settings): inactive, and nothing added although rows are present
    REQUIRE_FALSE(terms.active());
    Real e = Real(1.5);
    std::vector<Real> f = {0.1, 0.2, 0.3};
    terms.accumulate(1, coords.data(), types.data(), e, f.data());
    REQUIRE(e == Real(1.5));
    REQUIRE(f[0] == Real(0.1)); REQUIRE(f[1] == Real(0.2)); REQUIRE(f[2] == Real(0.3));

    // settings on: active, and the set's accumulate_as (the kernel form) is the term's own accumulate_as with the
    // same settings; accumulate (the CPU form) reads the set's own copy of the settings
    for (int bt : {BT_POS, BT_ALIGN}){
        TermSettings s; s.bias_type = bt; s.bias_k = Real(0.3);
        REQUIRE(terms.active_as(s));
        Real e_set = Real(1.5), e_term = Real(1.5);
        std::vector<Real> f_set = {0.1, 0.2, 0.3}, f_term = {0.1, 0.2, 0.3};
        terms.accumulate_as(s, 0, 1, 1, coords.data(), types.data(), e_set, f_set.data());
        terms.bias.accumulate_as(bt, Real(0.3), 0, 1, 1, coords.data(), types.data(), e_term, f_term.data());
        REQUIRE(e_set == e_term);
        REQUIRE(e_set != Real(1.5));
        for (int d = 0; d < 3; d++) REQUIRE(f_set[d] == f_term[d]);

        terms.settings = s;
        REQUIRE(terms.active());
        Real e_own = Real(1.5);
        std::vector<Real> f_own = {0.1, 0.2, 0.3};
        terms.accumulate(1, coords.data(), types.data(), e_own, f_own.data());
        REQUIRE(e_own == e_set);
        for (int d = 0; d < 3; d++) REQUIRE(f_own[d] == f_set[d]);
    }

    // settings on but no rows for this ligand (inds nullptr): inactive, so the kernel skips the call and its sync
    TermSet no_rows;
    no_rows.settings.bias_type = BT_ALIGN;
    REQUIRE_FALSE(no_rows.active());

    // term_settings_of: the DockParam fields the parser sets
    DockParam dp;
    dp.bias_type = BT_POS; dp.bias_k = Real(0.07);
    const TermSettings from_dp = term_settings_of(dp);
    REQUIRE(from_dp.bias_type == BT_POS);
    REQUIRE(from_dp.bias_k == Real(0.07));
    REQUIRE(term_settings_of(DockParam{}).bias_type == BT_NO);

    // FlexParamVina carries the set, every term off by default
    FlexParamVina fp{};
    REQUIRE(fp.terms.settings.bias_type == BT_NO);
    REQUIRE(fp.terms.bias.inds == nullptr);
    REQUIRE_FALSE(fp.terms.active());
}
