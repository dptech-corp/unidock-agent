//
// CPU Common Implementation
// Created for Uni-Dock2 CPU backend
//

#include "common_cpu.h"
#include <spdlog/spdlog.h>

// Global CPU constants
CPUConstants g_cpu_constants;

void init_constants_cpu(const DockParam& dock_param, const AtomTypeProps& type_props,
                        const std::vector<Real>& vdw_radii_vec) {
    spdlog::debug("Initializing CPU constants...");
    
    g_cpu_constants.FLAG_CONSTRAINT_DOCK = dock_param.policy.static_root;
    g_cpu_constants.weights = dock_param.weights;
    g_cpu_constants.BOX_X_LO = dock_param.box.x_lo;
    g_cpu_constants.BOX_X_HI = dock_param.box.x_hi;
    g_cpu_constants.BOX_Y_LO = dock_param.box.y_lo;
    g_cpu_constants.BOX_Y_HI = dock_param.box.y_hi;
    g_cpu_constants.BOX_Z_LO = dock_param.box.z_lo;
    g_cpu_constants.BOX_Z_HI = dock_param.box.z_hi;
    g_cpu_constants.USE_INNER_BOX = dock_param.use_inner_box;
    g_cpu_constants.INNER_X_LO = dock_param.inner_box.x_lo;
    g_cpu_constants.INNER_X_HI = dock_param.inner_box.x_hi;
    g_cpu_constants.INNER_Y_LO = dock_param.inner_box.y_lo;
    g_cpu_constants.INNER_Y_HI = dock_param.inner_box.y_hi;
    g_cpu_constants.INNER_Z_LO = dock_param.inner_box.z_lo;
    g_cpu_constants.INNER_Z_HI = dock_param.inner_box.z_hi;
    g_cpu_constants.TOR_PREC = dock_param.tor_prec;
    g_cpu_constants.BOX_PREC = dock_param.box_prec;
    g_cpu_constants.PENALTY_SLOPE = 10.0;  // Default penalty slope
    g_cpu_constants.atp = type_props;
    g_cpu_constants.USE_ANGULAR_HBOND = dock_param.angular_hbond;
    g_cpu_constants.USE_POST_BFGS_HBOND_CORR = !dock_param.angular_hbond;

    // Populate VDW radii: use JSON-provided values if available, else hardcoded defaults
    if (!vdw_radii_vec.empty()) {
        int n = std::min(static_cast<int>(vdw_radii_vec.size()), MAX_VN_TYPES);
        for (int i = 0; i < n; i++) g_cpu_constants.vdw_radii[i] = vdw_radii_vec[i];
        for (int i = n; i < MAX_VN_TYPES; i++) g_cpu_constants.vdw_radii[i] = Real(0);
    } else {
        int n = sizeof(VN_VDW_RADII) / sizeof(VN_VDW_RADII[0]);
        for (int i = 0; i < n; i++) g_cpu_constants.vdw_radii[i] = VN_VDW_RADII[i];
        for (int i = n; i < MAX_VN_TYPES; i++) g_cpu_constants.vdw_radii[i] = Real(0);
    }

    spdlog::debug("CPU constants initialized: box=[{},{},{},{},{},{}]",
                  g_cpu_constants.BOX_X_LO, g_cpu_constants.BOX_X_HI,
                  g_cpu_constants.BOX_Y_LO, g_cpu_constants.BOX_Y_HI,
                  g_cpu_constants.BOX_Z_LO, g_cpu_constants.BOX_Z_HI);
}
