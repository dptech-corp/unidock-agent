// Finite-difference check of torsion_penalty_ef in score/vina.h.
// The penalty is flat-bottomed, so a step that straddles a clamp boundary compares an analytic
// one-sided derivative with a numerical average across the kink. Those cases are skipped: they
// are a property of the flat-bottom form, not of the derivative.
#include "score/vina.h"
#include <cstdio>
#include <cmath>
#include <random>

static Real pen_only(const Real* p, int kind, Real tol){
    Real g[12];
    return torsion_penalty_ef(p, p + 3, p + 6, p + 9, kind, tol, g);
}

int main(){
    std::mt19937 rng(0);
    std::normal_distribution<double> N(0.0, 1.5);
    const Real h = (sizeof(Real) == 4) ? Real(1e-3) : Real(1e-6);
    // In float the check is noise-limited, not accuracy-limited: the penalty is O(1) and carries
    // ~1e-7 of roundoff, so a 1e-3 step leaves ~5e-5 of absolute noise in the numerical derivative,
    // which is several percent wherever the analytic gradient is small. The formula itself is
    // established by the double build (4e-08) and by the numpy reference (3e-09).
    const double tol_rel = (sizeof(Real) == 4) ? 5e-2 : 1e-5;
    double worst = 0;
    long checked = 0, skipped = 0;
    for (int t = 0; t < 2000; t++){
        Real p[12];
        for (int i = 0; i < 12; i++) p[i] = (Real)N(rng);
        const int kind = t % 3;
        const Real tol = 20.0;
        Real g[12];
        const Real v = torsion_penalty_ef(p, p + 3, p + 6, p + 9, kind, tol, g);
        if (v < 0 || v > 1){ printf("FAIL: value %f out of [0,1]\n", (double)v); return 1; }
        // interior only: at a clamp boundary the derivative is genuinely one-sided
        if (v <= Real(1e-6) || v >= Real(1) - Real(1e-6)) { skipped++; continue; }
        Real scale = Real(1e-12);
        for (int i = 0; i < 12; i++) if (fabs(g[i]) > scale) scale = fabs(g[i]);
        bool straddles = false;
        double err = 0;
        for (int i = 0; i < 12 && !straddles; i++){
            Real pp[12], pm[12];
            for (int k = 0; k < 12; k++){ pp[k] = p[k]; pm[k] = p[k]; }
            pp[i] += h; pm[i] -= h;
            const Real vp = pen_only(pp, kind, tol), vm = pen_only(pm, kind, tol);
            if (vp <= Real(1e-6) || vp >= Real(1) - Real(1e-6) ||
                vm <= Real(1e-6) || vm >= Real(1) - Real(1e-6)) { straddles = true; break; }
            const double num = (double)(vp - vm) / (2.0 * (double)h);
            err = fmax(err, fabs((double)g[i] - num) / (double)scale);
        }
        if (straddles){ skipped++; continue; }
        checked++;
        worst = fmax(worst, err);
    }
    printf("torsion_penalty_ef, Real=%s\n", sizeof(Real) == 4 ? "float" : "double");
    printf("  %ld quadruples checked, %ld skipped at a flat-bottom boundary\n", checked, skipped);
    printf("  max relative gradient error vs central FD: %.3e   (tolerance %.0e)\n", worst, tol_rel);
    if (checked < 100){ printf("FAIL: too few interior cases\n"); return 1; }
    return worst < tol_rel ? 0 : 1;
}
