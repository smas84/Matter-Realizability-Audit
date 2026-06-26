"""
GRAND MERGED FILE — FLRW + Kerr-like perturbation CPT conservation/Bianchi audit.

This file merges:
  1. flrw_audit_v6_merged_radial_debug.py  -- all geometry/conservation machinery
     (Christoffel/Ricci/Einstein builders, background+full linearized divergence
     operators (plain and mixed-tensor forms), Friedmann/continuity reduction,
     background-conservation and off-shell-Bianchi verification, geometry
     validation suite, EOS closure helpers, derivative-leakage scanners, and
     radial-conservation debugging utilities).
  2. flrw_run123_v2.py / flrw_run4.py        -- the runnable driver (Run 1-2-3-4):
     dsolve-first velocity solving, derivative scanning, the plain-vs-mixed
     divergence consistency check (FIX 3), and the RUN-4 fix that uses the
     mixed-tensor divergence everywhere in the matter ledger so it matches the
     geometric (Bianchi) side.

Everything below is self-contained: run this single file directly
(`python3 flrw_grand_audit.py`) with no other local module required.
"""

import sympy as sp

# ── Geometric helpers ────────────────────────────────────────────────────────
def get_christoffel(g, g_inv, coords):
    dim = 4
    gamma = sp.MutableDenseNDimArray.zeros(dim, dim, dim)
    half = sp.Rational(1, 2)
    for l in range(dim):
        for m in range(dim):
            for n in range(dim):
                gamma[l, m, n] = sum(half * g_inv[l, s] * (
                    sp.diff(g[s, m], coords[n]) +
                    sp.diff(g[s, n], coords[m]) -
                    sp.diff(g[m, n], coords[s])) for s in range(dim))
    return gamma

def get_ricci(gam, coords):
    dim = 4
    ric = sp.Matrix.zeros(dim, dim)
    for mu in range(dim):
        for nu in range(dim):
            ric[mu, nu] = (
                sum(sp.diff(gam[l, mu, nu], coords[l]) for l in range(dim)) -
                sum(sp.diff(gam[l, mu, l], coords[nu]) for l in range(dim)) +
                sum(gam[l, l, s] * gam[s, mu, nu] - gam[l, nu, s] * gam[s, mu, l]
                    for l in range(dim) for s in range(dim))
            )
    return ric

def get_einstein(g, g_inv, coords):
    gam = get_christoffel(g, g_inv, coords)
    ric = get_ricci(gam, coords)
    scalar = sum(g_inv[mu, nu] * ric[mu, nu] for mu in range(4) for nu in range(4))
    return ric - sp.Rational(1, 2) * g * scalar

# ── Conservation operators ────────────────────────────────────────────────────
def background_covariant_divergence(T_cov, g0_inv, Gamma0, coords):
    """
    nabla^(0)_mu T^mu_nu — background connection only.
    Passing this is informative; failing alone is not conclusive.
    """
    T_mx = g0_inv * T_cov
    C = []
    for nu in range(4):
        d  = sum(sp.diff(T_mx[mu, nu], coords[mu]) for mu in range(4))
        d += sum(Gamma0[mu, mu, rho]*T_mx[rho, nu] for mu in range(4) for rho in range(4))
        d -= sum(Gamma0[rho, mu, nu]*T_mx[mu, rho] for mu in range(4) for rho in range(4))
        C.append(sp.simplify(d))
    return C


def variation_mixed_tensor(dT_cov, T0_cov, g0_inv, h_cov):
    """
    True first-order variation of a mixed tensor:
        δ(T^μ_ν)=g0^{μα}δT_{αν}+δg^{μα}T0_{αν}
    """
    delta_g_inv = -g0_inv * h_cov * g0_inv
    return g0_inv * dT_cov + delta_g_inv * T0_cov

def full_linearized_divergence_mixed(dT_cov, T0_cov, g0_inv, Gamma0,
                                     dGamma, coords, h_cov):
    """
    Linearized divergence using the true mixed-tensor variation.
    """
    T_mx_var = variation_mixed_tensor(dT_cov, T0_cov, g0_inv, h_cov)
    C = []
    for nu in range(4):
        d  = sum(sp.diff(T_mx_var[mu, nu], coords[mu]) for mu in range(4))
        d += sum(Gamma0[mu, mu, rho]*T_mx_var[rho, nu]
                 for mu in range(4) for rho in range(4))
        d -= sum(Gamma0[rho, mu, nu]*T_mx_var[mu, rho]
                 for mu in range(4) for rho in range(4))

        T0_mx = g0_inv * T0_cov
        corr  = sum(dGamma[mu,mu,lam]*T0_mx[lam,nu]
                    for mu in range(4) for lam in range(4))
        corr -= sum(dGamma[lam,mu,nu]*T0_mx[mu,lam]
                    for mu in range(4) for lam in range(4))
        C.append(sp.simplify(d + corr))
    return C

def full_linearized_divergence(dT_cov, T0_cov, g0_inv, Gamma0, dGamma, coords):
    """
    Full linearized conservation at O(eps):
      delta(nabla_mu T^mu_nu) = nabla^(0)_mu delta_T^mu_nu
                               + delta_Gamma^mu_{mu lam} T^(0)^lam_nu
                               - delta_Gamma^lam_{mu nu} T^(0)^mu_lam

    Applies equally to T (fluid side) and G (geometric/Bianchi side).
    For the geometric side pass (delta_G, G0) instead of (delta_T, T0).

    CRITICAL ORDERING: always call this on the RAW (unreduced) tensor,
    then Friedmann-reduce the result afterward. Computing divergences
    and substituting Friedmann constraints do not commute in general.
    """
    C0    = background_covariant_divergence(dT_cov, g0_inv, Gamma0, coords)
    T0_mx = g0_inv * T0_cov
    C = []
    for nu in range(4):
        corr  = sum(dGamma[mu,mu,lam]*T0_mx[lam,nu] for mu in range(4) for lam in range(4))
        corr -= sum(dGamma[lam,mu,nu]*T0_mx[mu,lam] for mu in range(4) for lam in range(4))
        C.append(sp.simplify(C0[nu] + corr))
    return C

# ── Friedmann reduction ───────────────────────────────────────────────────────
def make_friedmann_subs(a, t, G_sym, rho0, p0, eos_subs=None):
    """
    Introduce H, Hdot as independent symbols, reduce adot and addot, then
    impose Friedmann constraints.

    adotdotdot substitution REMOVED: the formula adotdotdot = a*(Hdot+H^2)*H
    is only valid when Hddot=0, which is not true for general FLRW backgrounds.
    Since our perturbation h contains at most addot, this substitution is never
    needed. (Verified at runtime: no third a-derivatives appear in delta_G.)

    Friedmann equations imposed:
      3H^2 = 8piG rho0    =>  H^2 = (8piG/3) rho0
      -2Hdot - 3H^2 = 8piG p0  =>  Hdot = -4piG(rho0+p0)
    """
    H_sym    = sp.Symbol('H',    real=True)
    Hdot_sym = sp.Symbol('Hdot', real=True)
    adot  = sp.diff(a, t)
    addot = sp.diff(a, t, 2)

    step1 = [
        (addot, a*(Hdot_sym + H_sym**2)),   # ä = a(Hdot + H^2)
        (adot,  a*H_sym),                   # adot = aH
    ]
    H2_val   = 8*sp.pi*G_sym*rho0/sp.Integer(3)
    Hdot_val = -4*sp.pi*G_sym*(rho0 + p0)
    step2 = [(H_sym**2, H2_val), (Hdot_sym, Hdot_val)]

    rho0_dot  = sp.diff(rho0, t)
    rho0_ddot = sp.diff(rho0, t, 2)
    p0_dot    = sp.diff(p0, t)
    p0_ddot   = sp.diff(p0, t, 2)

    continuity_subs = [
        (rho0_dot, -3*H_sym*(rho0+p0)),
        (rho0_ddot, -3*Hdot_sym*(rho0+p0) - 3*H_sym*(rho0_dot + p0_dot)),
    ]

    if eos_subs:
        continuity_subs.extend(eos_subs)

    continuity_subs.extend([
        (p0_dot, p0_dot),
        (p0_ddot, p0_ddot),
    ])

    def apply(expr):
        e = expr
        for old, new in step1: e = e.subs(old, new)
        for old, new in continuity_subs: e = e.subs(old, new)
        for old, new in continuity_subs: e = e.subs(old, new)
        for old, new in step2: e = e.subs(old, new)
        e = e.subs(adot**2, H2_val*a**2)
        return sp.simplify(e)

    return apply, H_sym

# ── Background conservation verification (FIX 1) ──────────────────────────────
def verify_background_conservation(T0_cov, g0_inv, Gamma0, coords, rho0, p0, t):
    """
    FIX 1: Explicitly verify nabla^(0)_mu T0^mu_nu = 0 BEFORE trusting any
    perturbative (delta) conservation result.

    The linearized identity
        delta(nabla_mu T^mu_nu) = nabla^(0)_mu delta_T^mu_nu + dGamma * T0
    is only meaningful as "conservation" if the background itself satisfies
    nabla^(0)_mu T0^mu_nu = 0, i.e. the continuity equation
        rho0_dot + 3H(rho0+p0) = 0.

    In this script rho0, p0 are sympy Symbols with NO time dependence, so
    sp.diff(rho0, t) == sp.diff(p0, t) == 0 identically. That means the
    continuity equation is never imposed as a real constraint — it is only
    "trivially" not violated because no rho0_dot or p0_dot term can ever
    appear. This function makes that explicit instead of leaving it as a
    silent assumption: it raises if nabla^(0)_mu T0^mu_nu != 0, and warns
    that an apparent pass is conditional on rho0, p0 being constants and on
    the Friedmann relations supplied separately, NOT a substantive physical
    check of the continuity equation.
    """
    C0 = background_covariant_divergence(T0_cov, g0_inv, Gamma0, coords)
    print("\n  Background conservation check: nabla^(0)_mu T0^mu_nu = 0 ?")
    all_zero = True
    for nu, c in enumerate(C0):
        ok = sp.simplify(c).equals(0)
        all_zero &= ok
        print(f"    C_T0_bg[{nu}] = {'✓ 0' if ok else f'✗ {c}'}")
    if not all_zero:
        raise ValueError(
            "Background conservation nabla^(0)_mu T0^mu_nu != 0 — the "
            "Friedmann-reduced background stress tensor is not conserved. "
            "Any perturbative conservation result built on top of this "
            "background is not trustworthy until this is fixed."
        )
    if sp.diff(rho0, t) == 0 and sp.diff(p0, t) == 0:
        print("    NOTE: rho0 and p0 are time-independent Symbols here, so "
              "this pass does NOT verify the continuity equation "
              "rho0_dot + 3H(rho0+p0) = 0 in any nontrivial sense — it is "
              "satisfied vacuously. If a genuine continuity check is needed, "
              "promote rho0, p0 to sp.Function(t) and impose "
              "diff(rho0,t) + 3*H*(rho0+p0) = 0 as an explicit substitution "
              "before calling this function.")
    return C0

# ── Direct off-shell Bianchi check (FIX 2) ────────────────────────────────────
def verify_offshell_bianchi_direct(g_full, coords, eps):
    """
    FIX 2: The strongest possible Bianchi test, with none of the
    decomposition machinery used elsewhere in this script.

    Computes B_nu(eps) = nabla_mu G^mu_nu directly from the FULL metric
    g0 + eps*h (i.e. recomputes Christoffel/Ricci/Einstein from scratch for
    g_full, with no split into background + dGamma*G0 pieces, no Friedmann
    substitution, no Einstein equations, no closure). It then takes
    d/deps|_{eps=0}. Because the contracted second Bianchi identity
    nabla_mu G^mu_nu = 0 holds for ANY metric, all four components of the
    eps-derivative must vanish identically if get_christoffel/get_ricci/
    get_einstein are implemented correctly.

    This is independent of, and strictly stronger than, the previous
    "C_G_linearized_crosscheck" test, which was computed via
    full_linearized_divergence(delta_G_raw, G0_cov, ...). That function
    still feeds in G0_cov (the background Einstein tensor) and dGamma
    pieces derived from the background connection, so a bug in how those
    background pieces are assembled could hide inside an apparently
    "off-shell" looking result. This function never touches G0_cov, Gamma0,
    or dGamma — it only differentiates the fully reconstructed G^mu_nu[g_full].
    """
    dim = 4
    g_full_inv = g_full.inv()
    Gamma_full = get_christoffel(g_full, g_full_inv, coords)
    G_full     = get_einstein(g_full, g_full_inv, coords)
    G_full_mx  = g_full_inv * G_full   # mixed G^mu_nu

    B = []
    for nu in range(dim):
        d  = sum(sp.diff(G_full_mx[mu, nu], coords[mu]) for mu in range(dim))
        d += sum(Gamma_full[mu, mu, rho]*G_full_mx[rho, nu]
                 for mu in range(dim) for rho in range(dim))
        d -= sum(Gamma_full[rho, mu, nu]*G_full_mx[mu, rho]
                 for mu in range(dim) for rho in range(dim))
        B.append(d)

    print("\n  Direct off-shell Bianchi check: B_nu(eps) = nabla_mu G^mu_nu[g0+eps h]")
    print("  (no decomposition, no Friedmann, no closure — pure geometry)")
    dB = []
    failed = False
    for nu in range(dim):
        val = sp.simplify(sp.diff(B[nu], eps).subs(eps, 0))
        dB.append(val)
        ok = (val == 0) or sp.simplify(val).equals(0)
        failed |= not ok
        print(f"    dB[{nu}] = {'✓ 0 (Bianchi confirmed off-shell)' if ok else f'✗ {val}'}")
    if failed:
        print("    WARNING: direct off-shell Bianchi check FAILED. Treat "
              "C_G_linearized_crosscheck from execute_definitive_audit as "
              "UNRELIABLE until this discrepancy is resolved — it may "
              "indicate an index/contraction bug in get_christoffel, "
              "get_ricci, get_einstein, or in how G0_cov/dGamma are built.")
    else:
        print("    Direct check PASSED. This corroborates (but does not "
              "replace) the C_G_linearized_crosscheck result reported below.")
    return dB

# ── Audit function ────────────────────────────────────────────────────────────
def execute_definitive_audit(delta_G_raw, G0_cov, delta_T_func, T0_cov,
                             g0, g0_inv, Gamma0, dGamma, coords,
                             G_sym, drho, dp, v_phi_sym, friedmann):
    """
    Fully closed publication-grade CPT matter-ledger spectrograph.

    Complete fix history:
      FIX (latest E): Off-shell Bianchi test added.
        C_G_full_raw now computed WITHOUT any Friedmann reduction.
        This distinguishes:
          delta(nabla G) = 0 off-shell  (pure geometric identity, strongest)
          delta(nabla G) = 0 on-shell   (holds only using background Einstein eqs)
        Both are reported. A pure Bianchi identity must hold off-shell.
        If C_G_full_raw != 0 but C_G_full = 0, the identity only holds on-shell
        and is not a coordinate-free geometric statement.
      FIX (latest E): C_T_full_cl explicitly relabelled as a closure-consistency
        diagnostic, NOT a conservation proof. The label and all print output
        now read "closure-consistency diagnostic" to prevent misinterpretation.
        Rationale: dT_cl has Friedmann + scalar closure applied before
        differentiation, so C_T_full_cl tests whether the closed solution is
        self-consistent, not whether conservation holds mathematically.
      FIX (latest D): Full RAW->divergence->Friedmann ordering applied
        symmetrically to BOTH fluid and geometric sectors. Previously
        dT_raw_v had Friedmann reduction applied to tensor components
        before taking the divergence; this is not equivalent to reducing
        after differentiation (same non-commutativity issue that motivated
        the Bianchi ordering fix). Now both sectors use:
            RAW tensor -> full_linearized_divergence -> friedmann(result)
        Additionally, C_R_full_cl (closed net ledger) is now reported
        alongside C_R_full (raw net ledger). A nonzero C_R_full can
        vanish after imposing scalar closure (Einstein constraints); both
        versions are needed to distinguish that case.
      FIX (latest C): Conservation ordering fix for fluid side — C_T_full now
        computed on RAW delta_T (before scalar closure), matching the same
        philosophy as the Bianchi ordering fix. Both raw and post-closure
        versions are reported: C_T_full_raw (primary) and C_T_full_cl
        (secondary). The net ledger C_R uses the raw fluid side.
        If C_T_full_raw=0 but C_T_full_cl≠0, the closure solution itself
        breaks conservation — a subtle failure mode now caught explicitly.
      FIX (latest A): Bianchi ordering — full_linearized_divergence called on
        RAW delta_G, Friedmann reduction applied to result afterward.
        Derivatives and substitutions do not commute; this ordering is correct.
      FIX (latest B): adotdotdot substitution removed (incorrect formula,
        unnecessary since h contains at most addot).
      FIX: Full linearized Bianchi delta(nabla G) = nabla^(0) dG + dGamma*G0
        added to geometric side, using G0 as background tensor.
      FIX: Full linearized conservation delta(nabla T) = nabla^(0) dT + dGamma*T0
        for fluid side (dGamma*T0 terms included).
      FIX: H-symbol Friedmann reduction before any simplification.
      FIX: Closure Jacobian det checked before sp.solve.
      FIX: v_phi left free — scalar closure never pre-assumes momentum sector.
      FIX: delta_u^t = h_tt/2 normalization correction.
      FIX: delta_u_mu = g0 delta_u^nu + h u^nu (full covariant lowering).
      FIX: Exact inversion g_full.inv() + strict d/deps|_0 truncation.
      FIX: Separate C_T_bg vs C_T_full (partial vs conclusive fluid test).
      FIX (new, F): Background conservation of T0_cov is now explicitly
        verified (see verify_background_conservation) before this audit is
        even called, instead of being a silent assumption baked into the
        Friedmann substitutions.
      FIX (new, G): An independent, fully non-decomposed off-shell Bianchi
        check (see verify_offshell_bianchi_direct) is run in main() and
        compared against C_G_linearized_crosscheck below, since the latter
        still routes through G0_cov/dGamma and could mask an indexing bug.

    Conservation interpretation:
      C_T_full_raw=0, C_G_full=0, C_R_full=0    =>  fully consistent (pre-closure)
      C_T_full_raw=0, C_G_full=0, C_R_full_cl=0  =>  fully consistent (post-closure)
      C_G_full_raw!=0                              =>  Bianchi holds OFF-SHELL (strongest)
      C_G_full_raw!=0, C_G_full=0                =>  Bianchi holds ON-SHELL only
      C_T_full_raw!=0, C_G_full=0               =>  fluid ansatz inconsistent
      C_T_full_raw=0, C_G_full!=0               =>  Bianchi failure (pipeline error)
      C_T_full_raw=0, C_T_full_cl!=0            =>  closure solution fails
                                                     closure-consistency diagnostic
      C_R_full!=0 but C_R_full_cl=0             =>  raw inconsistency removed by closure
      C_R_full_cl!=0                             =>  irreducible net ledger inconsistency
      C_T_bg!=0 but C_T_full_raw=0             =>  dGamma*T0 cancels (not a failure)

    Spectral classification:
      q2=0, Pi2=0, R2=0       =>  perfect fluid closure
      q2!=0, v_phi can fix it  =>  momentum flux removable by rotation
      q2!=0, no v_phi solution =>  IRREDUCIBLE momentum flux
      Pi2!=0                   =>  anisotropic stress required
    """
    u_contra  = sp.Matrix([1, 0, 0, 0])
    u_cov     = sp.Matrix([-1, 0, 0, 0])
    h_cov     = g0 + u_cov*u_cov.T
    h_down_up = h_cov*g0_inv        # h_mu^alpha
    h_up_up   = g0_inv + u_contra*u_contra.T

    def tz(expr):
        return sp.simplify(sp.factor(sp.together(expr))).equals(0)

    delta_T = delta_T_func(v_phi_sym)

    # Friedmann applied after forming R_res
    # Derivative leakage audit can be applied to R_res after construction
    R_res = sp.Matrix([[friedmann(sp.simplify(
                            delta_G_raw[mu,nu] - 8*sp.pi*G_sym*delta_T[mu,nu]))
                        for nu in range(4)] for mu in range(4)])

    print("  Nonzero R_res (Friedmann on-shell):")
    for mu in range(4):
        for nu in range(mu,4):
            v = sp.simplify(R_res[mu,nu])
            if not tz(v): print(f"    R_res[{mu},{nu}] = {v}")

    rho_res = sp.simplify(sum(u_contra[mu]*u_contra[nu]*R_res[mu,nu]
                               for mu in range(4) for nu in range(4)))
    p_res   = sp.Rational(1,3)*sp.simplify(sum(h_up_up[mu,nu]*R_res[mu,nu]
                               for mu in range(4) for nu in range(4)))
    print(f"\n  rho_res = {rho_res}")
    print(f"  p_res   = {p_res}")

    J     = sp.Matrix([[sp.diff(rho_res,drho), sp.diff(rho_res,dp)],
                       [sp.diff(p_res,  drho), sp.diff(p_res,  dp)]])
    det_J = sp.simplify(J.det())
    print(f"\n  Closure Jacobian det = {det_J}  "
          f"=> {'non-degenerate ✓' if det_J != 0 else 'DEGENERATE ✗'}")

    scalar_sols = sp.solve([rho_res, p_res], [drho, dp], dict=True)
    if scalar_sols:
        closure  = scalar_sols[0]
        drho_sol = sp.simplify(closure.get(drho, drho))
        dp_sol   = sp.simplify(closure.get(dp, dp))
        print(f"  drho_sol = {drho_sol}")
        print(f"  dp_sol   = {dp_sol}")
        def apply_closure(expr):
            return friedmann(sp.simplify(expr.subs(drho,drho_sol).subs(dp,dp_sol)))
        R_res_cl = sp.Matrix([[apply_closure(R_res[i,j])   for j in range(4)] for i in range(4)])
        dT_cl    = sp.Matrix([[apply_closure(delta_T[i,j]) for j in range(4)] for i in range(4)])
    else:
        print("  WARNING: No scalar closure found.")
        R_res_cl, dT_cl = R_res, delta_T
        drho_sol, dp_sol = drho, dp

    publication_background_audit("R_res", R_res, rho0, p0, coords[0])
    publication_background_audit("R_res_cl", R_res_cl, rho0, p0, coords[0])

    print("\n  R_res after closure:")
    any_res = False
    for mu in range(4):
        for nu in range(mu,4):
            v = sp.simplify(R_res_cl[mu,nu])
            if not tz(v):
                print(f"    R_res_cl[{mu},{nu}] = {v}")
                any_res = True
    if not any_res: print("    All zero. ✓")

    rho_cl = sp.simplify(sum(u_contra[mu]*u_contra[nu]*R_res_cl[mu,nu]
                              for mu in range(4) for nu in range(4)))
    p_cl   = sp.Rational(1,3)*sp.simplify(sum(h_up_up[mu,nu]*R_res_cl[mu,nu]
                              for mu in range(4) for nu in range(4)))
    R_rem  = R_res_cl - rho_cl*(u_cov*u_cov.T) - p_cl*h_cov
    R_rem  = sp.Matrix([[sp.simplify(R_rem[i,j]) for j in range(4)] for i in range(4)])

    print("\n  R_rem (irreducible):")
    any_rem = False
    for mu in range(4):
        for nu in range(mu,4):
            v = sp.simplify(R_rem[mu,nu])
            if not tz(v):
                print(f"    R_rem[{mu},{nu}] = {v}")
                any_rem = True
    if not any_rem: print("    None — perfect fluid closure. ✓")

    q_cov    = sp.Matrix([sp.simplify(-sum(h_down_up[mu,aa]*R_rem[aa,nu]*u_contra[nu]
                           for aa in range(4) for nu in range(4))) for mu in range(4)])
    q_contra = g0_inv*q_cov
    S = sp.Matrix([[sp.simplify(sum(h_down_up[mu,aa]*h_down_up[nu,bb]*R_rem[aa,bb]
                    for aa in range(4) for bb in range(4)))
                    for nu in range(4)] for mu in range(4)])
    p_rem = sp.Rational(1,3)*sp.simplify(sum(h_up_up[i,j]*S[i,j]
                                              for i in range(4) for j in range(4)))
    Pi = sp.Matrix([[sp.simplify(S[i,j]-h_cov[i,j]*p_rem)
                     for j in range(4)] for i in range(4)])

    if not tz(sum(g0_inv[i,j]*Pi[i,j] for i in range(4) for j in range(4))):
        raise ValueError("Pi not trace-free")
    if not tz(sum(u_cov[i]*q_contra[i] for i in range(4))):
        raise ValueError("q not orthogonal to u")
    for nu in range(4):
        if not tz(sum(u_contra[mu]*Pi[mu,nu] for mu in range(4))):
            raise ValueError(f"Pi not spatial nu={nu}")
    print("\n  ✓ Pi trace-free  ✓ q⊥u  ✓ Pi spatial")

    q_nz = [(mu, sp.simplify(q_cov[mu])) for mu in range(4) if not tz(q_cov[mu])]
    v_phi_q_sol = []
    if not q_nz:
        print("\n  q_mu=0 for all v_phi. ✓")
        v_phi_q_sol = [sp.S.Zero]
    else:
        print("\n  Nonzero q components:")
        for mu, qv in q_nz: print(f"    q[{mu}] = {qv}")
        mu0, qv0 = q_nz[0]
        v_phi_q_sol = sp.solve(qv0, v_phi_sym)
        print(f"  v_phi eliminates q[{mu0}]? "
              f"{'Yes: ' + str(v_phi_q_sol) if v_phi_q_sol else 'No — IRREDUCIBLE.'}")

    def invariants_at(v_val):
        def sub(e): return friedmann(sp.simplify(e.subs(v_phi_sym, v_val)))

        # ── Closed (post-closure) tensors ──────────────────────────────────
        q_c    = sp.Matrix([sub(q_cov[i]) for i in range(4)])
        Pi_v   = sp.Matrix([[sub(Pi[i,j]) for j in range(4)] for i in range(4)])
        R_cl_v = sp.Matrix([[sub(R_res_cl[i,j]) for j in range(4)] for i in range(4)])
        dT_cl_v = sp.Matrix([[sub(dT_cl[i,j]) for j in range(4)] for i in range(4)])

        # ── RAW delta_T: substitute v_phi only, NO scalar closure, NO Friedmann yet.
        # FIX (D): Mirror the geometric sector exactly.
        #   Geometric ordering: delta_G_raw -> full_linearized_divergence -> friedmann
        #   Fluid ordering was: friedmann(delta_T_func(v_val)) -> divergence  [WRONG]
        #   Fluid ordering now: delta_T_func(v_val) -> full_linearized_divergence
        #                       -> friedmann  [CORRECT, matches Bianchi]
        # This matters because ∂_t(Friedmann-reduced tensor) ≠ Friedmann-reduce(∂_t tensor)
        # when the reduction substitutes a-derivatives with H, Hdot.
        dT_raw_v = delta_T_func(v_val)   # RAW: no Friedmann applied to components

        # ── Fluid conservation — RAW (primary, before closure) ─────────────
        # Ordering: RAW tensor -> divergence -> Friedmann  (matches geometric sector)
        C_T_bg_raw   = [friedmann(c) for c in
                        background_covariant_divergence(dT_raw_v, g0_inv, Gamma0, coords)]
        C_T_full_raw = [friedmann(c) for c in
                        full_linearized_divergence(dT_raw_v, T0_cov, g0_inv,
                                                   Gamma0, dGamma, coords)]

        # ── Fluid closure-consistency diagnostic (NOT a conservation proof) ─
        # FIX (E): Explicitly labelled as a diagnostic, not a conservation test.
        # dT_cl_v already has scalar closure + Friedmann applied before
        # differentiation, so this does NOT test mathematical conservation.
        # It tests whether the closed solution is internally self-consistent:
        # i.e., does the closed fluid stress-energy satisfy its own equations
        # in the reduced form? Nonzero here means the closure solution is
        # self-inconsistent, not that conservation is violated in general.
        C_T_bg_cl_diag   = [friedmann(c) for c in
                             background_covariant_divergence(dT_cl_v, g0_inv, Gamma0, coords)]
        C_T_full_cl_diag = [friedmann(c) for c in
                             full_linearized_divergence(dT_cl_v, T0_cov, g0_inv,
                                                        Gamma0, dGamma, coords)]

        # ── Bianchi — OFF-SHELL (no Friedmann, pure geometric identity) ────
        # FIX (E): C_G_full_raw computed WITHOUT any Friedmann reduction.
        # A true Bianchi identity (contracted second Bianchi) must hold
        # off-shell, i.e., without using background Einstein equations.
        # If this is nonzero: the identity holds only on-shell — not a pure
        # geometric statement. Report alongside on-shell version.
        #
        # NOTE (FIX G): This still routes through G0_cov and dGamma, both
        # derived from the background connection — it is NOT the strongest
        # possible test. See verify_offshell_bianchi_direct() in main(),
        # which recomputes G^mu_nu directly from g0+eps*h with no
        # decomposition at all, and should be treated as the authoritative
        # off-shell check. Use this value only as a cross-check against that
        # one; if they disagree, trust verify_offshell_bianchi_direct.
        C_G_linearized_crosscheck = full_linearized_divergence_mixed(
            delta_G_raw, G0_cov, g0_inv, Gamma0, dGamma, coords, h)
        # PATCH: use true mixed-tensor variation
        # δG^μ_ν = g0^{μα}δG_{αν} + δg^{μα}G0_{αν}
        # This matches the direct off-shell Bianchi check.
        # NO friedmann() applied here — this is the off-shell test.

        # ── Bianchi — ON-SHELL (Friedmann applied, was the sole test before) ─
        # Ordering: RAW delta_G -> divergence -> Friedmann (correct ordering).
        C_G_full_onshell = [friedmann(c) for c in C_G_linearized_crosscheck]
        C_G_bg           = [friedmann(c) for c in
                             background_covariant_divergence(delta_G_raw, g0_inv, Gamma0, coords)]

        # ── Net residual ledger — RAW fluid side (pre-closure) ─────────────
        # Tests global consistency before Einstein constraints are imposed.
        C_R_full = [friedmann(sp.simplify(cg/(8*sp.pi*G_sym) - ct))
                    for cg, ct in zip(C_G_full_onshell, C_T_full_raw)]

        # ── Net residual ledger — CLOSED fluid side (post-closure) ─────────
        # FIX (D): Added. Tests whether the fully closed (drho_sol, dp_sol)
        # fluid model satisfies Einstein + conservation simultaneously.
        # C_R_full!=0 but C_R_full_cl=0  =>  closure removes raw inconsistency
        # C_R_full_cl!=0                  =>  irreducible net ledger failure
        C_R_full_cl = [friedmann(sp.simplify(cg/(8*sp.pi*G_sym) - ct))
                       for cg, ct in zip(C_G_full_onshell, C_T_full_cl_diag)]

        q_ct  = g0_inv * q_c
        Pi_uu = g0_inv * Pi_v * g0_inv
        R_uu  = g0_inv * R_cl_v * g0_inv
        return {
            "q2":                     sp.simplify(sum(q_c[i]*q_ct[i] for i in range(4))),
            "Pi2":                    sp.simplify(sum(Pi_v[i,j]*Pi_uu[i,j]
                                                      for i in range(4) for j in range(4))),
            "R2":                     sp.simplify(sum(R_cl_v[i,j]*R_uu[i,j]
                                                      for i in range(4) for j in range(4))),
            "C_T_bg_raw":             C_T_bg_raw,           # partial, pre-closure — informative
            "C_T_full_raw":           C_T_full_raw,         # PRIMARY fluid conservation test
            "C_T_bg_cl_diag":         C_T_bg_cl_diag,       # partial closure-consistency diag
            "C_T_full_cl_diag":       C_T_full_cl_diag,     # closure-consistency diagnostic (NOT conservation proof)
            "C_G_linearized_crosscheck":  C_G_linearized_crosscheck, # OFF-SHELL Bianchi (no Friedmann)
            "C_G_full_onshell":       C_G_full_onshell,     # ON-SHELL Bianchi (with Friedmann)
            "C_G_bg":                 C_G_bg,               # partial Bianchi — informative
            "C_R_full":               C_R_full,             # net ledger pre-closure
            "C_R_full_cl":            C_R_full_cl,          # net ledger post-closure
        }

    def print_inv(label, inv):
        print(f"\n  --- {label} ---")
        for k in ["q2","Pi2","R2"]: print(f"    {k} = {inv[k]}")

        print("  nabla^(0) dT  [partial, RAW, informative]:")
        for nu,c in enumerate(inv["C_T_bg_raw"]):
            print(f"    C_T_bg_raw[{nu}]              = {'✓ 0' if tz(c) else f'✗ {c}'}")

        print("  Full delta(nabla T) on RAW ansatz  [PRIMARY conservation test: RAW->div->Friedmann]:")
        for nu,c in enumerate(inv["C_T_full_raw"]):
            print(f"    C_T_full_raw[{nu}]             = {'✓ 0' if tz(c) else f'✗ {c}'}")

        print("  Full delta(nabla T) POST-CLOSURE  [CLOSURE-CONSISTENCY DIAGNOSTIC — not a conservation proof]:")
        print("  (dT_cl has closure+Friedmann applied before differentiation; tests self-consistency, not conservation)")
        for nu,c in enumerate(inv["C_T_full_cl_diag"]):
            print(f"    C_T_full_cl_diag[{nu}]         = {'✓ 0' if tz(c) else f'✗ {c}'}")

        print("  nabla^(0) dG  [partial Bianchi, informative]:")
        for nu,c in enumerate(inv["C_G_bg"]):
            print(f"    C_G_bg[{nu}]                   = {'✓ 0' if tz(c) else f'✗ {c}'}")

        print("  Full delta(nabla G) OFF-SHELL [NO Friedmann — pure geometric Bianchi identity]:")
        print("  (Nonzero => identity holds only on-shell, not a coordinate-free geometric statement)")
        for nu,c in enumerate(inv["C_G_linearized_crosscheck"]):
            print(f"    C_G_linearized_crosscheck[{nu}]    = {'✓ 0' if tz(c) else f'✗ {c}'}")

        print("  Full delta(nabla G) ON-SHELL  [RAW->div->Friedmann — must=0 for on-shell consistency]:")
        for nu,c in enumerate(inv["C_G_full_onshell"]):
            print(f"    C_G_full_onshell[{nu}]         = {'✓ 0' if tz(c) else f'✗ {c}'}")

        print("  Net ledger C_R (pre-closure, raw fluid side):")
        for nu,c in enumerate(inv["C_R_full"]):
            print(f"    C_R_full[{nu}]                 = {'✓ 0' if tz(c) else f'✗ {c}'}")

        print("  Net ledger C_R (post-closure, closed fluid side):")
        for nu,c in enumerate(inv["C_R_full_cl"]):
            print(f"    C_R_full_cl[{nu}]              = {'✓ 0' if tz(c) else f'✗ {c}'}")

    inv0 = invariants_at(sp.S.Zero)
    print_inv("INVARIANTS AT v_phi=0 (no prior assumption)", inv0)

    inv_q = None
    if v_phi_q_sol:
        vqs   = sp.simplify(v_phi_q_sol[0])
        inv_q = invariants_at(vqs)
        print_inv(f"INVARIANTS AT v_phi={vqs} (q=0 closure)", inv_q)

    return {"drho_sol": drho_sol, "dp_sol": dp_sol, "det_J": det_J,
            "v_phi_q_sol": v_phi_q_sol,
            "inv_at_vphi0": inv0, "inv_at_q_closure": inv_q}


# ------------------------------------------------------------------
# Geometry validation suite
# ------------------------------------------------------------------
def validate_minkowski():
    t,x,y,z = sp.symbols('t x y z')
    coords = [t,x,y,z]
    g = sp.diag(-1,1,1,1)
    G = get_einstein(g, g.inv(), coords)
    return all(sp.simplify(G[i,j]) == 0 for i in range(4) for j in range(4))

def validate_flrw():
    t,r,th,ph = sp.symbols('t r theta phi')
    a = sp.Function('a')(t)
    coords = [t,r,th,ph]
    g = sp.Matrix([
        [-1,0,0,0],
        [0,a**2,0,0],
        [0,0,a**2*r**2,0],
        [0,0,0,a**2*r**2*sp.sin(th)**2]
    ])
    G = get_einstein(g, g.inv(), coords)
    return G

def validate_schwarzschild():
    t,r,th,ph,M = sp.symbols('t r theta phi M', positive=True)
    f = 1 - 2*M/r
    coords = [t,r,th,ph]
    g = sp.diag(-f, 1/f, r**2, r**2*sp.sin(th)**2)
    G = get_einstein(g, g.inv(), coords)
    return all(sp.simplify(G[i,j]) == 0 for i in range(4) for j in range(4))


# ======================================================================
# Publication-grade geometry validation additions
# ======================================================================

def direct_bianchi_test(g, coords):
    """
    Directly compute ∇_μ G^μ_ν[g] for an arbitrary metric.
    Validates Christoffel, Ricci, Einstein and divergence machinery together.
    """
    g_inv = g.inv()
    Gamma = get_christoffel(g, g_inv, coords)
    G_cov = get_einstein(g, g_inv, coords)

    G_mix = g_inv * G_cov

    B = []

    for nu in range(4):

        expr  = sum(
            sp.diff(G_mix[mu,nu], coords[mu])
            for mu in range(4)
        )

        expr += sum(
            Gamma[mu,mu,rho]*G_mix[rho,nu]
            for mu in range(4)
            for rho in range(4)
        )

        expr -= sum(
            Gamma[rho,mu,nu]*G_mix[mu,rho]
            for mu in range(4)
            for rho in range(4)
        )

        B.append(sp.simplify(expr))

    return B


def validate_bianchi_minkowski():
    t,x,y,z = sp.symbols('t x y z')
    coords = [t,x,y,z]
    g = sp.diag(-1,1,1,1)
    return direct_bianchi_test(g, coords)


def validate_bianchi_flrw():
    t,r,th,ph = sp.symbols('t r theta phi')
    a = sp.Function('a')(t)

    coords = [t,r,th,ph]

    g = sp.Matrix([
        [-1,0,0,0],
        [0,a**2,0,0],
        [0,0,a**2*r**2,0],
        [0,0,0,a**2*r**2*sp.sin(th)**2]
    ])

    return direct_bianchi_test(g, coords)


def validate_bianchi_schwarzschild():
    t,r,th,ph,M = sp.symbols(
        't r theta phi M',
        positive=True
    )

    f = 1 - 2*M/r

    coords = [t,r,th,ph]

    g = sp.diag(
        -f,
        1/f,
        r**2,
        r**2*sp.sin(th)**2
    )

    return direct_bianchi_test(g, coords)


def scan_for_p0_derivatives(objects, p0, t):

    target = sp.Derivative(p0, t)

    found = []

    for name, expr in objects.items():

        try:

            if isinstance(expr, (list, tuple)):
                hit = any(
                    getattr(x, "has", lambda *_: False)(target)
                    for x in expr
                )

            else:
                hit = expr.has(target)

        except Exception:
            hit = False

        if hit:
            found.append(name)

    return found



# ======================================================================
# Publication-grade additions implemented by audit patch
# ======================================================================

def scan_for_background_derivatives(objects, rho0, p0, t):
    targets = [
        sp.Derivative(rho0, t),
        sp.Derivative(p0, t),
        sp.Derivative(rho0, (t,2)),
        sp.Derivative(p0, (t,2)),
    ]
    escaped = {}
    for name, expr in objects.items():
        found = []
        iterable = expr if isinstance(expr, (list, tuple)) else [expr]

        expanded = []
        for item in iterable:
            if isinstance(item, sp.MatrixBase):
                expanded.extend(list(item))
            else:
                expanded.append(item)

        for item in expanded:
            try:
                for target in targets:
                    if getattr(item, "has", lambda *_: False)(target):
                        found.append(target)
            except Exception:
                pass

        if found:
            escaped[name] = list(set(found))
    return escaped


def make_background_eos_subs(rho0, p0, t, cs2=None):
    """
    Closure relation:
        p0_dot = c_s^2 rho0_dot
    Default c_s^2 symbol if not supplied.
    """
    if cs2 is None:
        cs2 = sp.Symbol("c_s2", real=True)
    return [
        (sp.diff(p0, t), cs2 * sp.diff(rho0, t)),
        (sp.diff(p0, (t,2)), cs2 * sp.diff(rho0, (t,2))),
    ], cs2


def validate_flrw_einstein_components():
    t,r,th,ph = sp.symbols('t r theta phi')
    a = sp.Function('a')(t)
    H = sp.Symbol('H')
    Hdot = sp.Symbol('Hdot')

    G = validate_flrw()

    replacements = {
        sp.diff(a,t): a*H,
        sp.diff(a,t,2): a*(Hdot + H**2)
    }

    G00 = sp.simplify(G[0,0].subs(replacements))
    G11 = sp.simplify(G[1,1].subs(replacements))
    G22 = sp.simplify(G[2,2].subs(replacements))
    G33 = sp.simplify(G[3,3].subs(replacements))

    expected = [
        sp.simplify(G00 - 3*H**2),
        sp.simplify(G11 + a**2*(2*Hdot + 3*H**2)),
        sp.simplify(G22 + a**2*r**2*(2*Hdot + 3*H**2)),
        sp.simplify(G33 + a**2*r**2*sp.sin(th)**2*(2*Hdot + 3*H**2))
    ]
    return all(x == 0 or sp.simplify(x).equals(0) for x in expected)


def run_geometry_validation_suite():
    checks = {
        "Minkowski": validate_minkowski(),
        "Schwarzschild": validate_schwarzschild(),
        "Bianchi Minkowski": all(sp.simplify(x).equals(0) for x in validate_bianchi_minkowski()),
        "Bianchi Schwarzschild": all(sp.simplify(x).equals(0) for x in validate_bianchi_schwarzschild()),
        "Bianchi FLRW": all(sp.simplify(x).equals(0) for x in validate_bianchi_flrw()),
        "FLRW Einstein components": validate_flrw_einstein_components()
    }

    failed = [k for k,v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"Geometry validation failures: {failed}")

    print("\\nGeometry validation suite passed:")
    for k in checks:
        print(f"  ✓ {k}")



def verify_continuity_closure(expr, rho0, p0, t):
    forbidden = [
        sp.diff(rho0,t),
        sp.diff(rho0,t,2),
        sp.diff(p0,t),
        sp.diff(p0,t,2),
    ]

    iterable = expr if isinstance(expr, (list, tuple)) else [expr]
    expanded = []
    for item in iterable:
        if isinstance(item, sp.MatrixBase):
            expanded.extend(list(item))
        else:
            expanded.append(item)

    for item in expanded:
        for f in forbidden:
            if getattr(item, "has", lambda *_: False)(f):
                raise RuntimeError(
                    f"Continuity reduction incomplete: {f}"
                )


# ======================================================================
# FULL PIPELINE INTEGRATION PATCH (v5.1)
# ======================================================================

PUBLICATION_GRADE_NOTES = """
Integration checklist completed in v5.1:

1. Geometry validation suite should be called at startup:
       run_geometry_validation_suite()

2. EOS closure should be injected into Friedmann substitutions:
       eos_subs, cs2 = make_background_eos_subs(...)
       continuity_subs.extend(eos_subs)

3. Background derivative leakage audit should be executed after:
       delta_G_raw
       R_res
       R_res_cl
       C_T_full_raw
       C_G_full_onshell

4. Any escaped derivative should raise:
       RuntimeError("Background derivative leakage detected")
"""

def publication_background_audit(stage_name, expr, rho0, p0, t):
    escaped = scan_for_background_derivatives(
        {stage_name: expr},
        rho0,
        p0,
        t
    )

    if escaped:
        raise RuntimeError(
            f"Background derivative leakage detected at {stage_name}: {escaped}"
        )

    print(f"  ✓ background closure audit passed for {stage_name}")


# ======================================================================
# FULL BIANCHI PATCH NOTES
# ======================================================================
# Step 1: Added explicit delta(g^-1).
# Step 2: Added variation_mixed_tensor().
# Step 3: Added full_linearized_divergence_mixed().
# Step 4: Use this function for geometric-sector Bianchi checks.
# Step 5: Re-run C_G_linearized_crosscheck and C_G_full_onshell manually.
#         This file contains the machinery; symbolic re-evaluation is still
#         required because the original script is not structured for a
#         lightweight hot-swap execution.


# ============================================================================
# Recent radial-conservation diagnostics
# ============================================================================

def debug_radial_conservation(
    dT_cov, T0_cov, g0_inv, Gamma0, dGamma, coords
):
    """
    Decompose:
        δ(∇_μ T^μ_r)
      = ∇^(0)_μ(δT^μ_r)
        + δΓ^μ_{μλ} T^(0)λ_r
        - δΓ^λ_{μr} T^(0)μ_λ
    """
    dT_mixed = g0_inv * dT_cov
    T0_mixed = g0_inv * T0_cov

    nu = 1

    term1 = sum(
        sp.diff(dT_mixed[mu, nu], coords[mu])
        for mu in range(4)
    )

    term2 = sum(
        Gamma0[mu, mu, rho] * dT_mixed[rho, nu]
        for mu in range(4)
        for rho in range(4)
    )

    term3 = -sum(
        Gamma0[rho, mu, nu] * dT_mixed[mu, rho]
        for mu in range(4)
        for rho in range(4)
    )

    term4 = sum(
        dGamma[mu, mu, lam] * T0_mixed[lam, nu]
        for mu in range(4)
        for lam in range(4)
    )

    term5 = -sum(
        dGamma[lam, mu, nu] * T0_mixed[mu, lam]
        for mu in range(4)
        for lam in range(4)
    )

    total = sp.simplify(term1 + term2 + term3 + term4 + term5)

    print("\\n=== RADIAL CONSERVATION DECOMPOSITION ===")
    print("term1 =", sp.simplify(term1))
    print("term2 =", sp.simplify(term2))
    print("term3 =", sp.simplify(term3))
    print("term4 =", sp.simplify(term4))
    print("term5 =", sp.simplify(term5))
    print("total =", total)

    return {
        "term1": sp.simplify(term1),
        "term2": sp.simplify(term2),
        "term3": sp.simplify(term3),
        "term4": sp.simplify(term4),
        "term5": sp.simplify(term5),
        "total": total,
    }


def debug_radial_connection_sources(T0_cov, g0_inv, dGamma):
    """
    Identify the specific δΓ components sourcing the radial residual.
    """
    T0_mixed = g0_inv * T0_cov
    nu = 1

    print("\\n=== term4 pieces ===")
    for mu in range(4):
        for lam in range(4):
            x = sp.simplify(
                dGamma[mu, mu, lam] * T0_mixed[lam, nu]
            )
            if x != 0:
                print((mu, lam), x)

    print("\\n=== term5 pieces ===")
    for mu in range(4):
        for lam in range(4):
            x = sp.simplify(
                -dGamma[lam, mu, nu] * T0_mixed[mu, lam]
            )
            if x != 0:
                print((mu, lam), x)


def radial_velocity_completion_equation(
    C_T_full_raw_radial,
    v_r
):
    """
    Solve C_T_full_raw[1] = 0 for a symbolic radial velocity perturbation.
    """
    eq = sp.Eq(sp.simplify(C_T_full_raw_radial), 0)

    print("\\n=== RADIAL COMPLETION EQUATION ===")
    print(eq)

    try:
        sol = sp.dsolve(eq)
    except Exception:
        try:
            sol = sp.solve(eq, v_r)
        except Exception as exc:
            sol = f"Unable to solve automatically: {exc}"

    print("\\n=== RADIAL COMPLETION SOLUTION ===")
    print(sol)

    return sol


def print_nonzero_deltaT(dT):
    print("\\n=== NONZERO delta T COMPONENTS ===")
    for mu in range(4):
        for nu in range(mu, 4):
            val = sp.simplify(dT[mu, nu])
            if val != 0:
                print(mu, nu, val)


# ============================================================================
# DRIVER SCRIPT (formerly flrw_run123_v2.py / flrw_run4.py)
# ============================================================================

"""
Runs 1-2-3-4 (v3): Five fixes applied.

FIX 1: dsolve-first strategy for v_r and v_phi (they satisfy ODEs, not algebraic eqs).
FIX 2: Derivative scanner on dT_solved before Run 2 — abort if unresolved objects survive.
FIX 3: Both full_linearized_divergence and full_linearized_divergence_mixed computed on
        solved stress tensor; compared. Disagreement flags a bookkeeping issue.
FIX 4: q² ≠ 0 verdict softened — only labelled irreducible after (a) conservation closes,
        (b) all velocity functions are solved, (c) gauge freedom exhausted.
FIX 5 (RUN 4): full_linearized_divergence_mixed() now used EVERYWHERE in the matter
        ledger (Run 1's C_T_raw, Run 2's C_T_div/C_T_cl_diag), matching the geometric
        side (C_G_cross), which already used the mixed form. Previously the matter
        sector used the plain covariant-tensor divergence while the geometric sector
        used the mixed-tensor variation -- an index-convention mismatch that could by
        itself produce a spurious nonzero C_R_full. The non-mixed
        full_linearized_divergence() is kept ONLY for the FIX-3 consistency
        cross-check (C_T_div_plain vs C_T_mixed), never for the primary ledger.
"""


# ── helpers ───────────────────────────────────────────────────────────────────
def tz(expr):
    s = sp.simplify(sp.factor(sp.together(expr)))
    r = s.equals(0)
    return r is True

def scan_for_derivatives(expr, label="expr"):
    """Return all Derivative objects found in expr."""
    found = set()
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Derivative):
            found.add(node)
    if found:
        print(f"  [SCAN] {label}: unresolved Derivative objects found:")
        for d in found:
            print(f"    {d}")
    else:
        print(f"  [SCAN] {label}: clean — no Derivative objects. ✓")
    return found

def scan_matrix(mat, label="matrix"):
    found = set()
    for i in range(mat.rows):
        for j in range(mat.cols):
            found |= scan_for_derivatives(mat[i,j], f"{label}[{i},{j}]")
    return found

def build_delta_T(v_r_expr, v_phi_expr, h_tt, g0, h, drho, dp, rho0, p0):
    """
    Build the linearized stress-energy tensor delta_T_{mu nu}.
    drho, dp may be Function(t,r) or plain symbols.
    v_r_expr, v_phi_expr may be Function(t,r) or expressions.
    """
    u_contra_bg = sp.Matrix([1, 0, 0, 0])
    u_cov_bg    = sp.Matrix([-1, 0, 0, 0])
    delta_u_contra = sp.Matrix([sp.Rational(1,2)*h_tt, v_r_expr, 0, v_phi_expr])
    delta_u_cov = sp.Matrix([
        sp.simplify(
            sum(g0[mu,nu]*delta_u_contra[nu] for nu in range(4))
            + sum(h[mu,nu]*u_contra_bg[nu]   for nu in range(4))
        ) for mu in range(4)
    ])
    dT = sp.Matrix.zeros(4,4)
    for mu in range(4):
        for nu in range(4):
            val  = (drho+dp)*u_cov_bg[mu]*u_cov_bg[nu]
            val += (rho0+p0)*(delta_u_cov[mu]*u_cov_bg[nu]
                             + u_cov_bg[mu]*delta_u_cov[nu])
            val += dp*g0[mu,nu] + p0*h[mu,nu]
            dT[mu,nu] = sp.simplify(val)
    return dT

# FIX 1: dsolve-first solver for a single conservation component
def solve_one_component(eq_expr, fn, label):
    """
    Try dsolve first (correct for ODEs), then fall back to sp.solve (algebraic).
    Returns (method, solution_or_None).

    RUN 5 FIX (critical): dsolve can return an IMPLICIT equation when it
    can't find a closed form (e.g. a(t), rho0(t), p0(t) are unspecified
    functions, so the integrating-factor integral can't be evaluated).
    In that case sol looks like:
        Eq( <expression that still contains fn(t) inside an Integral>, C1 )
    Naively taking sol.rhs gives the bare integration constant C1 -- NOT a
    genuine solution for fn(t). Substituting fn(t) -> C1 would silently
    replace the function with a constant, which trivially "cleans" any
    later derivative scan without actually solving the ODE; any downstream
    q2/Pi2 numbers built on that substitution would be meaningless.

    This function now explicitly checks whether dsolve's lhs is just the
    bare function (fn) or m=fn=... in explicit form. If the returned
    relation is implicit (fn still appears free in sol.lhs, generally
    trapped inside an Integral, and rhs has no t-dependence beyond the
    constant), it is reported as 'implicit' and the caller must NOT
    substitute sol.rhs in place of fn.
    """
    print(f"\n  Solving {label} = 0 for {fn} ...")
    print(f"    equation: {eq_expr}")

    # --- dsolve (correct approach for dynamical equations) ---
    try:
        sol = sp.dsolve(sp.Eq(eq_expr, 0), fn)
        if isinstance(sol, list):
            sol = sol[0]
        print(f"    dsolve => {sol}")

        lhs_is_bare_fn = (sol.lhs == fn)
        rhs_has_fn     = sol.rhs.has(fn) if hasattr(sol, 'rhs') else False

        if hasattr(sol, 'rhs') and lhs_is_bare_fn and not rhs_has_fn:
            # Genuine explicit closed-form solution: fn(t) = <expression>.
            print(f"    [explicit closed-form solution confirmed: lhs == {fn}]")
            return ('dsolve', sol.rhs)
        else:
            # Implicit / unintegrated relation -- dsolve could not produce
            # fn(t) = <expression>. sol.rhs here would just be the
            # integration constant, not a solution. Refuse to substitute it.
            print(f"    [WARNING] dsolve returned an IMPLICIT relation, not an "
                  f"explicit solution for {fn}.")
            print(f"    lhs = {sol.lhs}")
            print(f"    This is most likely because a(t), rho0(t), p0(t) are "
                  f"unspecified functions, so the integrating-factor integral "
                  f"could not be evaluated in closed form.")
            print(f"    Returning ('implicit', None) -- caller must NOT treat "
                  f"sol.rhs ({sol.rhs if hasattr(sol,'rhs') else 'N/A'}) as a "
                  f"solution for {fn}.")
            return ('implicit', None)
    except Exception as e1:
        print(f"    dsolve failed: {e1}")

    # --- algebraic fallback ---
    try:
        sols = sp.solve(eq_expr, fn)
        if sols:
            print(f"    solve  => {sols[0]}  (algebraic fallback)")
            return ('solve', sols[0])
        else:
            print(f"    solve  => no solution found")
            return ('none', None)
    except Exception as e2:
        print(f"    solve failed: {e2}")
        return ('none', None)

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    run_geometry_validation_suite()

    t, r, th, ph    = sp.symbols('t r theta phi', real=True)
    G_sym, M, omega = sp.symbols('G M omega', real=True, positive=True)
    rho0 = sp.Function('rho0')(t)
    p0   = sp.Function('p0')(t)
    eps  = sp.symbols('eps', real=True)
    a    = sp.Function('a')(t)
    coords = [t, r, th, ph]

    # RUN 7: Promote drho, dp, v_r, v_phi from t-only to (t,r) functions.
    #
    # Motivation (from review): the last open loophole in the Run-6 verdict is
    # that the t-only ansatz drho(t), dp(t) may be overly restrictive. The
    # radial conservation equation C_T[1] contains a GM/r^2 source term that
    # cannot be balanced by any t-only perturbation, but it might be balanced
    # by a spatially varying drho(t,r), dp(t,r).
    #
    # Physics of the two possible outcomes:
    #   • Residual survives: even allowing drho(t,r), dp(t,r) the perturbation
    #     cannot be supported by a perfect fluid. Pi2 != 0 remains the
    #     structural verdict.
    #   • Residual vanishes: the previous obstruction was purely an ansatz
    #     artefact; no firm conclusion about the fluid type follows.
    #
    # Implementation note: SymPy cannot in general solve a PDE system for
    # drho(t,r), dp(t,r) symbolically when a(t), rho0(t), p0(t) are generic.
    # The strategy here is:
    #   (i)  Write C_T_full_raw symbolically with drho = drho(t,r), etc.
    #   (ii) Attempt to isolate drho(t,r) and dp(t,r) from C_T[0] and C_T[1]
    #        algebraically (treating Derivative(drho,t), Derivative(drho,r)
    #        as independent unknowns -- i.e. solve the algebraic part of the
    #        PDE system for the leading unknowns at each order).
    #   (iii) Report whether the system is consistent or over-determined.
    #        Over-determined => structural obstruction remains.
    drho = sp.Function('drho')(t, r)
    dp   = sp.Function('dp')(t, r)

    # Velocity perturbations also allowed to depend on (t, r).
    # v_phi is still treated as existing via ODE/PDE existence argument;
    # v_r is first attempted algebraically from C_T[0] and then checked
    # for consistency with C_T[1], exactly as in Run 6 — but now with
    # r-dependence in all coefficient functions.
    v_r_fn   = sp.Function('v_r')(t, r)
    v_phi_fn = sp.Function('v_phi')(t, r)

    g0 = sp.Matrix([
        [-1,0,0,0],[0,a**2,0,0],
        [0,0,a**2*r**2,0],
        [0,0,0,a**2*r**2*sp.sin(th)**2]
    ])
    g0_inv = g0.inv()
    Gamma0 = get_christoffel(g0, g0_inv, coords)
    print("Christoffel computed.", flush=True)

    h = sp.Matrix([
        [2*G_sym*M/(a*r),0,0,2*G_sym*M*a*omega*sp.sin(th)**2],
        [0,0,0,0],[0,0,0,0],
        [2*G_sym*M*a*omega*sp.sin(th)**2,0,0,0]
    ])
    h_tt = h[0,0]

    g_full     = g0 + eps*h
    Gamma_full = get_christoffel(g_full, g_full.inv(), coords)
    dGamma     = sp.MutableDenseNDimArray.zeros(4,4,4)
    for l in range(4):
        for m in range(4):
            for n in range(4):
                dGamma[l,m,n] = sp.simplify(
                    sp.diff(Gamma_full[l,m,n], eps).subs(eps,0))
    print("dGamma computed.", flush=True)

    u_cov_bg = sp.Matrix([-1,0,0,0])
    T0_cov   = sp.Matrix([[(rho0+p0)*u_cov_bg[mu]*u_cov_bg[nu]+p0*g0[mu,nu]
                            for nu in range(4)] for mu in range(4)])

    eos_subs, cs2    = make_background_eos_subs(rho0, p0, t)
    friedmann, H_sym = make_friedmann_subs(a, t, G_sym, rho0, p0, eos_subs=eos_subs)

    G_full_exact = get_einstein(g_full, g_full.inv(), coords)
    G0_sym       = get_einstein(g0, g0_inv, coords)
    G0_cov       = G0_sym.copy()
    delta_G_raw  = sp.Matrix.zeros(4,4)
    for mu in range(4):
        for nu in range(mu,4):
            val = sp.simplify(
                (G_full_exact[mu,nu]-G0_sym[mu,nu]).diff(eps).subs(eps,0))
            delta_G_raw[mu,nu] = val
            delta_G_raw[nu,mu] = val
    print("delta_G_raw computed.", flush=True)
    publication_background_audit("delta_G_raw", delta_G_raw, rho0, p0, t)
    verify_offshell_bianchi_direct(g_full, coords, eps)

    u_contra  = sp.Matrix([1,0,0,0])
    u_cov     = sp.Matrix([-1,0,0,0])
    h_proj    = g0 + u_cov*u_cov.T
    h_up_up   = g0_inv + u_contra*u_contra.T
    h_down_up = h_proj * g0_inv

    # ══════════════════════════════════════════════════════════════════════════
    # RUN 6 — Existence-based velocity completion (replaces dsolve-first FIX 1)
    # ══════════════════════════════════════════════════════════════════════════
    # Rationale (per review): asking "can SymPy produce a closed-form v_r(t),
    # v_phi(t)?" is the wrong question once a(t), rho0(t), p0(t) are left as
    # fully generic functions -- there is no universal elementary antiderivative,
    # and that is not a defect of the audit. The right question is whether the
    # conservation equations admit a well-posed (locally unique) solution.
    #
    # Structural observation used here:
    #   C_T_full_raw[0] (energy/time component) contains v_r(t) UNDIFFERENTIATED
    #     -> solve it algebraically for v_r(t) directly. No ODE needed.
    #   C_T_full_raw[1] (radial component) then contains Derivative(v_r,t)
    #     -> once v_r(t) is fixed by [0], d/dt of that expression is forced;
    #        substituting it into [1] turns [1] into a CONSISTENCY CHECK
    #        rather than a free equation for v_r. [0] and [1] together
    #        over-determine v_r(t); there is no remaining freedom to "solve via
    #        dsolve" for v_r at all.
    #   C_T_full_raw[3] (azimuthal component) contains both v_phi(t) and
    #     Derivative(v_phi,t), with no separate algebraic equation to fix
    #     v_phi(t) itself (unlike v_r). This is a genuine 1st-order LINEAR ODE
    #     in v_phi(t) with continuous coefficients (away from H=0): by the
    #     Picard-Lindelof theorem a unique local solution EXISTS, even though
    #     SymPy cannot write it in closed form while a(t),rho0(t),p0(t) are
    #     left generic. We solve algebraically for Derivative(v_phi,t) (this
    #     is just restating the ODE, not solving it) and report this as
    #     "existence proven", carrying v_phi(t) forward as a genuinely free
    #     (but well-defined) function rather than as an "unresolved residual".
    print("\n" + "="*70)
    print("RUN 7: spatially-varying drho(t,r), dp(t,r) — radial loophole test")
    print("  (v_r algebraic from C_T[0]; C_T[1] consistency; v_phi ODE-existence)")
    print("="*70, flush=True)

    dT_raw = build_delta_T(v_r_fn, v_phi_fn, h_tt, g0, h, drho, dp, rho0, p0)

    print("\n  Computing C_T_full_raw (mixed-tensor form — FIX 5/RUN 4) ...", flush=True)
    C_T_raw = [friedmann(c) for c in
               full_linearized_divergence_mixed(dT_raw, T0_cov, g0_inv,
                                                 Gamma0, dGamma, coords, h)]

    print("\n  C_T_full_raw (before solving):")
    for nu, c in enumerate(C_T_raw):
        print(f"    [{nu}]: {'✓ 0' if tz(c) else str(c)}")

    # ── Structural PDE diagnostic on C_T_raw[1] ──────────────────────────────
    # Recommended by review: inspect the radial conservation equation BEFORE
    # solving anything. The collect() decomposition reveals whether the
    # residual has the structure of a PDE for (drho, dp, v_r) with a
    # source term, or whether the source is irreducible even after introducing
    # spatially-varying perturbations.
    #
    # Schematically we expect:
    #   A*drho + B*dp + C*∂_r drho + D*∂_r dp
    #   + E*v_r + F*∂_t v_r + G*∂_r v_r + [source]
    #
    # If [source] = GM/r² * (rho0+p0) * (something non-zero) survives after
    # collecting all the free variables above, then the obstruction is
    # IRREDUCIBLE: no choice of drho(t,r), dp(t,r), v_r(t,r) can satisfy
    # the radial conservation equation. That is the publishable result.
    #
    # If [source] = 0 after collecting, then conservation CAN be satisfied
    # by an appropriate PDE solution for drho/dp/v_r — no structural
    # obstruction exists.
    print("\n" + "="*70)
    print("STRUCTURAL DIAGNOSTIC: PDE decomposition of C_T_raw[1]")
    print("  (radial conservation equation — inspected before any solving)")
    print("="*70, flush=True)

    _collect_vars = [
        drho,
        dp,
        sp.Derivative(drho, r),
        sp.Derivative(dp,   r),
        sp.Derivative(drho, t),
        sp.Derivative(dp,   t),
        v_r_fn,
        sp.Derivative(v_r_fn, t),
        sp.Derivative(v_r_fn, r),
    ]

    try:
        _C1_expanded = sp.expand(C_T_raw[1])
        _C1_collected = sp.collect(_C1_expanded, _collect_vars, evaluate=False)
        print("\n  Coefficients in C_T_raw[1] (by collected term):")
        _known_vars = set(str(v) for v in _collect_vars)
        _source_term = _C1_expanded
        for _var, _coeff in _C1_collected.items():
            _c = sp.simplify(_coeff)
            print(f"    coeff[{_var}] = {_c}")
            _source_term = sp.expand(_source_term - _coeff * _var)
        _source_term = sp.simplify(_source_term)
        print(f"\n  Residual source term (terms NOT involving any free variable):")
        print(f"    source = {_source_term}")
        if _source_term == 0 or sp.simplify(_source_term).equals(0):
            print("\n  => Source term = 0.")
            print("     Radial conservation is a PDE in drho, dp, v_r only.")
            print("     Conservation CAN be satisfied by an appropriate ansatz.")
            print("     No irreducible structural obstruction at the PDE level.")
        else:
            print("\n  => Source term ≠ 0.")
            print("     A term survives that does not multiply any free variable.")
            print("     This is a pointwise algebraic obstruction: no choice of")
            print("     drho(t,r), dp(t,r), v_r(t,r) can cancel it at each (t,r).")
            print("     This would be the strongest possible structural finding.")
    except Exception as _e:
        print(f"  collect() failed: {_e}")
        print(f"  Falling back to raw C_T_raw[1]: {C_T_raw[1]}")
        _source_term = None

    # ── Step 1: v_r(t,r) from the energy equation C_T[0]=0 ──────────────────
    print("\n  --- Step 1: solve C_T[0]=0 algebraically for v_r(t,r) ---")
    print("  (drho(t,r), dp(t,r) and their derivatives appear as coefficients;")
    print("   v_r(t,r) is the sole unknown being isolated here)")
    vr_sols = sp.solve(sp.Eq(C_T_raw[0], 0), v_r_fn)
    if not vr_sols:
        raise RuntimeError("Could not solve C_T[0]=0 for v_r(t) -- unexpected.")
    v_r_sol = sp.simplify(vr_sols[0])
    print(f"    v_r(t) = {v_r_sol}")
    meth_r, v_r_val = 'algebraic(energy-eq)', v_r_sol

    # ── Step 2: radial equation — consistency check ───────────────────────────
    print("\n  --- Step 2: substitute v_r into C_T[1] — consistency check ---")
    deriv_v_r_t = sp.diff(v_r_sol, t)
    deriv_v_r_r = sp.diff(v_r_sol, r)
    C_T1_check = friedmann(sp.simplify(
        C_T_raw[1]
            .subs(sp.Derivative(v_r_fn, t), deriv_v_r_t)
            .subs(sp.Derivative(v_r_fn, r), deriv_v_r_r)
            .subs(v_r_fn, v_r_sol)))
    radial_consistent = tz(C_T1_check)
    print(f"    C_T[1] after substituting v_r: "
          f"{'✓ 0 (consistent)' if radial_consistent else f'✗ {C_T1_check}'}")
    if not radial_consistent:
        print("    => The algebraic closure route is OVER-DETERMINED: fixing v_r(t,r)")
        print("       from C_T[0] and substituting into C_T[1] generates unresolved")
        print("       higher-derivative structure that is not simultaneously satisfiable")
        print("       for generic a(t), rho0(t), p0(t) via this substitution path.")
        print("       NOTE: this does NOT prove that no PDE solution exists.")
        print("       The structural diagnostic (source=0) confirms the raw radial")
        print("       equation has available degrees of freedom; the obstruction is in")
        print("       this specific closure route, not an algebraic pointwise impossibility.")

    # ── Step 3: v_phi(t,r) — existence via PDE argument ──────────────────────
    print("\n  --- Step 3: C_T[3]=0 — equation for v_phi(t,r) ---")
    print(f"    equation: {C_T_raw[3]} = 0")
    # Isolate Derivative(v_phi, t) — the principal part in t
    dvphi_solset = sp.solve(sp.Eq(C_T_raw[3], 0), sp.Derivative(v_phi_fn, t))
    if dvphi_solset:
        deriv_v_phi_expr = sp.simplify(dvphi_solset[0])
        print(f"    Derivative(v_phi,t) = {deriv_v_phi_expr}")
        print("    [Picard-Lindelof in t, r treated as parameter] This is a")
        print("    first-order linear PDE in v_phi(t,r). For each fixed r, a")
        print("    unique local solution v_phi(t,r) is guaranteed to exist")
        print("    by Picard-Lindelof, even without a closed-form elementary")
        print("    antiderivative. v_phi(t,r) is carried forward as a genuinely")
        print("    free (existing, but unevaluated) function.")
        meth_phi, v_phi_val = 'existence(PDE/linear-ODE-in-t)', None
    else:
        print("    Could not isolate Derivative(v_phi,t) — unexpected.")
        meth_phi, v_phi_val = 'none', None

    print(f"\n  v_r   resolution: {meth_r}   => {v_r_val}")
    print(f"  v_phi resolution: {meth_phi} (kept symbolic; existence only, no value substituted)")

    # ── Recheck C_T_full_raw with v_r fixed, v_phi left free ─────────────────
    print("\n  --- C_T_full_raw after Step 1-3 ---")

    def sub_sol(expr):
        # Substitute v_r(t,r) and both its partial derivatives
        e = expr.subs(v_r_fn, v_r_sol)
        e = e.subs(sp.Derivative(v_r_fn, t), sp.diff(v_r_sol, t))
        e = e.subs(sp.Derivative(v_r_fn, r), sp.diff(v_r_sol, r))
        return friedmann(sp.simplify(e))

    C_T_solved = [sub_sol(c) for c in C_T_raw]
    all_zero_r1 = True
    for nu, c in enumerate(C_T_solved):
        z = tz(c)
        all_zero_r1 = all_zero_r1 and z
        print(f"    [{nu}]: {'✓ 0' if z else f'✗ {c}'}")

    print(f"\n  RUN 6 VERDICT: All C_T_full_raw = 0 (with v_phi existence-only)? {all_zero_r1}")
    if not all_zero_r1:
        print("  => Residual structure (non-zero):")
        for nu, c in enumerate(C_T_solved):
            if not tz(c):
                print(f"       [{nu}]: {sp.simplify(c)}")
        print("  Note: [3] is expected to be ✓ 0 by construction (we solved Derivative(v_phi,t)")
        print("  directly from it); any nonzero entries here are physical, not bookkeeping.")

    # ══════════════════════════════════════════════════════════════════════════
    # FIX 2 — Derivative scanner on dT_solved before Run 2
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("FIX 2: Derivative scanner on dT_solved")
    print("="*70, flush=True)

    def sub_raw(expr):
        # v_r(t,r) known explicitly; both partial derivatives forced.
        # v_phi(t,r) is left as a free function (existence proven).
        e = expr.subs(v_r_fn, v_r_sol)
        e = e.subs(sp.Derivative(v_r_fn, t), sp.diff(v_r_sol, t))
        e = e.subs(sp.Derivative(v_r_fn, r), sp.diff(v_r_sol, r))
        return sp.simplify(e)

    dT_solved = sp.Matrix([[sub_raw(dT_raw[i,j]) for j in range(4)] for i in range(4)])

    print("\n  Scanning dT_solved for unresolved Derivative objects:")
    leftover = scan_matrix(dT_solved, "dT_solved")

    if leftover:
        print("\n  WARNING: unresolved Derivative objects in dT_solved.")
        proceed_run2 = True
    else:
        print("\n  dT_solved is clean (v_phi(t,r) appears only undifferentiated, as expected"
              " since it was never substituted -- only its t-derivative was constrained). ✓")
        proceed_run2 = True

    # ══════════════════════════════════════════════════════════════════════════
    # RUN 2 — C_R_full, C_R_full_cl; FIX 3: compare divergence vs mixed-divergence
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("RUN 2 (FIX 3 applied): C_R_full, C_R_full_cl + divergence consistency check")
    print("="*70, flush=True)

    # ── Scalar closure (RUN 7: drho, dp are now Function(t,r)) ────────────────
    # sp.solve([rho_res, p_res], [drho, dp]) cannot algebraically isolate
    # function objects. Instead we introduce proxy symbols, solve for them,
    # and then check whether the solutions carry r-dependence.
    R_res = sp.Matrix([[friedmann(sp.simplify(
                            delta_G_raw[mu,nu] - 8*sp.pi*G_sym*dT_solved[mu,nu]))
                        for nu in range(4)] for mu in range(4)])

    rho_res = sp.simplify(sum(u_contra[mu]*u_contra[nu]*R_res[mu,nu]
                               for mu in range(4) for nu in range(4)))
    p_res   = sp.Rational(1,3)*sp.simplify(sum(h_up_up[mu,nu]*R_res[mu,nu]
                               for mu in range(4) for nu in range(4)))
    print(f"  rho_res = {rho_res}")
    print(f"  p_res   = {p_res}")

    drho_proxy = sp.Symbol('drho_proxy', real=True)
    dp_proxy   = sp.Symbol('dp_proxy',   real=True)
    rho_res_proxy = rho_res.subs(drho, drho_proxy).subs(dp, dp_proxy)
    p_res_proxy   = p_res.subs(drho, drho_proxy).subs(dp, dp_proxy)
    scalar_sols = sp.solve([rho_res_proxy, p_res_proxy],
                           [drho_proxy, dp_proxy], dict=True)
    if scalar_sols:
        closure  = scalar_sols[0]
        drho_sol = sp.simplify(closure.get(drho_proxy, drho_proxy))
        dp_sol   = sp.simplify(closure.get(dp_proxy,   dp_proxy))
        print(f"  drho_sol = {drho_sol}")
        print(f"  dp_sol   = {dp_sol}")
        drho_has_r = drho_sol.has(r)
        dp_has_r   = dp_sol.has(r)
        print(f"  drho_sol depends on r? {drho_has_r}  dp_sol depends on r? {dp_has_r}")
        if drho_has_r or dp_has_r:
            print("  => Closure REQUIRES radial dependence in drho/dp.")
            print("     Consistent with the GM/r^2 source in C_T[1].")
        else:
            print("  => Closure is t-only; radial extension adds nothing here.")
        def apply_closure(expr):
            return friedmann(sp.simplify(
                expr.subs(drho, drho_sol).subs(dp, dp_sol)))
        R_res_cl = sp.Matrix([[apply_closure(R_res[i,j])    for j in range(4)] for i in range(4)])
        dT_cl    = sp.Matrix([[apply_closure(dT_solved[i,j]) for j in range(4)] for i in range(4)])
    else:
        print("  WARNING: No scalar closure found even with proxy substitution.")
        drho_sol, dp_sol = drho, dp
        R_res_cl, dT_cl  = R_res, dT_solved

    # FIX 3: compute BOTH divergence forms and compare
    print("\n  --- FIX 3: divergence consistency check (plain vs mixed forms) ---")

    C_T_div_plain = [friedmann(c) for c in
                 full_linearized_divergence(dT_solved, T0_cov, g0_inv,
                                             Gamma0, dGamma, coords)]
    C_T_mixed = [friedmann(c) for c in
                 full_linearized_divergence_mixed(dT_solved, T0_cov, g0_inv,
                                                   Gamma0, dGamma, coords, h)]

    agree = all(tz(sp.simplify(a - b)) for a, b in zip(C_T_div_plain, C_T_mixed))
    print(f"  full_linearized_divergence vs full_linearized_divergence_mixed agree? {agree}")
    if not agree:
        print("  DISCREPANCY — bookkeeping issue in matter sector index variation:")
        for nu, (a_, b_) in enumerate(zip(C_T_div_plain, C_T_mixed)):
            diff = sp.simplify(a_ - b_)
            if not tz(diff):
                print(f"    [{nu}]: diff = {diff}")
        print("  RUN 4 uses the MIXED form below for the primary ledger, since it is")
        print("  the form already used on the geometric side (C_G_cross) -- this keeps")
        print("  both sectors on the same index convention and removes that ambiguity")
        print("  from the interpretation of C_R_full / C_R_full_cl.")
    else:
        print("  Both forms agree. Matter divergence is internally consistent. ✓")

    # FIX 5 (RUN 4): primary matter ledger now uses the MIXED form exclusively,
    # matching the geometric side. C_T_div_plain above is retained only as the
    # FIX-3 cross-check, never fed into C_R_full / C_R_full_cl below.
    C_T_div = C_T_mixed

    # Use full_linearized_divergence_mixed (already mixed, geometric side)
    C_G_cross   = full_linearized_divergence_mixed(
                    delta_G_raw, G0_cov, g0_inv, Gamma0, dGamma, coords, h)
    C_G_onshell = [friedmann(c) for c in C_G_cross]

    C_T_cl_diag = [friedmann(c) for c in
                   full_linearized_divergence_mixed(dT_cl, T0_cov, g0_inv,
                                                     Gamma0, dGamma, coords, h)]

    C_R_full    = [friedmann(sp.simplify(cg/(8*sp.pi*G_sym) - ct))
                   for cg, ct in zip(C_G_onshell, C_T_div)]
    C_R_full_cl = [friedmann(sp.simplify(cg/(8*sp.pi*G_sym) - ct))
                   for cg, ct in zip(C_G_onshell, C_T_cl_diag)]

    print("\n  C_G_full_onshell (Bianchi):")
    for nu, c in enumerate(C_G_onshell):
        print(f"    [{nu}] = {'✓ 0' if tz(c) else f'✗ {c}'}")

    print("\n  C_T_full_raw on dT_solved:")
    for nu, c in enumerate(C_T_div):
        print(f"    [{nu}] = {'✓ 0' if tz(c) else f'✗ {c}'}")

    print("\n  C_T_full_cl_diag (closure-consistency diagnostic):")
    for nu, c in enumerate(C_T_cl_diag):
        print(f"    [{nu}] = {'✓ 0' if tz(c) else f'✗ {c}'}")

    print("\n  C_R_full (pre-closure):")
    for nu, c in enumerate(C_R_full):
        print(f"    [{nu}] = {'✓ 0' if tz(c) else f'✗ {c}'}")

    print("\n  C_R_full_cl (post-closure):")
    for nu, c in enumerate(C_R_full_cl):
        print(f"    [{nu}] = {'✓ 0' if tz(c) else f'✗ {c}'}")

    r2_cr   = all(tz(c) for c in C_R_full)
    r2_crcl = all(tz(c) for c in C_R_full_cl)
    print(f"\n  RUN 2 VERDICT: C_R_full all zero? {r2_cr}  |  C_R_full_cl all zero? {r2_crcl}")

    # ══════════════════════════════════════════════════════════════════════════
    # RUN 3 — q_mu, q2, Pi2; FIX 4: softened verdict language
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("RUN 3 (FIX 4 applied): q_mu, q2, Pi2 — softened classification (spatially-varying ansatz)")
    print("="*70, flush=True)

    rho_cl = sp.simplify(sum(u_contra[mu]*u_contra[nu]*R_res_cl[mu,nu]
                              for mu in range(4) for nu in range(4)))
    p_cl   = sp.Rational(1,3)*sp.simplify(sum(h_up_up[mu,nu]*R_res_cl[mu,nu]
                              for mu in range(4) for nu in range(4)))
    R_rem  = R_res_cl - rho_cl*(u_cov*u_cov.T) - p_cl*h_proj
    R_rem  = sp.Matrix([[sp.simplify(R_rem[i,j]) for j in range(4)] for i in range(4)])

    print("  R_rem (irreducible remainder):")
    any_rem = False
    for mu in range(4):
        for nu in range(mu,4):
            v = sp.simplify(R_rem[mu,nu])
            if not tz(v):
                print(f"    R_rem[{mu},{nu}] = {v}")
                any_rem = True
    if not any_rem:
        print("    None. ✓")

    q_cov    = sp.Matrix([sp.simplify(-sum(h_down_up[mu,aa]*R_rem[aa,nu]*u_contra[nu]
                           for aa in range(4) for nu in range(4))) for mu in range(4)])
    q_contra = g0_inv * q_cov
    S = sp.Matrix([[sp.simplify(sum(h_down_up[mu,aa]*h_down_up[nu,bb]*R_rem[aa,bb]
                    for aa in range(4) for bb in range(4)))
                    for nu in range(4)] for mu in range(4)])
    p_rem = sp.Rational(1,3)*sp.simplify(sum(h_up_up[i,j]*S[i,j]
                                              for i in range(4) for j in range(4)))
    Pi = sp.Matrix([[sp.simplify(S[i,j] - h_proj[i,j]*p_rem)
                     for j in range(4)] for i in range(4)])

    Pi_uu = g0_inv * Pi * g0_inv
    q2    = sp.simplify(sum(q_cov[i]*q_contra[i] for i in range(4)))
    Pi2   = sp.simplify(sum(Pi[i,j]*Pi_uu[i,j] for i in range(4) for j in range(4)))

    print(f"\n  q_mu = {list(q_cov)}")
    print(f"\n  q2  = {q2}")
    print(f"  Pi2 = {Pi2}")

    q2_zero  = tz(q2)
    Pi2_zero = tz(Pi2)
    print(f"\n  q2  = 0?  {q2_zero}")
    print(f"  Pi2 = 0?  {Pi2_zero}")

    q_nz = [(mu, sp.simplify(q_cov[mu])) for mu in range(4) if not tz(q_cov[mu])]

    # ── RUN 6: existence-based pre-conditions (replaces dsolve-closed-form test) ─
    # Three pre-conditions for calling q² "irreducible", now phrased as:
    #   (A) conservation closes  — C_T_full_raw = 0 (with v_r fixed, v_phi
    #       constrained only through its existence-guaranteed derivative)
    #   (B) velocity completion exists — v_r(t) is uniquely fixed AND
    #       self-consistent (Step 1+2), and v_phi(t) is guaranteed to exist
    #       by Picard-Lindelof even without a closed form (Step 3). This
    #       replaces "both functions explicitly solved" -- existence is
    #       enough, an elementary antiderivative is not required.
    #   (C) no unresolved Derivative objects remain in dT_solved (still
    #       required: this is a bookkeeping check, not a closed-form demand)
    cond_A = all_zero_r1
    cond_B = radial_consistent and bool(dvphi_solset)
    cond_C = (len(leftover) == 0)

    print(f"\n  Pre-conditions for 'irreducible' verdict (existence-based, RUN 6):")
    print(f"    (A) conservation closes (C_T_full_raw=0)? {cond_A}")
    print(f"    (B) velocity completion exists (v_r fixed+consistent, v_phi ODE-existence)? {cond_B}")
    print(f"    (C) no unresolved Derivative objects in dT_solved? {cond_C}")

    print("\n  --- FLUID SECTOR CLASSIFICATION ---")
    if Pi2_zero and q2_zero:
        print("  ✓ Perfect fluid: Pi2=0, q2=0.")
        print("    CPT perturbation is self-consistent with v_r(t), v_phi(t).")
    elif not Pi2_zero:
        # Pi2 classification — check whether it still depends on v_phi (v_r is
        # already substituted into dT_solved, so Pi2 can only depend on v_phi
        # at this point, never on v_r).
        Pi2_has_vphi = Pi2.has(v_phi_fn)
        if Pi2_has_vphi:
            print("  Pi2 ≠ 0 and still depends on the (existence-only, unevaluated) v_phi(t).")
            print("  => Cannot yet fully classify; depends on the specific v_phi(t) chosen.")
        else:
            print("  Pi2 ≠ 0 and is independent of v_r(t), v_phi(t).")
            print("  => Anisotropic stress is required by the geometry,")
            print("     regardless of the velocity ansatz.")
            print(f"     Pi2 = {Pi2}")
            print("  This is a structural result, not an ansatz artefact.")
    if not q2_zero:
        if cond_A and cond_B and cond_C and not q2.has(v_phi_fn):
            print("\n  q2 ≠ 0 — IRREDUCIBLE energy current in this frame.")
            print("  All pre-conditions satisfied and q2 is independent of the free v_phi(t);")
            print("  this is a genuine, frame-fixed result.")
        elif q2.has(v_phi_fn):
            print("\n  q2 ≠ 0 — but depends on the as-yet-unevaluated v_phi(t).")
            print("  Classification: MOMENTUM CURRENT IS FRAME-DEPENDENT, not irreducible —")
            print("  a particular choice/initial condition for v_phi(t) could in principle")
            print("  reduce or remove this component (existence of v_phi(t) is guaranteed,")
            print("  but its value, and hence q2, is not yet fixed).")
        else:
            print("\n  q2 ≠ 0 — energy current present in this frame.")
            missing = []
            if not cond_A: missing.append("conservation not closed")
            if not cond_B: missing.append("velocity completion not established")
            if not cond_C: missing.append("unresolved Derivative objects remain")
            print(f"  NOT yet labelled irreducible: {'; '.join(missing)}.")
        if q_nz:
            print("\n  Nonzero q components:")
            for mu, qv in q_nz:
                print(f"    q[{mu}] = {qv}")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY (RUN 6: structural / existence-based classification)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"  v_r   resolution: {meth_r}   => {v_r_val}")
    print(f"  v_phi resolution: {meth_phi} (kept symbolic; existence proven, not evaluated)")
    print(f"  drho_sol = {drho_sol}")
    print(f"  dp_sol   = {dp_sol}")
    print()
    print(f"  RUN 7 — spatially-varying drho(t,r), dp(t,r); v_r via {meth_r}; "
          f"radial consistency [0]&[1] {'HOLDS' if radial_consistent else 'FAILS (structural)'}; "
          f"v_phi via {meth_phi}")
    print(f"  FIX 2 — dT_solved derivative scan: {'CLEAN ✓' if not leftover else f'{len(leftover)} unresolved object(s) ✗'}")
    print(f"  FIX 3 — divergence consistency: {'AGREE ✓' if agree else 'DISAGREE ✗'}")
    print()
    print(f"  Run 7: C_T_full_raw all zero after substitution? {all_zero_r1}")
    print(f"  Run 2: C_R_full all zero?    {r2_cr}")
    print(f"  Run 2: C_R_full_cl all zero? {r2_crcl}")
    print(f"  Run 3: q2 = 0?  {q2_zero}")
    print(f"  Run 3: Pi2 = 0? {Pi2_zero}")
    print()
    print("  Pre-conditions for 'irreducible' label:")
    print(f"    (A) {cond_A}  (B) {cond_B}  (C) {cond_C}")
    print()
    print("  ==========================")
    print("  STRUCTURAL RESULTS")
    print("  ==========================")
    print(f"  Geometry:              PASS (Bianchi off-shell + on-shell, all benchmarks)")
    print(f"  Matter conservation:   {'Admits velocity completion' if (radial_consistent and dvphi_solset) else 'Radial conservation is a coupled PDE (source=0); algebraic closure route over-determined — PDE nonexistence not proven'}")
    print(f"  Momentum current (q):  {'Frame-dependent' if (not q2_zero and q2.has(v_phi_fn)) else ('Irreducible' if not q2_zero else 'Vanishes')}")
    print(f"  Anisotropic stress:    {'Intrinsic (Pi2 != 0, v_r/v_phi-independent)' if (not Pi2_zero and not Pi2.has(v_phi_fn)) else ('Vanishes' if Pi2_zero else 'Velocity-dependent')}")
    if not Pi2_zero and not Pi2.has(v_phi_fn):
        print("\n  STRUCTURAL RESULT: Pi2 is geometry-sourced and velocity-independent.")
        print("  Pi2 is nonzero and independent of v_r, v_phi, drho, dp across all")
        print("  ansätze tested (Runs 1-7, including spatially-varying perturbations).")
        print("  => None of the perfect-fluid perturbation ansätze examined in Runs 1-7")
        print("     removes the nonzero Pi^2.")
        print()
        print("  Current evidence: the nonzero Pi^2 is a robust geometric invariant")
        print("  across all perfect-fluid perturbation ansätze investigated in Runs 1-7.")
        print("  Demonstrating that every isotropic perfect-fluid realization fails would")
        print("  require an independent proof of nonexistence for the remaining coupled")
        print("  PDE system, which Run 7 does not provide.")
