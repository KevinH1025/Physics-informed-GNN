"""Analytical small-signal physics formulas for the tower architecture.

All formulas are evaluated in log10 space for numerical stability.
"""

import math

import torch

_LN10 = 2.302585092994046  # math.log(10)


def log10_add(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """Compute log10(10^log_a + 10^log_b) in a numerically stable way."""
    return torch.logaddexp(log_a * _LN10, log_b * _LN10) / _LN10


def dc_gain_log10_formula(gm_key, gds_key, rout1_formula, openloop, r_in, r_f):
    """Three-stage opamp DC gain formula, evaluated entirely in log10 space.

    Args:
        gm_key: [B, 14] log10(gm) of the key signal-path MOSFETs
        gds_key: [B, 14] log10(gds) of the key signal-path MOSFETs
        rout1_formula: 'simple' or 'cascode' first-stage output resistance
        openloop: skip beta and R_load (no feedback network)
        r_in: feedback network input resistance (scalar tensor)
        r_f: feedback network feedback resistance (scalar tensor)

    Returns:
        (formula_feats, dc_gain_est_dB): [B, 7] formula intermediates
        (log_rout1, log_A1, log_A2_p1, log_A2, log_rout3, log_rout3_loaded,
        dc_gain_est_dB) and the [B] gain estimate in dB.
    """
    # All computation in log10 space — no pow(10,...) needed
    # Index map within the 14 key MOSFETs:
    # 0=M8, 1=M9, 2=M5, 3=M6, 4=M15, 5=M16, 6=M19, 7=M20
    # 8=M7, 9=M10, 10=M21, 11=M22, 12=M11, 13=M23

    # Stage 1: log10(Rout1) = -log10(gds6 + gds16) or cascode variant
    if rout1_formula == 'simple':
        log_rout1 = -log10_add(gds_key[:, 3], gds_key[:, 5])
    else:  # cascode: second term = gds16*gds20/gm16
        log_cascode_term = gds_key[:, 5] + gds_key[:, 7] - gm_key[:, 5]
        log_rout1 = -log10_add(gds_key[:, 3], log_cascode_term)
    log_A1 = gm_key[:, 0] + log_rout1  # log10(gm8 * Rout1)

    # Stage 2: A2 = [gm10/(gm21+gds21+gds10)] * [gm22/(gds7+gds22)]
    log_denom1 = log10_add(log10_add(gm_key[:, 10], gds_key[:, 10]), gds_key[:, 9])
    log_A2_p1 = gm_key[:, 9] - log_denom1
    log_denom2 = log10_add(gds_key[:, 8], gds_key[:, 11])
    log_A2 = log_A2_p1 + gm_key[:, 11] - log_denom2

    # Stage 3: Rout3 = 1/(gds11+gds23). For closed-loop the
    # output is loaded by R_in+R_f; for open-loop the load is the
    # output cap (open at DC) so we use the unloaded rout3 and
    # drop the feedback-factor beta.
    log_rout3 = -log10_add(gds_key[:, 12], gds_key[:, 13])
    if openloop:
        log_rout3_loaded = log_rout3
        log_beta_term = 0.0   # no feedback factor
    else:
        R_load = r_in + r_f
        log_inv_rload = torch.tensor(-math.log10(float(R_load)), device=gm_key.device)
        log_rout3_loaded = -log10_add(-log_rout3, log_inv_rload)
        beta = r_in / (r_in + r_f + 1e-15)
        log_beta_term = math.log10(float(beta))

    # T = A1 * Rout3_loaded * (gm11 + A2*gm23) * (beta or 1)
    log_sum_gm = log10_add(gm_key[:, 12], log_A2 + gm_key[:, 13])
    log_T = log_A1 + log_rout3_loaded + log_sum_gm + log_beta_term
    dc_gain_est_dB = 20.0 * log_T  # [B], already in log10

    formula_feats = torch.stack([
        log_rout1, log_A1, log_A2_p1, log_A2,
        log_rout3, log_rout3_loaded, dc_gain_est_dB,
    ], dim=-1)  # [B, 7]
    return formula_feats, dc_gain_est_dB


def ugbw_log10_estimate(z_gm_key, gm_mean, gm_std, cc_norm, cc_log10_min,
                        cc_log10_range, openloop, r_in, r_f):
    """Miller UGBW estimate ugbw ≈ beta * gm_M8 / (2π * Cc) in log10 space.

    Args:
        z_gm_key: [B, 14] z-scored log10(gm) of the key MOSFETs (M8 first)
        gm_mean: SS gm normalization mean (scalar tensor)
        gm_std: SS gm normalization std (scalar tensor)
        cc_norm: [B] normalized compensation cap node feature
        cc_log10_min: log10(Cc) at cc_norm = 0
        cc_log10_range: log10(Cc) span over cc_norm in [0, 1]
        openloop: β=1 (no feedback factor)
        r_in: feedback network input resistance (scalar tensor)
        r_f: feedback network feedback resistance (scalar tensor)

    Returns:
        log10_ugbw_est: [B] UGBW estimate in log10(Hz)
    """
    # gm_M8 in log10 space (for formula estimate)
    log10_gm_M8 = z_gm_key[:, 0] * gm_std + gm_mean
    log10_Cc = cc_norm * cc_log10_range + cc_log10_min

    # Formula in log10 space (no pow(10) needed!)
    # Openloop: β=1 (no feedback). Closed-loop: β = R_in/(R_in+R_f).
    if openloop:
        log10_beta_over_2pi = -math.log10(2 * math.pi)
    else:
        beta = r_in / (r_in + r_f + 1e-15)
        log10_beta_over_2pi = math.log10(float(beta) / (2 * math.pi))
    return log10_gm_M8 + log10_beta_over_2pi - log10_Cc
