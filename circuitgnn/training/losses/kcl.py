"""Kirchhoff current law losses, conservation projection and per-net debugging."""

import torch
import torch.nn.functional as F


def apply_kcl_conservation_projection(
    node_currents: torch.Tensor,
    edge_index: torch.Tensor,
    num_terminals: torch.Tensor,
    train_mask: torch.Tensor,
    ptr: torch.Tensor,
    terminal_current_sign: torch.Tensor,
    current_mean: float,
    current_std: float,
    kcl_include_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Project predicted currents onto the KCL-satisfying subspace.

    For each valid KCL net, adjusts terminal currents so that
    Σ sign_i * I_i = 0 (KCL exactly satisfied). Differentiable.

    Works uniformly for 2-term and 3+ term nets in denormalized space.
    """
    device = node_currents.device
    num_nodes = train_mask.size(0)

    # Build batch index for each node
    graph_sizes = ptr[1:] - ptr[:-1]
    batch_idx = torch.repeat_interleave(torch.arange(len(graph_sizes), device=device), graph_sizes)

    # Terminal mask
    local_idx = torch.arange(num_nodes, device=device) - ptr[batch_idx]
    if isinstance(num_terminals, int):
        terminal_mask = local_idx < num_terminals
    else:
        terminal_mask = local_idx < num_terminals[batch_idx]

    internal_net_mask = train_mask & ~terminal_mask

    # Valid edges: terminal -> internal net
    src, dst = edge_index
    valid_edges = terminal_mask[src] & internal_net_mask[dst]
    if not valid_edges.any():
        return node_currents

    valid_src = src[valid_edges]
    valid_dst = dst[valid_edges]

    # Filter by kcl_include_mask (exclude gate, bulk terminals with 0 DC current)
    if kcl_include_mask is not None:
        kcl_eligible = kcl_include_mask[valid_src]
        valid_src = valid_src[kcl_eligible]
        valid_dst = valid_dst[kcl_eligible]

    if not valid_src.numel():
        return node_currents

    # Filter: ≥2 terminals per net
    terms_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
    terms_per_net.scatter_add_(0, valid_dst, torch.ones_like(valid_dst))
    has_enough = terms_per_net >= 2

    # Filter: both positive and negative sign terminals
    if terminal_current_sign is not None:
        sign_at_src = terminal_current_sign[valid_src]
        pos_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
        neg_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
        pos_per_net.scatter_add_(0, valid_dst, (sign_at_src > 0).long())
        neg_per_net.scatter_add_(0, valid_dst, (sign_at_src < 0).long())
        has_both_signs = (pos_per_net > 0) & (neg_per_net > 0)
    else:
        return node_currents  # Need signs for conservation

    valid_net_mask = internal_net_mask & has_enough & has_both_signs
    if not valid_net_mask.any():
        return node_currents

    # Keep only edges to valid nets
    net_valid = valid_net_mask[valid_dst]
    v_src = valid_src[net_valid]
    v_dst = valid_dst[net_valid]
    v_sign = sign_at_src[net_valid]

    # Denormalize to amps
    z = node_currents[v_src]
    amps = torch.pow(10, z * current_std + current_mean)

    # Per-net signed sum (violation) and total magnitude
    signed_sum = torch.zeros(num_nodes, device=device)
    signed_sum.scatter_add_(0, v_dst, v_sign * amps)
    total_abs = torch.zeros(num_nodes, device=device)
    total_abs.scatter_add_(0, v_dst, amps)

    # Per-terminal relative violation and adjustment factor
    rel_viol = signed_sum[v_dst] / (total_abs[v_dst] + 1e-12)
    factor = (1.0 - v_sign * rel_viol).clamp(min=1e-6)

    # Z-space correction: z_adj = z + log10(factor) / std
    delta_z = torch.log10(factor) / current_std

    # Apply correction to affected terminals only
    result = node_currents.clone()
    result[v_src] = z + delta_z

    return result


def compute_kcl_loss(
    node_currents: torch.Tensor,
    edge_index: torch.Tensor,
    num_terminals: torch.Tensor,
    train_mask: torch.Tensor,
    ptr: torch.Tensor,
    terminal_current_sign: torch.Tensor = None,
    current_mean: float = 0.0,
    current_std: float = 1.0,
    gt_currents: torch.Tensor = None,
    kcl_threshold: float = 1e-6,  # 1µA threshold for ngspice numerical precision
    kcl_include_mask: torch.Tensor = None,
    kcl_min_current: float = 1e-9,  # 1nA min total current to include net in KCL
    return_stats: bool = False,
    kcl_mode: str = 'logsumexp',  # 'logsumexp', 'z_diff', or 'denorm'
    kcl_violation_threshold: float = 0.0,  # min relative violation to penalize (0 = penalize all)
    kcl_skip_two_term: bool = False,  # skip 2-term KCL loss (useful when 2-term is enforced in architecture)
    kcl_only_two_term: bool = False,  # only compute 2-term KCL loss, skip 3+ term nets
    kcl_huber_delta: float = 0.0,  # 0 = disabled (use MSE), >0 = Huber delta for 3-term logsumexp
    kcl_gt_filter: float = 0.1,  # max GT relative violation for multi-term nets (0 = disabled)
):
    """
    Compute Kirchhoff's Current Law loss on internal net nodes for batched graphs.
    Optimized vectorized implementation using scatter_add for GPU efficiency.

    KCL: Sum of signed currents at each internal net node = 0

    Only applies KCL to nets where ground truth satisfies KCL (truly internal nets).
    Nets where KCL is unsatisfiable (single terminal, all same sign) are filtered out.

    Args:
        node_currents: Predicted currents for all nodes in batch (normalized magnitudes)
        edge_index: Graph connectivity [2, num_edges]
        num_terminals: Number of terminals per graph [batch_size]
        train_mask: Mask for internal nodes (True = internal, where we predict voltage)
        ptr: Node boundaries between graphs [batch_size + 1]
        terminal_current_sign: Sign for each terminal (+1 for drain, -1 for source)
        current_mean: Mean for denormalization (log10 scale)
        current_std: Std for denormalization (log10 scale)
        gt_currents: Ground truth currents (normalized) for filtering valid KCL nets
        kcl_threshold: Threshold for GT sum to consider net valid for KCL (in Amps)
        kcl_include_mask: Mask for terminals to include in KCL (False for gate/bulk)
        kcl_min_current: Minimum total current for a net to be included in KCL (filters noise)

    Returns:
        Tuple of (loss, kcl_supervised_mask) where kcl_supervised_mask is a boolean
        tensor of shape [num_nodes] indicating which terminal nodes are supervised by KCL.
    """
    # Ensure float32 for scatter operations (model may output bfloat16 under AMP)
    node_currents = node_currents.float()
    if gt_currents is not None:
        gt_currents = gt_currents.float()
    device = node_currents.device
    num_nodes = train_mask.size(0)
    empty_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    # Build batch index for each node (vectorized)
    graph_sizes = ptr[1:] - ptr[:-1]
    batch_idx = torch.repeat_interleave(torch.arange(len(graph_sizes), device=device), graph_sizes)

    # Terminal mask: local_idx < num_terminals for that graph
    local_idx = torch.arange(num_nodes, device=device) - ptr[batch_idx]
    if isinstance(num_terminals, int):
        terminal_mask = local_idx < num_terminals
    else:
        terminal_mask = local_idx < num_terminals[batch_idx]

    # Net node mask: only nets where we predict voltages (internal + output)
    # Excludes supply (vdda/gnda) and input nets — matches apply_kcl_conservation_projection
    internal_net_mask = train_mask & ~terminal_mask

    # Valid edges: terminal -> internal net
    src, dst = edge_index
    valid_edges = terminal_mask[src] & internal_net_mask[dst]

    if not valid_edges.any():
        if return_stats:
            return torch.tensor(0.0, device=device), empty_mask, {}
        return torch.tensor(0.0, device=device), empty_mask

    valid_src = src[valid_edges]
    valid_dst = dst[valid_edges]

    # Filter to KCL-eligible terminals (exclude gate/bulk with 0 DC current)
    # This must happen BEFORE computing sums, so predicted sums and GT sums
    # are computed consistently over the same set of terminals
    if kcl_include_mask is not None:
        kcl_eligible_pred = kcl_include_mask[valid_src]
        valid_src = valid_src[kcl_eligible_pred]
        valid_dst = valid_dst[kcl_eligible_pred]

    if not valid_src.numel():
        if return_stats:
            return torch.tensor(0.0, device=device), empty_mask, {}
        return torch.tensor(0.0, device=device), empty_mask

    # Filter out nets with fewer than 2 included terminals (KCL unsatisfiable)
    terms_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
    terms_per_net.scatter_add_(0, valid_dst, torch.ones_like(valid_dst))
    has_enough_terms = terms_per_net >= 2

    # Filter out nets where all terminals have the same sign (KCL unsatisfiable)
    if terminal_current_sign is not None:
        sign_at_src = terminal_current_sign[valid_src]
        pos_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
        neg_per_net = torch.zeros(num_nodes, device=device, dtype=torch.long)
        pos_per_net.scatter_add_(0, valid_dst, (sign_at_src > 0).long())
        neg_per_net.scatter_add_(0, valid_dst, (sign_at_src < 0).long())
        has_both_signs = (pos_per_net > 0) & (neg_per_net > 0)
    else:
        has_both_signs = torch.ones(num_nodes, device=device, dtype=torch.bool)

    valid_net_mask = internal_net_mask & has_enough_terms & has_both_signs

    if not valid_net_mask.any():
        if return_stats:
            return torch.tensor(0.0, device=device), empty_mask, {}
        return torch.tensor(0.0, device=device), empty_mask

    # Split nets into 2-terminal and 3+ terminal groups
    two_term_mask = valid_net_mask & (terms_per_net == 2)
    multi_term_mask = valid_net_mask & (terms_per_net > 2)

    # Build kcl_supervised_mask: all terminal nodes participating in valid KCL equations
    # This includes terminals connected to valid 2-term nets and (if not z_diff) valid multi-term nets
    kcl_supervised_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    # Terminals connected to valid 2-term nets
    two_term_edges = two_term_mask[valid_dst]
    if two_term_edges.any():
        kcl_supervised_mask[valid_src[two_term_edges]] = True
    # Terminals connected to valid multi-term nets (only if we're actually computing multi-term loss)
    if kcl_mode != 'z_diff':
        multi_term_edges = multi_term_mask[valid_dst]
        if multi_term_edges.any():
            kcl_supervised_mask[valid_src[multi_term_edges]] = True

    # Get z-values (normalized predictions) at valid source terminals
    z_at_src = node_currents.squeeze()[valid_src]

    losses = []
    stats = {} if return_stats else None

    # --- 2-terminal nets: pure log-space loss (z_A - z_B)² ---
    # For 2-terminal nets, KCL means |I_A| = |I_B|, i.e. z_A = z_B in log space
    # Gradient is simply 2(z_A - z_B) — no exponential scaling
    if two_term_mask.any() and not kcl_skip_two_term:
        two_term_edges = two_term_mask[valid_dst]
        two_src = valid_src[two_term_edges]
        two_dst = valid_dst[two_term_edges]
        two_z = z_at_src[two_term_edges]

        # Sort by destination net to pair terminals
        sort_idx = torch.argsort(two_dst)
        two_z_sorted = two_z[sort_idx]

        # Each net has exactly 2 terminals, so reshape into pairs
        n_two_nets = two_z_sorted.numel() // 2
        if n_two_nets > 0:
            pairs = two_z_sorted[:n_two_nets * 2].reshape(n_two_nets, 2)

            two_term_loss = None
            if kcl_mode == 'denorm':
                # Denorm mode: relative violation in physical amps
                raw_pairs = torch.pow(10, pairs * current_std + current_mean)
                rel_viol_2t = (raw_pairs[:, 0] - raw_pairs[:, 1]) / (raw_pairs.sum(dim=1) + 1e-12)
                if kcl_violation_threshold > 0:
                    with torch.no_grad():
                        above_thresh = rel_viol_2t.abs() > kcl_violation_threshold
                    if stats is not None:
                        stats['n_two_term_above_thresh'] = above_thresh.sum().item()
                    if above_thresh.any():
                        two_term_loss = (rel_viol_2t[above_thresh] ** 2).mean()
                        losses.append(two_term_loss)
                else:
                    two_term_loss = (rel_viol_2t ** 2).mean()
                    losses.append(two_term_loss)
            else:
                # Z-space mode: (z_A - z_B)²
                if kcl_violation_threshold > 0:
                    with torch.no_grad():
                        raw_pairs = torch.pow(10, pairs * current_std + current_mean)
                        rel_viol = (raw_pairs[:, 0] - raw_pairs[:, 1]).abs() / (raw_pairs.sum(dim=1) + 1e-12)
                        above_thresh = rel_viol > kcl_violation_threshold
                    if stats is not None:
                        stats['n_two_term_above_thresh'] = above_thresh.sum().item()
                    if above_thresh.any():
                        two_term_loss = ((pairs[above_thresh, 0] - pairs[above_thresh, 1]) ** 2).mean()
                        losses.append(two_term_loss)
                else:
                    two_term_loss = ((pairs[:, 0] - pairs[:, 1]) ** 2).mean()
                    losses.append(two_term_loss)

            if stats is not None:
                stats['n_two_term'] = n_two_nets
                stats['two_term_loss'] = two_term_loss.item() if two_term_loss is not None else 0.0
                if gt_currents is not None:
                    with torch.no_grad():
                        gt_two_z = gt_currents.squeeze()[valid_src][two_term_edges]
                        gt_two_sorted = gt_two_z[sort_idx]
                        gt_pairs = gt_two_sorted[:n_two_nets * 2].reshape(n_two_nets, 2)
                        stats['gt_two_term_loss'] = ((gt_pairs[:, 0] - gt_pairs[:, 1]) ** 2).mean().item()

    # --- 3+ terminal nets: mode-dependent formulation ---
    # KCL: sum of positive-sign currents = sum of negative-sign currents
    if multi_term_mask.any() and terminal_current_sign is not None and kcl_mode != 'z_diff' and not kcl_only_two_term:
        multi_edges = multi_term_mask[valid_dst]
        multi_src = valid_src[multi_edges]
        multi_dst = valid_dst[multi_edges]
        multi_z = z_at_src[multi_edges]
        multi_sign = terminal_current_sign[multi_src]

        # Assign each multi-term net a compact index (0..N-1)
        unique_nets, net_indices = torch.unique(multi_dst, return_inverse=True)
        n_multi_nets = unique_nets.size(0)

        # GT filter: exclude nets where GT violates KCL beyond threshold
        valid_multi = torch.ones(n_multi_nets, device=device, dtype=torch.bool)
        n_multi_before_filter = n_multi_nets
        gt_z_at_src = gt_currents.squeeze()[valid_src] if gt_currents is not None else None
        if gt_currents is not None and kcl_gt_filter > 0:
            gt_multi_z = gt_z_at_src[multi_edges]
            gt_raw = torch.pow(10, gt_multi_z * current_std + current_mean)
            gt_signed = gt_raw * multi_sign
            gt_sums = torch.zeros(n_multi_nets, device=device)
            gt_sums.scatter_add_(0, net_indices, gt_signed)
            gt_abs_sums = torch.zeros(n_multi_nets, device=device)
            gt_abs_sums.scatter_add_(0, net_indices, gt_raw)
            gt_rel_viol = gt_sums.abs() / (gt_abs_sums + 1e-12)
            valid_multi = valid_multi & (gt_rel_viol < kcl_gt_filter)

        if stats is not None:
            stats['n_multi_total'] = n_multi_before_filter
            stats['n_multi_dropped'] = n_multi_before_filter - valid_multi.sum().item()

        if kcl_mode == 'denorm':
            # Denormalize to amps, compute relative violation: (Σsigned / Σabs)²
            raw_amps = torch.pow(10, multi_z * current_std + current_mean)
            signed_amps = raw_amps * multi_sign
            net_signed_sum = torch.zeros(n_multi_nets, device=device)
            net_signed_sum.scatter_add_(0, net_indices, signed_amps)
            net_abs_sum = torch.zeros(n_multi_nets, device=device)
            net_abs_sum.scatter_add_(0, net_indices, raw_amps)
            rel_viol = net_signed_sum / (net_abs_sum + 1e-12)

            if valid_multi.any():
                valid_rel_viol = rel_viol[valid_multi]
                if kcl_violation_threshold > 0:
                    with torch.no_grad():
                        above_thresh_denorm = valid_rel_viol.abs() > kcl_violation_threshold
                    if stats is not None:
                        stats['n_multi_above_thresh'] = above_thresh_denorm.sum().item()
                    if above_thresh_denorm.any():
                        multi_term_loss = (valid_rel_viol[above_thresh_denorm] ** 2).mean()
                        losses.append(multi_term_loss)
                        if stats is not None:
                            stats['multi_term_loss'] = multi_term_loss.item()
                else:
                    multi_term_loss = (valid_rel_viol ** 2).mean()
                    losses.append(multi_term_loss)
                    if stats is not None:
                        stats['multi_term_loss'] = multi_term_loss.item()

        elif kcl_mode == 'logsumexp':
            ln10 = 2.302585092994046  # ln(10)

            # Split into pos-sign and neg-sign edges
            pos_edge_mask = multi_sign > 0
            neg_edge_mask = multi_sign < 0

            NEG_INF = -1e9

            def scatter_logsumexp(z_vals, dst_idx, edge_mask, n_nets):
                """Compute logsumexp per net for a subset of edges.

                Converts z-scores back to log10|I| scale (z * std) before summing,
                so that logsumexp correctly computes log10(Σ|I|) up to a constant.
                The mean offset cancels when comparing pos vs neg sums.
                """
                if not edge_mask.any():
                    return torch.full((n_nets,), NEG_INF, device=device)
                z_sub = z_vals[edge_mask] * current_std * ln10
                dst_sub = dst_idx[edge_mask]
                max_vals = torch.full((n_nets,), NEG_INF, device=device)
                max_vals.scatter_reduce_(0, dst_sub, z_sub, reduce='amax', include_self=True)
                z_shifted = torch.exp(z_sub - max_vals[dst_sub])
                sum_exp = torch.zeros(n_nets, device=device)
                sum_exp.scatter_add_(0, dst_sub, z_shifted)
                result = (max_vals + torch.log(sum_exp + 1e-30)) / ln10
                has_edges = torch.zeros(n_nets, device=device, dtype=torch.bool)
                has_edges.scatter_(0, dst_sub, torch.ones_like(dst_sub, dtype=torch.bool))
                result[~has_edges] = NEG_INF
                return result

            log_pos_sum = scatter_logsumexp(multi_z, net_indices, pos_edge_mask, n_multi_nets)
            log_neg_sum = scatter_logsumexp(multi_z, net_indices, neg_edge_mask, n_multi_nets)

            # Valid nets: both pos and neg groups have edges
            valid_multi = valid_multi & (log_pos_sum > NEG_INF + 1) & (log_neg_sum > NEG_INF + 1)

            if valid_multi.any():
                diffs = log_pos_sum[valid_multi] - log_neg_sum[valid_multi]

                multi_term_loss = None
                if kcl_violation_threshold > 0:
                    # Only penalize nets where relative violation exceeds threshold
                    with torch.no_grad():
                        raw_pos = torch.pow(10, log_pos_sum[valid_multi])
                        raw_neg = torch.pow(10, log_neg_sum[valid_multi])
                        rel_viol_3t = (raw_pos - raw_neg).abs() / (raw_pos + raw_neg + 1e-12)
                        above_thresh_3t = rel_viol_3t > kcl_violation_threshold
                    if stats is not None:
                        stats['n_multi_above_thresh'] = above_thresh_3t.sum().item()
                    if above_thresh_3t.any():
                        d = diffs[above_thresh_3t]
                        if kcl_huber_delta > 0:
                            multi_term_loss = F.huber_loss(d, torch.zeros_like(d), delta=kcl_huber_delta, reduction='mean')
                        else:
                            multi_term_loss = (d ** 2).mean()
                        losses.append(multi_term_loss)
                else:
                    if kcl_huber_delta > 0:
                        multi_term_loss = F.huber_loss(diffs, torch.zeros_like(diffs), delta=kcl_huber_delta, reduction='mean')
                    else:
                        multi_term_loss = (diffs ** 2).mean()
                    losses.append(multi_term_loss)

                if stats is not None:
                    stats['multi_term_loss'] = multi_term_loss.item() if multi_term_loss is not None else 0.0

                    # GT KCL floor: same logsumexp but with GT currents
                    if gt_currents is not None:
                        with torch.no_grad():
                            gt_multi_z_norm = gt_z_at_src[multi_edges]
                            gt_log_pos = scatter_logsumexp(gt_multi_z_norm, net_indices, pos_edge_mask, n_multi_nets)
                            gt_log_neg = scatter_logsumexp(gt_multi_z_norm, net_indices, neg_edge_mask, n_multi_nets)
                            gt_diffs = gt_log_pos[valid_multi] - gt_log_neg[valid_multi]
                            stats['gt_multi_term_loss'] = (gt_diffs ** 2).mean().item()

    if not losses:
        if return_stats:
            return torch.tensor(0.0, device=device), kcl_supervised_mask, stats
        return torch.tensor(0.0, device=device), kcl_supervised_mask

    # Average the losses (both are scale-invariant, similar magnitude)
    total_loss = sum(losses) / len(losses)
    if return_stats:
        return total_loss, kcl_supervised_mask, stats
    return total_loss, kcl_supervised_mask


def compute_kcl_per_net_debug(
    node_currents: torch.Tensor,
    edge_index: torch.Tensor,
    num_terminals: torch.Tensor,
    train_mask: torch.Tensor,
    ptr: torch.Tensor,
    terminal_current_sign: torch.Tensor,
    current_mean: float,
    current_std: float,
    node_names: list,
    kcl_include_mask: torch.Tensor = None,
    graph_idx: int = 0,
) -> dict:
    """
    Compute per-net KCL violations for debugging.

    Returns dict of {net_name: relative_violation} for internal nets.
    """
    # Get graph boundaries
    start = ptr[graph_idx].item()
    end = ptr[graph_idx + 1].item()
    n_terms = num_terminals[graph_idx].item() if hasattr(num_terminals[graph_idx], 'item') else num_terminals[graph_idx]

    # Get names for this graph
    if isinstance(node_names[0], list):
        names = node_names[graph_idx]
    else:
        names = node_names[start:end]

    results = {}

    # For each internal net
    for local_idx in range(n_terms, end - start):
        global_idx = start + local_idx
        if not train_mask[global_idx]:
            continue
        net_name = names[local_idx] if local_idx < len(names) else f"net_{local_idx}"

        # Find terminals connected to this net
        src, dst = edge_index
        mask = (dst == global_idx) & (src >= start) & (src < start + n_terms)
        term_indices = src[mask]

        if len(term_indices) == 0:
            continue

        # Filter to eligible terminals
        eligible = [t for t in term_indices if kcl_include_mask is None or kcl_include_mask[t]]
        if len(eligible) < 2:
            continue  # Can't check KCL with fewer than 2 terminals

        # Compute sum
        total_signed = 0.0
        total_abs = 0.0

        for term_idx in eligible:
            pred_norm = node_currents[term_idx].item()
            pred_raw = 10 ** (pred_norm * current_std + current_mean)
            sign = terminal_current_sign[term_idx].item()

            total_signed += pred_raw * sign
            total_abs += pred_raw

        if total_abs > 1e-9:  # Filter nets with < 1nA total current (e.g., vc, vg2)
            rel_viol = abs(total_signed) / total_abs
            results[net_name] = rel_viol

    return results
