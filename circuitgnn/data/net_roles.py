"""Net role classification from SPICE net names.

Shared core used by both classification paths:
  - ``CircuitGraphBuilder._classify_net`` (graph build time) — returns the
    string category 'gnd'/'vdd'/'input'/'output'/'internal'.
  - ``PretrainCombinedLoader._classify_net`` (and
    scripts/patch_dataset_net_role.py) — returns the 5-class integer role
    used for the ``net_role`` one-hot dataset attribute.

The two callers agree on every predicate except one: the pretrain/patch
variant additionally treats any name starting with 'gnd' as ground
(``gnd_prefix=True``); the graph builder only matches the exact names
'0'/'gnd'/'vss' plus the 'gnda' substring. The difference is parameterized
here rather than merged.

Templates should use consistent names: vdd/vdda for supply, gnd/gnda/0 for
ground, vin*/vp/vn for inputs, vout* for outputs. Everything else is internal.
"""

# Net role classification: 5 classes (VDD/GND/SIG_IN/SIG_OUT/INTERNAL).
NET_ROLE_DIM = 5
ROLE_VDD, ROLE_GND, ROLE_SIG_IN, ROLE_SIG_OUT, ROLE_INTERNAL = 0, 1, 2, 3, 4

# String category (graph builder vocabulary) → integer role.
NET_TYPE_TO_ROLE = {
    'gnd': ROLE_GND,
    'vdd': ROLE_VDD,
    'input': ROLE_SIG_IN,
    'output': ROLE_SIG_OUT,
    'internal': ROLE_INTERNAL,
}


def classify_net(net_name: str, gnd_prefix: bool = False) -> str:
    """Classify net type based on standard SPICE naming conventions.

    Args:
        net_name: net name from the netlist.
        gnd_prefix: if True, any name starting with 'gnd' also counts as
            ground (pretrain loader / net_role patch behavior).

    Returns:
        One of 'gnd', 'vdd', 'input', 'output', 'internal'.
    """
    net_lower = net_name.lower()
    if (net_lower in ('0', 'gnd', 'vss') or 'gnda' in net_lower
            or (gnd_prefix and net_lower.startswith('gnd'))):
        return 'gnd'
    if 'vdd' in net_lower or 'vcc' in net_lower:
        return 'vdd'
    if 'vin' in net_lower or net_lower in ('vp', 'vn', 'vsig', 'inp', 'inn'):
        return 'input'
    if 'vout' in net_lower:
        return 'output'
    return 'internal'


def classify_net_role(name: str) -> int:
    """Return the 5-class integer role for a net name (pretrain semantics)."""
    return NET_TYPE_TO_ROLE[classify_net(name, gnd_prefix=True)]
