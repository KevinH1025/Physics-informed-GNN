"""Fixed 3-stage opamp topology constants shared by the tower architecture."""

# Default 3-stage opamp subcircuit groups (mosfet_info indices)
DEFAULT_SUBCIRCUIT_GROUPS = {
    'bias_pmos': [0, 1, 2, 3, 4, 7],
    'cmfb': [5, 6],
    'diff_pair': [8, 9],
    'cascode': [12, 13, 10, 11],
    'bias_nmos': [14, 16, 18, 15, 17],
    'stage2': [19, 20, 21],
    'output': [22, 23],
}

# Default DAG edges: parent → child (forward signal flow)
# Map: bias_pmos=0, cmfb=1, diff_pair=2, cascode=3, bias_nmos=4, stage2=5, output=6
DEFAULT_DAG_EDGES = [
    (0, 1),  # bias_pmos → cmfb
    (0, 2),  # bias_pmos → diff_pair
    (0, 3),  # bias_pmos → cascode
    (4, 3),  # bias_nmos → cascode
    (1, 3),  # cmfb → cascode
    (2, 3),  # diff_pair → cascode
    (0, 5),  # bias_pmos → stage2
    (3, 5),  # cascode → stage2
    (3, 6),  # cascode → output
    (5, 6),  # stage2 → output
]

# 14 key signal-path MOSFETs (mosfet_info indices):
#   Stage 1: M8(8), M9(9), M5(5), M6(6), M15(12), M16(13), M19(10), M20(11)
#   Stage 2: M10(19), M21(20), M22(21), M7(7)
#   Stage 3: M11(22), M23(23)
SIGNAL_PATH_MOSFET_INDICES = [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23]
