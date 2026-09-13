"""
Circuit parsing and simulation utilities.

Modules:
- parser: SPICE netlist parsing
- simulator: NgSpice circuit simulation

The simulator (and its PySpice dependency) is loaded lazily so parser-only
consumers don't need PySpice installed.
"""

from .parser import SPICENetlistParser, Component, NetNode

__all__ = [
    # Parser
    'SPICENetlistParser',
    'Component',
    'NetNode',
    # Simulator
    'CircuitSimulator',
]


def __getattr__(name):
    if name == 'CircuitSimulator':
        from .simulator import CircuitSimulator
        return CircuitSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
