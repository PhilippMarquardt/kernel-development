"""Static SASS analyser (M9, M13, M15). Owner: P5. Pure Python over cuobjdump/nvdisasm text; no GPU.

Fixture: tests/fixtures/sass/vjc_k_L1_F0_M1.sass (cuobjdump -sass of a PDL attention kernel: contains
ACQBULK and LDG.E.CONSTANT at /*0080*/ before it -> positive case for the PDL hoist audit).
"""
