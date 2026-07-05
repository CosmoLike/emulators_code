"""Polynomial-chaos (NPCE) emulator variant: a closed-form
sparse-Legendre PCE base (emulator_designs.py) plus the losses that
put a neural refiner on top of it (loss_functions.py, additive and
multiplicative). Deprioritized for cosmic-shear xi -- a smoothness
prior cannot lower a data-coverage floor -- and kept for reuse; the
history lives in notes/npce-and-ia-template-factoring.md."""
