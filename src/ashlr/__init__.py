"""ASH-LR: Adaptive and numerically stable optimization for ill-conditioned logistic regression."""
from .objectives import LogisticObjective
from .optimizers import METHODS, OptResult, ash_lr, gd, gd_fixed, lbfgs, newton, newton_cg
from .data import Dataset, load_spambase, make_synthetic

__all__ = ["LogisticObjective", "METHODS", "OptResult", "ash_lr", "gd", "gd_fixed", "lbfgs",
           "newton", "newton_cg", "Dataset", "load_spambase", "make_synthetic"]
__version__ = "1.0.0"
