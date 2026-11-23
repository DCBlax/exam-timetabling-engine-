"""Exam Timetabling Engine (ETE): CSP / graph-colouring solvers."""
import sys

# the exact solvers recurse once per exam (n up to 500+)
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

from .backtracking import solve_fc_mrv, solve_naive          # noqa: E402
from .bnb import solve_bnb                                    # noqa: E402
from .common import Result, allocate_venues, is_valid, violations  # noqa: E402
from .generators import load_instance, save_instance          # noqa: E402
from .greedy import solve_dsatur, solve_greedy                # noqa: E402
from .hybrid import solve_hybrid                              # noqa: E402
from .model import Instance, lower_bound                      # noqa: E402

SOLVERS = {
    "greedy": solve_greedy,
    "dsatur": solve_dsatur,
    "naive": solve_naive,
    "fcmrv": solve_fc_mrv,
    "bnb": solve_bnb,
    "hybrid": solve_hybrid,
}
