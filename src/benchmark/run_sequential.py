import os
import sys
from time import time

import bbobbenchmarks as bn
import fgeneric
import numpy as np

np.random.seed(42)


def run_optimizer(
    optimizer, dim, fID, instance, logfile, lb, ub, max_FEs, data_path, bbob_opt
):
    """Experiment wrapper"""
    # Set different seed for different processes
    start = time()
    seed = np.mod(int(start) + os.getpid(), 1000)
    np.random.seed(seed)

    data_path = os.path.join(data_path, str(instance))
    max_FEs = eval(max_FEs)

    f = fgeneric.LoggingFunction(data_path, **bbob_opt)
    f.setfun(*bn.instantiate(fID, iinstance=instance))

    # use f.evalfun
    opt = optimizer(dim, f.evalfun, f.ftarget, max_FEs, lb, ub, logfile)
    opt.run()

    f.finalizerun()
    with open(logfile, "a") as fout:
        fout.write(
            "{} on f{} in {}D, instance {}: FEs={}, fbest-ftarget={:.4e}, "
            "elapsed time [m]: {:.3f}\n".format(
                optimizer,
                fID,
                dim,
                instance,
                f.evaluations,
                f.fbest - f.ftarget,
                (time() - start) / 60.0,
            )
        )


if __name__ == "__main__":
    dims = (2,)
    fIDs = bn.nfreeIDs[6:]  # for all fcts
    instance = [1] * 10

    algorithm = test_BO

    opts = {
        "max_FEs": "50",
        "lb": -5,
        "ub": 5,
        "data_path": "./bbob_data/%s" % algorithm.__name__,
    }
    opts["bbob_opt"] = {
        "comments": "max_FEs={0}".format(opts["max_FEs"]),
        "algid": algorithm.__name__,
    }

    for dim in dims:
        for fID in fIDs:
            for i in instance:
                run_optimizer(algorithm, dim, fID, i, logfile="./log", **opts)
