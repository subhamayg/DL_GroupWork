from Optimizers.adam import Adam
from Optimizers.sgd import SGD
from Optimizers.sgd_momentum import SGDMomentum


# ── Optimizer factories ──────────────────────────────────────────────────────
#
# NOTE: `adam` sets lr=1.0 because its learning rate is entirely controlled by
# the paired `warmup_lambda` scheduler (which outputs the actual lr values).
# `sgd` and `sgd_momentum` use args.learning_rate directly and should be
# paired with `cosine` or `step` schedulers.

def adam(params, args):
    return Adam(
        params=params,
        # *EXP-003
        # *hypothesis: lower base learning rates under the warmup-plus-cosine scheduler will further reduce loss explosion and improve training stability
        # *intervention: changed learning_rate while keeping the warmup-plus-cosine scheduler fixed
        # *control: kept optimizer, repaired codebase, scheduler design, dataset, batch size, num_steps, loss, seed, and evaluation protocol fixed
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=getattr(args, "eps", 1e-7),
        weight_decay=args.weight_decay,
    )


def sgd(params, args):
    return SGD(
        params=params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def sgd_momentum(params, args):
    return SGDMomentum(
        params=params,
        lr=args.learning_rate,
        momentum=getattr(args, "momentum", 0.9),
        weight_decay=args.weight_decay,
    )


# ── Registry ─────────────────────────────────────────────────────────────────

optimizers = {
    "adam":          adam,
    "sgd":           sgd,
    "sgd_momentum":  sgd_momentum,
}
