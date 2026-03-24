from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR
import math


# ── Scheduler factories ──────────────────────────────────────────────────────

def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )

# *EXP-002                                                                                                                                                                                                        
# *hypothesis: a warmup phase followed by cosine decay will improve early optimization stability
# *intervention: changed the scheduler to the warmup-plus-cosine lambda schedule and enabled warmup_steps=20
# *control: kept the optimizer, dataset, batch size, num_steps, seed, loss, and evaluation setup fixed
# *result: losses decreased substantially relative to the previous Adam baseline, but QA performance remained weak
class WarmupCosineSchedule:
    def __init__(self, warmup_steps, total_steps):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)

    def __call__(self, step):
        # Warmup phase
        if step < self.warmup_steps:
            return float(step + 1) / self.warmup_steps

        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        progress = float(step - self.warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)

        # *EXP-004                                                                                                                                                                                                        
        # *hypothesis: raising the scheduler's minimum learning-rate floor will prevent the final LR from collapsing too far and may improve late-stage optimization
        # *intervention: increased the minimum LR floor in the warmup-plus-cosine lambda schedule
        # *control: kept the training recipe and evaluation setup fixed
        # *result: changing the floor raised the final LR but did not materially change losses or QA metrics, so the original floor factor 1e-6 was retained
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

def lambda_scheduler(optimizer, args):
    # *EXP-002                                                                                                                                                                                                        
    # *hypothesis: a warmup phase followed by cosine decay will improve early optimization stability
    # *intervention: changed the scheduler to the warmup-plus-cosine lambda schedule and enabled warmup_steps=20
    # *control: kept the optimizer, dataset, batch size, num_steps, seed, loss, and evaluation setup fixed
    # *result: losses decreased substantially relative to the previous Adam baseline, but QA performance remained weak
    warmup_steps = getattr(args, "warmup_steps", 20)
    total_steps = getattr(args, "num_steps", 60000)

    schedule = WarmupCosineSchedule(warmup_steps, total_steps)

    return LambdaLR(optimizer, lr_lambda=schedule)

# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
}
