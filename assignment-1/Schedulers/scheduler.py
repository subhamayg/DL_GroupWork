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

# *EXP-003                                                                                                                                                                                                    
# *hypothesis: a warmup followed by cosine decay learning rate schedule will stabilise early training and ensure smoother convergence than fixed or non-warmup schedules                                          
# *intervention: replaced the previous lambda scheduler with a warmup-plus-cosine schedule and added a small minimum learning-rate floor                                                                          
# *control: kept optimizer, repaired codebase, dataset, batch size, num_steps, loss, seed, and evaluation protocol fixed  
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
        # *intervention: increase the minimum LR floor in the warmup-plus-cosine lambda schedule while keeping optimizer_name="adam", scheduler_name="lambda", warmup_steps=20, and learning_rate=1e-4 fixed              
        # *control: same repaired codebase, dataset split, batch_size=4, num_steps=200, loss, seed, and evaluation protocol
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

def lambda_scheduler(optimizer, args):
    # *EXP-003                                                                                                                                                                                                    
    # *hypothesis: a warmup followed by cosine decay learning rate schedule will stabilise early training and ensure smoother convergence than fixed or non-warmup schedules                                          
    # *intervention: replaced the previous lambda scheduler with a warmup-plus-cosine schedule and added a small minimum learning-rate floor                                                                          
    # *control: kept optimizer, repaired codebase, dataset, batch size, num_steps, loss, seed, and evaluation protocol fixed  
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
