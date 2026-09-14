"""Per-loss warmup schedules for train_v3.

These classes replace the copy-pasted linear warmup ramp blocks that lived in
train_v3's epoch loop. The arithmetic is reproduced exactly:

- ``LossSchedule.weight`` computes ``target * (e + 1) / warmup_epochs``
  (left-associative, like the original inline expressions).
- ``PairedLossSchedule.weights`` computes the warmup fraction once and
  multiplies it onto each target (``target * frac``), matching the original
  supervised gm/gds ramp which shared one fraction between the two weights.
  This is kept separate from LossSchedule because ``a * ((e + 1) / w)`` and
  ``a * (e + 1) / w`` are not guaranteed to be bitwise-equal floats.

The effective start epoch is ``start_epoch_cfg`` when > 0, else
``start_default`` (train_v3 passes the current-loss warmup length as the
default for most physics losses, and 0 for the DC gain and region losses).
"""


class LossSchedule:
    """Linear warmup ramp for one loss weight.

    weight(epoch) is:
      - 0.0 before the start epoch,
      - ``target * (e + 1) / warmup_epochs`` for e = epoch - start within the
        warmup window (only when warmup_epochs > 0),
      - ``target`` afterwards (and always when warmup_epochs == 0 and
        epoch >= start).
    """

    def __init__(self, target, warmup_epochs=0, start_epoch_cfg=0, start_default=0):
        self.target = target
        self.warmup_epochs = warmup_epochs
        self.start_epoch_cfg = start_epoch_cfg
        self.start_default = start_default

    @property
    def start_epoch(self):
        return self.start_epoch_cfg if self.start_epoch_cfg > 0 else self.start_default

    def weight(self, epoch):
        start = self.start_epoch
        if self.warmup_epochs > 0 and epoch >= start:
            e = epoch - start
            if e < self.warmup_epochs:
                return self.target * (e + 1) / self.warmup_epochs
            else:
                return self.target
        elif epoch < start:
            return 0.0
        else:
            return self.target


class PairedLossSchedule:
    """Shared linear warmup ramp for two loss weights (supervised gm/gds).

    The warmup fraction is computed once and applied to both targets,
    reproducing the original in-place code exactly (``target * frac``).
    """

    def __init__(self, target_a, target_b, warmup_epochs=0, start_epoch_cfg=0, start_default=0):
        self.target_a = target_a
        self.target_b = target_b
        self.warmup_epochs = warmup_epochs
        self.start_epoch_cfg = start_epoch_cfg
        self.start_default = start_default

    @property
    def start_epoch(self):
        return self.start_epoch_cfg if self.start_epoch_cfg > 0 else self.start_default

    def weights(self, epoch):
        start = self.start_epoch
        if self.warmup_epochs > 0 and epoch >= start:
            e = epoch - start
            if e < self.warmup_epochs:
                frac = (e + 1) / self.warmup_epochs
                return self.target_a * frac, self.target_b * frac
            else:
                return self.target_a, self.target_b
        elif epoch < start:
            return 0.0, 0.0
        else:
            return self.target_a, self.target_b
