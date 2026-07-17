"""Online metrics used by all three modules."""

from collections import deque


class RollingF1:
    """Calculates binary F1 over the last W predictions."""

    def __init__(self, window_size=5000, pos_label="ATTACK"):
        self.window = deque(maxlen=window_size)
        self.pos_label = pos_label

    def update(self, y_true, y_pred):
        self.window.append((y_true, y_pred))

    def get(self):
        if len(self.window) < 100:
            return 0.0
        tp = fp = fn = 0
        for yt, yp in self.window:
            if yp == self.pos_label and yt == self.pos_label:
                tp += 1
            elif yp == self.pos_label and yt != self.pos_label:
                fp += 1
            elif yp != self.pos_label and yt == self.pos_label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
