from .log_normal_mean import log_normal_mean


def log_normal_height(w, position=None):
    return len(w)*log_normal_mean(w, position=position)
