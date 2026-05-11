from __future__ import annotations


def split_name(session_index: int, total_sessions: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> str:
    train_cut = int(total_sessions * train_ratio)
    val_cut = train_cut + int(total_sessions * val_ratio)
    if session_index < train_cut:
        return "train"
    if session_index < val_cut:
        return "val"
    return "test"


def split_name_for_class(sample_index: int, samples_per_class: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> str:
    train_cut = int(samples_per_class * train_ratio)
    val_cut = train_cut + int(samples_per_class * val_ratio)
    if sample_index < train_cut:
        return "train"
    if sample_index < val_cut:
        return "val"
    return "test"
