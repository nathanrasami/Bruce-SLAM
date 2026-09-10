#!/usr/bin/env python3
"""Inspection d'un ROS bag SANS ROS (lib rosbags) — LE premier réflexe sur tout
nouveau dataset (cf. HOLOOCEAN_GARDE_FOU.md §3 et DATASETS.md).

Affiche : topics, types, nombre de messages, cadence, durée ; puis pour chaque
topic un APERÇU du premier message (champs utiles : tailles de tableaux,
range_max des LaserScan, frame_id, etc.).

Usage : python3 analysis/inspect_bag.py <fichier.bag> [--topic /un/topic]
"""
import argparse
import numpy as np

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


def apercu(msg, indent="    "):
    """Aperçu compact des champs d'un message désérialisé."""
    out = []
    for name in getattr(msg, "__dataclass_fields__", {}):
        v = getattr(msg, name)
        if isinstance(v, (int, float, str, bool)):
            out.append(f"{indent}{name} = {v!r}")
        elif isinstance(v, np.ndarray) or isinstance(v, (list, tuple)):
            a = np.asarray(v)
            extra = ""
            if a.size and a.dtype.kind in "if":
                extra = f", min {a.min():.3g}, max {a.max():.3g}"
            out.append(f"{indent}{name} : tableau {a.shape} {a.dtype}{extra}")
        else:
            out.append(f"{indent}{name} : {type(v).__name__}")
            if hasattr(v, "__dataclass_fields__"):
                out.extend(apercu(v, indent + "  ")[:6])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--topic", default=None, help="détailler seulement ce topic")
    a = ap.parse_args()
    ts = get_typestore(Stores.ROS1_NOETIC)

    with Reader(a.bag) as r:
        dur = (r.end_time - r.start_time) * 1e-9
        print(f"=== {a.bag} — {dur:.1f} s ({dur/60:.1f} min), "
              f"{sum(c.msgcount for c in r.connections)} messages\n")
        print(f"{'topic':35s} {'type':42s} {'n':>8s} {'Hz':>7s}")
        for c in sorted(r.connections, key=lambda c: c.topic):
            print(f"{c.topic:35s} {c.msgtype:42s} {c.msgcount:8d} "
                  f"{c.msgcount/dur:7.1f}")
        print()
        vus = set()
        for c, t, raw in r.messages():
            if c.topic in vus or (a.topic and c.topic != a.topic):
                continue
            vus.add(c.topic)
            print(f"--- {c.topic} ({c.msgtype}) — 1er message :")
            try:
                msg = ts.deserialize_ros1(raw, c.msgtype)
                print("\n".join(apercu(msg)[:14]))
            except Exception as e:
                print(f"    (désérialisation impossible : {e} — type custom ? "
                      f"utiliser les topics *_ros standards)")
            print()
            if len(vus) == len(r.connections) or a.topic:
                break


if __name__ == "__main__":
    main()
