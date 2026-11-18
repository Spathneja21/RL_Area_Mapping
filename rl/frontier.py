"""Frontier candidates, BFS planner and path follower (numpy + stdlib only).

Shared by MacroExploreEnv (training), the greedy baseline (evaluate.py) and the ROS node, so the
policy chooses among exactly the same candidates and drives with the same executor everywhere.
"""
from collections import deque
import numpy as np
from mapping import RES, FREE, OCC, UNKNOWN, dilate, frontier

K = 8              # candidates offered to the policy (nearest first; slot 0 = greedy nearest frontier)
N_FEAT = 6         # per candidate: valid, path length, forward, left, frontier size, unknown area nearby
BIN = 10           # cells (1 m): one candidate per 1 m x 1 m bin of frontier cells
GAIN_BINS = 2      # unknown area is counted within +-2 bins (5 m x 5 m) of the candidate
MIN_DIST = 5       # cells: ignore frontier cells the robot is already standing on
REACHED = 3        # cells: goal reached


def plan(belief, bumps, start):
    """BFS path length (cells) from start through known free space; -1 = unreachable.
    Prefers 0.3 m clearance from obstacles; falls back to 0.2 m (the collision radius) for tight doors."""
    occ = belief == OCC
    safe = (belief == FREE) & ~dilate(occ, 2) & ~bumps     # bumps (remembered hits, blacklisted goals) un-inflated here
    front = frontier(belief)
    for trav in (safe & ~dilate(occ | bumps, 3), safe):
        dist = _bfs(trav, safe, start)
        if (front & (dist > MIN_DIST)).any():
            break
    return dist


def _bfs(trav, safe, start):
    h, w = trav.shape
    ok = trav.copy()
    r0, c0 = start
    near = np.zeros_like(ok)                          # escape zone: robot may start inside the clearance
    near[max(r0 - 3, 0):r0 + 4, max(c0 - 3, 0):c0 + 4] = True
    ok = (ok | (near & safe)).ravel().tolist()        # python lists: much faster than numpy scalar access
    dist = [-1] * (h * w)
    s = r0 * w + c0
    ok[s] = True
    dist[s] = 0
    q = deque([s])
    while q:
        i = q.popleft()
        d = dist[i] + 1
        c = i % w
        for j in (i - w, i + w, i - 1 if c else -1, i + 1 if c < w - 1 else -1):
            if 0 <= j < h * w and ok[j] and dist[j] < 0:
                dist[j] = d
                q.append(j)
    return np.array(dist, np.int32).reshape(h, w)


def _bin_sum(mask, k):
    h, w = -(-mask.shape[0] // k) * k, -(-mask.shape[1] // k) * k
    m = np.zeros((h, w), np.int32)
    m[:mask.shape[0], :mask.shape[1]] = mask
    return m.reshape(h // k, k, w // k, k).sum(axis=(1, 3))


def candidates(belief, dist, pose, origin):
    """-> (goals int[n, 2] (row, col), feats float32[K, N_FEAT]), nearest first, n <= K."""
    feats = np.zeros((K, N_FEAT), np.float32)
    front = frontier(belief) & (dist > MIN_DIST)
    cells = np.argwhere(front)
    if not len(cells):
        return cells, feats
    d = dist[cells[:, 0], cells[:, 1]]
    key = (cells[:, 0] // BIN) * 100000 + cells[:, 1] // BIN
    order = np.lexsort((d, key))                      # by bin, nearest cell first within each bin
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    idx = order[first]
    size = np.diff(np.r_[np.flatnonzero(first), len(order)])
    near = np.argsort(d[idx], kind='stable')[:K]
    goals, gd, size = cells[idx[near]], d[idx[near]], size[near]
    unk = np.pad(_bin_sum(belief == UNKNOWN, BIN), GAIN_BINS)
    box = sum(unk[GAIN_BINS + dr:unk.shape[0] - GAIN_BINS + dr, GAIN_BINS + dc:unk.shape[1] - GAIN_BINS + dc]
              for dr in range(-GAIN_BINS, GAIN_BINS + 1) for dc in range(-GAIN_BINS, GAIN_BINS + 1))
    gain = box[goals[:, 0] // BIN, goals[:, 1] // BIN]
    x, y, th = pose
    dx = origin[0] + (goals[:, 1] + 0.5) * RES - x
    dy = origin[1] + (goals[:, 0] + 0.5) * RES - y
    n = len(goals)
    feats[:n] = np.stack([np.ones(n), gd * RES / 20, (dx * np.cos(th) + dy * np.sin(th)) / 20,
                          (-dx * np.sin(th) + dy * np.cos(th)) / 20, size / 50, gain * RES * RES / 100], 1)
    return goals, feats


def is_frontier(belief, cell):
    r, c = cell
    if belief[r, c] != FREE:
        return False
    return any(0 <= rr < belief.shape[0] and 0 <= cc < belief.shape[1] and belief[rr, cc] == UNKNOWN
               for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)))


def path_to(dist, goal):
    """Cells from the robot (dist 0) to goal, following the BFS distance field downhill."""
    p = [tuple(goal)]
    while dist[p[-1]] > 0:
        r, c = p[-1]
        d = dist[r, c]
        for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= n[0] < dist.shape[0] and 0 <= n[1] < dist.shape[1] and dist[n] == d - 1:
                p.append(n)
                break
    return np.array(p[::-1])


def steer(pose, path, origin, lookahead=6):
    """Motion primitive index (env.ACTIONS) toward the path cell `lookahead` cells past the closest one."""
    x, y, th = pose
    r, c = (y - origin[1]) / RES, (x - origin[0]) / RES
    i = int(np.argmin(np.abs(path[:, 0] - r) + np.abs(path[:, 1] - c)))
    tr, tc = path[min(i + lookahead, len(path) - 1)]
    err = np.arctan2(origin[1] + (tr + 0.5) * RES - y, origin[0] + (tc + 0.5) * RES - x) - th
    err = (err + np.pi) % (2 * np.pi) - np.pi
    if abs(err) > 0.6:
        return 3 if err > 0 else 4                    # rotate left / right
    if abs(err) > 0.15:
        return 1 if err > 0 else 2                    # arc left / right
    return 0                                          # forward
