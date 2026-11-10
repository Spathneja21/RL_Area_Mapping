"""Numpy-only core shared by the training env and the ROS node.

Grid convention: grid[row, col], row = (y - oy) / RES, col = (x - ox) / RES (same as nav_msgs/OccupancyGrid).
Belief values: -1 unknown, 0 free, 1 occupied.
"""
import os
import numpy as np

RES = 0.1          # m per cell, belief/truth resolution
CROP = 64          # observation crop size (px)
SCALES = (1, 8)    # cells per px: local 6.4 m window, global 51.2 m window
UNKNOWN, FREE, OCC = -1, 0, 1


def load_truth(yaml_path):
    """ROS map_server yaml+pgm -> (occupied bool grid at RES, origin xy). Unknown counts as wall."""
    meta = {}
    for line in open(yaml_path):
        if ':' in line:
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip()
    pgm = os.path.join(os.path.dirname(yaml_path), meta['image'])
    img = _read_pgm(pgm)[::-1]                      # pgm row 0 is the top (max y)
    src_res = float(meta['resolution'])
    k = int(round(RES / src_res))
    h, w = img.shape[0] // k * k, img.shape[1] // k * k
    free = (img[:h, :w] > 250).reshape(h // k, k, w // k, k).all(axis=(1, 3))
    origin = [float(t) for t in meta['origin'].strip('[]').split(',')[:2]]
    return ~free, np.array(origin)


def _read_pgm(path):
    data = open(path, 'rb').read()
    tokens, i = [], 0
    while len(tokens) < 4:                          # magic, width, height, maxval (skip comments)
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b'#':
            i = data.index(b'\n', i)
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    assert tokens[0] == b'P5', 'only binary PGM supported'
    w, h = int(tokens[1]), int(tokens[2])
    return np.frombuffer(data, np.uint8, w * h, i + 1).reshape(h, w)


def dilate(mask, r_cells):
    """Binary dilation with a disk (mask shifted by every offset in the disk)."""
    out = mask.copy()
    h, w = mask.shape
    for dr in range(-r_cells, r_cells + 1):
        for dc in range(-r_cells, r_cells + 1):
            if dr * dr + dc * dc <= r_cells * r_cells and (dr or dc):
                out[max(dr, 0):h + min(dr, 0), max(dc, 0):w + min(dc, 0)] |= \
                    mask[max(-dr, 0):h + min(-dr, 0), max(-dc, 0):w + min(-dc, 0)]
    return out


def flood(free, seed):
    """Connected component of `free` containing seed (row, col), 4-connected."""
    comp = np.zeros_like(free)
    comp[seed] = True
    while True:
        grown = comp.copy()
        grown[1:] |= comp[:-1]; grown[:-1] |= comp[1:]
        grown[:, 1:] |= comp[:, :-1]; grown[:, :-1] |= comp[:, 1:]
        grown &= free
        if grown.sum() == comp.sum():
            return comp
        comp = grown


MAX_RANGE = 10.0
_D = np.arange(0.0, MAX_RANGE, RES / 2)            # ray sample spacing: half a cell


def _ray_cells(origin, pose, angles, shape):
    x, y, th = pose
    ang = th + np.asarray(angles)
    r = np.floor((y + np.sin(ang)[:, None] * _D - origin[1]) / RES).astype(int)
    c = np.floor((x + np.cos(ang)[:, None] * _D - origin[0]) / RES).astype(int)
    inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    return np.clip(r, 0, shape[0] - 1), np.clip(c, 0, shape[1] - 1), inside


def cast(occ, origin, pose, n_beams=180):
    """Simulated lidar on the truth grid -> (beam angles in robot frame, ranges; inf = no return)."""
    angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    r, c, _ = _ray_cells(origin, pose, angles, occ.shape)
    hit = occ[r, c]
    any_hit = hit.any(1)
    return angles, np.where(any_hit, _D[hit.argmax(1)], np.inf)


def integrate(belief, origin, pose, angles, ranges):
    """Write one scan into the belief: cells before the return -> free, the return cell -> occupied.
    No return (inf / beyond range) -> free out to MAX_RANGE. Same code for the sim and the real /scan."""
    rng = np.nan_to_num(np.asarray(ranges, float), nan=np.inf, posinf=np.inf)
    r, c, inside = _ray_cells(origin, pose, angles, belief.shape)
    free = (_D[None, :] < rng[:, None]) & inside
    belief[r[free], c[free]] = FREE
    hit = np.isfinite(rng) & (rng < MAX_RANGE)
    k = np.minimum(np.round(rng[hit] / (RES / 2)).astype(int), len(_D) - 1)
    ok = inside[hit, k]
    belief[r[hit, k][ok], c[hit, k][ok]] = OCC


def _pool(grid, k):
    """Downsample by k: (any free, any occupied)."""
    h, w = grid.shape[0] // k * k, grid.shape[1] // k * k
    g = grid[:h, :w].reshape(h // k, k, w // k, k)
    return (g == FREE).any(axis=(1, 3)), (g == OCC).any(axis=(1, 3))


def observe(belief, trail, origin, pose):
    """Egocentric multi-scale crops, robot at centre facing up. -> uint8 (6, CROP, CROP):
    per scale [free, occupied, visited]."""
    x, y, th = pose
    idx = (CROP / 2 - np.arange(CROP) - 0.5)
    ahead, left = np.meshgrid(idx, idx, indexing='ij')        # row 0 = ahead, col 0 = left
    chans = []
    for k in SCALES:
        s = k * RES
        wx = x + (ahead * np.cos(th) - left * np.sin(th)) * s - origin[0]
        wy = y + (ahead * np.sin(th) + left * np.cos(th)) * s - origin[1]
        free, occ = _pool(belief, k)
        vis = _pool(np.where(trail, FREE, UNKNOWN), k)[0]
        r, c = np.floor(wy / s).astype(int), np.floor(wx / s).astype(int)
        inside = (r >= 0) & (r < free.shape[0]) & (c >= 0) & (c < free.shape[1])
        r, c = np.clip(r, 0, free.shape[0] - 1), np.clip(c, 0, free.shape[1] - 1)
        for g in (free, occ, vis):
            chans.append(np.where(inside, g[r, c], False))
    return (np.stack(chans) * 255).astype(np.uint8)
