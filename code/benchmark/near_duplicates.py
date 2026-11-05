"""Find perceptual-hash candidates efficiently; candidates are not proof of duplicates."""
from collections import defaultdict


def candidates(rows, distance=4):
    # Five disjoint chunks: <=4 differing bits guarantees an equal chunk.
    buckets = defaultdict(list)
    output = []
    values = [int(r['phash'], 16) for r in rows]
    for i, value in enumerate(values):
        possible = set()
        for part in range(5):
            key = (part, (value >> (part * 13)) & 8191)
            possible.update(buckets[key])
        for j in sorted(possible):
            diff = (value ^ values[j]).bit_count()
            if diff <= distance:
                output.append({'left': rows[j]['path'], 'right': rows[i]['path'],
                               'distance': diff,
                               'same_pixels': rows[j]['pixel_sha256'] == rows[i]['pixel_sha256']})
        for part in range(5):
            buckets[(part, (value >> (part * 13)) & 8191)].append(i)
    return output


def connected_groups(rows, pairs):
    parent = {r['path']: r['path'] for r in rows}

    def find(path):
        while path != parent[path]:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    for pair in pairs:
        left, right = find(pair['left']), find(pair['right'])
        parent[max(left, right)] = min(left, right)
    return {p: find(p) for p in parent}
