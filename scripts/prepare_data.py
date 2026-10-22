"""Split whole real capture sessions; map source labels to 13 ranks.

Input: data/raw/SESSION/image.jpg + image.txt (YOLO box labels).
Optional synthetic root: image/label pairs, used in training ONLY.
"""
import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from blackjack import RANKS

EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def label_rank(name):
    # Accept ranks and common suit-specific names: AS, 10H, ace_of_spades.
    s = str(name).strip().upper()
    if s in RANKS:
        return s
    match = re.fullmatch(r'(10|[2-9AJQKT])[_ -]?[CDHS]', s)
    if match:
        return '10' if match[1] == 'T' else match[1]
    match = re.fullmatch(r'(ACE|KING|QUEEN|JACK|TEN|[2-9]|10)[_ -]+(?:OF[_ -]+)?(?:CLUBS|DIAMONDS|HEARTS|SPADES)', s)
    if match:
        return {'ACE':'A', 'KING':'K', 'QUEEN':'Q', 'JACK':'J', 'TEN':'10'}.get(match[1], match[1])
    raise ValueError(f'Unrecognized source class {name!r}; edit classes JSON to explicit ranks.')


def convert_labels(path, mapping):
    if not path.exists():
        raise ValueError(f'Missing labels: {path}. Empty images need an empty .txt file.')
    converted = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'{path}: expected class cx cy width height')
        cls = int(parts[0])
        if not 0 <= cls < len(mapping):
            raise ValueError(f'{path}: invalid class ID {cls}')
        cx, cy, bw, bh = map(float, parts[1:])
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f'{path}: invalid normalized box')
        if cx-bw/2 < -0.001 or cy-bh/2 < -0.001 or cx+bw/2 > 1.001 or cy+bh/2 > 1.001:
            raise ValueError(f'{path}: box outside image')
        converted.append(f'{mapping[cls]} {cx} {cy} {bw} {bh}')
    return '\n'.join(converted) + ('\n' if converted else '')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', default='data/raw')
    p.add_argument('--classes', default='data/classes.json', help='JSON list in source class-ID order')
    p.add_argument('--output', default='data/prepared')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--synthetic', help='Optional root with paired images and labels')
    p.add_argument('--synthetic-classes', help='Separate source class list for synthetic data')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Output exists. Choose a NEW --output to preserve previous splits.')
    sessions = sorted(s for s in Path(args.raw).iterdir() if s.is_dir())
    if len(sessions) < 3:
        raise ValueError('Need at least 3 independent real capture sessions; 10+ recommended.')
    random.Random(args.seed).shuffle(sessions)
    n_val = n_test = max(1, round(len(sessions) * 0.15))
    assigned = [('test', sessions[:n_test]), ('val', sessions[n_test:n_test+n_val]),
                ('train', sessions[n_test+n_val:])]
    source_names = json.loads(Path(args.classes).read_text())
    mapping = [RANKS.index(label_rank(n)) for n in source_names]
    # Validate ALL inputs before creating output. Hash guard catches exact leakage.
    jobs, manifest, seen = [], [], {}
    def collect(image, split, session, class_map, prefix):
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in seen:
            if seen[digest] != split:
                raise ValueError(f'Exact duplicate crosses splits: {image}')
            return
        seen[digest] = split
        labels = convert_labels(image.with_suffix('.txt'), class_map)
        name = f'{prefix}_{len(jobs):06d}{image.suffix.lower()}'
        jobs.append((image, split, name, labels))
        manifest.append({'image':name, 'source':str(image), 'session':session,
                         'split':split, 'sha256':digest})
    for split, group in assigned:
        for session in group:
            images = sorted(p for p in session.iterdir() if p.suffix.lower() in EXTENSIONS)
            if not images:
                raise ValueError(f'Empty session: {session}')
            for image in images:
                collect(image, split, session.name, mapping, 'real')
    if args.synthetic:
        if not args.synthetic_classes:
            p.error('--synthetic requires --synthetic-classes')
        smap = [RANKS.index(label_rank(n)) for n in json.loads(Path(args.synthetic_classes).read_text())]
        for image in sorted(Path(args.synthetic).rglob('*')):
            if image.suffix.lower() in EXTENSIONS:
                collect(image, 'train', 'synthetic', smap, 'synthetic')
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (output / kind / split).mkdir(parents=True)
    for image, split, name, labels in jobs:
        shutil.copy2(image, output / 'images' / split / name)
        (output / 'labels' / split / Path(name).with_suffix('.txt')).write_text(labels)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    # JSON flow sequences and quoted strings are valid YAML.
    yaml = f'path: {json.dumps(str(output.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {json.dumps(list(RANKS))}\n'
    (output / 'cards.yaml').write_text(yaml)
    print(f'Prepared {len(jobs)} images. Train using {output / "cards.yaml"}')
    for split in ('train', 'val', 'test'):
        print(split, sum(row['split'] == split for row in manifest))


if __name__ == '__main__':
    main()
