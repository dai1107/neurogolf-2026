# task084 High-Cost Diagnosis

- Current estimated cost: 1390970
- Current file size: 1127799 bytes
- Input shapes: 15x15;3x3;7x7
- Output shapes: 15x15;3x3;7x7
- Same shape: True
- Shape relation: same_shape
- Colors: 0 2 4 5 6 8
- Likely rule families: identity|color_map|mirror_or_rotate|local_neighborhood|object_edit|mask_recolor
- Recommended action: run formal same-shape rules, then inspect changed cells for compact mask algebra
- Expected cost after replacement: 10000

## Train Cases

### Case 0

- Input shape: 15x15
- Output shape: 15x15
- Input colors: 0 6
- Output colors: 0 2 4 6

### Case 1

- Input shape: 3x3
- Output shape: 3x3
- Input colors: 0 5
- Output colors: 0 2 4 5

### Case 2

- Input shape: 7x7
- Output shape: 7x7
- Input colors: 0 8
- Output colors: 0 2 4 8
