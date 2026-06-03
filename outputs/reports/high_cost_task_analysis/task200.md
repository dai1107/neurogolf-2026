# task200 High-Cost Diagnosis

- Current estimated cost: 990050
- Current file size: 797905 bytes
- Input shapes: 10x10;10x10;10x10
- Output shapes: 10x10;10x10;10x10
- Same shape: True
- Shape relation: same_shape
- Colors: 0 2 3 4 5
- Likely rule families: identity|color_map|mirror_or_rotate|local_neighborhood|object_edit|mask_recolor
- Recommended action: run formal same-shape rules, then inspect changed cells for compact mask algebra
- Expected cost after replacement: 10000

## Train Cases

### Case 0

- Input shape: 10x10
- Output shape: 10x10
- Input colors: 0 2
- Output colors: 0 2 5

### Case 1

- Input shape: 10x10
- Output shape: 10x10
- Input colors: 0 3
- Output colors: 0 3 5

### Case 2

- Input shape: 10x10
- Output shape: 10x10
- Input colors: 0 4
- Output colors: 0 4 5
