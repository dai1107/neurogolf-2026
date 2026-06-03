# task157 High-Cost Diagnosis

- Current estimated cost: 1023477
- Current file size: 857911 bytes
- Input shapes: 10x15;10x15
- Output shapes: 10x15;10x15
- Same shape: True
- Shape relation: same_shape
- Colors: 0 1 2 5
- Likely rule families: identity|color_map|mirror_or_rotate|local_neighborhood|object_edit|mask_recolor
- Recommended action: run formal same-shape rules, then inspect changed cells for compact mask algebra
- Expected cost after replacement: 10000

## Train Cases

### Case 0

- Input shape: 10x15
- Output shape: 10x15
- Input colors: 0 2 5
- Output colors: 0 1 2

### Case 1

- Input shape: 10x15
- Output shape: 10x15
- Input colors: 0 2 5
- Output colors: 0 1 2
