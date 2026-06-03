# task025 High-Cost Diagnosis

- Current estimated cost: 332565
- Current file size: 826752 bytes
- Input shapes: 18x19;15x14;15x16
- Output shapes: 18x19;15x14;15x16
- Same shape: True
- Shape relation: same_shape
- Colors: 0 1 2 3 4 8
- Likely rule families: identity|color_map|mirror_or_rotate|local_neighborhood|object_edit|mask_recolor
- Recommended action: run formal same-shape rules, then inspect changed cells for compact mask algebra
- Expected cost after replacement: 10000

## Train Cases

### Case 0

- Input shape: 18x19
- Output shape: 18x19
- Input colors: 0 2 3 4
- Output colors: 0 3 4

### Case 1

- Input shape: 15x14
- Output shape: 15x14
- Input colors: 0 1 2 4
- Output colors: 0 1 2

### Case 2

- Input shape: 15x16
- Output shape: 15x16
- Input colors: 0 1 8
- Output colors: 0 8
