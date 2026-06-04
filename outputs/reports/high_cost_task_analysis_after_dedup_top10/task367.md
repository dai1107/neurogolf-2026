# task367 High-Cost Diagnosis

- Current estimated cost: 295949
- Current file size: 293653 bytes
- Input shapes: 12x19;13x16;15x17
- Output shapes: 12x19;13x16;15x17
- Same shape: True
- Shape relation: same_shape
- Colors: 0 4 5
- Likely rule families: identity|color_map|mirror_or_rotate|local_neighborhood|object_edit|mask_recolor
- Recommended action: run formal same-shape rules, then inspect changed cells for compact mask algebra
- Expected cost after replacement: 10000

## Train Cases

### Case 0

- Input shape: 12x19
- Output shape: 12x19
- Input colors: 0 5
- Output colors: 0 4 5

### Case 1

- Input shape: 13x16
- Output shape: 13x16
- Input colors: 0 5
- Output colors: 0 4 5

### Case 2

- Input shape: 15x17
- Output shape: 15x17
- Input colors: 0 5
- Output colors: 0 4 5
