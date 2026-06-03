# task133 High-Cost Diagnosis

- Current estimated cost: 1406822
- Current file size: 783434 bytes
- Input shapes: 16x12;16x18;17x18;15x18
- Output shapes: 16x12;16x18;17x18;15x18
- Same shape: True
- Shape relation: same_shape
- Colors: 0 1 2 3 4 6 8
- Likely rule families: identity|color_map|mirror_or_rotate|local_neighborhood
- Recommended action: run formal same-shape rules, then inspect changed cells for compact mask algebra
- Expected cost after replacement: 10000

## Train Cases

### Case 0

- Input shape: 16x12
- Output shape: 16x12
- Input colors: 0 1 3 4
- Output colors: 0 1 3 4

### Case 1

- Input shape: 16x18
- Output shape: 16x18
- Input colors: 0 2 3 6 8
- Output colors: 0 2 3 6 8

### Case 2

- Input shape: 17x18
- Output shape: 17x18
- Input colors: 0 1 4 8
- Output colors: 0 1 4 8

### Case 3

- Input shape: 15x18
- Output shape: 15x18
- Input colors: 0 2 3 4 8
- Output colors: 0 2 3 4 8
