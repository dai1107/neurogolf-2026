# task209 High-Cost Diagnosis

- Current estimated cost: 144226
- Current file size: 349736 bytes
- Input shapes: 17x17;17x18;16x18
- Output shapes: 9x14;7x7;11x11
- Same shape: False
- Shape relation: shrinks_or_crop
- Colors: 0 1 2 3 4 8
- Likely rule families: fixed_crop|dynamic_bbox_crop|frame_or_substructure_extract
- Recommended action: try crop, dynamic bbox, frame interior, and substructure extraction builders
- Expected cost after replacement: 5000

## Train Cases

### Case 0

- Input shape: 17x17
- Output shape: 9x14
- Input colors: 0 1 2 3 4
- Output colors: 0 1 2 3 4

### Case 1

- Input shape: 17x18
- Output shape: 7x7
- Input colors: 0 2 3 4 8
- Output colors: 0 2 3 4 8

### Case 2

- Input shape: 16x18
- Output shape: 11x11
- Input colors: 0 1 2 3 4
- Output colors: 0 1 2 3 4
