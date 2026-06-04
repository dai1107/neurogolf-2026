# task366 High-Cost Diagnosis

- Current estimated cost: 266691
- Current file size: 1269927 bytes
- Input shapes: 30x17;11x20;10x16
- Output shapes: 15x17;11x10;10x8
- Same shape: False
- Shape relation: shrinks_or_crop
- Colors: 0 1 2 3 4 6 8
- Likely rule families: fixed_crop|dynamic_bbox_crop|frame_or_substructure_extract
- Recommended action: try crop, dynamic bbox, frame interior, and substructure extraction builders
- Expected cost after replacement: 5000

## Train Cases

### Case 0

- Input shape: 30x17
- Output shape: 15x17
- Input colors: 0 1 2 3 8
- Output colors: 0 1 2 3

### Case 1

- Input shape: 11x20
- Output shape: 11x10
- Input colors: 1 2 3 6 8
- Output colors: 1 2 3 8

### Case 2

- Input shape: 10x16
- Output shape: 10x8
- Input colors: 1 2 4 6 8
- Output colors: 1 2 4 6
