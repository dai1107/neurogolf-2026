# task319 High-Cost Diagnosis

- Current estimated cost: 78739
- Current file size: 247340 bytes
- Input shapes: 17x17;18x18;19x17;15x17
- Output shapes: 5x5;5x3;5x3;3x5
- Same shape: False
- Shape relation: shrinks_or_crop
- Colors: 1 2 3 4 6 8
- Likely rule families: fixed_crop|dynamic_bbox_crop|frame_or_substructure_extract
- Recommended action: try crop, dynamic bbox, frame interior, and substructure extraction builders
- Expected cost after replacement: 5000

## Train Cases

### Case 0

- Input shape: 17x17
- Output shape: 5x5
- Input colors: 1 2 3 8
- Output colors: 1 2

### Case 1

- Input shape: 18x18
- Output shape: 5x3
- Input colors: 3 4 6 8
- Output colors: 4 8

### Case 2

- Input shape: 19x17
- Output shape: 5x3
- Input colors: 1 2 3 8
- Output colors: 2 8

### Case 3

- Input shape: 15x17
- Output shape: 3x5
- Input colors: 1 2 3 8
- Output colors: 1 3
