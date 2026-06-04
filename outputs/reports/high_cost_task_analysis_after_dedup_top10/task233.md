# task233 High-Cost Diagnosis

- Current estimated cost: 661410
- Current file size: 924434 bytes
- Input shapes: 24x19;20x10;16x12
- Output shapes: 17x9;8x8;9x9
- Same shape: False
- Shape relation: shrinks_or_crop
- Colors: 0 1 2 3 4 5 8
- Likely rule families: fixed_crop|dynamic_bbox_crop|frame_or_substructure_extract
- Recommended action: try crop, dynamic bbox, frame interior, and substructure extraction builders
- Expected cost after replacement: 5000

## Train Cases

### Case 0

- Input shape: 24x19
- Output shape: 17x9
- Input colors: 0 1 2 3 4 5 8
- Output colors: 1 2 3 4 5 8

### Case 1

- Input shape: 20x10
- Output shape: 8x8
- Input colors: 0 2 3 4
- Output colors: 2 3 4

### Case 2

- Input shape: 16x12
- Output shape: 9x9
- Input colors: 0 2 8
- Output colors: 2 8
