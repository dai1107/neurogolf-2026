# task396 High-Cost Diagnosis

- Current estimated cost: 115080
- Current file size: 148123 bytes
- Input shapes: 13x13;13x16;15x16
- Output shapes: 7x7;6x7;7x7
- Same shape: False
- Shape relation: shrinks_or_crop
- Colors: 0 1 2 3 4
- Likely rule families: fixed_crop|dynamic_bbox_crop|frame_or_substructure_extract
- Recommended action: try crop, dynamic bbox, frame interior, and substructure extraction builders
- Expected cost after replacement: 5000

## Train Cases

### Case 0

- Input shape: 13x13
- Output shape: 7x7
- Input colors: 0 2 4
- Output colors: 0 4

### Case 1

- Input shape: 13x16
- Output shape: 6x7
- Input colors: 0 1 3
- Output colors: 0 3

### Case 2

- Input shape: 15x16
- Output shape: 7x7
- Input colors: 0 2 3
- Output colors: 0 2
