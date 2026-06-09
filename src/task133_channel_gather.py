"""Rewrite task133 anchor_vector_bank: one-hot table -> Gather along axis.

anchor_vector_bank [8100,1,10,1,1] float32 stores one-hot color vectors.
Each row encodes a single color (0-9). Consumed by:
  Gather(bank, best_index) -> Reshape -> Mul(input, vector) -> ReduceSum

Since the vector is one-hot at position c, Mul(input, vector).sum(axis=1) = input[:,c,:,:].
We replace with:
  Gather(idx_table[8100] int64, best_index) -> scalar c
  Gather(input, [c], axis=1) -> [1,1,H,W]

Deletes: 1 bank (324KB), 1 Gather, 1 Reshape, 1 Mul, 1 ReduceSum
Adds:    1 idx table (32KB int64), 1 Gather, 1 Gather(axis=1)
Net:     -2 nodes, -292KB
"""

from __future__ import annotations

import argparse, csv, json, shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .cost_estimator import estimate_model_cost

FIELDS = ["task_id","source_model_path","output_model_path","source_cost",
          "output_cost","cost_delta","source_file_size_bytes","output_file_size_bytes",
          "file_size_delta","failure_reason"]


def _color_index_from_onehot(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(arr.shape[0], -1)
    return flat.argmax(axis=1).astype(np.int64)


def rewrite_task133(source_model: str, output_model: str) -> dict[str, Any]:
    src = Path(source_model); dst = Path(output_model)
    dst.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(src))
    onnx.checker.check_model(model)
    src_cost = estimate_model_cost(str(src))

    # Extract color index from anchor_vector_bank
    bank_arr = None
    for init in model.graph.initializer:
        if init.name == "anchor_vector_bank":
            bank_arr = numpy_helper.to_array(init)
            break
    if bank_arr is None:
        shutil.copyfile(src, dst)
        return {"task_id":"task133","source_cost":int(src_cost["estimated_cost"]),
                "output_cost":int(src_cost["estimated_cost"]),"cost_delta":0,
                "failure_reason":"anchor_vector_bank not found"}

    color_idx = _color_index_from_onehot(bank_arr)

    # Find the Mul's data input (not from the vector)
    mul_data = None
    for node in model.graph.node:
        if node.name == "anchor_color_input_out":
            for inp in node.input:
                if inp != "anchor_vector":
                    mul_data = inp
            break
    if mul_data is None:
        shutil.copyfile(src, dst)
        return {"task_id":"task133","source_cost":int(src_cost["estimated_cost"]),
                "output_cost":int(src_cost["estimated_cost"]),"cost_delta":0,
                "failure_reason":"Mul not found"}

    # Delete old nodes: anchor_vector_gather, anchor_vector, anchor_color_input_out, anchor_color_cells_out
    remove_nodes = {"anchor_vector_gather", "anchor_vector", "anchor_color_input_out", "anchor_color_cells_out"}
    remove_inits = {"anchor_vector_bank"}

    # Build new index table
    idx_table = numpy_helper.from_array(color_idx, name="anchor_color_idx")

    # New: Gather(idx_table, best_index) -> scalar color
    gather_idx = helper.make_node("Gather", ["anchor_color_idx", "best_index"],
                                  ["anchor_color_idx_out"], name="anchor_color_idx_gather")

    # New: Gather(input, scalar, axis=1) -> [1,1,H,W] replaces Mul+ReduceSum
    gather_ch = helper.make_node("Gather", [mul_data, "anchor_color_idx_out"],
                                 ["anchor_color_cells_out"], name="anchor_channel_gather", axis=1)

    # Build new node list
    old_nodes = list(model.graph.node)
    new_nodes = []
    # Find insertion point: after 'input' is produced (it's the model input, so position 0)
    # but before any consumer of anchor_color_cells_out
    insert_pos = 0
    for i, node in enumerate(old_nodes):
        # Find where best_index is produced — insert after it
        if "best_index" in node.output:
            insert_pos = i + 1

    for node in old_nodes:
        if node.name in remove_nodes:
            continue
        new_nodes.append(node)

    # Insert new nodes at the right position
    # Find the earliest consumer of the new output (anchor_color_cells_out)
    # and insert before it
    first_consumer = len(new_nodes)
    for i, node in enumerate(new_nodes):
        if "anchor_color_cells_out" in node.input:
            first_consumer = min(first_consumer, i)

    if first_consumer < len(new_nodes):
        new_nodes.insert(first_consumer, gather_idx)
        new_nodes.insert(first_consumer + 1, gather_ch)
    else:
        new_nodes.insert(insert_pos, gather_idx)
        new_nodes.insert(insert_pos + 1, gather_ch)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # Update initializers
    kept_inits = [i for i in model.graph.initializer if i.name not in remove_inits]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)
    model.graph.initializer.append(idx_table)

    # Keep existing value_info; only remove entries whose producer node was deleted
    stale_outputs = {"anchor_vector_gather", "anchor_vector", "anchor_color_input_out"}
    kept_vi = [vi for vi in model.graph.value_info if vi.name not in stale_outputs]
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()
    model.graph.value_info.extend(kept_vi)

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))

    out_cost = estimate_model_cost(str(dst))
    return {"task_id":"task133","source_model_path":str(src),"output_model_path":str(dst),
            "source_cost":int(src_cost["estimated_cost"]),"output_cost":int(out_cost["estimated_cost"]),
            "cost_delta":int(out_cost["estimated_cost"]-src_cost["estimated_cost"]),
            "source_file_size_bytes":int(src_cost["file_size_bytes"]),
            "output_file_size_bytes":int(out_cost["file_size_bytes"]),
            "file_size_delta":int(out_cost["file_size_bytes"]-src_cost["file_size_bytes"]),
            "failure_reason":""}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="outputs/onnx/task133.onnx")
    p.add_argument("--output", default="outputs/candidates/task133_channel_gather/task133_ChannelGather.onnx")
    p.add_argument("--report", default="outputs/reports/task133_channel_gather.csv")
    a = p.parse_args()
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    r = rewrite_task133(a.source, a.output)
    with open(a.report,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows([r])
    print(json.dumps(r, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
