import torch
from nets import get_model
from torch.export import export
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
import os

WEIGHTS_PATH = "export/weights/best_model.pth"
OUTPUT_DIR = "export/quantized"
PTE_MODEL_NAME = "rpsmodel.pte"
COMPATIBLE_INPUT_SIZE = 48

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, PTE_MODEL_NAME)

model = get_model("CNN", input_size=COMPATIBLE_INPUT_SIZE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

example_input = (torch.randn(1, 1, COMPATIBLE_INPUT_SIZE, COMPATIBLE_INPUT_SIZE),)
dynamic_shapes_spec = None
exported_program = export(model, example_input, dynamic_shapes=dynamic_shapes_spec)
edge_program = to_edge_transform_and_lower(exported_program, partitioner=[XnnpackPartitioner()])
executorch_program = edge_program.to_executorch()

with open(output_path, "wb") as f:
    f.write(executorch_program.buffer)
