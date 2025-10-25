from .nodes import Create360SweepVideoFramesNode, Create360SweepVideoFramesTensorNode

NODE_CLASS_MAPPINGS = {
    "Create360SweepVideoFramesNode": Create360SweepVideoFramesNode,
    "Create360SweepVideoFramesTensorNode": Create360SweepVideoFramesTensorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Create 360 Sweep Frames": "Create360SweepVideoFramesNode",
    "Create 360 Sweep Frames Tensor": "Create360SweepVideoFramesTensorNode",
}
