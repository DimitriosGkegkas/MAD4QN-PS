from torchviz import make_dot
import torch

from dqn.dueling_net import DuelingDQNetwork

# Example: Visualize a random tensor through your model
model = DuelingDQNetwork(0.0010, 2, 'test', (9, 48, 48), 'tmp/dqn')
dummy_input = torch.randn(1, 9, 48, 48)
model.train()
output = model(dummy_input)

# Generate visualization
dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
dot.render("model_architecture", format="png")  # Save as an image
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11, training=torch.onnx.TrainingMode.TRAINING)  # Save as an ONNX file,
