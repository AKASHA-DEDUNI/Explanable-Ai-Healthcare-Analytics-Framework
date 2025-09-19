# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        # For ResNet-50, default target_layer is the last conv in layer4
        if target_layer is None:
            self.target_layer = self.model.cnn.layer4[-1].conv3
        else:
            self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_image, metadata, text_emb, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_image, metadata, text_emb)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]     # (C, H, W)
        activations = self.activations[0] # (C, H, W)

        weights = gradients.mean(dim=(1, 2))  # Global Avg Pooling

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()
        return cam.cpu().numpy()

    def close(self):
        for handle in self.hook_handles:
            handle.remove()


def apply_heatmap(img_pil, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img_pil.width, img_pil.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed_img)



