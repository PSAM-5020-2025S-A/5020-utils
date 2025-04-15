import torch

def get_num_params(m):
  psum = sum(p.numel() for p in m.parameters() if p.requires_grad)
  return psum

def get_labels(model, inputs):
  model.eval()
  with torch.no_grad():
    y_pred = model(inputs).argmax(dim=1)
    return [l.item() for l in y_pred]
