from model import *
import numpy as np

model = SiteHapModel(input_dim=2, hidden_dim=64, n_heads=4).to("cuda")
head = SampleLevelHead(hidden_dim=64, output_dim=1).to("cuda")
model.load_state_dict(torch.load("sitehap_pretrain.pth"))
head.load_state_dict(torch.load("sitehap_head_finetune.pth"))
model.eval()
head.eval()

x = np.load("../disc-interpret/dataset-CEU/X.npy")
x[:, :, :, 0][x[:, :, :, 0] == -1] = 0  # replace -1 with 0
y = np.load("../disc-interpret/dataset-CEU/y.npy")

embed = model(torch.tensor(x[:1], dtype=torch.float32).to("cuda")).detach().cpu().numpy()
print(embed[0])

# import torch
# import matplotlib.pyplot as plt

# def predict_logits(x_batch):
#     with torch.no_grad():
#         x_tensor = torch.tensor(x_batch, dtype=torch.float32).to("cuda")
#         feats = model(x_tensor)
#         logits = head(feats).squeeze(-1).cpu().numpy()
#     return logits

# def predict_logits_batched(x_data, batch_size=256):
#     logits = []
#     for i in range(0, len(x_data), batch_size):
#         x_batch = x_data[i:i+batch_size]
#         logits_batch = predict_logits(x_batch)
#         logits.append(logits_batch)
#     return np.concatenate(logits)

# # First 10000 samples (batched)
# x_first = x[:10000]
# logits_first = predict_logits_batched(x_first)

# # Last 10000 samples (batched)
# x_last = x[-10000:]
# logits_last = predict_logits_batched(x_last)

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.hist(logits_first, bins=50, alpha=0.7)
# plt.title("Logits Distribution (First 10,000)")
# plt.subplot(1,2,2)
# plt.hist(logits_last, bins=50, alpha=0.7)
# plt.title("Logits Distribution (Last 10,000)")
# plt.tight_layout()
# plt.savefig("logits_distribution.pdf")
