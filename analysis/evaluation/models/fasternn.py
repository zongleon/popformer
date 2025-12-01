"""
This might one day contain code reproducing FASTER-NN.

It does! Mostly correct, hopefully.
"""

from math import floor
from ..core import BaseModel
from tqdm import tqdm
import torch

from popformer.collators import RawMatrixCollator

class SweepNet1DDet(torch.nn.Module):
    """I believe this is the FASTER-NN architecture.
    https://github.com/SjoerdvandenBelt/FASTER-NN/blob/main/sources/pytorch-sources/Logic/models.py
    """
    def __init__(self, H, W, outputs, channels):
        super(SweepNet1DDet, self).__init__()
        
        conv1_kernel = 3
        conv2_kernel = 3
        conv3_kernel = 3
        conv4_kernel = 6
        conv5_kernel = 6
        conv6_kernel = 6
        
        conv1_stride = 1
        conv2_stride = 1
        conv3_stride = 1
        conv4_stride = 2
        conv5_stride = 2
        conv6_stride = 2
        
        conv1_channels = 32
        conv2_channels = 32
        conv3_channels = 32
        conv4_channels = 32
        conv5_channels = 32
        conv6_channels = 32
        
        pool1_kernel = 2
        pool2_kernel = 2
        pool3_kernel = 2
        
        pool1_stride = 2
        pool2_stride = 2
        pool3_stride = 2
                
        self.outputs = outputs
        self.channels = channels
        self.conv1 = torch.nn.Conv1d(channels, conv1_channels, kernel_size=conv1_kernel, stride=conv1_stride, padding=0)
        self.pool1 = torch.nn.MaxPool1d(pool1_kernel, pool1_stride)
        self.relu1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv1d(conv1_channels, conv2_channels, kernel_size=conv2_kernel, stride=conv2_stride, padding=0)
        self.pool2 = torch.nn.MaxPool1d(pool2_kernel, pool2_stride)
        self.relu2 = torch.nn.ReLU()
        
        self.conv3 = torch.nn.Conv1d(conv2_channels, conv3_channels, kernel_size=conv3_kernel, stride=conv3_stride, padding=0)
        self.pool3 = torch.nn.MaxPool1d(pool3_kernel, pool3_stride)
        self.relu3 = torch.nn.ReLU()
        
        self.conv4 = torch.nn.Conv1d(conv3_channels, conv4_channels, kernel_size=conv4_kernel, stride=conv4_stride, padding=0) # padding was 3
        self.relu4 = torch.nn.ReLU()
        
        self.conv5 = torch.nn.Conv1d(conv4_channels, conv5_channels, kernel_size=conv5_kernel, stride=conv5_stride, padding=0) # padding was 3
        self.relu5 = torch.nn.ReLU()
        
        self.conv6 = torch.nn.Conv1d(conv5_channels, conv6_channels, kernel_size=conv6_kernel, stride=conv6_stride, padding=0) # padding was 3
        self.relu6 = torch.nn.ReLU()
        
        
        # compute output size to FC
        out_conv1 = self.compute_out_shape((H, W), (conv1_kernel, conv1_kernel), (conv1_stride, conv1_stride))
        out_pool1 = self.compute_out_shape(out_conv1, (pool1_kernel, pool1_kernel), (pool1_stride, pool1_stride))
        out_conv2 = self.compute_out_shape(out_pool1, (conv2_kernel, conv2_kernel), (conv2_stride, conv2_stride))
        out_pool2 = self.compute_out_shape(out_conv2, (pool2_kernel, pool2_kernel), (pool2_stride, pool2_stride))
        out_conv3 = self.compute_out_shape(out_pool2, (conv3_kernel, conv3_kernel), (conv3_stride, conv3_stride))
        out_pool3 = self.compute_out_shape(out_conv3, (pool3_kernel, pool3_kernel), (pool3_stride, pool3_stride))
        out_conv4 = self.compute_out_shape((out_pool3[0], out_pool3[1]), (conv4_kernel, conv4_kernel), (conv4_stride, conv4_stride))
        out_conv5 = self.compute_out_shape((out_conv4[0], out_conv4[1]), (conv5_kernel, conv5_kernel), (conv5_stride, conv5_stride))
        out_conv6 = self.compute_out_shape((out_conv5[0], out_conv5[1]), (conv5_kernel, conv5_kernel), (conv5_stride, conv5_stride))
        
        # init output layer
        self.fc = torch.nn.Linear(out_conv6[1]*32, 2*self.outputs)
        
        
    def compute_out_shape(self, I, k, s):
        out_H = floor(((I[0] - (k[0] - 1) - 1) / s[0]) + 1)
        out_W = floor(((I[1] - (k[1] - 1) - 1) / s[1]) + 1)
        return (out_H, out_W)
        
    def forward(self, x):
        # x = torch.mean(x, dim=2)
        
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.relu6(x)

        x = x.flatten(1)
        x = self.fc(x)
        x = x.reshape(-1, self.outputs, 2)
        x = x.squeeze(1)

        return x

class FasterNNModel(BaseModel):
    """Re-implementation of Faster-NN model for evaluation.
    From https://github.com/SjoerdvandenBelt/FASTER-NN/tree/main/sources/pytorch-sources

    We've retrained Faster-NN on our training data to ensure fair comparisons.
    """

    def __init__(
        self, model_path: str, model_name: str, device: torch.device | None = None, from_init: bool = False
    ):
        if not model_path.endswith(".pt"):  
            model_path += ".pt"
        self.model_path = model_path
        self.model_name = model_name
        # 2 channels = -x ("use base pair distance")
        self.width = 512

        self.model = SweepNet1DDet(H=128, W=self.width, outputs=1, channels=2)
        if not from_init:
            self.model.load_state_dict(torch.load(self.model_path))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

        self.collator = RawMatrixCollator()

    def preprocess(self, batch):
        """Preprocess the batch data, by first collating and then computing
        1. Vector of minor allele frequencies
        2. Vector of distances between segregating sites
        These come together in shape (batch_size, 2, num_snps)
        """
        batch = self.collator(batch)

        inputs = []

        # batch contains input_ids, distances
        for mat, dist in zip(batch["input_ids"], batch["distances"]):
            # compute MAFs
            freqs = torch.mean(mat == 1, dim=0, dtype=torch.float16)
            mafs = torch.minimum(freqs, 1 - freqs)

            combined = torch.stack([mafs, dist.float()], dim=0)
            
            # pad with 0s to self.width
            if combined.shape[1] < self.width:
                pad_width = self.width - combined.shape[1]
                padding = torch.zeros((2, pad_width), dtype=combined.dtype)
                combined = torch.cat([combined, padding], dim=1)
            else:
                combined = combined[:, :self.width]

            inputs.append(combined.unsqueeze(0))  # add batch dim

        batch["inputs"] = torch.cat(inputs, dim=0) # shape (batch_size, 2, num_snps)
        batch["labels"] = torch.tensor([ex for ex in batch["labels"]])
            
        return batch


    def train(self, train_loader, val_loader, epochs: int = 10, lr: float = 1e-4, patience: int = 5):
        """Train the model with early stopping based on val_loss."""
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)

        best_val_loss = float("inf")
        best_state_dict = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            for batch in tqdm(train_loader):
                inputs = batch["inputs"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
                
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / (len(train_loader.dataset))

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            for batch in tqdm(val_loader):
                inputs = batch["inputs"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(inputs)
                    val_loss += loss_fn(outputs, labels).item()
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / (len(val_loader.dataset))

            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # save best model
        if best_state_dict is not None:
            torch.save(best_state_dict, self.model_path)
        else:
            torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def run(self, batch):
        """Make predictions on the given batch of data."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)

        output = self.model(batch["inputs"])

        preds = torch.softmax(output, dim=1)

        return preds.cpu().numpy()

