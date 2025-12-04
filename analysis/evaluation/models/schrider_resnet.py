"""
This file contains code to replicate the Schrider Tree Sequences
ResNet model for selection detection.

Mostly copied from 
https://github.com/SchriderLab/TreeSeqPopGenInference/blob/main/src/models/torchvision_mod_layers.py
"""

from ..core import BaseModel
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional, Callable
from scipy.spatial.distance import pdist, squareform
from seriate import seriate

from popformer.collators import RawMatrixCollator


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: BasicBlock,
        layers: List[int],
        in_channels: int = 3,
        num_classes: int = 512,
        zero_init_residual: bool = False,
        groups: int = 1,
        final_dim: int = 512,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            final_dim,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(final_dim * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class SchriderResnet(BaseModel):
    """Re-implementation of Schrider's resnet model for evaluation.
    """

    def __init__(
        self, model_path: str, model_name: str, device: torch.device | None = None, from_init: bool = False
    ):
        if not model_path.endswith(".pt"):
            model_path += ".pt"
        self.model_path = model_path
        self.model_name = model_name
        # 1 population, 2 classes (neutral, selection)
        # Resnet34 layers according to Schrider's train_cnn.py
        self.width = 512
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=1, num_classes=2, final_dim=512)
        # self.model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels=1, num_classes=2, final_dim=512)

        if not from_init:
            self.model.load_state_dict(torch.load(self.model_path))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)

        self.collator = RawMatrixCollator()

    def preprocess(self, batch):
        """Preprocess the batch data, by first collating and then computing
        the row-sorted haplotype matrix.
        """
        batch = self.collator(batch)

        inputs = []

        # batch contains input_ids, distances
        for mat in batch["input_ids"]:
            if mat.shape[1] > self.width:
                ii = torch.randint(0, mat.shape[1] - self.width, (1,), device=mat.device).item()
                mat = mat[:, ii:ii + self.width]
            else:
                to_pad = self.width - mat.shape[1]
                if to_pad % 2 == 0:
                    left = right = to_pad // 2
                else:
                    left = to_pad // 2 + 1
                    right = to_pad // 2
                mat = torch.nn.functional.pad(mat, (left, right))
                    
            # D = squareform(pdist(mat, metric = 'cosine'))
            # # print(D)
            # D[np.isnan(D)] = 0.
            
            # ii = seriate(D, timeout = 0.)
            
            # # print(ii)
            # mat = mat[ii,:]

            # sort by computing hamming distances, picking the haplotype with
            # the smallest total distance to all other haplotypes, then iteratively
            # picking the next closest haplotype
            n_haps = mat.shape[0]
            D = torch.cdist(mat.float(), mat.float(), p=0)  # shape (n_haps, n_haps)
            D /= mat.shape[1]  # normalize
            D_np = D.cpu().numpy()
            selected = []
            unselected = set(range(n_haps))
            # pick first haplotype with smallest total distance
            first_hap = np.argmin(D_np.sum(axis=1))
            selected.append(first_hap)
            unselected.remove(first_hap)
            while unselected:
                last_hap = selected[-1]
                unselected_list = list(unselected)
                next_hap = unselected_list[np.argmin(D_np[last_hap, unselected_list])]
                selected.append(next_hap)
                unselected.remove(next_hap)

            inputs.append(mat[selected, :].unsqueeze(0).unsqueeze(0))  # add batch + channel dim

        batch["inputs"] = torch.cat(inputs, dim=0).to(torch.float32)  # shape (batch_size, 2, num_snps)
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
                f"Epoch {epoch + 1}/{epochs}, "
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

        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)

        output = self.model(batch["inputs"])

        preds = torch.softmax(output, dim=1)

        return preds.cpu().numpy()
