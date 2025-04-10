from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolo_base import YOLOBase
from torch.ao.quantization import QuantStub, DeQuantStub

class Network(YOLOBase):

    def __init__(self,
                    in_channels = 4,
                    num_classes: int = 2,
                    anchors: List[List[Tuple[float, float]]] = [
                        [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
                    ],
                    threshold: float = 0.1,
                    tau_grad: float = 0.1,
                    scale_grad: float = 0.1,
                    clamp_max: float = 5.0) -> None:
        
        super().__init__(num_classes=num_classes,
                            anchors=anchors,
                            clamp_max=clamp_max)
        
        neuron_stride_params = {
                'v_threshold'     : 1.0,
                'current_decay' : 1.0,
                'voltage_decay' : 0.03,
                'requires_grad' : False,
        }

        neuron_pool_params = {
                'v_threshold'     : 1.0,
                'current_decay' : 1.0,
                'voltage_decay' : 1.0,
                'requires_grad' : False,
        }

        self.backend_blocks = torch.nn.ModuleList([
            
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            #nn.Dropout(0.2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            #nn.Dropout(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        
        ])

        self.head1_backend = torch.nn.ModuleList([

            nn.Conv2d(64,64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(0.2),

            nn.Conv2d(64,128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(0.2),
            
        ])

        self.head1_blocks = torch.nn.ModuleList([

            nn.Conv2d(128,128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.Dropout(0.2),

            nn.Conv2d(128,128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,self.num_output, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(self.num_output),
            nn.ReLU(),

        ])

        """self.backend_blocks[1].synapse.weight.requires_grad = False #pool
        self.backend_blocks[6].synapse.weight.requires_grad = False #pool
        self.head2_backend[2].synapse.weight.requires_grad = False #unpool
        self.head2_backend[3].synapse.weight.requires_grad = False"""
    
    def forward(self,
        input,sparsity_montior = None
    ):

        count = []
        """for block in self.input_blocks:
            input = block(input)
            count.append(slayer.utils.event_rate(input))
            
        backend = input"""

        backend = input
        for block in self.backend_blocks:
            backend = block(backend)
        
        h1_backend = backend
        for block in self.head1_backend:
            h1_backend = block(h1_backend)
        
        head1 = h1_backend
        for block in self.head1_blocks:
            head1 = block(head1)
            """if has_sparisty_loss and isinstance(block,
                                                slayer.block.sigma_delta.Conv):
                sparsity_monitor.append(head1)"""

            """if has_sparisty_loss and isinstance(block,
                                                slayer.block.sigma_delta.Conv):
                sparsity_monitor.append(head2)"""

        ## Get the summation or averaging over all the time steps, and preserve the last dimension of head1 tensor as time but Time = 1

        head1 = self.yolo_raw(head1)
        
        if not self.training:

            """head1 = torch.sum(head1,dim=-1).unsqueeze(-1)
            head2 = torch.sum(head2,dim=-1).unsqueeze(-1)
            print("head1 shape ",head1.shape)
            print("head2 shaoe ",head2.shape)
            output = torch.concat([self.yolo(head1, self.anchors[0]),self.yolo(head2, self.anchors[1])], dim=1)"""

            output = [head1]

        else:
            output = [head1]

        return (output,torch.FloatTensor(count).reshape((1, -1)).to(input.device))
    
    def initialize_model_weights(self) -> None:

        for i in range(len(self.backend_blocks)):
            if isinstance(self.backend_blocks[i],nn.Conv2d):
                torch.nn.init.kaiming_normal_(self.backend_blocks[i].weight)
                if self.backend_blocks[i].bias is not None:
                    torch.nn.init.constant_(self.backend_blocks[i].bias, 0)
            """if isinstance(self.backend_blocks[i],nn.MaxPool2d):
                torch.nn.init.constant_(self.backend_blocks[i].weight,1.)"""

        
        for i in range(len(self.head1_backend)):
            if isinstance(self.head1_backend[i],nn.Conv2d):
                torch.nn.init.kaiming_normal_(self.head1_backend[i].weight)
                if self.head1_backend[i].bias is not None:
                    torch.nn.init.constant_(self.head1_backend[i].bias, 0)

        for i in range(len(self.head1_blocks)):
            if isinstance(self.head1_blocks[i],nn.Conv2d):
                torch.nn.init.kaiming_normal_(self.head1_blocks[i].weight)

    def fuse_model(self,quant_aware_train = False):
        # Specify which layers to fuse
        torch.quantization.fuse_modules(self.backend_blocks, ['0', '1', '2'], inplace=True)  # Conv + BN + ReLU
        torch.quantization.fuse_modules(self.backend_blocks, ['3', '4', '5'], inplace=True)  # Conv + BN + ReLU
        # Pooling layers do not need to be fused
        torch.quantization.fuse_modules(self.backend_blocks, ['7', '8', '9'], inplace=True)  # Conv + BN + ReLU
        torch.quantization.fuse_modules(self.backend_blocks, ['10', '11', '12'], inplace=True)  # Conv + BN + ReLU
        torch.quantization.fuse_modules(self.backend_blocks, ['14', '15', '16'], inplace=True)  # Conv + BN + ReLU

        torch.quantization.fuse_modules(self.head1_backend, ['0', '1', '2'], inplace=True)  # Conv + BN + ReLU
        torch.quantization.fuse_modules(self.head1_backend, ['3', '4', '5'], inplace=True)  # Conv + BN + ReLU

        torch.quantization.fuse_modules(self.head1_blocks, ['0', '1', '2'], inplace=True)  # Conv + BN + ReLU
        torch.quantization.fuse_modules(self.head1_blocks, ['3', '4', '5'], inplace=True)  # Conv + BN + ReLU