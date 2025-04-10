# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

from typing import List, Tuple, Union

"""import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .yolo_base import YOLOBase

#from spikingjelly.activation_based import neuron, surrogate,functional, encoding, layer

class Network(YOLOBase):

    def __init__(self,
                    in_channels = 4,
                    num_classes: int = 2,
                    anchors: List[List[Tuple[float, float]]] = [
                        [(0.28, 0.22), (0.38, 0.48), (0.90, 0.78)],
                        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                    ],
                    threshold: float = 0.1,
                    tau_grad: float = 0.1,
                    scale_grad: float = 0.1,
                    clamp_max: float = 5.0) -> None:
        super().__init__(num_classes=num_classes,
                            anchors=anchors,
                            clamp_max=clamp_max)
        
        """sigma_params = {  # sigma-delta neuron parameters
            'threshold'     : threshold,   # delta unit threshold
            'tau_grad'      : tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,       # trainable threshold
            'shared_param'  : True,        # layer wise threshold
        }"""

        """neuron_params = {
                'threshold'     : 1.0,
                'current_decay' : 0.2,
                'voltage_decay' : 0.03,
                'tau_grad'      : 0.03,
                'scale_grad'    : 3,
                'requires_grad' : False,
            }"""
        """neuron_params_drop = {
                 **neuron_params,
                 'dropout' : slayer.neuron.Dropout(p=0.05),
        }


        sdnn_params = {
            **neuron_params,
        }"""

        neuron_args = {'surrogate_function' : surrogate.ATan(), 'decay_input' : False, 'detach_reset' : True,'v_threshold' : 1.0}
    
        self.backend_blocks = torch.nn.ModuleList([
            
            #layer.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=False), #We expect this to behave as the normalizer

            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=2, bias=False), #2
            nn.BatchNorm2d(16),
            #nn.AvgPool2d(2,2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(16,32, kernel_size=3, padding=1,stride=2, bias=False), #2
            nn.BatchNorm2d(32),
            #nn.AvgPool2d(2,2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(32,64, kernel_size=3, padding=1,stride=2, bias=False),
            nn.BatchNorm2d(64),
            #nn.AvgPool2d(2,2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(64,128, kernel_size=3, padding=1,stride=2, bias=False),
            nn.BatchNorm2d(128),
            #nn.AvgPool2d(2,2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(128,256, kernel_size=3, padding=1,stride=1, bias=False),
            nn.BatchNorm2d(256),
            #nn.AvgPool2d(2,2),
            nn.ReLU(),
            nn.Dropout(0.2)

        ])

        self.head1_backend = torch.nn.ModuleList([
            
            nn.Conv2d(256,256, kernel_size=3, padding=1,stride=2, bias=False),
            nn.BatchNorm2d(256),
            #nn.AvgPool2d(2,2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(256,512, kernel_size=3, padding=1,stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(512,1024, kernel_size=3, padding=1,stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(1024,256, kernel_size=1, padding=0,stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        ])
        
        self.head1_blocks = torch.nn.ModuleList([

            nn.Conv2d(256,512, kernel_size=3, padding=1,stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv2d(512,self.num_output, kernel_size=1, padding=0,stride=1, bias=False),
            nn.BatchNorm2d(self.num_output),
            nn.ReLU(),
            
        ])

        self.head2_backend = torch.nn.ModuleList([
            
            nn.Conv2d(256,128, kernel_size=1, padding=0,stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        ])

        self.head2_blocks = torch.nn.ModuleList([

            nn.Conv2d(384,256, kernel_size=3, padding=1,stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Conv2d(256,self.num_output, kernel_size=1, padding=0,stride=1, bias=False),
            nn.BatchNorm2d(self.num_output),
            nn.ReLU(),
        ])
    
    def forward(self,
        input,
        sparsity_monitor = None
    ) -> Tuple[Union[torch.tensor, List[torch.tensor]], torch.tensor]:
         
        """Forward computation step of the network module.

        Parameters
        ----------
        input : torch.tensor
            Input frames tensor.
        sparsity_monitor : slayer.loss.SparsityEnforcer, optional
            Sparsity monitor module. If None, sparisty is not enforced.
            By default None.

        Returns
        -------
        Union[torch.tensor, List[torch.tensor]]
            Output of the network.

            * If the network is in training mode, the output is a list of
            raw output tensors of the different heads of the network.
            * If the network is in testing mode, the output is the consolidated
            prediction bounding boxes tensor.

            Note: the difference in the output behavior is done to apply
            loss to the raw tensor for better training stability.
        torch.tensor
            Event rate statistics.
        """

        has_sparisty_loss = sparsity_monitor is not None

        count = []
        """for block in self.input_blocks:
            input = block(input)
            count.append(slayer.utils.event_rate(input))
            
        backend = input"""
        backend = input
        for block in self.backend_blocks:
            backend = block(backend)
            if has_sparisty_loss:
                sparsity_monitor.append(backend)
        
        h1_backend = backend
        for block in self.head1_backend:
            h1_backend = block(h1_backend)
            if has_sparisty_loss:
                sparsity_monitor.append(h1_backend)
        
        head1 = h1_backend
        for block in self.head1_blocks:
            head1 = block(head1)
            if has_sparisty_loss and isinstance(block,
                                                slayer.block.sigma_delta.Conv):
                sparsity_monitor.append(head1)

        h2_backend = h1_backend
        for block in self.head2_backend:
            h2_backend = block(h2_backend)
            if has_sparisty_loss:
                sparsity_monitor.append(h2_backend)

        print("backend shape ",backend.shape)
        print("h2_backend ",h2_backend.shape)
        
        head2 = torch.concat([h2_backend, backend], dim=1)
        for block in self.head2_blocks:
            head2 = block(head2)
            if has_sparisty_loss and isinstance(block,
                                                slayer.block.sigma_delta.Conv):
                sparsity_monitor.append(head2)

        ## Get the summation or averaging over all the time steps, and preserve the last dimension of head1 tensor as time but Time = 1
        head1 = self.yolo_raw(head1)
        head2 = self.yolo_raw(head2)
        
        if not self.training:

            """head1 = torch.sum(head1,dim=-1).unsqueeze(-1)
            head2 = torch.sum(head2,dim=-1).unsqueeze(-1)
            print("head1 shape ",head1.shape)
            print("head2 shaoe ",head2.shape)
            output = torch.concat([self.yolo(head1, self.anchors[0]),self.yolo(head2, self.anchors[1])], dim=1)"""

            output = [head1, head2]

        else:
            output = [head1, head2]

        return (output,
                torch.FloatTensor(count).reshape((1, -1)).to(input.device))

    def grad_flow(self, path: str) -> List[torch.tensor]:
        """Montiors gradient flow along the layers.

        Parameters
        ----------
        path : str
            Path for output plot export.

        Returns
        -------
        List[torch.tensor]
            List of gradient norm per layer.
        """
        # helps monitor the gradient flow
        def block_grad_norm(blocks):
            return [b.synapse.grad_norm
                    for b in blocks if hasattr(b, 'synapse')
                    and b.synapse.weight.requires_grad]

        grad = block_grad_norm(self.backend_blocks)
        grad += block_grad_norm(self.head1_backend)
        grad += block_grad_norm(self.head2_backend)
        grad += block_grad_norm(self.head1_blocks)
        grad += block_grad_norm(self.head2_blocks)

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad  


    def initialize_model_weights(self) -> None:
        """Selectively loads the model from save pytorch state dictionary.
        If the number of output layer does not match, it will ignore the last
        layer in the head. Other states should match, if not, there might be
        some mismatch with the model file.

        Parameters
        ----------
        model_file : str
            Path to pytorch model file.
        """
  
        for i in range(len(self.backend_blocks)):
            if isinstance(self.backend_blocks[i],nn.Conv2d):
                torch.nn.init.kaiming_normal_(self.backend_blocks[i].weight)
                if self.backend_blocks[i].bias is not None:
                    torch.nn.init.constant_(self.backend_blocks[i].bias, 0)
        
        for i in range(len(self.head1_backend)):
            if isinstance(self.head1_backend[i],nn.Conv2d):
                torch.nn.init.kaiming_normal_(self.head1_backend[i].weight)
                if self.head1_backend[i].bias is not None:
                    torch.nn.init.constant_(self.head1_backend[i].bias, 0)

        for i in range(len(self.head2_backend)):
            if isinstance(self.head2_backend[i],nn.Conv2d):
                torch.nn.init.kaiming_normal_(self.head2_backend[i].weight)
                if self.head2_backend[i].bias is not None:
                    torch.nn.init.constant_(self.head2_backend[i].bias, 0)


        for i in range(len(self.head1_blocks)):
            if isinstance(self.head1_blocks[i],nn.Conv2d):
                print("444444444444444444 ")
                torch.nn.init.kaiming_normal_(self.head1_blocks[i].weight)

        for i in range(len(self.head2_blocks)):
            if isinstance(self.head2_blocks[i],nn.Conv2d):
                print("5555555555555555 ")
                torch.nn.init.kaiming_normal_(self.head2_blocks[i].weight)
                

    def load_model(self, model_file: str) -> None:
        """Selectively loads the model from save pytorch state dictionary.
        If the number of output layer does not match, it will ignore the last
        layer in the head. Other states should match, if not, there might be
        some mismatch with the model file.

        Parameters
        ----------
        model_file : str
            Path to pytorch model file.
        """
        return












