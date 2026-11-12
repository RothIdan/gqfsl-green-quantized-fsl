import torch
import torch.nn as nn
import torchvision
from thop import profile
import pprint

# ==========================================
#       Models for FSL Profiling
# ==========================================

class BaseCNN_Split(nn.Module):
    """
    The 8-Layer CNN ('gfl'), restructured for the profiler (MNIST), that was considered in: 
    M. Kim, W. Saad, M. Mozaffari, and M. Debbah, “Green, quantized federated learning over wireless networks: An energy-efficient design.”
    """
    def __init__(self, in_features=1, num_classes=10):
        super().__init__()

        # Blocks map exactly to your 8 split points
        self.block1 = nn.Sequential(nn.Conv2d(in_features, 128, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(3, 1))
        self.block2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 0), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 0), nn.BatchNorm2d(64), nn.ReLU())
        self.block4 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 0), nn.BatchNorm2d(32), nn.ReLU())
        self.block5 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 0), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3, 2))
        self.block6 = nn.Sequential(nn.Flatten(), nn.Linear(32, 2000), nn.ReLU())
        self.block7 = nn.Sequential(nn.Linear(2000, 100), nn.ReLU())
        self.block8 = nn.Sequential(nn.Linear(100, num_classes))

        self.net = nn.Sequential(
            self.block1, self.block2, self.block3, self.block4, 
            self.block5, self.block6, self.block7, self.block8
        )

    def forward(self, x):
        return self.net(x)



class ResNet18Split_CIFAR10(nn.Module):
    """ResNet18 restructured into 10 blocks (CIFAR-10)."""
    def __init__(self, num_classes=10):
        super().__init__()
        base_model = torchvision.models.resnet18(weights=None)
        
        # CIFAR-10 Tweaks: 3x3 initial conv, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

        self.block1 = nn.Sequential(self.conv1, base_model.bn1, base_model.relu, nn.Identity())  # No maxpool for CIFAR-10
        self.block2 = base_model.layer1[0]
        self.block3 = base_model.layer1[1]
        self.block4 = base_model.layer2[0]
        self.block5 = base_model.layer2[1]
        self.block6 = base_model.layer3[0]
        self.block7 = base_model.layer3[1]
        self.block8 = base_model.layer4[0]
        self.block9 = base_model.layer4[1]
        self.block10 = nn.Sequential(base_model.avgpool, nn.Flatten(), self.fc)

        self.net = nn.Sequential(
            self.block1, self.block2, self.block3, self.block4, self.block5,
            self.block6, self.block7, self.block8, self.block9, self.block10
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
#       THE DICTIONARY GENERATOR
# ==========================================


# Define a custom rule to ignore computation for specific layers 
def ignore_macs(m, x, y):
    m.total_ops += torch.DoubleTensor([0])

# Map the layers we want to ignore computationally to the rule above
# Focus on the dominant Conv2d and Linear layers for MACs (ignore BatchNorm, ReLU, Pooling, etc.)
custom_ops_dict = {
    nn.BatchNorm2d: ignore_macs,
    nn.BatchNorm1d: ignore_macs,
    nn.ReLU: ignore_macs,
    nn.MaxPool2d: ignore_macs,
    nn.AvgPool2d: ignore_macs,
    nn.Identity: ignore_macs
}


def generate_fsl_dictionary(model, input_size):
    device = torch.device('cpu')
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    block_results = []
    x_current = dummy_input
    
    print("Profiling blocks (Convs and FCs only)...")
    for name, block in model.net.named_children():
        macs, params = profile(
            block, 
            inputs=(x_current,), 
            custom_ops=custom_ops_dict, 
            verbose=False
        )
        
        x_current = block(x_current)
        act = x_current.numel()
        
        block_results.append({
            'macs': macs,
            'params': params,
            'act': act
        })
        
    M_total = sum(b['macs'] for b in block_results)
    d_total = sum(b['params'] for b in block_results)
    O_total = sum(b['act'] for b in block_results)
    
    fsl_params = {}
    Mc_cum, dc_cum, Oc_cum = 0, 0, 0
    total_blocks = len(block_results)
    
    for i, b in enumerate(block_results):
        split_idx = i + 1
        
        Mc_cum += b['macs']
        dc_cum += b['params']
        Oc_cum += b['act']
        
        if split_idx == total_blocks:
            final_act = 0
            Ms, Os, ds = 0, 0, 0 
        else:
            final_act = int(b['act'])
            Ms = int(M_total - Mc_cum)
            Os = int(O_total - Oc_cum)
            ds = int(d_total - dc_cum)
        
        fsl_params[split_idx] = {
            'client': {'Mc': int(Mc_cum), 'Oc': int(Oc_cum), 'dc': int(dc_cum)},
            'server': {'Ms': Ms, 'Os': Os, 'ds': ds},
            'act': final_act
        }
        
    return fsl_params




if __name__ == '__main__':
    print("--- 1. Base CNN on MNIST (1x28x28) ---")
    cnn_model = BaseCNN_Split()
    cnn_dict = generate_fsl_dictionary(cnn_model, (1, 1, 28, 28))
    pprint.pprint(cnn_dict, sort_dicts=False)
    
    print("\n--- 2. ResNet18 on CIFAR-10 (3x32x32) ---")
    resnet_model = ResNet18Split_CIFAR10()
    resnet_dict = generate_fsl_dictionary(resnet_model, (1, 3, 32, 32))
    pprint.pprint(resnet_dict, sort_dicts=False)



