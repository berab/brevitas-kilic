from brevitas.export import FINNManager
import torch

from models import LowPrecisionLeNet as Net
PATH = './saved-models/cifar_net_e2.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net() #init the model
net.load_state_dict(torch.load(PATH))

FINNManager.export(net, input_shape=(1, 3, 32, 32), export_path='finn_lenet.onnx')