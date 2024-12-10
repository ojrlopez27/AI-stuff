import torch
import time

def create_torch_tensors(device):
	x = torch.rand((100000, 100000), dtype=torch.float32)
	y = torch.rand((100000, 100000), dtype=torch.float32)
	x = x.to(device)
	y = y.to(device)
	
	return x, y

def compute_with_device(device):
	device = torch.device(device)
	x, y = create_torch_tensors(device)

	start = time.time()
	print('start')
	x * y
	print('end')
	end = time.time()
	print('time for {device}:', end - start)


compute_with_device('cpu')
compute_with_device('mps')


