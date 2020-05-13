import os

def makeDirectory(dirpath):
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus