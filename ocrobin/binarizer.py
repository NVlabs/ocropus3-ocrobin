from numpy import *
import torch
from torch.autograd import Variable

class Binarizer(object):
    def __init__(self, mname):
        self.model = torch.load(mname).cuda()
    def binarize(self, image):
        if image.ndim==3 and image.shape[2]==3:
            image = mean(image, 2)
        assert image.ndim==2, image.shape
        timage = torch.FloatTensor(image).cuda()[None,None,:,:]
        bimage = self.model.forward(Variable(timage, requires_grad=False))
        result = array(bimage.data.cpu(), 'f')[0, 0]
        return result
    def binarize_batch(self, batch):
        if batch.ndim==4 and batch.shape[2]==3:
            batch = mean(batch, 3)
        assert batch.ndim==3, batch.shape
        timage = torch.FloatTensor(batch).cuda()[:,None,:,:]
        bimage = self.model.forward(Variable(timage, requires_grad=False))
        assert bimage.size(1) == 1
        result = array(bimage.data.cpu(), 'f')[:, 0, :, :]
        return result
