# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe

class RoIDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        self.count=0

    def forward(self, bottom, top):
        if self.count%caffe.solver_count()==caffe.solver_rank():
            print self.count
        self.count+=1

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        top[0].reshape(1, 3, 28, 28)
        top[1].reshape(1)
