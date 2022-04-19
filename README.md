# ENet_PyTorch
Implementation of ENet using PyTorch (v.1.10)

Implementation of [*ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation*](https://arxiv.org/abs/1606.02147)

This implementation is validated on the CamVid dataset.
The pre-trained of ENet trained with CamVid is available [here](https://github.com/davidtvs/PyTorch-ENet/tree/master/save)

|                               Dataset                                | Classes <sup>1</sup> | Input resolution | Batch size | Epochs |   Mean IoU (%)    | GPU memory (GiB) | Type |
| :------------------------------------------------------------------: | :------------------: | :--------------: | :--------: | :----: | :---------------: | :--------------: | :-------------------------------: |
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |          12          |     480x360      |     10     |  -     | 52.1<sup>3</sup> | 
| [CamVid](https://www.cityscapes-dataset.com/)               |          12          |     480x360      |     10     |  100   | 59.5<sup>4</sup> |
