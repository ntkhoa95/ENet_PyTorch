# ENet_PyTorch

Implementation of [*ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation*](https://arxiv.org/abs/1606.02147) using PyTorch (v.1.10)

This implementation is validated on the CamVid dataset.
The pre-trained of ENet trained with CamVid is available [here](https://github.com/ntkhoa95/ENet_PyTorch/tree/main/content/checkpoint/camvid/)

|                               Dataset                                |       Type     | Classes  | Input resolution | Batch size | Epochs |   Mean IoU (%)   | GFLOPS | Parameters|
| :------------------------------------------------------------------: |:--------------:| :------------------: | :--------------: | :--------: | :----: | :---------------: | :-------: | :-------: | 
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |    Original    |          11          |     480x360      |     10     |  -     | 58.3| 3.83 | 0.37M|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) | Implementation |          11          |     480x360      |     10     |  100   | 59.5| 2.34 | 0.35M|
