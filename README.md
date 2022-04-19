# ENet_PyTorch

Implementation of [*ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation*](https://arxiv.org/abs/1606.02147) using PyTorch (v.1.10)

This implementation is validated on the CamVid dataset.
The pre-trained of ENet trained with CamVid is available [here](https://github.com/ntkhoa95/ENet_PyTorch/tree/main/content/checkpoint/camvid/)

|                               Dataset                                |       Type     | Classes  | Input resolution | Batch size | Epochs |   Mean IoU (%)   | GFLOPS | Parameters|
| :------------------------------------------------------------------: |:--------------:| :------------------: | :--------------: | :--------: | :----: | :---------------: | :-------: | :-------: | 
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) |    Original    |          11          |     480x360      |     10     |  -     | 58.3| 3.83 | 0.37M|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) | Implementation |          11          |     480x360      |     10     |  100   | 59.5| 2.34 | 0.35M|

### To Use
1. Clone the repository
`git clone https://github.com/ntkhoa95/ENet_PyTorch.git`
`cd ENet_PyTorch`

2. Download the CamVid datasets
Download the CamVid dataset and unzip to ./content/camvid/
`wget https://www.dropbox.com/s/pxcz2wdz04zxocq/CamVid.zip?dl=1 -O CamVid.zip`
`unzip CamVid.zip`

3. Use command to train the model
`python init.py --mode train -iptr ./content/camvid/train/ -lptr ./content/camvid/trainannot/`

4. Use command to test model
`python init.py --mode test -m ./content/checkpoint/camvid/best_model.pth -i ./content/camvid/test/0001TP_008550.png`

5. Use `--help` to get more commands
`python init.py --help`

