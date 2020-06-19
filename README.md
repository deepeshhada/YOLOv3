# You Only look Once (YOLO) v3

The repository contains code for object detection on the [MS COCO](http://cocodataset.org/#home) dataset using [YOLO v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf). 

The pretrained weights can be found [here](https://drive.google.com/file/d/1tazw2Ar1ubJvAa7PutOrJDaiRnAjdIay/view?usp=sharing
). Please replace the empty weights file in the `weights` folder with the downloaded file (~237 MB).

`test.ipynb` contains code for testing the model. Sample images are placed under the `images` directory. Simply run all the cells of this notebook to test the model. Here's a sample result:
![dog_cycle_truck](https://i.imgur.com/e39u6t8.png "Testing YOLO v3")
