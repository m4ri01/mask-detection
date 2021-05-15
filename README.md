# Mask Detection Based On MobileNet Architecture  
  
## Decryption  
This program is about mask detection using opencv and mobile-net. I used transfer learning method to increase the accuration of the model.  
The dataset that i used to created this models provide by [prajnasb](https://github.com/prajnasb/observations/tree/master/experiements/data). And the streaming detection program i used  from [mk-gurucharan](https://github.com/mk-gurucharan/Face-Mask-Detection).  
  
## Requirements  
- Python 3.x  
- Tensorflow 2.x
- keras  
- Numpy  
- OpenCV  


## Architecture  
I used pretrained model Mobile Net architecture. For the output layer i add some Dense layer and GlobalAveragePooling2D layer. For the output layer activation i used softmax. This models use Adam optimizer and categorical loss entropy to calculate the loss. 