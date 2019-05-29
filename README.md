# Easy to understand Image Style Transfer in Pytorch

This is a minimalistic Pytorch implementation of style transfer by Gatys Et al. Meant to be run in google colab, though you can turn the COLAB flag off if running locally.  

The idea is retain a candidate images 'content' while redering it in the 'style' of another reference image. This is best explained by looking at the images and the result themselves:
   
![What's Style Transfer](https://raw.githubusercontent.com/tumble-weed/style-transfer/master/style_transfer_flowchart.png)


[A timelapse of style being transferred](https://drive.google.com/open?id=1cIsETWlD2u2ceiUAt1K7NjnFpTRtKmtO)

### Some points about the code:

Style transfer requires the convolutional features of a standard classfier (most commonly VGG19, same as the one used here). This code can serve as a good example of getting these by using 'backward hooks'.

