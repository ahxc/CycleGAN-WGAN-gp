## CycleGAN-WGAN-gp
An implementation of CycleGan using TensorFlow, I added the WGAN-GP(gradient penalty) to the discriminator, but the result was not satisfactory, it produces two results that are almost identical to the input image. The reasons are still being explored, but it will works well without GP.

## Environment
python 3.6.3

windows 10

## Data preparing
[man2woman](https://pan.baidu.com/s/1i5qY3yt)

## Training
The "[]" means an optional operation, please refer to the tf.flag settings for details and other optional operation
```
python train.py [--X the path of domain X] [--Y the path of domain Y] [--image_size 256]
```

## Results preview
cycle-GP:

<p align="center">
  <img src="/Related images/step-17100.png.png">
  <img src="/Related images/step-17200.png.png">
  <img src="/Related images/step-17300.png.png">
  <img src="/Related images/step-17400.png.png">
</p>


cycleGAN:



## References
code: [github-CycleGAN](https://github.com/vanhuyz/CycleGAN-TensorFlow)

code: [github-WGAN-gp](https://github.com/igul222/improved_wgan_training)

code: [csdn-CycleGAN](https://blog.csdn.net/jiongnima/article/details/80113976)

blog: [tencentcloud-dataset](https://cloud.tencent.com/developer/article/1064970)
