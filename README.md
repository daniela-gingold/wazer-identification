# wazer-identification

This project identifies whether an image includes a wazer or not, excluding the waze logo.

- **input:** an image
- **output:** "wazer" in case at least one wazer was detected in an image, "no wazer" otherwise .

For example, the image below contains wazers so the output will be "wazer":

<img src="/images/img1.jpg" height = "250" width="250">

However, the following image doesn't contain any wazer except for the waze logo. Thus, the expected result will be "no wazer"


<img src="/images/img5.png" height = "250" width="500">

In order to run this project, besides the files in this repository, download [model_1](https://drive.google.com/drive/folders/1grwC-OLDYLe3nwEdrdYi75Q5ZnSG85aM?usp=sharingo).

The solution includes two models:
- model_1.h5 is a transfer learning Mask R-CNN to detect wazers
- model_2.h5 includes a CNN to classify detected objects

This project has two solutions to identify and disqualify logo wazers:
- according to coordinates of model_1 outputs of wazers, if a logo overlaps a wazer area, then that wazer is disqualified. This is the default solution.
- using the matching template algorithm. This can be enabled using a command line argument.

For running an inference, run:

python test.py --test_image_path images\img1.jpg --model_path model_1.h5 --secondary_model_path model_2.h5 --use_match_template 1

* the model’s paths can be skipped as they are default
* test_image_path can either be an image url link or an image saved locally 

Please note that TensorFlow 1 is required for mrcnn. See full requirements.txt file attached to the folder.
