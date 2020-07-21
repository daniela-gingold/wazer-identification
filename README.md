# wazer-identification

This project identifies whether an image includes a wazer or not, excluding wazer logo.

- **input** is an image
- **output** is "wazer" in case at least one wazer was detected in an image, "no wazer" otherwise .

For example, the image below conatins wazers, the output will be "wazer":
![image](/images/img1.jpg)

Hovewer, the following image doesn't contain any wazer except for waze logo. Thus, the expected result will be "no wazer"
![img5](/images/img5.jpg)

In order to run this project, beside the files in this repository, download [model_1](https://drive.google.com/drive/folders/1grwC-OLDYLe3nwEdrdYi75Q5ZnSG85aM?usp=sharingo).

The solution includes two consequent models:
- model_1.h5 and mrcnn folder is a transfer learning to detect wazers
- model_2.h5 includes a CNN to classify detected objects

This project has two solutions to identify and disqualify logos:
- according to model_1 locating outputs, i.e. when a logo overlaps a wazer area. This is a default solution.
- using matching template algorithm. There is an argument to apply this solution.

For running an inference:

python test.py --test_image_path images\img1.jpg --model_path model_1.h5 --secondary_model_path model_2.h5 --use_match_template 1

* the model’s paths can be skipped as they are default
* test_image_path can be link to an image saved locally or a web link 

Please note, that TensorFlow 1 is required for mrcnn. See full requirements.txt file attached to the folder.
