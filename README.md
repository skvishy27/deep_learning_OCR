# Text-Extraction-using-CRAFT-and-Deep-Text-Recognition
Extracting Text from US Driving Licence Using CRAFT and Deep Text Recognition

The Code is in initial stage.

## Getting started
### Install dependencies
#### Requirements
- lmdb==0.98
- natsort==7.0.1
- nltk==3.5
- numpy==1.18.3
- opencv-contrib-python==4.2.0.34
- Pillow==7.1.2
- pip==20.1
- scikit-image==0.16.2
- scipy==1.4.1
- setuptools==39.0.1
- six==1.14.0
- torch==1.5.0+cpu
- torchvision==0.6.0+cpu
- wheel==0.34.2
```
pip install -r requirements.txt
```
### Model
Download Pre-trained Model/Weight
- [craft_mlt_25k.pth](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ),<br>
- [TPS-ResNet-BiLSTM-Attn.pth]((https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW))

### Test
Run the below code:<br>
```
python extract.py -i data/AL.jpeg -o result
```
where 
"extract.py" -> filename<br>
"data/AL.jpeg" -> data is input directory and AL.jpeg is image file<br>
"result" -> output directory where extracted text will be saved

## Acknowledgements
This implementation has been based on these repository [clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch), [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

## Links
- WebDemo : https://demo.ocr.clova.ai/ <br>
Combination of Clova AI detection and recognition, additional/advanced features used for KOR/JPN.

