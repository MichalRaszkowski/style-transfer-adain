Neural Style Transfer (AdaIN)

Implementation of **Arbitrary Style Transfer** using **AdaIN**. This project enables real-time artistic stylization of any content image using any style image, based on Encoder-Decoder architecture with a VGG-19.


## Results

| Content | Style | Result |
| :---: | :---: | :---: |
| ![Content](images/content1.jpg) | ![Style](images/styl1.jpg) | ![Result](images/ostateczny1.jpg) |
| ![Content](images/content2.jpg) | ![Style](images/styl2.jpg) | ![Result](images/ostateczny2.jpg) |
| ![Content](images/content2.jpg) | ![Style](images/styl2.jpg) | ![Result](images/ostateczny3.jpg) |

## Quick Start

Run the setup script that installs dependencies and downloads the pre-trained model.

```bash
git clone [https://github.com/MichalRaszkowski/style-transfer-adain.git](https://github.com/MichalRaszkowski/style-transfer-adain.git)
cd style-transfer-adain

python setup.py
```

## You can stylize images from the terminal using stylize.py:
```bash
python stylize.py --content images/my_photo.jpg --style images/picasso.jpg
```
You can also use:
--output: Filename for result
--alpha: Stylization strenght between 0.0 (original image) to 1.0 (most stylized)
--size Resize content/style to this size before processing, default: 512.

## To run the web-based interface use:
```bash
python app.py
```
You can access it at [http://127.0.0.1:7860](http://127.0.0.1:7860)

## The project is also deployed and available for testing online on Hugging Face Spaces:
[https://huggingface.co/spaces/Michal-Raszkowski/style-transfer-adain](https://huggingface.co/spaces/Michal-Raszkowski/style-transfer-adain)

## Project structure
```text
style-transfer-adain/
├─ checkpoints/
├─ images/
├─ src/
│  ├─ models/
│  │  ├─ decoder.py
│  │  ├─ encoder.py
│  │  ├─ functional.py
│  ├─ callbacks.py
│  ├─ data_module.py
│  ├─ dataset.py
│  ├─ lightning_module.py
├─ .gitignore
├─ app.py
├─ benchmark.py
├─ main.py
├─ README.md
├─ requirements.txt
├─ setup.py
├─ stylize.py
```
