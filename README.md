# Steganography detection in images

## Background
Steganography refers to an evasive technique that aims to conceal a file within another file – in this case, an image – without altering the appearance of the original file to ensure secrecy.

The most common techniques are being used now is called LSB(Least Significant Bits) by manipulating with the least significant bits to hide messages.
In general, there are two methods. One is based on spatial domain: hidden in pixels like LSB technique. Another is based on transformed domain by tampering with DCT or DWT.

In this project, I mainly implemented three components:
1. **LSB decoder**: only work with .png images as JPEG/JPG is a lossy compression method thus cannot really accommodate LSB encoding.
2. **EXIF viewer**: We know a raw image can come with a rich set of metadata like 'location', 'aperture' and etc. However, cybercriminals can take advantage of this and hide asome malicious content there. Thus, an EXIF viewer
comes important, which can extract all metadata of an image.
3. **DCT-based steganography classifier**: An classifier has been trained on JMiPOD, JUNIWARd, UERD embedding algorithms to identify the existence of them.

More details like model performance and constraints are mentioned [here](https://drive.google.com/file/d/1XONCLq2AGtTrX5dw-aQmdoh6A30EKifX/view?usp=sharing).
## How to use
In your terminal, after having located to where the project is saved:
1. prepare the python environment

```pip install -r requirements.txt```

2. export the data environment
```./Test_images``` should be replaced by the path which is the parent directory of your image folder

```export DATA_ROOT_PATH=./Test_images```

3. launch the Flask interface

```python3 app.py```

This is a screenshot on the interface.

![image](https://user-images.githubusercontent.com/77568908/196149563-b228bee7-e722-400e-ba8e-38baccac5656.png)
### Details
- You can either upload images, which will be shown on the interface once the results are returned or put down a folder name and the final results will be saved as .csv in the project directory.
- You can change the parameters like device, batch_size or number of workers for DCT-based classifier.
