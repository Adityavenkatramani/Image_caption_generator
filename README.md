

## Image Captioning with Vision Encoder-Decoder Model

This code uses the `transformers` library to perform image captioning using a Vision Encoder-Decoder Model based on the ViT-GPT2 architecture. It generates captions for input images by leveraging the pre-trained model.

### Installation

Before running the code, you need to install the required dependencies. You can do this by running the following command:

```python
!pip install transformers
```

This command will install the `transformers` library, which is used for working with pre-trained models.

### Usage

1. Import necessary libraries:

```python
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
```

2. Load the pre-trained model and tokenizer:

```python
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

3. Prepare the input image:

```python
image_path = "/path/to/image.jpg"
image = Image.open(image_path)
```

4. Generate captions for the image:

```python
images = [image]
pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

output_ids = model.generate(pixel_values)

preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
preds = [pred.strip() for pred in preds]

print(preds)
```

Make sure to replace `/path/to/image.jpg` with the actual path to your image file.

### Pre-trained Model

The code uses a pre-trained Vision Encoder-Decoder Model based on the ViT-GPT2 architecture, which is downloaded using the `from_pretrained` method. The model is specifically designed for image captioning tasks.

### Example Output

The generated captions will be printed to the console. Each caption represents a possible description of the input image.

### GPU Support

The code checks if a CUDA-enabled GPU is available and utilizes it for faster processing. If no GPU is available, it falls back to CPU execution.

### Note

This README assumes that you have basic knowledge of working with Python and have the necessary image files to perform image captioning.

That's it! You now have a README explaining how to use the image captioning code with the Vision Encoder-Decoder Model.
