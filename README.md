# Plant Species Classifier

## Dataset readiness

The current dataset is organized as one folder per plant species under
`dataset/`. It contains 5,539 image files; inspection excludes one exact
duplicate from the generated 5,538-image manifest and stratified splits. It
builds `dataset/labels.csv` and train/validation/test splits from those
folders. Training regenerates splits if they no longer match the images on
disk. Images are decoded as RGB, resized to 224x224, and processed with ResNetV2
preprocessing; augmentation is applied only to training data.

Any existing checkpoint from the previous dataset must be replaced by a
complete training run before it is used for inference.

## Train the final model

From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m src.dataset_inspection
.\.venv\Scripts\python.exe -m src.train
```

This runs frozen-backbone training followed by fine-tuning. Both stages use
early stopping and save their best validation-loss checkpoints. The selected
best checkpoint is exported to:

- `models/weed_resnet152v2_final.keras`
- `models/weed_resnet152v2_best.keras`
- `models/class_names.json`
- `models/model_metadata.json`

Omit `--sample-fraction` for final training; it is only for experiments. You
can adjust epoch limits, for example `--epochs 15 --fine-tune-epochs 15`.
Native Windows TensorFlow on this machine does not detect a GPU, so full
training may be very slow. Use a supported GPU environment (such as Linux/WSL2
with compatible TensorFlow/CUDA) for practical training speed.

## Run inference

After training completes, launch the upload page from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The browser opens a local page where you can choose an image and request a
prediction. The command-line option is also available:

```powershell
.\.venv\Scripts\python.exe predict.py --image "dataset\Black-grass\1.png"
```

## Sharing

The public GitHub repository contains the source code and dataset metadata.
Download the published assets from
[GitHub Releases](https://github.com/gouravkunder/veedd-classification/releases/latest):

- Download `dataset.zip` and extract it in the project root to create `dataset/`.
- Download `weed_resnet152v2_final.keras` into `models/`.

The final model and full dataset are release downloads rather than ordinary
repository files because they exceed GitHub's browser-upload file-size limit.

The original images remain in `dataset/<class>/`; `dataset/labels.csv` and
`dataset/split_files/` contain the generated manifest and splits. The images
are preprocessed on the fly during training (decode, resize, augment training
images only, and ResNetV2 normalization); separate preprocessed image copies
are not saved.

To run inference, install the packages in `requirements_inference.txt` and
launch the upload page from the project root:

```powershell
python -m pip install -r requirements_inference.txt
python -m streamlit run app.py
```

Training is not required when the published final model is downloaded.
