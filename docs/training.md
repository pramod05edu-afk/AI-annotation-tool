# Training the detector

The detector that pre-labels images is a COCO-pretrained **YOLO11** fine-tuned on the
prepared Safety Helmet dataset (`helmet`, `head`). Training runs on a free Google Colab
T4 GPU; the backend later runs the exported model on CPU with OpenVINO.

## Steps

1. **Prepare and package the dataset** (from `backend/`):
   ```bash
   uv run python scripts/prepare_dataset.py --src D:/datasets/hard-hat-detection --out D:/datasets/hardhat-prepared
   uv run python scripts/package_dataset.py --src D:/datasets/hardhat-prepared --out D:/datasets/hardhat-prepared.zip
   ```
   The zip (~1.2 GB) contains train / val / test; the `export` split stays local for the demo.
2. **Upload** `hardhat-prepared.zip` to Google Drive: `MyDrive/annotator/hardhat-prepared.zip`.
3. **Open** [`eval/colab_finetune.ipynb`](../eval/colab_finetune.ipynb) in Colab
   (File → Upload notebook), select *Runtime → Change runtime type → T4 GPU*, then *Run all*.
   Allow the Google Drive access prompt.
4. **Wait** about 1.5–2 hours (two models). If Colab disconnects, *Run all* again — training
   resumes from the last epoch, because everything is saved to Drive.
5. **Download** from `MyDrive/annotator/` into the repo's `weights/` folder (git-ignored):
   - `yolo11?_helmet_openvino_model/` — the model the backend uses
   - `yolo11?_helmet.pt` — the PyTorch weights (backup)

   and copy `detector_metrics.json` to `eval/results/` (committed: the report's numbers).

## Design choices

| Choice | Why |
|---|---|
| Fine-tune, not train from scratch | 3,800 images are too few to learn features from zero; COCO weights already know people and heads |
| Compare `yolo11n` and `yolo11s` | Accuracy vs. speed trade-off; the app needs fast CPU inference |
| 640 px input | Many heads are small (20–40 px); upscaling the ~416 px images helps |
| Disk cache (`cache="disk"`) | Fast loading on 2 CPU cores, reproducible, and no out-of-memory crash when training two models in one session |
| Early stopping (patience 15) | Stops when validation mAP stops improving; avoids overfitting and saves GPU time |
| Metrics on the **test** split | Validation was used to pick the best epoch, so only test is unbiased |
| Pinned library versions | Reproducible results; the backend must load the model with the same ultralytics version |
| MLflow tracking | Every run's settings, metrics and files in one place, comparable later |
| OpenVINO export | Intel's CPU runtime; typically several times faster than PyTorch on the laptop |

## Viewing MLflow locally

Download `MyDrive/annotator/mlflow.db`, then:
```bash
uvx mlflow ui --backend-store-uri sqlite:///mlflow.db
```
and open http://localhost:5000 (metrics and parameters; artifacts stay on Drive).
