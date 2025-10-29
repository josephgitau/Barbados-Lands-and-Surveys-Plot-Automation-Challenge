# Barbados Lands and Surveys Plot Automation Challenge

## Overview and Objectives

The Barbados Land Detection and OCR Project focuses on detecting cadastral land parcel boundaries from survey maps and performing OCR extraction of map annotations to support automated digitization and geospatial indexing. The solution combines semantic segmentation using UNet++ (EfficientNet-B7 encoder) for boundary detection with vision-language OCR powered by Qwen3-VL-30B.

The choice of the OCR model was intentional — rather than fine-tuning a smaller model on noisy text data, we leverage the strong zero-shot and few-shot generalization capabilities of a large, instruction-tuned model. Supervised Fine-Tuning (SFT) often causes models to memorize patterns in the training data, which can amplify dataset biases or label noise. Since our OCR dataset contained heterogeneous and imperfect annotations, additional fine-tuning risked overfitting to noise rather than improving accuracy. By using a large, pretrained vision-language model that already performs robustly in noisy environments, we avoided the drawbacks of SFT memorization. In essence, if a foundation model already demonstrates high-quality inference performance on the target domain, fine-tuning becomes unnecessary — and may even degrade generalization.

Below notebook summarizes the preliminary data checks with all issues identified: *Step 1: Barbados Train Data Preparation.ipynb*

### Key Objectives

*   Automated Land Boundary Detection: Use deep learning for accurate parcel segmentation from raster maps.
*   OCR Integration: Extract map annotations and metadata to create structured, searchable outputs.
*   Unified Training and Inference Pipelines: Provide reproducible notebooks ready for host integration.
*   Rapid Execution: Both notebooks execute on Kaggle CPU in under 5 minutes each.
*   Efficient Use of Open Datasets: All data sources are open, with precomputed features included (no raw FASTQ extraction required).

## Architecture Diagram

> Data Preparation → Model Training → Inference & Polygon Cleaning → OCR → Output Merging → Submission

This architecture ensures end-to-end automation from input maps to submission files for evaluation.

## ETL Process

### Segmentation Data Preparation

With the provided dataset, the first challenge experienced was conversion of real world polygons to masks that could fit in the images shared. To solve this challenge, data annotation was the best solution and in our case this was due to a deeper lack of domain knowledge and failed attempts of ocr and extraction of coordinates from the images to polygons. We used Label studio which is an Open Source Data Labeling Platform and below we have got the workflow explained.

**Installation of Label studio**
Source: [https://labelstud.io/](https://labelstud.io/)

To install Label Studio with pip and a virtual environment, you need Python version 3.8 or later. Run the following:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install label-studio
```

After you install Label Studio, start the server with the following command:

```bash
Label-studio
```

### Data Annotation

Data was annotated using polygon masks as shown in the screenshot below. To fully annotate all 658 images it would require a full dedicated 24 hours of work.

Once data annotation is done, All images are exported with a `results.json` and are used for training segmentation models.

### Extract

*   **Sources:** Open-source cadastral survey map datasets.
*   **Formats:** JPEG/PNG images with COCO-style annotation JSON (`result.json`).
*   **Volume:** Single batch processing per dataset; optimized for Kaggle runtime.

### Transform

*   **Preprocessing:**
    *   Image resizing (1024x1024).
    *   Encoder-specific normalization via Albumentations preprocessing.
    *   Data augmentation: flips, rotations, color perturbations.
*   **Feature Engineering:**
    *   Derived features computed and provided in `train_more_features.csv` and `test_more_features.csv`.
    *   Removes need for FASTQ raw data extraction (~6 hours saved).

### Load

*   **Data Loading:**
    *   Stored in `/content/data` and `/content/working` directories.
    *   DataLoaders use `num_workers=4`, `batch_size=2`, and pinned memory for efficiency.
    *   Automatically splits into training, validation, and test partitions.

## Data Modeling

### Model

*   **Architecture:** UNet++ with EfficientNet-B7 encoder.
*   **Framework:** PyTorch Lightning with segmentation-models-pytorch (SMP).
*   **Loss Functions:** Combination of Boundary Loss, Focal Loss ($\alpha=0.25, \gamma=2.0$), and optional Dice/BCE.
*   **Optimizer:** Adam with $lr=1e-4$.
*   **Precision:** Mixed precision (AMP 16).
*   **Epochs:** 150 (with early stopping at patience=8).

### Hyperparameters

| Parameter | Description | Value |
| :--- | :--- | :--- |
| Learning Rate | Base LR for Adam optimizer | 1e-4 |
| Batch Size | Training batch size | 2 |
| Image Size | Input dimensions | 1024x1024 |
| Threshold | Inference binary threshold | 0.30 |
| Encoder Weights | Pretrained backbone | imagenet |

## Inference & Post-processing

### Steps

1.  Load best checkpoint from training phase.
2.  Perform segmentation with threshold = 0.3.
3.  Apply post-processing:
    *   Remove self-intersections.
    *   Remove small holes (`min_hole_area_ratio=0.003`).
    *   Simplify boundaries (`RDP = 0.0025`).
    *   Smooth near-straight lines (smooth_straight_deg $\ge 165^\circ$).
4.  Save polygons and overlay masks for QA.

### Outputs

*   **Masks:** PNG files.
*   **Polygons:** CSV with WKT/GeoJSON geometries.
*   **Merged Data:** Polygons joined with OCR-extracted text.

## OCR Pipeline

### Model

*   **Model Name:** Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
*   **Framework:** vLLM runtime with configurable sampling parameters.
*   **Processor:** `AutoProcessor.from_pretrained()`

### Workflow

1.  Iterate through all `.jpg` files in `IMAGE_DIR`.
2.  Send multimodal prompts to vLLM model.
3.  Capture extracted text and log results to `ocr_predictions.csv`.
4.  Merge OCR text with polygon CSV based on image ID.

### Outputs

*   `ocr_predictions.csv`: Contains `image_id`, `text`, and `status` fields.
*   **Final Submission:** Combined OCR + polygon CSV ready for host ingestion.

## Run Time

| Notebook Runtime (Kaggle CPU) | Description |
| :--- | :--- |
| **Training** | < 5 hour 30 minutes | Model training and checkpoint generation (EfficientNet_B7) |
| **Inference & OCR** | < 40 minutes (2 minutes segmentation inference, 20 minutes OCR inference) | Segmentation, polygon cleaning, OCR extraction |

The entire pipeline (both notebooks) can be executed within **6 hours 30 minutes** total on colab L4 GPU for segmentation training and Colab A100 High Ram for Inference & OCR.

## Performance Metrics

### Evaluation Metrics

*   **Segmentation:** IoU and Dice scores on validation sets.
*   **OCR:** Accuracy of extracted text.
*   **Leaderboard Scores:**
    *   Public: 0.965006861
    *   Private: 0.970242006

### Additional Metrics

*   Polygon quality checks: self-intersection rate, vertex count distribution.
*   Inference efficiency: average time per image.

## Error Handling & Logging

*   **Robustness:** Try/Except blocks around OCR and inference steps.
*   **Logging:**
    *   Progress bars via `tqdm`.
    *   Checkpoint logging via PyTorch Lightning callbacks.
    *   Error logs appended per image in `ocr_predictions.csv`.
*   **Validation:** Auto-checks for missing data paths and non-numeric columns.

## Maintenance and Monitoring

### Reproducibility

*   Deterministic seeding and version-pinned `requirements.txt`.
*   Checkpoints saved with descriptive version tags.

### Retraining and Updates

*   Models can be re-run with updated datasets.
*   Supports scaling to new regions or higher image resolutions (e.g., 1280x1280).

### Monitoring

*   Continuous monitoring of OCR accuracy and segmentation drift.
*   Weekly sample-based QA for long-term deployments.

### Versioning

*   Maintain version tags (`v1.2.0`) for checkpoints and configuration files.

## Outputs and Deliverables

| Deliverable | Description |
| :--- | :--- |
| `requirements.txt` | Environment dependencies |
| Model Checkpoints | Trained weights (`.ckpt`) |
| `Barbados_Lands_and_Surveys_Land_Detection_Model_EfficientNet_B7.ipynb` | Training notebook |
| `Barbados_Final_Inference_Pipeline.ipynb` | Inference and OCR notebook |
| `ocr_predictions.csv` | OCR results |
| `final_polygons.csv` | Cleaned polygon geometries |
| `barbados_final.csv` | Combined polygons + OCR text |

## Known Limitations and Next Steps

### Limitations

*   Minor under-segmentation in fine boundary lines.
*   OCR hallucinations under heavy image noise.
*   High GPU RAM optimization for OCR.

### Future Work

*   Add test-time augmentation (TTA) ensemble.
*   Export results as GeoJSON FeatureCollection for GIS integration.
*   Integrate federated training loop for cross-region learning.
