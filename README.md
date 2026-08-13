# RadiantScan

**Explainable AI for Chest X-Ray Diagnosis**

RadiantScan is a deep learning system that classifies chest X-rays (Pneumonia vs Normal)
using a fine-tuned CNN, and explains *why* it made each prediction using Grad-CAM
visual heatmaps — so predictions aren't a black box.

> This is a research/educational project. It is **not** a certified diagnostic tool
> and should never be used for real clinical decision-making.

---

## Project Status

- [ ] Phase 1 — Setup & Data
- [ ] Phase 2 — Baseline Model
- [ ] Phase 3 — Handling Class Imbalance
- [ ] Phase 4 — Explainability (Grad-CAM)
- [ ] Phase 5 — Evaluation
- [ ] Phase 6 — Demo App
- [ ] Phase 7 — Documentation & Polish

## Problem Statement

_(fill in after Phase 1 EDA: dataset size, class distribution, task definition)_

## Architecture

_(fill in during Phase 2: backbone, head, training config)_

## Results

_(fill in during Phase 5: accuracy, F1, ROC-AUC, comparison table)_

## Explainability

_(fill in during Phase 4: sample Grad-CAM overlays, failure case analysis)_

## Demo

_(fill in during Phase 6: HuggingFace Spaces link, screenshot/gif)_

## Project Structure

```
RadiantScan/
├── data/
│   ├── raw/            # original downloaded dataset (gitignored)
│   └── processed/      # train/val/test splits, manifests
├── notebooks/          # EDA and experiment notebooks
├── src/
│   ├── data_pipeline.py
│   ├── model.py
│   ├── train.py
│   ├── gradcam.py
│   └── evaluate.py
├── models/
│   └── checkpoints/    # saved weights (gitignored)
├── app/
│   └── app.py          # Gradio demo
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd RadiantScan
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Chest X-Ray Images (Pneumonia) — [Kaggle link]
Download via Kaggle API:
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/raw/
```

## Ethics & Limitations

_(fill in during Phase 7: dataset bias, known failure modes, disclaimer)_

## Author

Hardik Agrawal
