# EECE5645-Course-Project

Project Structure:
```
├── data            <- Data files.
├── notebooks       <- Jupyter notebooks.
├── output          <- Model predictions, logs, etc.
└── src             <- Source code(.py) for use in this project.
    ├── data        <- Scripts to load and preprocess data.
    ├── features    <- Scripts to turn raw data into features for modeling.
    └── models      <- Scripts to train models and make predictions.
```

Default data directory is `data/` and default config directory is `configs/`.

Run logistic regression pipeline locally:
```bash
python local.py --model LR
```

Run logistic regression pipeline in a spark cluster:
```bash
spark-submit --packages com.microsoft.azure:synapseml_2.12:1.0.1 parallel.py --model LR
```