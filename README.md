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

Run pipeline locally:
```bash
python local.py --data_dir data/ --model LR --config ./configs/local.yaml
```

Run pipeline in a spark cluster:
```bash
spark-submit --packages com.microsoft.azure:synapseml_2.12:1.0.1 parallel.py
```