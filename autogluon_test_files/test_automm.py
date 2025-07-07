from autogluon.core.utils import show_versions

def test_automm():
    show_versions()
    import os
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    np.random.seed(123)

    download_dir = './ag_automm_tutorial'
    zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip'
    from autogluon.core.utils.loaders import load_zip
    load_zip.unzip(zip_file, unzip_dir=download_dir)

    import pandas as pd
    dataset_path = download_dir + '/petfinder_processed'
    train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
    test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)
    label_col = 'AdoptionSpeed'

    image_col = 'Images'
    train_data[image_col] = train_data[image_col].apply(
        lambda ele: ele.split(';')[0])  # Use the first image for a quick tutorial
    test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

    def path_expander(path, base_folder):
        path_l = path.split(';')
        return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

    train_data = train_data.sample(500, random_state=0)
    test_data = test_data.sample(100, random_state=0)

    from autogluon.multimodal import MultiModalPredictor
    predictor = MultiModalPredictor(label=label_col)
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "model.names": ["clip"],
            "env.num_gpus": 1,
        },
        time_limit=20,  # seconds
    )
    scores = predictor.evaluate(test_data, metrics=["accuracy"])
    print(scores)
    from autogluon.multimodal import MultiModalPredictor
    predictor = MultiModalPredictor(label=label_col)
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "model.names": ["clip","timm_image", "hf_text", "numerical_mlp", "fusion_mlp"],
            "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
            "env.num_gpus": 1,
        },
        time_limit=20,  # seconds
    )
    scores = predictor.evaluate(test_data, metrics=["accuracy"])
    print(scores)
    from autogluon.multimodal import MultiModalPredictor
    predictor = MultiModalPredictor(label=label_col)
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "model.names": ["timm_image"],
            "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
            "env.num_gpus": 1,
        },
        time_limit=20,  # seconds
    )
    scores = predictor.evaluate(test_data, metrics=["accuracy"])
    print(scores)
    from autogluon.multimodal import MultiModalPredictor
    predictor = MultiModalPredictor(label=label_col)
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "model.names": ["hf_text"],
            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
            "env.num_gpus": 1,
        },
        time_limit=20,  # seconds
    )
    scores = predictor.evaluate(test_data, metrics=["accuracy"])
    print(scores)