import subprocess

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.utils import show_versions


def test_tabular():
    show_versions()
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
    train_data = train_data.sample(n=subsample_size, random_state=0)
    label = 'class'
    print("Summary of class variable: \n", train_data[label].describe())
    save_path = 'agModels-predictClass'  # specifies folder to store trained models
    predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
    subprocess.run(["tar", "-C", save_path, "-czf", f"model.tar.gz", "."], check=True)
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    y_test = test_data[label]  # values to predict
    test_data_nolab = test_data.drop(columns=[label])  # delete label column to prove we're not cheating
    test_data_nolab.head()
    predictor = TabularPredictor.load(
        save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

    y_pred = predictor.predict(test_data_nolab)
    print("Predictions:  \n", y_pred)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    print(predictor.leaderboard(test_data, silent=True))

def test_tabular_automm():
    show_versions()
    download_dir = './ag_petfinder_tutorial'
    zip_file = 'https://automl-mm-bench.s3.amazonaws.com/petfinder_kaggle.zip'

    from autogluon.core.utils.loaders import load_zip
    load_zip.unzip(zip_file, unzip_dir=download_dir)

    import os
    dataset_path = download_dir + '/petfinder_processed'

    import pandas as pd

    train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
    test_data = pd.read_csv(f'{dataset_path}/dev.csv', index_col=0)

    label = 'AdoptionSpeed'
    image_col = 'Images'

    train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
    test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])

    def path_expander(path, base_folder):
        path_l = path.split(';')
        return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

    train_data[image_col] = train_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))
    test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder=dataset_path))

    train_data = train_data.sample(500, random_state=0)

    from autogluon.tabular import FeatureMetadata
    feature_metadata = FeatureMetadata.from_df(train_data)
    feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
    print(feature_metadata)

    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
    hyperparameters = get_hyperparameter_config('multimodal')
    hyperparameters['AG_IMAGE_NN'] = {'model': 'resnet18'}
    hyperparameters['AG_TEXT_NN'] = ['lower_quality_fast_train']
    hyperparameters

    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor(label=label).fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        feature_metadata=feature_metadata,
        time_limit=900,
    )

    print(predictor.leaderboard(test_data))
    
