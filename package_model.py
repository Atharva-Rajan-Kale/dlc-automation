import os
import time
import warnings
import subprocess
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.configs.config_helper import ConfigBuilder
import shutil
warnings.filterwarnings('ignore')

version = '1.3.1'

subprocess.run(['python', '-m', 'pip', 'install', '-Uq', 'pip'])
subprocess.run(['python', '-m', 'pip', 'install', '-Uq', 'setuptools', 'wheel'])
subprocess.run(['python', '-m', 'pip', 'install', f'autogluon.tabular[all]=={version}'])

os.system('rm -rf data-sm-package')

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data[:100]

label = 'class'
metric = 'accuracy'

config = ConfigBuilder().hyperparameters('toy').build()

time_start = time.time()
predictor = TabularPredictor(label=label, path='data-sm-package/')
predictor = predictor.fit(
    train_data,
    **config,
    verbosity=2,
)
time_elapsed = time.time() - time_start
predictor.leaderboard(silent=True)

os.system(f'rm model_{version}.tar.gz')
os.system(f'tar -C data-sm-package/ -czf model_{version}.tar.gz .')
os.system(f'ls -la model_{version}.tar.gz')

shutil.rmtree('data-sm-package')
