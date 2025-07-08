import os
import time
import warnings
import subprocess
import shutil
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

version = '1.3.1'

def run_command(cmd, check=True, timeout=600):
    """Run command with better error handling"""
    logger.info(f"🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            check=check, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")
        if result.stderr:
            logger.warning(f"STDERR: {result.stderr}")
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Command timed out after {timeout} seconds: {' '.join(cmd)}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        raise

def main():
    """Main function with better error handling"""
    try:
        logger.info(f"🚀 Starting model packaging for AutoGluon {version}")
        
        # Update pip and setuptools
        logger.info("📦 Updating pip and setuptools...")
        run_command(['python', '-m', 'pip', 'install', '-Uq', 'pip'], timeout=300)
        run_command(['python', '-m', 'pip', 'install', '-Uq', 'setuptools', 'wheel'], timeout=300)
        
        # Install AutoGluon with retries
        logger.info(f"📦 Installing autogluon.tabular[all]=={version}...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                run_command([
                    'python', '-m', 'pip', 'install', 
                    f'autogluon.tabular[all]=={version}',
                    '--timeout', '300'
                ], timeout=900)  # 15 minutes for installation
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Installation attempt {attempt + 1} failed: {e}")
                    logger.info(f"🔄 Retrying installation (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(30)  # Wait 30 seconds before retry
                else:
                    logger.error(f"❌ All installation attempts failed")
                    raise
        
        # Clean up previous data
        data_dir = Path('data-sm-package')
        if data_dir.exists():
            logger.info("🧹 Cleaning up previous data...")
            shutil.rmtree(data_dir)
        
        # Import AutoGluon after installation
        logger.info("📥 Importing AutoGluon...")
        try:
            from autogluon.tabular import TabularDataset, TabularPredictor
            from autogluon.tabular.configs.config_helper import ConfigBuilder
            logger.info("✅ AutoGluon imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import AutoGluon: {e}")
            raise
        
        # Load training data with retry logic
        logger.info("📊 Loading training data...")
        max_retries = 3
        train_data = None
        for attempt in range(max_retries):
            try:
                train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
                train_data = train_data[:100]  # Use only first 100 rows for faster training
                logger.info(f"✅ Loaded {len(train_data)} training samples")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Data loading attempt {attempt + 1} failed: {e}")
                    logger.info(f"🔄 Retrying data loading (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(10)
                else:
                    logger.error(f"❌ All data loading attempts failed")
                    raise
        
        # Configure training
        label = 'class'
        metric = 'accuracy'
        config = ConfigBuilder().hyperparameters('toy').build()
        
        # Train model
        logger.info("🤖 Training model...")
        time_start = time.time()
        
        predictor = TabularPredictor(label=label, path='data-sm-package/')
        predictor = predictor.fit(
            train_data,
            **config,
            verbosity=2,
        )
        
        time_elapsed = time.time() - time_start
        logger.info(f"✅ Model training completed in {time_elapsed:.2f} seconds")
        
        # Generate leaderboard
        try:
            leaderboard = predictor.leaderboard(silent=True)
            logger.info("📊 Model leaderboard generated")
        except Exception as e:
            logger.warning(f"⚠️ Could not generate leaderboard: {e}")
        
        # Create tar.gz archive
        model_file = f'model_{version}.tar.gz'
        
        # Remove existing model file if it exists
        if Path(model_file).exists():
            logger.info(f"🗑️ Removing existing {model_file}")
            Path(model_file).unlink()
        
        # Create archive using Python instead of os.system
        logger.info(f"📦 Creating {model_file}...")
        try:
            result = run_command([
                'tar', '-C', 'data-sm-package/', '-czf', model_file, '.'
            ], timeout=300)
            
            # Verify the archive was created
            if Path(model_file).exists():
                file_size = Path(model_file).stat().st_size
                logger.info(f"✅ Archive created: {model_file} ({file_size} bytes)")
                
                # List archive contents for verification
                try:
                    result = run_command(['ls', '-la', model_file], check=False)
                except:
                    pass  # Don't fail if ls command fails
            else:
                raise FileNotFoundError(f"Archive {model_file} was not created")
                
        except Exception as e:
            logger.error(f"❌ Failed to create archive: {e}")
            raise
        
        # Clean up training data directory
        logger.info("🧹 Cleaning up training data...")
        try:
            if data_dir.exists():
                shutil.rmtree(data_dir)
                logger.info("✅ Training data cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up training data: {e}")
        
        logger.info(f"🎉 Model packaging completed successfully! Created {model_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model packaging failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
