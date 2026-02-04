import sys
import os

# Add the project root to sys.path to simulate running from root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # parent of src
sys.path.append(project_root)

print(f"Testing imports from project root: {project_root}")

try:
    print("Attempting to import src.models.unet...")
    from src.models import unet
    print("✅ Successfully imported src.models.unet")
except Exception as e:
    print(f"❌ Failed to import src.models.unet: {e}")

try:
    print("Attempting to import src.models.feature_extractor...")
    from src.models import feature_extractor
    print("✅ Successfully imported src.models.feature_extractor")
except Exception as e:
    print(f"❌ Failed to import src.models.feature_extractor: {e}")

try:
    print("Attempting to import src.utils.iam_dataset...")
    from src.utils import iam_dataset
    print("✅ Successfully imported src.utils.iam_dataset")
except Exception as e:
    print(f"❌ Failed to import src.utils.iam_dataset: {e}")

try:
    print("Attempting to import src.training.eng_train...")
    from src.training import eng_train
    print("✅ Successfully imported src.training.eng_train")
except Exception as e:
    print(f"❌ Failed to import src.training.eng_train: {e}")
    
try:
    print("Attempting to import src.inference.infer...")
    from src.inference import infer
    print("✅ Successfully imported src.inference.infer")
except Exception as e:
    print(f"❌ Failed to import src.inference.infer: {e}")

print("\nImport verification complete.")
