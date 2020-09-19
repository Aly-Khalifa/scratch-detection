from scratch_detector import  ScratchDetector
from scratch_dataset import ScratchDataset
from tensorflow.keras.models import Model


#Load scratch detector model
app = ScratchDetector()

#load weights from the most recent training checkpoint
app.load_most_recent_model()


data = ScratchDataset()
data.load_dataset()
data.display_random_patches(128)

#display segmentaiton result
mask = app.display_predicted_mask('frames/mlts_025.png', threshold = 0.7)

