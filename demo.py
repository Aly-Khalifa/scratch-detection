from scratch_detector import  ScratchDetector
from scratch_dataset import ScratchDataset
from tensorflow.keras.models import Model


#Load scratch detector model
app = ScratchDetector()

#load weights from the most recent training checkpoint
app.load_most_recent_model()

#display segmentaiton result
mask = app.display_predicted_mask('frames/mlts_056.png', threshold = 0.7)
