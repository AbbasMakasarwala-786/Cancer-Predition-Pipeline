import os

PROCESSED_DIR = 'artifacts/processed'
MODEL_DIR = 'artifacts/models'

X_TRAIN_ARRAY =os.path.join(PROCESSED_DIR,"X_train.pkl")
X_TEST_ARRAY = os.path.join(PROCESSED_DIR,'X_test.pkl')
Y_TRAIN =os.path.join(PROCESSED_DIR,"y_train.pkl")
Y_TEST = os.path.join(PROCESSED_DIR,'y_test.pkl')
SCALER= os.path.join(PROCESSED_DIR,"scaler.pkl")