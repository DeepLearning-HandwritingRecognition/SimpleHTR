
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

LABELS = [
    "ReachToShelf", "RetractFromShelf"
]
# DATASET_PATH = "data/HAR_pose_activities/database/"
DATASET_PATH = "/home/son/workspace/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input/data/HAR_pose_activities/setting_4/"

X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH + "X_test.txt"

y_train_path = DATASET_PATH + "Y_train.txt"
y_test_path = DATASET_PATH + "Y_test.txt"


def load_X(X_path):
    n_frames = 5  # 32 timesteps per series

    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.float32
    ) / 960.0
    file.close()
    blocks = int(len(X_) / n_frames)

    x_2d = []
    for i in X_:
        x1 = i[0::2]
        x2 = i[1::2]
        x_2d.append([x1, x2])
    X_ = np.array(x_2d)
    X_ = np.array(np.split(X_, blocks))
    return X_


def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    return y_ - 1


X_train = load_X(X_train_path)
X_test = load_X(X_test_path)
# print X_test

y_train = to_categorical(load_y(y_train_path))
y_test = to_categorical(load_y(y_test_path))
# print(X_train)
training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 1197 test series
n_input = len(X_train[0][0])  # num input parameters per timestep
print(n_input)
n_hidden = 32  # Hidden layer num of features
n_classes = 2
n_frames = 5  # 32 timesteps per series
n_dim = 2
n_joints = 18

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(n_frames, 1, n_dim, n_joints)))
# model.add(Bidirectional(LSTM(64,return_sequences=True)))  # out = (?,5,64)
# model.add(Bidirectional(LSTM(32,return_sequences=True)))  # out = (?,5,64)
# model.add(Dropout(0.2))

model.add(LSTM(n_hidden, return_sequences=True))  # out = (?,5,32)
model.add(GlobalMaxPool1D(data_format='channels_last'))
# model.add(Dropout(0.25))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 1000
callbacks = []
weights_file = 'weights_bilstm_4_2bidirection_{epoch:03d}-{val_acc:.4f}.h5'
callbacks.append(
    ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=50))
callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto'))

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, verbose=1,
          callbacks=callbacks)

# model.evaluate(X_test,y_test,verbose=1)
weights_file = 'weights_bilstm_3_{epoch:03d}.h5'
model.save_weights(weights_file.format(epoch=epochs))
