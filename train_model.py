from preprocess import generate_training_sequences, SEQUENCE_LENGTH
import keras
from keras.layers import Input, Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint


OUTPUT_UNITS = 1
NUM_UNITS = [256]
LOSS = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64
SAVE_MODEL_PATH = 'model.h5'

def build_model(output_units, num_units, loss, learning_rate):
    
    input = Input(shape=(None, output_units))
    x = LSTM(num_units[0])(input)
    x = Dropout(0.2)(x)

    output = Dense(101, activation='softmax')(x)

    model = keras.Model(input, output)

    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    model.summary()

    return model

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):

    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    model = build_model(output_units, num_units, loss, learning_rate)

    filepath = 'model-{epoch:02d}-{loss:.2f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)

    model.save(SAVE_MODEL_PATH)


if __name__ == '__main__':
    train()
