from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from MyDataManager import MyDataManager

def main():

    # Create instance of our class
    dm = MyDataManager("2014-01-01")

    # Assign x and y train values
    x_train, y_train = dm.create_train_data()

    # Build the model
    model = Sequential()
    model.add(LSTM(units=4, activation="sigmoid", input_shape=(None, 1)))
    model.add(Dropout(0.25))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(x_train, y_train, epochs=200, batch_size=32)

    # Assign the test value
    test_set = dm.create_test_data()

    # Plot
    dm.plot(test_set, model)


if __name__ == '__main__':
    main()
