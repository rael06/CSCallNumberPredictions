import os
import sys
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_predictions(model_path, draws_path):
    result = ''
    model = load_model(model_path)
    draws = os.listdir(draws_path)
    filenames_test = []
    categories_test = []

    for file in draws:
        filenames_test.append(file)
        categories_test.append('')

    df_draws = pd.DataFrame({
        'filename': filenames_test,
        'category': categories_test
    })
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    for i in range(df_draws.shape[0]):
        with suppress_stdout():
            df_draw = df_draws.iloc[[i]]
            draw_generator = test_datagen.flow_from_dataframe(
                df_draw,
                draws_path,
                x_col='filename',
                y_col='category',
                target_size=(28, 28),
                class_mode='categorical',
                color_mode='grayscale'
            )

        predict = model.predict_generator(draw_generator, steps=1)
        prediction = predict.argmax()
        balance = ''
        for b in range(len(predict[0])):
            balance += str('{:0.2f}'.format(predict[0, b] * 100)) + '\n'

        result += df_draw.iloc[0, 0] + '\n' + str(prediction) + '\n' + balance + '\n'
    return result


def main(argv):
    result = get_predictions(argv[0], argv[1])
    print(result)


if __name__ == "__main__":
    main(sys.argv[1:])
