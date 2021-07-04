import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

import tts

tts.speak("Getting Data", print_text=True)
x = numpy.load('image.npz')['arr_0']
y = pandas.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv")["labels"]
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
           "W", "X", "Y", "Z"]
tts.speak("Done Getting Data", print_text=True)

tts.speak("Performing Train Test Split", print_text=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=9, train_size=7500, test_size=2500)
tts.speak("Done Performing Train Test Split", print_text=True)

tts.speak("Scaling X Train and Test", print_text=True)
x_train_scaled = x_train / 255.0
x_test_scaled = x_test / 255.0
tts.speak("Done Scaling X Train and Test", print_text=True)

tts.speak("Performing Logistic Regression", print_text=True)
log_red = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train_scaled, y_train)
tts.speak("Done Performing Logistic Regression", print_text=True)


def get_prediction(image):
    tts.speak("Getting Prediction", print_text=True)

    im_pil = Image.open(image)

    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20

    min_pixel = numpy.percentile(image_bw_resized_inverted, pixel_filter)

    image_bw_resized_inverted_scaled = numpy.clip(image_bw_resized_inverted - min_pixel, 0, 255)
    max_pixel = numpy.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled = numpy.asarray(image_bw_resized_inverted_scaled) / max_pixel

    test_sample = numpy.array(image_bw_resized_inverted_scaled).reshape(1, 660)
    test_pred = log_red.predict(test_sample)

    tts.speak("Done Getting Prediction", print_text=True)
    return test_pred
