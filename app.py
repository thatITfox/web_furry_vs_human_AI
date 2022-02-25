import numpy as np
import torch
import cv2
from flask import Flask, render_template, request, redirect, url_for

# set up labels furry and human
labels = ['furry', 'human']

# create flask app
app = Flask(__name__)

# load the pytorch script model
model = torch.load('furry_vs_human_ai_torchscript.pt')
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # get image from form
        image = request.files['file']
        # read image
        image = np.asarray(bytearray(image.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # print image shape
        print(image.shape)
        # turn the image to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resize image
        image = cv2.resize(image, (50, 50))
        # convert image to tensor
        output = model(torch.from_numpy(image).float().reshape(1, 1, 50, 50)/255)
        print(output)
        return("{}".format(labels[output.argmax()]))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')