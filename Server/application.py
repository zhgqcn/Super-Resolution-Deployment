from flask import Flask
from flask import render_template, send_from_directory
from flask import request
import os
import cv2
from predict import *
from model import *
from prepare_images import calculate_psnr, compare_images, calculate_ssim
from PIL import Image
from flask_ngrok import run_with_ngrok


app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

UPLOAD_FOLDER = 'static/input/'
OUTPUT_FOLDER = 'static/output/'


global model_g
global model_r
model_r = MSASRN(5, 5, 4).cuda()
model_g = MSASRN(5, 5, 4).cuda()

optimizer_r = optim.Adagrad(model_r.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer_g = optim.Adagrad(model_g.parameters(), lr=1e-3, weight_decay=1e-5)

model_r, optimizer_r, epochs_r = load_model(model_r, optimizer_r, './model/MSASRN_model_epoch_r_250.pth')
model_g, optimizer_g, epochs_g = load_model(model_g, optimizer_g, './model/MSASRN_model_epoch_g_250.pth')


@app.route("/", methods=['GET', 'POST'])
def upload_predict():
    return render_template('index.html', image_name=None)


@app.route("/get_result", methods=['GET', 'POST'])
def get_result():
    if request.method == 'POST':
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            if not os.path.exists(image_location):
                image_file.save(image_location)

                output2, output4 = predict(model_r, model_g, image_file.filename)

                output_location_x2 = os.path.join(OUTPUT_FOLDER + '/X2', 'X2_' + image_file.filename + '.png')
                output_location_x4 = os.path.join(OUTPUT_FOLDER + '/X4', 'X4_' + image_file.filename + '.png')
                output2.save(output_location_x2)
                output4.save(output_location_x4)
            return render_template('result.html', image_name=image_file.filename)

    return render_template('index.html', image_name=None)


@app.route('/static/output/<string:filename>', methods=['GET', 'POST'])
def download_img(filename):
    return send_from_directory('/static/output/', filename, as_attachment=True)


if __name__ == "__main__":
    # app.run(host='127.0.0.1', debug=True)
    # app.run(host='0.0.0.0', port=8080, debug=False)
    app.run()

