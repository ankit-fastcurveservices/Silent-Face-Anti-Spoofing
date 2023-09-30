# Solve last issue: apt install libgl1-mesa-glx
import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"

def check_image_classify(image):
    height, width, channel = image.shape
    print(f'Height={height}, Width={width}')
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True
    
    
def classify(image, model_dir, device_id):
    # from cv2 import cv
    
    # fd = open(SAMPLE_IMAGE_PATH + image_name)
    # img_str = fd.read()
    # fd.close()

    # # CV2
    # nparr = np.fromstring(img_str, np.uint8)
    # image = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    # image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    # result = check_image_classify(image)
    # if result is False:
    #     return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format("passed image", value))
        result_text = "RealFace Score: {:.2f}".format(value)
        return {"imageLive": True, "confidence": value}
        # color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format("passed image", value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        return {"imageLive": False, "confidence": value}
        # color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    # cv2.rectangle(
    #     image,
    #     (image_bbox[0], image_bbox[1]),
    #     (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
    #     color, 2)
    # cv2.putText(
    #     image,
    #     result_text,
    #     (image_bbox[0], image_bbox[1] - 5),
    #     cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    # format_ = os.path.splitext(image_name)[-1]
    # result_image_name = image_name.replace(format_, "_result" + format_)
    # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)


# if __name__ == "__main__":
#     desc = "test"
#     parser = argparse.ArgumentParser(description=desc)
#     parser.add_argument(
#         "--device_id",
#         type=int,
#         default=0,
#         help="which gpu id, [0/1/2/3]")
#     parser.add_argument(
#         "--model_dir",
#         type=str,
#         default="./resources/anti_spoof_models",
#         help="model_lib used to test")
#     parser.add_argument(
#         "--image_name",
#         type=str,
#         default="image_F1.jpg",
#         help="image used to test")
#     args = parser.parse_args()
#     classify(args.image_name, args.model_dir, args.device_id)



from flask import Flask, request, jsonify

app = Flask(__name__)
# from images import Image
# import images

@app.route("/liveness/classify", methods=["POST"])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    # img = Image.open(file.stream)
    # print(file)
    nparr = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print(img)
    result = classify(img, "./resources/anti_spoof_models", 0)

    return jsonify(result)#, 'size': [img.width, img.height]})


if __name__ == "__main__":
    app.run(host=os.environ.get('LIVENESS_HOST', '127.0.0.1'), port = os.environ.get('LIVENESS_PORT', 8080), debug=True)
