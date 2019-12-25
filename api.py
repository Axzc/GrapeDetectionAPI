from flask import Flask, request, abort, jsonify

import os
import logging.config

import utils
import get3dpoint
import inference


if not os.path.exists("./logs"):
    os.mkdir("./logs")
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs/logconf.ini')
logging.config.fileConfig(log_path, defaults=None, disable_existing_loggers=True)
logger = logging.getLogger("log")

app = Flask(__name__)


@app.errorhandler(400)
def error_400(error):
    return jsonify({"status": "1x0000", "data": "", "message": "Incoming parameters are incorrect"})


@app.route('/get5GDispose1', methods=['POST'])
def detection():
    """
    Detect if there are targets
    :return: robotCode, ImageNumber, isGoal
    """

    robotCode = request.json.get('robotCode')
    temperature = request.json.get('temperature')
    carbonDioxide = request.json.get('carbonDioxide')
    humidity = request.json.get('humidity')
    beam = request.json.get('beam')
    Lat = request.json.get('Lat')
    Lon = request.json.get('Lat')
    ImagerNumber = request.json.get('ImageNumber')
    imageFilesLeft = request.json.get('imageFilesLeft')

    utils.verify_param(
        abort, logger, error_code=400, robotCode=robotCode, temperature=temperature, carbonDioxide=carbonDioxide,
        humidity=humidity, beam=beam, Lat=Lat, Lon=Lon, ImagerNumber=ImagerNumber, imageFilesLeft=imageFilesLeft,
        method_name='detection')

    cv2_image = utils.convert_to_cv2(abort, logger, imageFilesLeft)  # Convert to cv2 format
    # result = fd.predict(cv2_image)  # Get prediction results
    infer = inference.Inference()
    bboxes, probs, ratio = infer.pre_predict(cv2_image)

    # Determine if the image has a target
    if bboxes.values():
        isGoal = 1
    else:
        isGoal = 0

    return jsonify({
        'robotCode': robotCode,
        'ImageNumber': ImagerNumber,
        'isGoal': isGoal
    })


@app.route('/get5GDispose2', methods=['POST'])
def get_position():
    """
    Returns picking points based on target coordinates
    :return: robotCode, ImageNumber, isGoal, Pickingpoint
    """

    robotCode = request.json.get('robotCode')
    temperature = request.json.get('temperature')
    carbonDioxide = request.json.get('carbonDioxide')
    humidity = request.json.get('humidity')
    beam = request.json.get('beam')
    Lat = request.json.get('Lat')
    Lon = request.json.get('Lon')
    stereoImageNumber = request.json.get('stereoImageNumber')
    imageFilesLeft = request.json.get('imageFilesLeft')
    imageFilesRight = request.json.get('imageFilesRight')

    utils.verify_param(
        abort, logger, robotCode=robotCode, temperature=temperature, carbonDioxide=carbonDioxide,
        humidity=humidity, beam=beam, Lat=Lat, Lon=Lon, stereoImageNumber=stereoImageNumber, imageFilesLeft=imageFilesLeft,
        imageFilesRight=imageFilesRight, method_name='get_position')

    # Convert to cv2 format
    imageFilesLeft_cv2 = utils.convert_to_cv2(abort, logger, imageFilesLeft)
    imageFilesRight_cv2 = utils.convert_to_cv2(abort, logger, imageFilesRight)

    # Get predictions
    inter = inference.Inference()
    left_real_boxes = inter.post_predict(img=imageFilesLeft_cv2)
    print(left_real_boxes)

    right_real_boxes = inter.post_predict(img=imageFilesRight_cv2)

    # Determine if there is a goal
    if left_real_boxes is not None and right_real_boxes is not None:
        isGoal = 1
    else:
        isGoal = 0

    # Get centroid
    Left_DetectPoints, Right_DetectPoints = [], []
    for boxes in left_real_boxes:
        Left_DetectPoints.append([0.5 * (boxes[0] + boxes[2]), 0.5 * (boxes[1] + boxes[3]), 1])
    for boxes in right_real_boxes:
        Right_DetectPoints.append([0.5 * (boxes[0] + boxes[2]), 0.5 * (boxes[1] + boxes[3]), 1])

    # Calculate the picking point
    Position = get3dpoint.get3dpointwuliu(Left_DetectPoints, Right_DetectPoints)
    PickingPosition = []
    for p in Position:
        PickingPosition.append("Point3:" + str(p[0]) + "," + str(p[1]) + "," + str(p[2]) + "*")

    return jsonify({
        'robotCode': robotCode,
        'ImageNumber': stereoImageNumber,
        'isGoal': isGoal,
        'PickingPosition': PickingPosition
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
