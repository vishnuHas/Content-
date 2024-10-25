from flask import Flask, render_template, Response
import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory
app = Flask(__name__)

# Model files paths
faceProto = r"opencv_face_detector.pbtxt"
faceModel = r"opencv_face_detector_uint8.pb"
ageProto = r"age_deploy.prototxt"
ageModel = r"age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)','(0-4)','(8-12)','(15-20)','(20-25)','(25-30)','(30-40)','(40-50)','(60-80)','(80-100)']

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (255, 255, 255), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

def generate_frames():
    video = cv2.VideoCapture(0)
    padding = 20

    while True:
        success, frame = video.read()
        if not success:
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if faceBoxes:
            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                             max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]
                
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                cv2.putText(resultImg, f'Age: {age[1:-1]} years', (faceBox[0], faceBox[1] - 10), cv2.FONT_ITALIC , 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', resultImg)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return send_file('web/index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)
if __name__ == "__main__":
    app.run(debug=True)
