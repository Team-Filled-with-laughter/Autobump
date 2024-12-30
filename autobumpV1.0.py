import argparse
import time
import signal
import sys
from flask import Flask, Response, request, jsonify
from flask_cors import CORS  # CORS를 import 합니다
from pathlib import Path
import PulseIn
import serial
import cv2
import torch
from numpy import random
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import Jetson.GPIO as GPIO
import threading

app = Flask(__name__)

# React 앱의 출처만 허용하고, credentials: 'include'도 허용
CORS(app, supports_credentials=False)

# GPIO 핀 설정
sigPin1 = 21
speed_pin = 33
In1 = 23
In2 = 24

# GPIO 모드 설정
GPIO.setmode(GPIO.BOARD)
GPIO.setup(sigPin1, GPIO.IN)
GPIO.setup(speed_pin, GPIO.OUT)
GPIO.setup(In1, GPIO.OUT)
GPIO.setup(In2, GPIO.OUT)

# 초기값 설정
overSpeed = 8
isOverSpeed = False
isEmergency = False
listSize = 20
listA = [0 for i in range(listSize)]

# 시리얼 포트 설정 (적절한 포트를 확인하세요)
arduino = serial.Serial('/dev/ttyACM0', 9600)  # 아두이노가 연결된 포트로 변경하세요
time.sleep(2)  # 시리얼 포트 안정화 대기


def read_arduino():  # actuatorPlay()에 speed = read_arduino() 추가
    if arduino.in_waiting > 0:  # 4바이트가 수신될 때까지 대기
        # 4바이트 데이터를 읽음
        data = arduino.readline().decode('utf-8')
        print(f"Raw data received: {data}")
        data = data.strip()

        try:
            speed = float(data)
            return speed
        except ValueError:
            print("Received invalid data, could not convert to int")
    else:
        return 0


def isListFull(listType):
    if listType[0] >= len(listType) - 1:
        return True
    else:
        return False


def clear(listType):
    for i in range(len(listType) - 1):
        listType[i + 1] = 0
    listType[0] = 1


def putList(val, listType):
    if isListFull(listType):
        clear(listType)
        putList(val, listType)
    else:
        listType[listType[0]] = val
        listType[0] += 1


def getMax(listType):
    max = 0
    for i in range(listType[0] - 1):
        if (listType[i + 1] > max):
            max = listType[i]
    return max


# 액추에이터 상승
def actuatorUp():
    GPIO.output(In1, GPIO.LOW)
    GPIO.output(In2, GPIO.HIGH)
    GPIO.output(speed_pin, GPIO.HIGH)


# 액추에이터 하강
def actuatorDown():
    GPIO.output(In1, GPIO.HIGH)
    GPIO.output(In2, GPIO.LOW)
    GPIO.output(speed_pin, GPIO.HIGH)


def actuatorPlay():
    global listA, isOverSpeed, overSpeed, isEmergency
    while 1:
        v1 = getMax(listA)
        speed = read_arduino()
        isOverSpeed = False
        if (speed > 0):
            speed2 = int(speed)
            data = v1 - speed2
            if(data!=0):
                data = data/2
                v1 = v1 - data
            if v1>overSpeed and speed2 < overSpeed:
                v1 = 4.3
            isOverSpeed = (speed >= overSpeed) or (v1 >= overSpeed)
            yield f"data: {{\"v1\": {v1}, \"v2\": {speed}}}\n\n".encode('utf-8')
            print('isOverSpeed: ', isOverSpeed)
            print('speed :', speed)
            print('v1 :', v1)
            clear(listA)
        if isOverSpeed:
            if not isEmergency:
                actuatorDown()
                print('Down success')
                time.sleep(3)
                actuatorUp()
                print('Up success')
                time.sleep(3)
                isOverSpeed = False
                isEmergency = False
            else:
                print('detect emergency')
        time.sleep(0.5)


def speedCheck1():
    while 1:
        v1 = 0
        while GPIO.input(sigPin1):
            pass
        while not GPIO.input(sigPin1):
            pass
        duration = PulseIn.pulse_in(sigPin1, GPIO.HIGH) + PulseIn.pulse_in(sigPin1, GPIO.LOW)

        if duration != 0:
            frequency = 1.0 / duration
            v1 = int((frequency * 1e5) / 90.0)
            if v1 > 15:
                 v1 = 0
            putList(v1, listA)
        else:
            v1 = 0
        time.sleep(0.5)


####################################################
def apply_clahe(image):
    """Apply CLAHE to the input image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


####################################################




def detect(save_img=False):
    global isEmergency
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

  

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        #################################
        if isinstance(im0s, list):
            im0s = [apply_clahe(im0) for im0 in im0s]  # 리스트인 경우 각 프레임에 CLAHE 적용
        else:
            im0s = apply_clahe(im0s)  # 단일 이미지인 경우 바로 CLAHE 적용
        #################################
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    ################################################################################
                  # send signal
                    if names[int(c)] == "Ambulance" or names[int(c)] == "Police":
                        isEmergency = True  # catch am pol
                        print('emergency change')
                    #################################################################################

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            ######################################################
            else:
                isEmergency = False  # catch null
                print('emergency is false')
            ######################################################

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            yield im0

    print(f'Done. ({time.time() - t0:.3f}s)')


def generate_frames():
    frame = detect()

    for dectection_ in frame:
        ret, buffer = cv2.imencode('.jpg', dectection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/speed_feed')
def speed_feed():
    speed_check_thread = threading.Thread(target=speedCheck1)
    speed_check_thread.daemon = True  # 이 옵션을 설정하면 메인 프로세스가 종료될 때 해당 스레드도 종료됩니다.
    speed_check_thread.start()
    return Response(actuatorPlay(), content_type='text/event-stream')  # 이렇게하면 web실행할때만 될듯?

@app.route('/update_threshold', methods=['POST', 'OPTIONS'])
def update_threshold():
    # CORS preflight 요청 처리
    if request.method == 'OPTIONS':
        response = Response()
        origin = request.headers.get('Origin')
        allowed_origins = ['http://ec2-13-209-45-118.ap-northeast-2.compute.amazonaws.com:8080' ]  # List of allowed origins

        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            response.headers['Access-Control-Allow-Origin'] = '*'

        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'  # 허용할 메서드
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # 허용할 헤더
        response.headers['Access-Control-Allow-Credentials'] = 'False'  # 쿠키 포함 허용
        return response, 200  # 200 OK 응답

    # POST 요청 처리
    data = request.get_json()  # JSON 데이터 받기
    threshold = data.get('threshold')

    if threshold is not None:
        global overSpeed
        overSpeed = threshold  # threshold 값으로 overSpeed 업데이트
        print(f"overSpeed 값이 {threshold}로 변경되었습니다.")  # 변경된 값 출력
        return jsonify({"message": "임계값 업데이트 성공", "threshold": overSpeed}), 200
    else:
        return jsonify({"error": "임계값을 찾을 수 없습니다."}), 400

@app.route('/update_actuator', methods=['POST', 'OPTIONS'])
def update_actuator():
    # CORS preflight 요청 처리
    if request.method == 'OPTIONS':
        response = Response()
        origin = request.headers.get('Origin')
        allowed_origins = ['http://ec2-13-209-45-118.ap-northeast-2.compute.amazonaws.com:8080']  # List of allowed origins

        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'  # 허용할 메서드
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # 허용할 헤더
        response.headers['Access-Control-Allow-Credentials'] = 'False'  # 쿠키 포함 허용
        return response, 200  # 200 OK 응답

    # POST 요청 처리
    data = request.get_json()  # JSON 데이터 받기
    up = data.get('up')

    if up is not None:
        if up==1:
            actuatorUp()
            print('Up success')

            return jsonify({"message": "엑추에이터 상승"}), 200
        else:
            actuatorDown()
            print('Down success')
            return jsonify({"message": "엑추에이터 하강"}), 200
    else:
        return jsonify({"error": "엑추에이터 버튼 오류."}), 400

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csi-cam', action='store_true', help='CSI camera (True) or USB camera (False, default)')
    parser.add_argument('--weights', nargs='+', type=str, default='weights/v5lite-s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='sample', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    opt = parser.parse_args()
    print(opt)
    clear(listA)


    def cleanup():
        GPIO.cleanup()
        print("GPIO cleaned up and program exited.")
        sys.exit(0)


    signal.signal(signal.SIGINT, lambda s, f: cleanup())

    try:
        app.run(host='0.0.0.0', threaded=True, port=5001)  # React 앱과 포트 충돌 피하려면 다른 포트 사용
    except Exception as e:
        print(f"An error occurred: {e}")
        cleanup()
