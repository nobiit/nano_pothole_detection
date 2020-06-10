#!/usr/bin/env python3


# noinspection SpellCheckingInspection
from random import randint


# noinspection SpellCheckingInspection
class NanoPotholeDetection(object):
    DATA_DIR = 'data'
    TEST_DIR = 'tests'
    _classifier = None

    def __init__(self, scale=4):
        from os import environ
        self.scale = scale
        environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        environ['CUDA_VISIBLE_DEVICES'] = str(0)
        environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)

    @property
    def classifier(self):
        if not self._classifier:
            from frontend import YOLO
            self._classifier = YOLO(
                backend='Full Yolo',
                input_size=416,
                anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                max_box_per_image=10,
                labels=['Pothole'],
            )
            self._classifier.load_weights('%s/classifier.h5' % NanoPotholeDetection.DATA_DIR)
        return self._classifier

    @staticmethod
    def clean_classifier():
        from os import remove
        remove('%s/classifier.h5' % NanoPotholeDetection.DATA_DIR)

    def run(self, test=False, webcam=False):
        print('Khởi động Nano PotholeDetection !')
        try:
            if test:
                self.processing_test(self.classifier)
            if webcam:
                self.processing_webcam(self.classifier)
        except KeyboardInterrupt:
            pass

    @staticmethod
    def draw_info(image, boxes):
        from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX
        image_h, image_w, _ = image.shape

        index = 0
        for box in boxes:
            index += 1
            xmin = int(box.xmin * image_w)
            ymin = int(box.ymin * image_h)
            xmax = int(box.xmax * image_w)
            ymax = int(box.ymax * image_h)
            color = (randint(0, 255), randint(0, 255), randint(0, 255))

            text = [
                'Pothole',
                '%d/%d' % (index, len(boxes)),
                str(int(box.get_score() * 10000) / 100) + '%',
                # str(int(((xmax - xmin) * (ymax - ymin) * 100) / (image_w * image_h))) + '%'
            ]

            rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
            putText(
                image,
                ' '.join(text),
                (xmin, ymin - 13),
                FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

    def processing_test(self, classifier):
        from os import environ
        from utils import image_files_in_folder
        from cv2 import imread, imshow, waitKey
        try:
            images = list(filter(
                lambda image_path: image_path.startswith('test') and image_path.endswith('.jpg'),
                image_files_in_folder(self.TEST_DIR)
            ))
        except FileNotFoundError:
            return
        images.sort()
        for path in images:
            print('Tìm thấy ảnh %s - Đang thực hiện phân tích ...' % path)
            image = imread(path)
            boxes = classifier.predict(image)
            if len(boxes) > 0:
                print('Phát hiện %s ổ gà trong ảnh %s' % (len(boxes), path))
                self.draw_info(image, boxes)
            else:
                print('Không phát hiện ổ gà trong ảnh %s' % path)
            if environ.get('DISPLAY'):
                imshow('Image ' + path, image)
                waitKey(1)
        while True:
            if waitKey(1) & 0xFF == ord('q'):
                break

    def processing_webcam(self, classifier):
        from os import environ
        from cv2 import VideoCapture, resize, imshow, waitKey
        video_capture = VideoCapture(0)
        is_empting = False
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            _frame = resize(frame, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
            _frame = _frame[:, :, ::-1]
            boxes = classifier.predict(_frame)
            if len(boxes) > 0:
                print('Phát hiện %s ổ gà trong ảnh' % len(boxes))
                self.draw_info(frame, boxes)
                is_empting = False
            else:
                if not is_empting:
                    print('Không phát hiện ổ gà trong ảnh')
                is_empting = True
            if environ.get('DISPLAY'):
                imshow('Video', frame)
                waitKey(1)

    def __str__(self):
        return 'NanoPotholeDetection'


# noinspection SpellCheckingInspection
if __name__ == '__main__':
    from sys import argv

    nano = NanoPotholeDetection()
    # if len(argv) == 1:
    #     exit(nano.run(test=True, webcam=True))
    if len(argv) == 2:
        command = argv.copy().pop()
        # if command == 'reset':
        #     exit(nano.clean_classifier())
        if command == 'test':
            exit(nano.run(test=True))
        # noinspection SpellCheckingInspection
        if command == 'webcam':
            exit(nano.run(webcam=True))
    # noinspection SpellCheckingInspection
    print('Lệnh không hợp lệ !')
    # noinspection SpellCheckingInspection
    print('Cấu trúc: run-pothole-detection [test|webcam]')
