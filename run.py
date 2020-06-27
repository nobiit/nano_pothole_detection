#!/usr/bin/env python3
# noinspection SpellCheckingInspection
from random import randint


# noinspection SpellCheckingInspection
class NanoPotholeDetection(object):
    # Thư mục chứa dữ liệu
    DATA_DIR = 'data'
    # Thư mục chứa các file test
    TEST_DIR = 'tests'
    # File test
    TEST_FILE = TEST_DIR + '/test.jpg'
    # Model nhận dạng
    _classifier = None

    # Khởi tạo NanoPotholeDetection
    # - slace: Hệ số thu nhỏ (mặc định giảm ảnh đi 4 lần)
    def __init__(self, scale=4):
        from os import environ
        self.scale = scale
        # Thiết lập cho Nano
        environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        environ['CUDA_VISIBLE_DEVICES'] = str(0)
        # Tắt bớt log của Tensoflow
        environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)

    # Lấy model nhận diện hiện tại hoặc khởi tạo
    @property
    def classifier(self):
        if not self._classifier:
            from frontend import YOLO
            # Khởi tạo Yolo
            self._classifier = YOLO(
                backend='Full Yolo',
                input_size=416,
                anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
                max_box_per_image=10,
                labels=['Pothole'],
            )
            # Load dữ liệu cho model nhận diện
            self._classifier.load_weights('%s/classifier.h5' % NanoPotholeDetection.DATA_DIR)
        return self._classifier

    # Xoá dữ liệu
    @staticmethod
    def clean_classifier():
        from os import remove
        remove('%s/classifier.h5' % NanoPotholeDetection.DATA_DIR)

    # Chạy
    def run(self, test=False, webcam=False, only=False):
        print('Khởi động Nano PotholeDetection !')
        try:
            if test:
                self.processing_test(self.classifier, only=only)
            if webcam:
                self.processing_webcam(self.classifier)
        except KeyboardInterrupt:
            pass

    # Vẽ các ô vuông vào hình nơi tìm thấy ổ gà
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
            # Lấy ngẫu nhiên màu
            color = (randint(0, 255), randint(0, 255), randint(0, 255))

            # Nội dung sẽ được viết lên trên các ô vuông
            text = [
                'Pothole',
                '%d/%d' % (index, len(boxes)),
                str(int(box.get_score() * 10000) / 100) + '%',
                # str(int(((xmax - xmin) * (ymax - ymin) * 100) / (image_w * image_h))) + '%'
            ]

            # Vẽ ô vuông
            rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
            # Vẽ chữ
            putText(
                image,
                ' '.join(text),
                (xmin, ymin - 13),
                FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

    # Chạy các hình ảnh trong thư mục test
    def processing_test(self, classifier, only=False):
        from os import environ
        from utils import image_files_in_folder
        from cv2 import imread, imshow, waitKey
        try:
            if only:
                images = [self.TEST_FILE]
            else:
                images = list(filter(
                    lambda image_path: image_path.startswith('test') and image_path.endswith('.jpg'),
                    image_files_in_folder(self.TEST_DIR)
                ))
        except FileNotFoundError:
            return
        images.sort()
        for path in images:
            print('Tìm thấy ảnh %s - Đang thực hiện phân tích ...' % path)
            # Đọc ảnh
            image = imread(path)
            # Phân tích ảnh
            boxes = classifier.predict(image)
            if len(boxes) > 0:
                print('Phát hiện %s ổ gà trong ảnh %s' % (len(boxes), path))
                self.draw_info(image, boxes)
            else:
                print('Không phát hiện ổ gà trong ảnh %s' % path)
            # Nếu hỗ trợ hiển thị thì hiện ảnh
            if environ.get('DISPLAY'):
                imshow('Image ' + path, image)
                waitKey(1)
        # Giữ chương trình không bị đóng
        while True:
            if waitKey(1) & 0xFF == ord('q'):
                break

    def processing_webcam(self, classifier):
        from os import environ
        from cv2 import VideoCapture, resize, imshow, waitKey
        # Khởi tạo hàm đọc Camera
        video_capture = VideoCapture(0)
        # Đánh dấu trạng thái
        is_empting = False
        # Nếu Camera đang mở
        while video_capture.isOpened():
            # Đọc ảnh từ Camera
            ret, frame = video_capture.read()
            # Giảm ảnh đi scale lần
            _frame = resize(frame, (0, 0), fx=1 / self.scale, fy=1 / self.scale)
            # Đảo ngược màu của ảnh (do ảnh chụp là RGB mà OpenCV dùng BGR)
            _frame = _frame[:, :, ::-1]
            # Phân tích ảnh
            boxes = classifier.predict(_frame)
            if len(boxes) > 0:
                print('Phát hiện %s ổ gà trong ảnh' % len(boxes))
                # Vẽ lên ảnh
                self.draw_info(frame, boxes)
                # Đánh dấu là đang tìm thấy ổ gà
                is_empting = False
            else:
                # Nếu trạng thái trước đó là không có ổ gà thì thông báo
                # -> Sẽ không thông báo liên tiếp nếu không có ổ gà
                if not is_empting:
                    print('Không phát hiện ổ gà trong ảnh')
                # Đánh dấu là đã tìm thấy ổ gà
                is_empting = True
            # Nếu hỗ trợ hiển thị thì hiện ảnh
            if environ.get('DISPLAY'):
                imshow('Video', frame)
                waitKey(1)

    # Tên class
    def __str__(self):
        return 'NanoPotholeDetection'


# Nếu file này đang được chạy trực tiếp
# noinspection SpellCheckingInspection
if __name__ == '__main__':
    from sys import argv

    # Khởi tạo
    nano = NanoPotholeDetection()
    # if len(argv) == 1:
    #     exit(nano.run(test=True, webcam=True))
    # Nếu có tham số
    if len(argv) == 2:
        # Lấy tham sô
        command = argv.copy().pop()
        # Nếu tham số là reset
        # if command == 'reset':
        #     exit(nano.clean_classifier())
        # Nếu tham số là test
        if command == 'test':
            exit(nano.run(test=True))
        # Nếu tham số là webcam
        # noinspection SpellCheckingInspection
        if command == 'webcam':
            exit(nano.run(webcam=True))
        # Nếu là 1 link web
        if command.startswith('http://') or command.startswith('https://'):
            from requests import get

            # Tải file
            res = get(command)
            # Mở file test
            with open(self.TEST_FILE, 'wb') as f:
                # Lưu file vào file test
                f.write(res.content)
                # Đóng file
                f.close()

            # Chạy test
            exit(nano.run(test=True, only=True))
    # Ngược lại nếu không có tham số
    # noinspection SpellCheckingInspection
    print('Lệnh không hợp lệ !')
    # noinspection SpellCheckingInspection
    print('Cấu trúc: run-pothole-detection [test|webcam|<url>]')
