#!/usr/bin/env python

import cv2
import numpy as np
import pytesseract

from module import APA_Status
from module.ROI import ROI

PERPENDICULAR_LEFT = 1
PERPENDICULAR_RIGHT = 2
PARALLEL_LEFT = 3
PARALLEL_RIGHT = 4


class APA_Recognizer():
    config = None
    checkStatus = False
    input_string = ''
    input_list = []
    arrival_list = [1, 2, 3, 4, 9, 13, 17, 18, 19, 22, 27, 28, 35, 36, 37, 38, 39, 40, 41, 42]
    steer_list = [6, 7, 8, 10, 11, 12, 14, 15, 16, 20, 21, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34]
    apa_status_dict = APA_Status.get_APA_status()
    status_str_dict = APA_Status.get_APA_status_string()
    result_list = []
    result_dict = {}
    index = 0
    upper_text_roi = ''
    bottom_text_roi = ''
    first_para_perp = ''
    second_para_perp = ''
    third_para_perp = ''
    forth_para_perp = ''
    rate_para_perp = 0
    limit_drive_speed = ''

    orange_hsv_low = ''
    orange_hsv_high = ''

    def __init__(self, config=None):
        self.config = config
        if self.config is not None:
            self.upper_text_roi = config.get('apa-config', 'upper_text_image')
            self.bottom_text_roi = config.get('apa-config', 'bottom_text_image')
            self.first_para_perp = config.get('apa-config', 'first_para_perp')
            self.second_para_perp = config.get('apa-config', 'second_para_perp')
            self.third_para_perp = config.get('apa-config', 'third_para_perp')
            self.forth_para_perp = config.get('apa-config', 'forth_para_perp')
            self.limit_drive_speed = config.get('apa-config', 'limit_drive_speed')
            self.rate_para_perp = config.getfloat('apa-config', 'rate_para_perp')
            self.orange_hsv_low = config.get('apa-config', 'orange_hsv_low')
            self.orange_hsv_high = config.get('apa-config', 'orange_hsv_high')

    def thresholdHSV(self, orig_image, low, high):
        lower_range = np.array(low, dtype=np.uint8)
        upper_range = np.array(high, dtype=np.uint8)
        hsv = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_range, upper_range)
        return mask

    def check_progress_bar(self, image):
        lower_orange = np.array([11, 43, 46])
        upper_orange = np.array([77, 255, 255])  # orange and green
        image_roi_rc = self.thresholdHSV(image, lower_orange, upper_orange)
        nz = np.count_nonzero(image_roi_rc)
        sz = image_roi_rc.size
        print(nz * 1.0 / sz)

    def set_roi_image(self, image, rio_name, roi_xy):
        roi = ROI(rio_name, roi_xy)
        roi_image = image[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
        return roi_image

    def check_parallel_perpendicular(self, image):
        image_roi_rc1 = self.set_roi_image(image, "first", self.first_para_perp)
        image_roi_rc2 = self.set_roi_image(image, "second", self.second_para_perp)
        image_roi_rc3 = self.set_roi_image(image, "third", self.third_para_perp)
        image_roi_rc4 = self.set_roi_image(image, "forth", self.forth_para_perp)

        sl = self.orange_hsv_low.split(',')
        bg_hsv_low = list(map(int, sl))
        sh = self.orange_hsv_high.split(',')
        bg_hsv_high = list(map(int, sh))
        lower_orange = np.array(bg_hsv_low)
        upper_orange = np.array(bg_hsv_high)  # orange
        image_roi_rc1 = self.thresholdHSV(image_roi_rc1, lower_orange, upper_orange)
        image_roi_rc2 = self.thresholdHSV(image_roi_rc2, lower_orange, upper_orange)
        image_roi_rc3 = self.thresholdHSV(image_roi_rc3, lower_orange, upper_orange)
        image_roi_rc4 = self.thresholdHSV(image_roi_rc4, lower_orange, upper_orange)
        nz1 = np.count_nonzero(image_roi_rc1)
        nz2 = np.count_nonzero(image_roi_rc2)
        nz3 = np.count_nonzero(image_roi_rc3)
        nz4 = np.count_nonzero(image_roi_rc4)
        sz1 = image_roi_rc1.size
        sz2 = image_roi_rc2.size
        sz3 = image_roi_rc3.size
        sz4 = image_roi_rc4.size
        list_parking = [nz1 * 1.0 / sz1, nz2 * 1.0 / sz2, nz3 * 1.0 / sz3, nz4 * 1.0 / sz4]
        if list_parking[0] > self.rate_para_perp and list_parking[2] > self.rate_para_perp:
            return PARALLEL_RIGHT
        elif list_parking[0] > self.rate_para_perp and list_parking[3] > self.rate_para_perp:
            return PARALLEL_LEFT
        elif list_parking[1] > self.rate_para_perp and list_parking[2] > self.rate_para_perp:
            return PERPENDICULAR_RIGHT
        elif list_parking[1] > self.rate_para_perp and list_parking[3] > self.rate_para_perp:
            return PERPENDICULAR_LEFT

    def check_status_arrival_name(self, text, image):
        if self.input_list[self.index] in [2, 4, 18]:
            if 'Searching' in text and 'parallel' in text:
                if 'on right' not in text:
                    self.check_next()
            elif 'Searching' in text and 'perpendicular' in text:
                if 'on right' not in text:
                    self.check_next()
        elif self.status_str_dict.get(self.input_list[self.index]) in text:
            if self.input_list[self.index] in [9, 35, 39]:
                if self.check_parallel_perpendicular(image) == PARALLEL_RIGHT:
                    self.check_next()
            elif self.input_list[self.index] in [13, 36, 40]:
                if self.check_parallel_perpendicular(image) == PARALLEL_LEFT:
                    self.check_next()
            elif self.input_list[self.index] in [19, 37, 41]:
                if self.check_parallel_perpendicular(image) == PERPENDICULAR_RIGHT:
                    self.check_next()
            elif self.input_list[self.index] in [22, 38, 42]:
                if self.check_parallel_perpendicular(image) == PERPENDICULAR_LEFT:
                    self.check_next()
            else:
                self.check_next()

    def check_next(self):
        self.result_list.append(self.apa_status_dict.get(self.input_list[self.index]))
        print(self.apa_status_dict.get(self.input_list[self.index]))
        self.index += 1

    def check_image_to_str(self, image):
        text = pytesseract.image_to_string(image)
        status = text.replace('\n', ' ')
        return status

    def set_check_points(self, input_status):
        self.input_string = input_status.strip().split(';')
        for s in self.input_string:
            if s != "":
                for key, value in self.apa_status_dict.items():
                    if s.strip() == value:
                        self.input_list.append(key)
                if s.strip() not in self.apa_status_dict.values():
                    raise RuntimeError(s + ':Input error!!')

    def start_check(self):
        self.checkStatus = True

    def stop_check(self):
        self.checkStatus = False

    def get_result(self):
        result = ''
        for s in self.input_string:
            if s in self.result_list:
                result += s + ':True;'
            elif s != "":
                result += s + ':False;'
        self.result_list = []
        self.index = 0
        return result

    def handleFrame(self, image):
        if self.index == len(self.input_list):
            return self.result_list
        if not self.checkStatus:
            return
        upper_text_image = self.set_roi_image(image, "upper_text", self.upper_text_roi)
        bottom_text_image = self.set_roi_image(image, "bottom_text", self.bottom_text_roi)
        limit_speed_image = self.set_roi_image(image, "limit_drive_speed", self.limit_drive_speed)
        if self.input_list[self.index] in self.arrival_list:
            text = self.check_image_to_str(upper_text_image)
        elif self.input_list[self.index] in [5]:
            text = self.check_image_to_str(limit_speed_image)
        else:
            text = self.check_image_to_str(bottom_text_image)
        self.check_status_arrival_name(text, image)


if __name__ == "__main__":
    apa_recognizer = APA_Recognizer()
    videoCapture = cv2.VideoCapture("20190129180145775.avi")
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # apa_recognizer.set_check_points('6-AutoSteeringActivated;7-AutoSteeringActivatedREABOff;8-ParkingCompleted;10'
    #                                 '-ParallelParkingRightProgressForward;11-ParallelParkingProgressBackward;12'
    #                                 '-ParallelParkingRightStop;14-ParallelParkingLeftProgressForward;'
    #                                 '15-ParallelParkingProgressBackward')
    apa_recognizer.start_check()
    apa_recognizer.input_list = [6, 7, 8, 10, 11, 12, 14, 15, 16, 20, 21, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34]
    while True:
        # time.sleep(0.05)
        ret, frame = videoCapture.read()
        # cv2.imshow('frame', frame)
        # print check_status(frame)
        if ret:
            cv2.imshow('cap video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                videoCapture.release()
                cv2.destroyAllWindows()
                break
        else:
            break
        apa_recognizer.handleFrame(frame)
    print(apa_recognizer.get_result())
