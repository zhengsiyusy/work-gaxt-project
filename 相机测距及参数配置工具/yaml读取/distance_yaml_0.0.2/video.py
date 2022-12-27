import cv2
import threading


class RecordingThread(threading.Thread):

    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True
        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        self.out = cv2.VideoWriter('./video001.avi', fourcc, 20.0, (640, 480))


    def start(self):  
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                return frame
            #     self.out.write(frame)
        self.out.release()
    
    def write(self,frame):
        if frame is not None:
            self.out.write(frame)


    def stop(self):
        self.isRunning = False


    def __del__(self):
        self.out.release()


class VideoCamera(object):

    def __init__(self):
        self.cap = cv2.VideoCapture(8)
        self.is_record = False
        self.out = None
        self.recordingThread = None


    def __del__(self):
        self.cap.release()


    def close(self):
        if self.cap.isOpened():
            self.cap.release()


    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        frame = self.recordingThread.start()
        return frame

    def write_frame(self,frame):
        self.is_record = True
        # self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.write(frame)

    def stop_record(self):
        self.is_record = False
        if self.recordingThread != None:
            self.recordingThread.stop()



# import time

# camera = VideoCamera()

# camera.start_record()

# time.sleep(10)

# camera.stop_record() 