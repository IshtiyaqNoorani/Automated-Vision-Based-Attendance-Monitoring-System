from backend.insightface_engine import engine

class AttendanceSystem:

    def recognize_frame(self, frame):

        return engine.recognize(frame)


system = AttendanceSystem()
