# Automated-Vision-Based-Attendance-Monitoring-System
Problem Statement -> Manual attendance marking in classrooms is time-consuming, inefficient, and prone to errors or proxy attendance. With the growing class sizes and need for accurate monitoring, an automated solution is required.
This project aims to develop a camera-based attendance system using computer vision and deep learning that can automatically detect and recognize students’ faces in real time without any manual intervention. The system will capture video from a camera, identify registered students using facial recognition models, and record attendance with timestamps in a digital format.
By leveraging Python, OpenCV, and pre-trained deep learning models such as FaceNet or VGG-Face, the proposed system ensures improved accuracy, reduced human effort, and reliable attendance tracking. The project demonstrates the practical application of Artificial Intelligence and Machine Learning in automating routine academic processes.

Goal
Build a simple, efficient attendance system that works on mobile phones and stores attendance digitally.

Recommended approach
Use QR code based attendance as the main system. Face recognition can be added later as an optional improvement.

System flow
	1.	Teacher generates a QR code for the class.
	2.	Students scan the QR code using the mobile app.
	3.	The app sends attendance data to the server.
	4.	Server stores attendance in the database.

Tools to use
	1.	Mobile app development
Use Flutter so one app works on Android and iOS.

Useful Flutter packages
qr_flutter for generating QR codes
mobile_scanner for scanning QR codes
http for backend communication
	2.	Backend development
Use Django with Django REST Framework.

Backend tools
Django
Django REST Framework
django cors headers
	3.	Database
Use SQLite for simplicity and easy setup.
Can upgrade to PostgreSQL later if needed.

QR code logic
Teacher generates a QR containing class id, date, time, and a random token.
Student scans and sends student id and token to backend.
Backend verifies token and marks attendance.
QR expires after about 30 to 60 seconds to prevent misuse.

Optional future feature
Face recognition using OpenCV and Python can be added later. Capture student face at registration and compare during attendance.

Final technology stack
Mobile app Flutter
Backend Django and Django REST Framework
Database SQLite
QR generation qr_flutter
QR scanning mobile_scanner
Optional face recognition OpenCV

Security basics
QR code expires quickly.
Each student can mark attendance only once per class.
Students and teachers log in before using the system.

Minimum features to build
Teacher
Login
Generate QR code
View attendance

Student
Login
Scan QR code
See attendance confirmation

Admin
View all attendance records
Export attendance data

Development roadmap
Week 1
Setup Django backend and database models.

Week 2
Build Flutter login and QR scanning. Connect backend.

Week 3
Add QR expiration and admin dashboard. Testing.

Week 4 optional
Add face recognition prototype.
