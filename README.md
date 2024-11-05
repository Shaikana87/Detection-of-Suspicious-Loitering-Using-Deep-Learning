# 🚨 Detection of Suspicious Loitering Using Deep Learning 👀

## 👥 Team Members
- **Shaik Anas** - (20N31A12E3)
- **Shaik Mohammed Faiyaz Khan** - (20N31A12E4)
- **Yata Goutham** - (20N31A12H1)

### 🎓 Guide
- **Mr. Munnangi Vazralu**, Associate Professor, MRCET - IT Dept

---

## 📅 Project Outline
- **Introduction**
- **Methodology**
- **Implementation**
- **Conclusion**
- **References**

---

## 🌟 Project Highlights

Explore key features and snapshots of our project in action:

1. **User Login Interface** 🗝️ - Secure login portal for verified users.  
   ![User Login](images/login.png)

2. **Registration Module** ✍️ - Streamlined registration for new users.  
   ![User Registration](images/registration.png)

3. **Detection Model Selection** 📷 - Choose between standard YOLO and YOLO Tiny for optimized detection.  
   ![Model Selection](images/select-yolo.png)

4. **Video Feed Integration** 🎥 - Upload video files or connect a live camera feed for real-time analysis.  
   ![Video Feed](images/browse-video.png)

5. **Real-Time Detection** 🕵️‍♂️ - Identify and monitor suspicious loitering behavior with dynamic bounding boxes.  
   ![Real-Time Detection](images/detection.png)

6. **Automated Email Alerts** 📧 - Receive instant alerts for any detected suspicious activity.  
   ![Email Alert](images/email-alert.png)

---

## 📜 Introduction
**Project Description:**  
This project leverages **YOLOv5** for real-time loitering detection, aiming to enhance public security through automated surveillance. It provides a scalable solution for monitoring public spaces, identifying unusual behaviors, and triggering timely alerts.

### Key Features
- **Smart Surveillance Solution**: Proactively detect loitering in public areas.
- **YOLOv5 Integration**: Accurate, real-time object detection.
- **Scalability**: Suitable for various surveillance environments.
- **Customizable Monitoring**: Adjustable detection thresholds for different needs.

---

## 💻 Software Requirements
- **Operating System**: Windows (7, 8, 10, 11)
- **Programming Language**: Python
- **Front-end**: HTML, CSS, JavaScript
- **Back-end**: Firebase

## ⚙️ Hardware Requirements
- **Processor**: Minimum 1 GHz; Recommended 2 GHz or more
- **Network**: Ethernet connection (LAN) or Wi-Fi
- **Hard Drive**: Minimum 32 GB; Recommended 64 GB or more
- **Memory (RAM)**: Minimum 1 GB; Recommended 4 GB or above

---

## 🔧 Software Requirement Specification

### Functional Requirements
1. **Object Detection**: Identify objects in video feeds.
2. **YOLOv5 Integration**: Real-time object recognition.
3. **Tracking Objects**: Continuous tracking across frames.
4. **Loitering Identification**: Detect suspicious loitering activity.
5. **Alert Triggering**: Send notifications when loitering is detected.

### Nonfunctional Requirements
1. **Performance Metrics**: Real-time accuracy.
2. **Parameter Tuning**: Adjustable thresholds for detection.
3. **Default Duration Settings**: Pre-set times for loitering detection.
4. **Cost-effectiveness**: Efficient on standard hardware.
5. **Scalability**: Adapts to multiple environments.

---

## 🔍 Existing System vs Proposed System

| **Existing System**            | **Proposed System**           |
|--------------------------------|-------------------------------|
| Basic Surveillance Monitoring   | Advanced Loitering Detection  |
| Limited Object Detection        | YOLOv5-Powered Detection      |
| Lacks Duration-Based Alerts     | Time-Based Suspicion Detection|
| Basic Video Analytics           | Multi-task Training Integration|

---

## 🛠️ Methodology
![Architecture Diagram](images/architecture-diagram.png)

### Key Libraries Used:
- **YOLOv5**: For efficient object detection.
- **Kalman Filtering**: Maintains object tracking across frames.
- **OpenCV**: Processes video feeds and images.
- **Yagmail**: Automates email alerting.
- **NumPy**: Efficiently handles numerical data.

---

## 🔄 Procedure

Here’s how our system operates step-by-step:

1. **User Login** 🗝️: Existing users securely log in to access features.  
2. **User Registration** ✍️: New users register with an easy sign-up process.  
3. **Select Detection Model** 📷:
   - Choose between YOLO or YOLO Tiny models for optimized detection.
4. **Upload Video or Connect Camera** 🎥:
   - Upload a video or connect to a camera for real-time analysis.
5. **Detection and Tracking** 🕵️‍♂️:
   - The system detects and monitors loitering behavior, marking individuals with bounding boxes.
6. **Alerting** 📧:
   - Alerts are sent to registered emails if suspicious loitering is detected.

---

### 📝 Instructions to Run
To start the project, use this command:
```bash
python start.py
