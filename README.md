# 🏆 Football Analysis Project

## 🚀 Introduction
This project is a **computer vision and machine learning pipeline** designed to analyze football matches. By leveraging **YOLO for object detection, KMeans for color-based segmentation, Optical Flow for motion tracking, and Perspective Transformation for depth analysis**, we accurately detect and track **players, referees, and the football** in match footage.

🔹 **Track players, referees, and the ball** using YOLO 🏃‍♂️⚽👨‍⚖️  
🔹 **Assign teams** based on jersey color using KMeans 🎨  
🔹 **Measure ball possession percentage** per team 📊  
🔹 **Estimate camera motion** with Optical Flow 🎥  
🔹 **Map real-world movement** using Perspective Transformation 📏  
🔹 **Calculate player speed & distance covered** ⏱️  

This project is ideal for **sports analysts, coaches, and AI enthusiasts** looking to integrate **AI-driven insights into football analysis.**

---

## 📂 Project Structure

The project follows a **step-by-step pipeline** for organization and efficiency:

1️⃣ **Data Preparation**  
   📁 Organizing input videos and setting up training data  

2️⃣ **YOLO Inference & Model Training**  
   🎯 Running **yolo_inference.py** to detect objects and fine-tune YOLO with custom annotations  

3️⃣ **Tracking & Visual Enhancements**  
   🔄 Implementing tracking with bounding boxes, ellipses, and labels  

4️⃣ **Advanced Analysis**  
   📊 Estimating ball possession, player movement in meters, speed, and distance covered  

5️⃣ **Final Output**  
   🎥 Generating an annotated video with AI-powered insights  

---

## 🛠️ Technologies Used

The project integrates multiple AI and computer vision techniques:

✅ **YOLO** – Object detection for identifying players, referees, and the ball 🎯  
✅ **KMeans Clustering** – Segmentation of player jerseys for team classification 🎨  
✅ **Optical Flow** – Camera movement estimation to enhance tracking 🎥  
✅ **Perspective Transformation** – Mapping pixels to real-world meters 📏  
✅ **Tracking Algorithms** – Enhancing motion tracking and player identification 🔄  

---

## 🔍 Trained Models

This project uses a **custom-trained YOLO model** with a dataset containing labeled images of **players, referees, and the ball**.

---

## 📦 Installation & Requirements

To run this project, install the following dependencies:

```bash
pip install ultralytics supervision opencv-python numpy matplotlib pandas
```

✅ **Python 3.x**  
✅ **Ultralytics (YOLO framework)**  
✅ **Supervision** – Video processing  
✅ **OpenCV** – Image and video processing  
✅ **NumPy & Pandas** – Data handling  
✅ **Matplotlib** – Visualization  

---

