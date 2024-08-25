import tkinter as tk
from tkinter import messagebox
import smtplib
import pyttsx3
import os
import geocoder
# from googleapiclient.discovery import build
import pywhatkit as kit
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
# import pycaw.pycaw as pca

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tkinter import filedialog, messagebox

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


import streamlit as st
import pickle
import numpy as np

import tkinter as tk
import subprocess
import os
import cv2
import matplotlib.pyplot as plt

# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# import mysql.connector
# from linkedin_api import Linkedin
# import boto3


# Stub functions for each task

def send_email():
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        sender_email = "anaryadav123@gmail.com"
        sender_password = "xdfr ptss koiw wtyv"
        receiver_email = "1anjaliyadav5@gmail.com"
        subject = "Test Email"
        body = "This is a test email from Python Tkinter app."
        
        # Login to the email server
        server.login(sender_email, sender_password)
        
        # Prepare email
        email_message = f"Subject: {subject}\n\n{body}"
        server.sendmail(sender_email, receiver_email, email_message)
        server.quit()
        
        messagebox.showinfo("Send Email", "Email sent successfully!")
    except Exception as e:
        messagebox.showerror("Send Email", f"Failed to send email: {e}")
        
        
def send_sms():
    try:
        # Example using pywhatkit to send a WhatsApp message as SMS
        phone_number = "+919571747482"
        message = "This is a test SMS sent via Python!"
        kit.sendwhatmsg_instantly(phone_number, message)
        
        messagebox.showinfo("Send SMS", "SMS sent successfully!")
    except Exception as e:
        messagebox.showerror("Send SMS", f"Failed to send SMS: {e}")
        
        
def scrape_google():
    try:
        query = "Python programming"
        driver = webdriver.Chrome()
        driver.get("https://www.google.com")
        
        search_box = driver.find_element("name", "q")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        time.sleep(2)
        results = driver.find_elements("xpath", "//div[@class='g']//h3")
        
        top_5 = [result.text for result in results[:5]]
        # driver.quit()
        
        messagebox.showinfo("Google Search", "\n".join(top_5))
    except Exception as e:
        messagebox.showerror("Google Search", f"Failed to scrape Google: {e}")
    
    
def find_location():
    try:
        g = geocoder.ip('me')
        location_info = f"Coordinates: {g.latlng}\nAddress: {g.address}"
        messagebox.showinfo("Find Location", location_info)
    except Exception as e:
        messagebox.showerror("Find Location", f"Failed to get location: {e}")

def text_to_audio():
    
    try:
        text = text_input.get()  # Get the user input text
        if text:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            messagebox.showinfo("Text to Audio", "Audio played successfully!")
        else:
            messagebox.showwarning("Text to Audio", "Please enter text to convert to audio.")
    except Exception as e:
        messagebox.showerror("Text to Audio", f"Failed to convert text to audio: {e}")
def control_volume():
    try:
        volume_level = float(volume_input.get())  # Get the user input volume level
        if 0.0 <= volume_level <= 1.0:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(volume_level, None)
            messagebox.showinfo("Control Volume", f"Volume set to {int(volume_level * 100)}%!")
        else:
            messagebox.showwarning("Control Volume", "Please enter a volume level between 0.0 and 1.0.")
    except ValueError:
        messagebox.showwarning("Control Volume", "Please enter a valid number for volume.")
    except Exception as e:
        messagebox.showerror("Control Volume", f"Failed to control volume: {e}")

def connect_and_send_sms():
    try:
        # Example command to send SMS using adb
        phone_number = "+919571747482"  # Replace with the recipient's phone number
        sms_message = "This is a test SMS sent via adb from Python!"
        
        # adb command to send SMS
        adb_command = f'adb shell service call isms 7 i32 1 s16 "com.android.mms.service" s16 "{phone_number}" s16 "null" s16 "{sms_message}" s16 "null" s16 "null"'
        os.system(adb_command)
        
        messagebox.showinfo("Connect to Mobile", "SMS sent via mobile device!")
    except Exception as e:
        messagebox.showerror("Connect to Mobile", f"Failed to send SMS: {e}")

def send_bulk_email():
    try:
        sender_email = "anaryadav123@gmail.com"
        sender_password = "xdfr ptss koiw wtyv"
        subject = "Bulk Email Test"
        body = "This is a test bulk email sent from Python!"
        
        # List of recipients
        recipient_list = ["anaryadav123@gmail.com", "1anjaliyadav5@gmail.com"]  # Add as many as you want
        
        # Login to SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        
        for recipient in recipient_list:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server.sendmail(sender_email, recipient, msg.as_string())
        
        server.quit()
        
        messagebox.showinfo("Send Bulk Email", "Bulk emails sent successfully!")
    except Exception as e:
        messagebox.showerror("Send Bulk Email", f"Failed to send bulk emails: {e}")

def data_processing():
    file_path = filedialog.askopenfilename()
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            df.fillna(df.mean(), inplace=True)
            
            # Encoding categorical data
            label_encoders = {}
            for col in df.select_dtypes(include=['object']).columns:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col])
            
            # Normalizing numerical data
            scaler = StandardScaler()
            df[df.columns] = scaler.fit_transform(df)
            
            # Splitting the data
            train, test = train_test_split(df, test_size=0.2)
            
            messagebox.showinfo("Data Processing", "Dataset processed successfully!")
            return train, test
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process dataset: {e}")
    else:
        messagebox.showerror("Error", "Please select a valid CSV file.")
        
        
def model_integration():
    try:
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Save the model
        with open("model.pkl", "wb") as model_file:
            pickle.dump(model, model_file)
        
        messagebox.showinfo("Model Training", "Model trained and saved successfully!")
    except Exception as e:
        messagebox.showerror("Model Training Error", f"Error occurred while training: {e}")
        
def run_streamlit_app():
    try:
        # Use subprocess to run the Streamlit app
        subprocess.Popen(['streamlit', 'run', 'streamlit-web-app.py'])
    except Exception as e:
        print(f"Failed to launch Streamlit app: {e}")



def image_crop():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            cv2.imshow('Face', face)
        
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        messagebox.showerror("Error", "Failed to capture image.")
        
def apply_filters():
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path)
    
    if image is None:
        messagebox.showerror("Error", "Invalid image file.")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (15, 15), 0)
    
    cv2.imshow('Original', image)
    cv2.imshow('Gray', gray)
    cv2.imshow('Blur', blur)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def custom_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [255, 0, 0]  # Red square in the center
    
    plt.imshow(img)
    plt.show()

def cool_image_filters():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    if ret:
        # Load a sunglasses filter image with alpha channel
        sunglasses = cv2.imread("sunglasses.webp", cv2.IMREAD_UNCHANGED)
        
        # Detect face and overlay sunglasses (simplified here, improve with proper alignment)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            messagebox.showinfo("No Face Detected", "No face was detected in the image.")
        else:
            for (x, y, w, h) in faces:
                # Overlay sunglasses on the detected face
                overlay_image(frame, sunglasses, x, y, w, h)
            
            # Show the result
            cv2.imshow('Cool Filter', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        messagebox.showerror("Error", "Failed to capture image.")

def overlay_image(background, overlay, x, y, w, h):
    # Resize the overlay to fit the face
     # Resize the overlay to fit the face width
    overlay_resized = cv2.resize(overlay, (w, int(overlay.shape[0] * (w / overlay.shape[1]))))

    # Check if the overlay has an alpha channel (4 channels: BGR + Alpha)
    if overlay_resized.shape[2] == 4:
        # Iterate through the pixels of the overlay
        for i in range(overlay_resized.shape[0]):
            for j in range(overlay_resized.shape[1]):
                # Extract the alpha channel value (transparency)
                alpha = overlay_resized[i, j, 3] / 255.0  # Normalize to 0-1 range
                if alpha > 0:  # Only apply non-transparent parts
                    # Apply overlay using transparency
                    background[y + i, x + j] = (
                        alpha * overlay_resized[i, j, :3] +
                        (1 - alpha) * background[y + i, x + j]
                    )
    else:
        # If no alpha channel, overlay directly (no transparency)
        for i in range(overlay_resized.shape[0]):
            for j in range(overlay_resized.shape[1]):
                background[y + i, x + j] = overlay_resized[i, j]
                


def search_google_drive(query):
    # Authenticate with Google Drive
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    
    # Build the Drive API service
    service = build('drive', 'v3', credentials=creds)

    # Search for files in Google Drive
    results = service.files().list(q=f"name contains '{query}'", fields="files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print(f"{item['name']} ({item['id']})")

def google_drive_search():
    query = entry.get()
    search_google_drive(query)
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    
    # Build the Drive API service
    service = build('drive', 'v3', credentials=creds)

    # Search for files in Google Drive
    results = service.files().list(q=f"name contains '{query}'", fields="files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print(f"{item['name']} ({item['id']})")


def search_database(query):
    # Connect to the MySQL database
    conn = mysql.connector.connect(
        host="your_host",
        user="your_user",
        password="your_password",
        database="your_database"
    )
    
    cursor = conn.cursor()
    search_query = f"SELECT * FROM your_table WHERE your_column LIKE '%{query}%'"
    
    cursor.execute(search_query)
    results = cursor.fetchall()

    if not results:
        print("No results found.")
    else:
        for result in results:
            print(result)

    cursor.close()
    conn.close()

def database_search():
    query = entry.get()
    search_database(query)

def search_aws_s3(query):
    # Initialize S3 client
    s3 = boto3.client('s3')

    # List all objects in a specified bucket
    bucket_name = 'your-bucket-name'
    response = s3.list_objects_v2(Bucket=bucket_name)
    
    if 'Contents' in response:
        for obj in response['Contents']:
            if query in obj['Key']:
                print(f"Found: {obj['Key']}")
    else:
        print("No objects found.")

def aws_search():
    query = entry.get()
    search_aws_s3(query)

def search_linkedin(query):
    # Authenticate with LinkedIn
    api = Linkedin('your_email', 'your_password')

    # Search for people on LinkedIn
    results = api.search_people(keywords=query)

    if not results:
        print("No profiles found.")
    else:
        for person in results:
            print(f"{person['firstName']} {person['lastName']} - {person['headline']}")

def linkedin_search():
    query = entry.get()
    search_linkedin(query)

# Tkinter setup
root = tk.Tk()
root.title("Task Menu")
root.geometry("400x600")

# Create a frame and canvas
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create a frame inside the canvas
second_frame = tk.Frame(canvas)

# Add that new frame to a window in the canvas
canvas.create_window((0, 0), window=second_frame, anchor="nw")

# Add widgets inside the second_frame instead of root
task_dict = {
    "Send Email": send_email,
    "Send SMS": send_sms,
    "Scrape Google": scrape_google,
    "Find Location": find_location,
    "Text to Audio": text_to_audio,
}

for task_name, task_func in task_dict.items():
    button = tk.Button(second_frame, text=task_name, command=task_func)
    button.pack(pady=10)

# Text-to-Audio section
tk.Label(second_frame, text="Enter text for Text-to-Audio:").pack(pady=5)
text_input = tk.Entry(second_frame, width=40)
text_input.pack(pady=5)
tk.Button(second_frame, text="Convert Text to Audio", command=text_to_audio).pack(pady=10)

# Volume Control section
tk.Label(second_frame, text="Enter volume level (0.0 to 1.0):").pack(pady=5)
volume_input = tk.Entry(second_frame, width=10)
volume_input.pack(pady=5)
tk.Button(second_frame, text="Set Volume", command=control_volume).pack(pady=10)

task_dict2 = {
    "Data Processing": data_processing,
    "Model Integration": model_integration,
}

for task_name, task_func in task_dict2.items():
    button = tk.Button(second_frame, text=task_name, command=task_func)
    button.pack(pady=10)

tk.Button(second_frame, text="Run Streamlit Web App", command=run_streamlit_app).pack(pady=10)

task_dict2 = {
    "Image Crop": image_crop,
    "Apply Filters": apply_filters,
    "Custom Image": custom_image,
    "Cool Image Filters": cool_image_filters,
}

for task_name, task_func in task_dict2.items():
    button = tk.Button(second_frame, text=task_name, command=task_func)
    button.pack(pady=10)

task_dict3 = {
    "Google Drive Search": google_drive_search,
}

for task_name, task_func in task_dict3.items():
    button = tk.Button(second_frame, text=task_name, command=task_func)
    button.pack(pady=10)


entry = tk.Entry(second_frame)
entry.pack()

task_dict4 = {
    
    "Database Search": database_search,
}

for task_name, task_func in task_dict4.items():
    button = tk.Button(second_frame, text=task_name, command=task_func)
    button.pack(pady=10)
entry = tk.Entry(second_frame)
entry.pack()

task_dict5 = {
    "AWS Search": aws_search,
}

for task_name, task_func in task_dict5.items():
    button = tk.Button(second_frame, text=task_name, command=task_func)
    button.pack(pady=10)
entry = tk.Entry(second_frame)
entry.pack()

task_dict6 = {
    
    "LinkedIn Search": linkedin_search,
}

for task_name, task_func in task_dict6.items():
    button = tk.Button(second_frame, text=task_name, command=task_func)
    button.pack(pady=10)
entry = tk.Entry(second_frame)
entry.pack()

# Start the Tkinter loop
root.mainloop()
