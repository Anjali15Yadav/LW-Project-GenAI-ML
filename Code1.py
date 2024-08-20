import tkinter as tk
from tkinter import messagebox
import smtplib
import pyttsx3
import os
import geocoder
import pywhatkit as kit
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email():
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        sender_email = ""
        sender_password = ""
        receiver_email = ""
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
        phone_number = ""
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
        phone_number = "+XXXXXXXXXX"  # Replace with the recipient's phone number
        sms_message = "This is a test SMS sent via adb from Python!"
        
        # adb command to send SMS
        adb_command = f'adb shell service call isms 7 i32 1 s16 "com.android.mms.service" s16 "{phone_number}" s16 "null" s16 "{sms_message}" s16 "null" s16 "null"'
        os.system(adb_command)
        
        messagebox.showinfo("Connect to Mobile", "SMS sent via mobile device!")
    except Exception as e:
        messagebox.showerror("Connect to Mobile", f"Failed to send SMS: {e}")

def send_bulk_email():
    try:
        sender_email = ""
        sender_password = ""
        subject = "Bulk Email Test"
        body = "This is a test bulk email sent from Python!"
        
        # List of recipients
        recipient_list = ["", ""]  # Add as many as you want
        
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


# Tkinter setup
root = tk.Tk()
root.title("Task Menu")
root.geometry("400x600")

# Create menu buttons
task_dict = {
    "Send Email": send_email,
    "Send SMS": send_sms,
    "Scrape Google": scrape_google,
    "Find Location": find_location,
    
}

for task_name, task_func in task_dict.items():
    button = tk.Button(root, text=task_name, command=task_func)
    button.pack(pady=10)

# Text-to-Audio section
tk.Label(root, text="Enter text for Text-to-Audio:").pack(pady=5)
text_input = tk.Entry(root, width=40)
text_input.pack(pady=5)
tk.Button(root, text="Convert Text to Audio", command=text_to_audio).pack(pady=10)


# Volume Control section
tk.Label(root, text="Enter volume level (0.0 to 1.0):").pack(pady=5)
volume_input = tk.Entry(root, width=10)
volume_input.pack(pady=5)
tk.Button(root, text="Set Volume", command=control_volume).pack(pady=10)

task_dict2 = {
    "Connect and Send SMS": connect_and_send_sms,
    "Send Bulk Email": send_bulk_email,
}

for task_name, task_func in task_dict2.items():
    button = tk.Button(root, text=task_name, command=task_func)
    button.pack(pady=10)


# Start the Tkinter loop
root.mainloop()
