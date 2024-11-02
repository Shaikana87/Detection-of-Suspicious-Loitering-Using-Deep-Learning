import yagmail
import time
# using datetime module
import datetime;

def sendAlertMail(img_path):
    try:                
        current_time = datetime.datetime.now()
        yag = yagmail.SMTP("autoemailsender2@gmail.com", "tczewxnxfrpviped")
        yag.send(to='shaikanas2401@gmail.com', subject="Loitering Alert", contents=f'Loitering Detected at {current_time}', attachments=img_path)
        yag.close()
        print('Mail sent successfully')
    except Exception as e:
        print('Message not sent due to:', e)

# Android - Python Java, Kotlin, Python



