import os
import cv2

import sqlite3

import numpy as np
import pandas as pd

 
import seaborn as sns 
import matplotlib.pyplot as plt

from ultralytics import YOLO 


# folder's project
ROOT = os.path.dirname(os.path.abspath(os.getcwd()))
print(f"Folder: {ROOT}")

MODEL_PATH = os.path.join(ROOT, 'models', 'yolo11n.pt')
INPUT_VIDEO_PATH = os.path.join(ROOT, "asesrt", "sample.mp4")   
MODEL = YOLO(MODEL_PATH)


def predict_on_frame (frame, device: str = 'cpu'): 
    
    
    try: 
        results = MODEL.predict(source= frame, 
                            conf=0.25, 
                            device= device, 
                            verbose= False)

    except Exception as e: 
        print(f"ERROR: {e}")
    return results


def process_video(conn, file_path, rows= 4, cols= 6):
    
    # Load video  
    cap = cv2.VideoCapture(file_path) 
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    
    try: 
        index = 0
        while True:
            # Đọc từng khung hình
            ret, frame = cap.read()

            # Nếu không đọc được khung hình nào nữa thì dừng lại
            if not ret:
                print("Đã kết thúc")
                break
            
            
            results = predict_on_frame(frame) 
            if results and len(results)>0:
                index+= 1
                result = results[0] 
                im_bgr = result.plot() 
                
                try:
                    print(f"Frame {index}| Count: {len(result.boxes)}")
                    insert = '''INSERT INTO track (frame, count) VALUES (? , ?);'''
                    cursor = conn.cursor()
                    cursor.execute(insert, (index, len(result.boxes)))         
                    conn.commit() 
                except Exception as e : 
                    pass
                
            # Hiển thị khung hình trong cửa sổ 'Video'
            cv2.imshow('Video', im_bgr)
            index+= 1

            # Thoát vòng lặp khi nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                  
    except Exception as e: 
        print(f"ERROR: {e}")
        
    # Giải phóng đối tượng video và đóng tất cả các cửa sổ
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__": 
        
    try: 
        conn = sqlite3.Connection(database= os.path.join(ROOT, 'data', 'test.db'))
    except Exception as e: 
        pass

    cursor = conn.cursor() 
    camera = '''CREATE TABLE 
                IF NOT EXISTS track 
                (id INTERGER PRIMARY KEY, 
                frame INTEGER, 
                count INTEGER
                );
            '''
    cursor.execute(camera)
    conn.commit()
    
    print("--Process--")
    process_video(conn = conn, file_path= INPUT_VIDEO_PATH)
    conn.close()

