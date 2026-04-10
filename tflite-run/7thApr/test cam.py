import cv2

def check_available_cameras():
    print("Quet cong cam.")
    available_ports = []
    
    # 
    for i in range(10):
        # D
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
        if cap is None or not cap.isOpened():
            pass
        else:
            print(f"Tim thay cam tai cong: {i}")
            available_ports.append(i)
            cap.release()
            
    if not available_ports:
        print("Nothing!")
    else:
        print(f"Port active: {available_ports}")

check_available_cameras()