import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path="res_guitar/guitar/guitar_amp25.mp4"

points_of_interest=[]
# Function called when the user clicks a point in an image
def mouse_callback(event, x, y, flags, image):
    
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Pointed clicked at x, y = ", x, y)
        points_of_interest.append([x, y])
        cv2.circle(image, (x, y), 3, (0, 0, 0), -1)
        cv2.imshow("Frame", image)
        if len(points_of_interest) == 2:
            print("Press enter within the OpenCV window to continue.")
    

def get_point_of_interest(frame):
     # Draw line between points clicked by the user
    
    cv2.setMouseCallback("Frame", mouse_callback, frame)
    cv2.waitKey(0)
    
     
     






cap = cv2.VideoCapture(video_path) # Retrieve frames using recorded video
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
if not cap.isOpened():
    print("Cannot open video capture")
    exit()

first_frame=True

while True:
        
        ret, frame = cap.read() # Capture frame-by-frame. If frame is read correctly ret is True
        
        if first_frame:
            cv2.imshow("Frame", frame)
            get_point_of_interest(frame)
            print(points_of_interest)
            points=np.array(points_of_interest)
            luminance=[]

        
        try:
            first_frame=False
            cv2.imshow("frame", frame)
            luminance_current_frame=0
            for point in points:
                luminance_current_frame+=np.mean(frame[point[1],point[0]])
            luminance.append(luminance_current_frame/len(points))
            
            
            # YOUR CODE: Save frame to video file
        except Exception as e:
             break
        
            
        key = 0xFF & cv2.waitKey(1)
        if key == ord('q'):
            break

luminance=np.array(luminance)
luminance=luminance-np.mean(luminance)
fft_result = np.fft.fft(luminance)
fft_result[1]=0
####
max_index = np.argmax(np.abs(fft_result[:len(fft_result)//2]))
max_value = np.abs(fft_result[:len(fft_result)//2][max_index])




### Plot ####
x=((2*np.pi)/fps)*np.arange(0,len(fft_result)//2)
plt.plot(x,np.abs(fft_result[:len(fft_result)//2]))
plt.scatter(x[max_index], max_value, color='red')
plt.annotate(f'Max Value: {max_value:.2f}\nFrequency: {x[max_index]}', xy=(x[max_index], max_value), xytext=(20, -20), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
plt.xlabel("Frequency")
plt.ylabel("Intensity")

plt.show()
cap.release() # When everything done, release the capture
cv2.destroyAllWindows()

