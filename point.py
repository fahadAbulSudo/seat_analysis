import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import cv2
image = cv2.imread("/home/fahadabul/mask_rcnn_skyhub/Subtitles/taper.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots()
ax.imshow(image_rgb)
plt.title("Click to get coordinates (zoom supported)")

coords = []

def onclick(event):
    if event.xdata and event.ydata:
        x, y = int(event.xdata), int(event.ydata)
        coords.append((x, y))
        print(f"Clicked at: x={x}, y={y}")

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
