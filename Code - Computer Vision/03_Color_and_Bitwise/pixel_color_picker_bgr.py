# Lesson: click a pixel on a rescaled image to read/display its BGR value live.
import cv2 as cv

# 1. Load your image (replace with your file path)
img_path = "../assets/col.jpg"
img = cv.imread(img_path)

if img is None:
    print(
        f"Error: Could not open image '{img_path}'. Make sure the file exists!"
    )
    exit()

def rescaleFrame(frame, scale):
    # .shape = height[0], width[1], layers/channel [2]
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimension=(width,height)
    return cv.resize(frame,dimension, interpolation=cv.INTER_AREA)

test = rescaleFrame(img, 0.5)

def click_bgr_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        # OpenCV images are indexed as [row/y, column/x]
        b, g, r = img[y, x]
        bgr_text = f"BGR: ({b}, {g}, {r})"

        print(f"Clicked at ({x}, {y}) -> {bgr_text}")

        # Display values directly on the image at the clicked position
        display_img = test.copy()
        cv.putText(
            display_img,
            bgr_text,
            (x + 10, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        cv.circle(
            display_img, (x, y), 3, (0, 0, 255), -1
        )  # Red dot on clicked pixel
        cv.imshow("Click for BGR", display_img)


cv.namedWindow("Click for BGR")
cv.setMouseCallback("Click for BGR", click_bgr_callback)

print("Click anywhere on the image to view the BGR values. Press 'q' to exit.")

while True:
    cv.imshow("Click for BGR", test)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cv.destroyAllWindows()