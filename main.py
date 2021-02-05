from image_segment import GaussianBlur as Blur
from image_segment import AdaptiveThreshold as AThresh
from image_segment import RemoveBG as App
import cv2 as cv

# Init the preprocessor
kernel_size = (13, 13)
preprocessor = Blur(kernel_size)

# Init stragegy for remove background
stragegy = AThresh(
    max=255,
    mode=cv.THRESH_BINARY_INV,
    adaptive=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    C=2,
    blocksize=13
    # preprocessor = preprocessor
)

# Set app's stragegy
app = App(stragegy)

# Show results
for i in range(1, 8):
    app.show_img_after_remove_bg('img/img' + str(i) + '.jpg')
