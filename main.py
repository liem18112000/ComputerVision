from image_segment import GaussianBlur as Blur
from image_segment import Mean_Shift 
from image_segment import RemoveBG as App
import cv2 as cv

# Init the preprocessor
kernel_size = (13, 13)
preprocessor = Blur(kernel_size)

# Init stragegy for remove background
stragegy = Mean_Shift(
    # preprocessor = preprocessor
)

# Set app's stragegy
app = App(stragegy)

# Show results
for i in range(1, 3):
    app.show_img_after_remove_bg('img/img' + str(i) + '.jpg')
