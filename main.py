from image_segment import *

blur = GaussianBlur((13, 13))

# preprocessor = CannyEdgeDetection(100, 200, preprocessor=blur)

stragegy = MeanShift()

stragegy = GThresh(
    min=127,
    max=255,
    mode=cv.THRESH_BINARY_INV + cv.THRESH_OTSU,
    preprocessor=preprocessor
)

for i in range(1, 8):
    app.show_img_after_remove_bg('img/img' + str(i) + '.jpg')
