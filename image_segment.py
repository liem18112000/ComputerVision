import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class RemoveBGInterfaces(object):

    def remove_bg(self, src):
        pass

    def set_stragegy(self, stragegy):
        pass

    def show_img_after_remove_bg(self, src):
        pass
            
class RemoveBG(RemoveBGInterfaces):
    def __init__(self, stragegy):
        self.set_stragegy(stragegy)

    def remove_bg(self, src):
        return self._stragegy_.apply(src)

    def set_stragegy(self, stragegy):
        if stragegy is not None and isinstance(stragegy, StragegyInterfaces):
            self._stragegy_ = stragegy

    def show_img_after_remove_bg(self, src):
        images = self.remove_bg(src)
        titles = ['Original Image', 'Remove BG']
        for i in range(len(images)):
            plt.subplot(len(images) / 2, 2, i + 1), plt.imshow(images[i], vmin=0, vmax=255)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()



class StragegyInterfaces(object):

    def apply(self, src):
        pass



class Preprocessor:

    def apply_preprocess(self, img):
        pass



class BlurStragegy(StragegyInterfaces):

    def apply(self, src):
        pass

    def _blur_(self, img):
        pass

    def _preprocess_(self, img):
        pass

class GaussianBlur(BlurStragegy, Preprocessor):

    def __init__(self, kernel_size, preprocessor = None):
        self._kernel_size = kernel_size
        self._preprocessor_ = preprocessor

    def apply(self, src):
        img = cv.cvtColor(cv.imread(src), cv.COLOR_BGR2RGB)
        return [img, self._blur_(self._preprocess_(img))]

    def apply_preprocess(self, img):
        return self._blur_(self._preprocess_(img))

    def _blur_(self, img):
        return cv.GaussianBlur(img, self._kernel_size, 0)

    def _preprocess_(self, img):
        if self._preprocessor_ is None:
            return img
        return self._preprocessor_.apply_preprocess(img)



class EdgeDetectionStrategy(StragegyInterfaces):
    def apply(self, src):
        pass

    def _edge_detection_(self, img):
        pass

class CannyEdgeDetection(EdgeDetectionStrategy, Preprocessor):

    def __init__(self, min, max, preprocessor = None):
        self._max_ = max
        self._min_ = min
        self._preprocessor_ = preprocessor

    def apply(self, src):
        img = cv.cvtColor(cv.imread(src), cv.COLOR_BGR2RGB)
        return [img, self._edge_detection_(self._preprocess_(img))]

    def apply_preprocess(self, img):
        return self._edge_detection_(self._preprocess_(img))

    def _edge_detection_(self, img):
        return cv.Canny(img, self._min_, self._max_)

    def _preprocess_(self, img):
        if self._preprocessor_ is None:
            return img
        return self._preprocessor_.apply_preprocess(img)



class ThresholdStragegyInterfaces(StragegyInterfaces):

    def apply(self, src):
        pass

    def _threshold_(self, img):
        pass

    def _preprocess_(self, img):
        pass

class GlobalThreshold(ThresholdStragegyInterfaces):

    def __init__(self, min, max, mode, preprocessor = None):
        self._min_ = min
        self._max_ = max
        self._mode_ = mode
        self._preprocessor_ = preprocessor

    def _threshold_(self, img):
        ret, thresh = cv.threshold(img, self._min_, self._max_, self._mode_)
        return thresh

    def apply(self, src):
        gray_img = cv.imread(src, 0)
        img = cv.imread(src)
        original_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        thresh = self._threshold_(self._preprocess_(gray_img))
        remove_bg_img = original_img * np.reshape((thresh // 255), (original_img.shape[0], original_img.shape[1], 1))
        return [original_img, remove_bg_img]

    def _preprocess_(self, img):
        if self._preprocessor_ is None:
            return img
        return self._preprocessor_.apply_preprocess(img)

class AdaptiveThreshold(ThresholdStragegyInterfaces):

    def __init__(self, max, mode, adaptive, preprocessor = None, C = 2, blocksize = 11):
        self._max_ = max
        self._adaptive_ = adaptive
        self._mode_ = mode
        self._C_ = C
        self._blocksize_ = blocksize
        self._preprocessor_ = preprocessor

    def _threshold_(self, img):
        thresh = cv.adaptiveThreshold(img, self._max_, self._adaptive_, self._mode_, self._blocksize_, self._C_)
        return thresh

    def apply(self, src):
        gray_img = cv.imread(src, 0)
        img = cv.imread(src)
        original_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        thresh = self._threshold_(self._preprocess_(gray_img))
        remove_bg_img = original_img * np.reshape((thresh // 255), (original_img.shape[0], original_img.shape[1], 1))
        return [original_img, remove_bg_img]

    def _preprocess_(self, img):
        if self._preprocessor_ is None:
            return img
        return self._preprocessor_.apply_preprocess(img)



class ClusteringStrategy(StragegyInterfaces):
    def apply(self, src):
        pass

    def _clustering_(self,img):
        pass

class KMeamClustering(ClusteringStrategy):
    def __init__(self, numberOfClusters, preprocessor = None):
        self._numberOfClusters_ = numberOfClusters
        self._preprocessor_ = preprocessor

    def apply(self, src):
        img = cv.imread(src)
        original_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cluster_img = self._clustering_(self._preprocess_(original_img))
        return [original_img, cluster_img]

    def _clustering_(self, img):
        reshape_img = np.float32(img.reshape(-1, 3))
        stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        ret, labels, clusters = cv.kmeans(reshape_img, self._numberOfClusters_, None, stopCriteria, 10, cv.KMEANS_RANDOM_CENTERS)
        reshape_img[labels.flatten() == 1] = [0, 0, 0]
        reshape_img = np.uint8(reshape_img)
        return np.reshape(reshape_img, np.shape(img))

    def _preprocess_(self, img):
        if self._preprocessor_ is None:
            return img
        return self._preprocessor_.apply_preprocess(img)



class GrabCut(StragegyInterfaces):

    def __init__(self, preprocessor=None):
        self._preprocessor_ = preprocessor

    def apply(self, src):
        img = cv.imread(src)
        original_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        grabcut_img = self._graph_cut(self._preprocess_(original_img))
        return [original_img, grabcut_img]

    def _preprocess_(self, img):
        if self._preprocessor_ is None:
            return img
        return self._preprocessor_.apply_preprocess(img)

    def _graph_cut(self, img):
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (0, 0, img.shape[0], img.shape[1])

        cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img*mask2[:, :, np.newaxis]
        return img


class MeanShift(StragegyInterfaces):
    def __init__(self, preprocessor=None):
        self._preprocessor_ = preprocessor

    def apply(self, src):
        img = cv.imread(src)
        original_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        meanshift_img = self._mean_shift(self._preprocess_(original_img))
        return [original_img, meanshift_img]

    def _preprocess_(self, img):
        if self._preprocessor_ is None:
            return img
        return self._preprocessor_.apply_preprocess(img)

    def _mean_shift(self, img):
        # setup initial location of window
        x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
        track_window = (x, y, w, h)
        # set up the ROI for tracking
        roi = img[y:y+h, x:x+w]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)),
                        np.array((180., 255., 255.)))
        roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, w, h = track_window
        img2 = cv.rectangle(img, (x, y), (x+w, y+h), 255, 2)
        return img2
