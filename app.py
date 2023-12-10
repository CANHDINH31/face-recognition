import cv2
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(5)

img = cv2.imread('anh1.jpg')
# Resize anh ve dang 200*200*3 (3 layer)
img = cv2.resize(img, (200,200))

# Chuyển về 1 layer nàu xám
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img_gray.shape)


# Thiết kế lớp Convolution
class Conv2d:
    def __init__(self, input,numOfKernel = 8 ,kernelSize=3 ,padding =0 ,  stride = 1):
        self.input = np.pad(input,((padding,padding),(padding,padding)), 'constant')
        self.stride = stride
        self.kernel = np.random.randn(numOfKernel,kernelSize, kernelSize)

        # print(kernel)

        self.results = np.zeros((int((self.input.shape[0] - self.kernel.shape[1]) / self.stride) + 1,
                                 int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1,
                                 self.kernel.shape[0]
                                 ))

    def getROI(self):
        for row in range(int((self.input.shape[0] - self.kernel.shape[1]) / self.stride) + 1):
            for col in range(int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1):
                roi = self.input[row * self.stride: row * self.stride + self.kernel.shape[1],
                      col* self.stride: col * self.stride + self.kernel.shape[2]]
                yield row, col, roi

    def operate(self):
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.getROI():
                self.results[row, col, layer] = np.sum(roi * self.kernel[layer])

        return self.results;

class Relu:
    def __init__(self, input):
        self.input = input
        self.result = np.zeros((self.input.shape[0],
                                self.input.shape[1],
                                self.input.shape[2]
                                ))
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row,col,layer] = 0 if self.input[row,col, layer] < 0 else self.input[row,col, layer]

        return self.result


class LeakyRelu:
    def __init__(self, input):
        self.input = input
        self.result = np.zeros((self.input.shape[0],
                                self.input.shape[1],
                                self.input.shape[2]))
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row, col, layer] = 0.1 * self.input[row, col, layer] if self.input[row, col, layer] < 0 else self.input[row, col, layer]

            return self.result


class MaxPooling:
    def __init__(self, input, poolingSize = 2):
        self.input = input
        self.poolingSize = poolingSize
        self.result = np.zeros((
            int(self.input.shape[0] / self.poolingSize),
            int(self.input.shape[1] / self.poolingSize),
            self.input.shape[2]))

    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(int(self.input.shape[0] / self.poolingSize)):
                for col in range(int(self.input.shape[1] / self.poolingSize)):
                    self.result[row, col, layer] =np.max(self.input[row * self.poolingSize: row*self.poolingSize + self.poolingSize,
                                                   col * self.poolingSize: col * self.poolingSize + self.poolingSize,
                                                   layer
                                                   ])
            return self.result



class Softmax:
    def __init__(self, input, nodes):
        self.input = input
        self.nodes = nodes

        self.flatten = self.input.flatten()
        self.weights = np.random.randn(self.flatten.shape[0]) / self.flatten.shape[0]
        self.bias = np.random.randn(nodes)

    def operate(self):
        total = np.dot(self.flatten, self.weights) + self.bias
        exp = np.exp(total)
        return exp/sum(exp)



img_gray_conv2d = Conv2d(img_gray, 16 ,3,padding=0 , stride=1).operate()
img_gray_conv2d_relu = Relu(img_gray_conv2d).operate()
img_gray_conv2d_leakyrelu = LeakyRelu(img_gray_conv2d).operate()
img_gray_conv2d_relu_maxpooling = MaxPooling(img_gray_conv2d_relu, 3).operate()

softmax = Softmax(img_gray_conv2d_relu_maxpooling, 10)
print(softmax.operate())

# fig = plt.figure(figsize=(10,10))
#
# for i in range(16):
#     plt.subplot(4,4, i+1)
#     plt.imshow(img_gray_conv2d_leakyrelu[:,:,i], cmap="gray")
#     plt.axis('off')
#
# # plt.savefig('img_gray_conv2d_relu_maxpooling.jpg')
# plt.show()

