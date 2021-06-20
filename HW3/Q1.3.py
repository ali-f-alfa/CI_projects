# Q1.3_graded
# Do not change the above line.
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageFilter

!wget "https://www.freebestfonts.com/yone//down/arial.ttf"
!mv arial.ttf /usr/share/fonts/truetype

font_size = 64
font = ImageFont.truetype("arial.ttf", font_size)
# This cell is for your imports.


# Q1.3_graded
# Do not change the above line.

# This cell is for your codes.



class Hopfield:
    def __init__(self, dim, input_patterns):
        # set dimension of network (number of nuerons)
        self.dim = dim
        # initialize weight matrix with zeros
        self.weights = np.zeros((dim, dim))
        self.count = 0
        self.input_patterns = input_patterns

    def Save(self):

        for i in range(len(self.input_patterns)):
            input_list = self.input_patterns[i]
            input_list = np.array(input_list)

            # create reverse of input
            input_list_r = input_list.reshape(-1,1)

            # calculate new weights that our input generates
            new_weights = input_list_r * input_list

            # remove diagonal of matrix
            np.fill_diagonal(new_weights, 0)

            self.weights += new_weights

        # update our final weights matrix
        self.weights = np.divide(self.weights, self.dim)
        # print("updated weights by input: ", input_list, "is :\n", self.weights) 
 

    def CheckInput(self, input, i=10):
        
        input = np.array(input)
        e = self.energy(input)
        
        for j in range(i):
            idx = np.random.randint(input.size)
            input = self.update_async(input, idx)
        
        show_image(input)

        new_e = self.energy(input)
        
        if new_e == e:
            print('Network find an answer')
            print("result image is :")
            show_image(input)
            return input,-1

        return input,1
       

    def update_async(self, input, idx=None):

        if idx==None:
            new_input = np.matmul(self.weights,input)
            new_input[new_input < 0] = -1
            new_input[new_input > 0] = 1
            new_input[new_input == 0] = input[new_input == 0]
            input = new_input
        else:
            new_input = np.matmul(self.weights[idx],input)
            if new_input < 0:
                input[idx] = -1
            elif new_input > 0:
                input[idx] = 1
        return input

    def energy(self,input):
        e = -0.5*np.matmul(np.matmul(input.T,self.weights),input)
        return e



def cal_acc(input, expected):
    mistakes = 0
    for i in range(font_size * font_size):
        if expected[i] != input[i]:
            mistakes += 1
    return 1 - (mistakes/(font_size * font_size))


def add_noise(image, percentage):
    gauss = np.random.normal(0,percentage,font_size*font_size)
    gauss = gauss.reshape(font_size,font_size)
    return image + gauss


def image_to_array(image):
    I = np.array(image).astype('int')
    I[I <= 30] = -1
    I[I > 30] = 1
    input_font = I.reshape(1,-1)
    return input_font

def show_image(input_array):
    x = input_array.copy().reshape(font_size,font_size)
    x[x == 1] = 255
    x[x == -1] = 0
    im = Image.fromarray(np.uint8(x), 'L')
    plt.imshow(im)
    plt.show()
    return


if __name__=='__main__':

    font_patterns = []
    noisy10 = []
    noisy30 = []
    noisy60 = []

    for char in "ABCDEFGHIJ":
        im = Image.Image()._new(font.getmask(char))
        im = im.resize((font_size,font_size))

        noisy10.append(add_noise(im, 10))
        noisy30.append(add_noise(im, 30))
        noisy60.append(add_noise(im, 60))
        
        input_font = image_to_array(im)
        font_patterns.append(input_font)


    hop = Hopfield(font_size*font_size, font_patterns)
    hop.Save()


    img_num = 0
    xx = image_to_array(noisy60[img_num])[0]
    # xx = font_patterns[8][0]

    show_image(xx)
    for i in range(1000):
        print("iteration: ", i)
        # xxx = int(2000/(i+1)) + 5
        xx,tamam = hop.CheckInput(xx, i = 100)
        if (tamam == -1):
            print("finished in:",i)
            print("ACC is: ", cal_acc(xx, font_patterns[img_num][0]))
            break



