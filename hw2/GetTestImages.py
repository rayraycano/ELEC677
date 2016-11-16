"""
Save the test images in a grid like format for presentation
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def save_test_images():
    for j in range(0, 1000, 100):
        plt.figure(1, figsize=(30, 30))
        for i in range(0, 100, 10):
            num = str(i)
            if i == 0:
                num = '0' + num
            img = mpimg.imread('data/CIFAR10/Test/' + str(j/100) + '/Image000' + num + '.png')
            plt.subplot(7, 6, i / 10 + 1)
            plt.imshow(img, cmap=plt.get_cmap('gray'))

        plt.savefig('figures/testImgs/class' + str(j/100))
        # plt.clf()


if __name__ == '__main__':
    save_test_images()