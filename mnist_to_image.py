from PIL import Image
import struct
from tqdm import tqdm

def read_image(filename):
    f = open(filename,'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')


    for i in tqdm(range(images)):
        image = Image.new('L', (columns, rows))
        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

        # print('save ' + str(i) + 'image')
        image.save('data_imgs/' + str(i) + '_' + str(labelArr[i]) + '.png')


def read_label(filename, saveFilename):
    global labelArr
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    labelArr = [0] * labels
    for x in tqdm(range(labels)):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')
        # save = open(saveFilename, 'w')
        # save.write(','.join(map(lambda x: str(x), labelArr)))
        # save.write('\n')
        # save.close()
        # print('save labels success')


if __name__ == '__main__':
    labelArr = []
    read_label('data/raw/t10k-labels-idx1-ubyte', 'data_imgs/label.txt')
    print(labelArr)
    read_image('data/raw/t10k-images-idx3-ubyte')
