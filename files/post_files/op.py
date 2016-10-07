import cv2

def show_image(image):
    cv2.imshow('Hi', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

if( __name__ == '__main__'):
    img = cv2.imread("girl.png")
    show_image(img)
