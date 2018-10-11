
import Image

imageFile = "6_test.png"
im1 = Image.open(imageFile)
width = 28
height = 28
im5 = im1.resize((width, height), Image.ANTIALIAS)
ext = ".png"
im5.save("6" + ext)
