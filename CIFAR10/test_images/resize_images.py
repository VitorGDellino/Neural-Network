from PIL import Image

name = "horse"
imageFile = name + ".jpg"
im1 = Image.open(imageFile)
width = 32
height = 32
im5 = im1.resize((width, height), Image.ANTIALIAS)
ext = "_.jpg"
im5.save(name + ext)
