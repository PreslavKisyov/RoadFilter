from PIL import Image
import os

out_path = "../Road Networks/"
in_path = "../Road Images/map/"
counter = 1

print("Converting...")
for im_path in os.listdir(in_path):
    image = Image.open(in_path+im_path)
    image.save(out_path+str(counter)+".jpg")
    print("Converting image " + str(counter))
    counter += 1
print("Finished!")

