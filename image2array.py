import imageio

image_file_name = "my_images\\7-01.png"

image_array = imageio.imread(image_file_name, as_gray="F")
image_data = 255.0 - image_array.reshape(784)
print(image_array)
