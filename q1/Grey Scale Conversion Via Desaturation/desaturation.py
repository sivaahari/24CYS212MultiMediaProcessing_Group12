#Lightness Method
# We import the Image class from PIL so we can open, read and save images
from PIL import Image
# This function converts a color image into grayscale using the DESATURATION method
def desaturate_grayscale(input_path, output_path):
    # Open the original image and convert it to RGB format
    img = Image.open(input_path).convert("RGB") #converts to RGB if img is in other format like CJMK
    # Get the width and height of the image (number of pixels horizontally and vertically)
    width, height = img.size
    # Create a new blank image with same size where we will store grayscale pixels
    gray_img = Image.new("RGB", (width, height))
    # Loop through every column (x-coordinate) of the image
    for x in range(width):
        # Loop through every row (y-coordinate) in that column
        for y in range(height):
            # Get the R, G, B values of the current pixel
            r, g, b = img.getpixel((x, y))
            # DESATURATION GRAYSCALE RULE:
            #   gray = (max(R,G,B) + min(R,G,B)) / 2
            # Find the brightest (maximum) among R, G, B
            max_val = max(r, g, b)
            # Find the darkest (minimum) among R, G, B
            min_val = min(r, g, b)
            # Apply the desaturation formula and convert to integer
            gray = int((max_val + min_val) / 2)
            # Write this computed grayscale value back to the output image
            # All channels are same for grayscale: (gray, gray, gray)
            gray_img.putpixel((x, y), (gray, gray, gray))
    # Save the final grayscale image to the given output path
    gray_img.save(output_path)
    # Print message to show that conversion is complete
    print("Saved:", output_path)
desaturate_grayscale("input.jpg", "out.jpg")
