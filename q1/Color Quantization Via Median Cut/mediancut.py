from PIL import Image
# This function reduces the number of colors in an image using the MEDIAN CUT method.
def median_cut_quantize(input_path, output_path, num_colors):
    # Open the image and ensure it is in RGB format
    img = Image.open(input_path).convert("RGB")
    # Apply PIL's built-in Median Cut quantization
    # method = 0 → Median Cut
    quantized_img = img.quantize(colors=num_colors, method=0)
    # Convert back to RGB so it saves correctly in JPG/PNG
    quantized_img = quantized_img.convert("RGB")
    # Save the output
    quantized_img.save(output_path)
    print("Saved quantized image:", output_path)
num_colors=int(input("Enter no.of colors: "))
# Example usage
median_cut_quantize("input.jpg", "out_median.jpg", num_colors)
