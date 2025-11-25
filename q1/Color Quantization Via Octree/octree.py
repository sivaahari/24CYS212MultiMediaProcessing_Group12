from PIL import Image
def octree_quantize_image(input_path, output_path, max_colors):
    #Perform Octree color quantization on an image using Pillow's FASTOCTREE method.
    # Load the input image and convert to RGB mode.
    # Converting ensures the image uses standard red, green, blue channels,
    img = Image.open(input_path).convert('RGB')
    print(f"Quantizing image to max {max_colors} colors using FASTOCTREE method...")
    # Perform quantization using Pillow's built-in octree quantizer.
    # How octree quantization works internally:
    # - The RGB color space is represented as a tree with up to 8 children per node (an octree).
    # - Each level of the tree examines one bit of the R, G, and B channels, dividing color space into smaller cubes.
    # - Colors from the image are inserted into the tree according to their bits, accumulating counts and sums.
    # - To reduce colors, the tree is pruned by merging nodes with similar colors until only max_colors leaves remain.
    # - The leaves of the tree become the palette colors used to represent the image.
    # This method efficiently groups similar colors and keeps those most visually relevant,
    # resulting in a palette-based (mode 'P') image with quality close to the original but fewer colors.
    quantized_img = img.quantize(colors=max_colors, method=Image.Quantize.FASTOCTREE)
    # Save the quantized image in a file format supporting palette mode (PNG recommended).
    quantized_img.save(output_path)
    print(f"Quantized image saved to: {output_path}")
max_colors = int(input("Enter maximum number of colors (e.g., 256): "))
octree_quantize_image("octree_input.jpg","octree_output.png", max_colors)
