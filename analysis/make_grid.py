from PIL import Image, ImageDraw


def create_grid_image(
    size=(560, 560),
    grid_size=4,
    background_color=(0, 0, 0),
    line_color=(255, 255, 255),
    output_path="grid_560x560.png",
):
    # Create a black background image
    img = Image.new("RGB", size, background_color)
    draw = ImageDraw.Draw(img)

    # Draw the evenly spaced grid lines
    width, height = size
    for i in range(1, grid_size):
        # Vertical line at x = i * width / grid_size
        x = i * width / grid_size
        draw.line([(x, 0), (x, height)], fill=line_color)
        # Horizontal line at y = i * height / grid_size
        y = i * height / grid_size
        draw.line([(0, y), (width, y)], fill=line_color)

    # Save to a file
    img.save(output_path)
    print(f"Saved grid image to {output_path}")

if __name__ == "__main__":
    create_grid_image()