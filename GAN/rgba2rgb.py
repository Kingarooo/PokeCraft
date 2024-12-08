from PIL import Image
import os

src = "./data/images/all"
dst = "./data/processed/all"

if not os.path.exists(dst):
    os.makedirs(dst)

for each in os.listdir(src):
    png = Image.open(os.path.join(src, each))
    if png.mode == 'RGBA':
        png.load()  # Required for png.split()
        background = Image.new("RGB", png.size, (0, 0, 0))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        background.save(os.path.join(dst, each.split('.')[0] + '.jpg'), 'JPEG')
    elif png.mode == 'P':  # Check for palette-based images
        png = png.convert('RGB')  # Convert to RGB mode
        png.save(os.path.join(dst, each.split('.')[0] + '.jpg'), 'JPEG')
    else:
        png = png.convert('RGB')  # Ensure conversion to RGB for safety
        png.save(os.path.join(dst, each.split('.')[0] + '.jpg'), 'JPEG')