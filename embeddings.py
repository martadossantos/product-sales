# Pad the images so that they're squares
# To then use in the image embedder
# Image is centered and white padding added around it
# All images are RGBA

def make_square(img, fill=(255, 255, 255)):
    background = Image.new('RGBA', img.size, fill + (255,))
    img = Image.alpha_composite(background, img).convert('RGB')

    w, h = img.size

    if w == h:
        return img

    s = max(w, h)
    canvas = Image.new('RGB', (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))

    return canvas
