from PIL import Image

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



# Encode images
def encode_images():
    return

# Construct text feature from the product's metadata
def build_text_feature(df):
    df = df.copy()

    # pick the columns
    columns = ['category', 'color', 'fabric']
    assert not df[columns].isnull().values.any(), "nulls in text columns"

    row_count = len(df)

    # concanate text
    for col in columns:
        df[col] = df[col].str.strip()

    df['text'] = df[columns].agg(', '.join, axis=1)

    # check row count is correct
    assert len(df) == row_count

    # return df with code + new text feature
    return df[['external_code', 'text']]

# Encode text
def encode_text():
    return
