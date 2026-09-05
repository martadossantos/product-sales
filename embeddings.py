from PIL import Image
import numpy as np
import pandas as pd
import torch

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
def encode_images(df, model, processor, device, batch_size, image_root):
    img_paths = df['image_path'].tolist()
    codes = df['external_code'].tolist()

    # list to hold all the new vectors
    all_features = []

    # process the images in batches
    for n in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[n:n + batch_size]

        # open and square each image in the batch
        batch_images = []

        for p in batch_paths:
            img = Image.open(image_root / p)
            batch_images.append(make_square(img))

        inputs = processor(images=batch_images, return_tensors='pt')
        inputs = inputs.to(device)

        # to return the 768 embedding
        with torch.no_grad():
            features = model.get_image_features(**inputs).pooler_output
    
        features = features.cpu().numpy()
        all_features.append(features)

    # stack features vertically    
    matrix = np.vstack(all_features)

    # build df to return
    columns = [f'image_{i}' for i in range(768)]
    encoded_df = pd.DataFrame(matrix, columns=columns)
    encoded_df.insert(0, 'external_code', codes)

    return encoded_df


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
def encode_text(df, model, processor, device, batch_size):
    texts = df['text'].tolist()
    codes = df['external_code'].tolist()

    # list to hold all the new vectors
    all_features = []

    # process the texts in batches
    for n in range(0, len(texts), batch_size):
        batch = texts[n:n + batch_size]
        inputs = processor(text=batch, padding=True, truncation=True, return_tensors='pt')

        inputs = inputs.to(device)

        # to return the 768 embedding
        with torch.no_grad():
            features = model.get_text_features(**inputs).pooler_output
    
        features = features.cpu().numpy()
        all_features.append(features)

    # stack features vertically    
    matrix = np.vstack(all_features)

    # build df to return
    columns = [f'text_{i}' for i in range(768)]
    encoded_df = pd.DataFrame(matrix, columns=columns)
    encoded_df.insert(0, 'external_code', codes)

    return encoded_df
