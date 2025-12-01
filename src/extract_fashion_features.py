"""
Extract Fashion-Related Features from Image Descriptions

This script filters image descriptions to keep only fashion-related information,
removing noise like background, setting, poses, etc.

Methods:
1. Rule-based extraction using fashion keywords
2. (Optional) LLM-based extraction for better quality
"""

import pandas as pd
import re
import os

# Fashion-related keywords and patterns
CLOTHING_ITEMS = [
    # Tops
    'shirt', 'blouse', 'top', 'sweater', 'cardigan', 'hoodie', 'jacket', 'blazer',
    'coat', 'vest', 'tank top', 'crop top', 't-shirt', 'tee', 'pullover', 'sweatshirt',
    'turtleneck', 'polo', 'tunic', 'camisole',
    # Bottoms
    'pants', 'jeans', 'trousers', 'shorts', 'skirt', 'leggings', 'joggers',
    'cargo pants', 'wide-leg', 'straight-leg', 'skinny', 'flare',
    # Dresses & Full body
    'dress', 'gown', 'jumpsuit', 'romper', 'overalls', 'suit', 'uniform',
    # Outerwear
    'coat', 'jacket', 'puffer', 'bomber', 'trench', 'parka', 'windbreaker',
    'denim jacket', 'leather jacket', 'raincoat',
    # Accessories
    'bag', 'handbag', 'purse', 'tote', 'clutch', 'backpack', 'crossbody',
    'hat', 'cap', 'beanie', 'scarf', 'belt', 'watch', 'bracelet', 'necklace',
    'earrings', 'ring', 'sunglasses', 'glasses',
    # Footwear
    'shoes', 'sneakers', 'boots', 'heels', 'sandals', 'flats', 'loafers',
    'slippers', 'ankle boots', 'high heels', 'stilettos', 'wedges',
    # Materials & Patterns
    'leather', 'denim', 'silk', 'cotton', 'wool', 'linen', 'velvet',
    'sequins', 'lace', 'fur', 'suede', 'satin',
    'floral', 'striped', 'plaid', 'checkered', 'polka dot', 'leopard print',
    'geometric', 'abstract', 'solid', 'printed',
]

COLORS = [
    'white', 'black', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
    'pink', 'brown', 'grey', 'gray', 'beige', 'navy', 'burgundy', 'maroon',
    'cream', 'ivory', 'gold', 'silver', 'bronze', 'tan', 'khaki', 'olive',
    'coral', 'turquoise', 'teal', 'lavender', 'mint', 'peach', 'nude',
    'dark', 'light', 'bright', 'pastel', 'neon', 'metallic',
]

STYLE_DESCRIPTORS = [
    'sleeveless', 'long-sleeved', 'short-sleeved', 'strapless', 'off-the-shoulder',
    'v-neckline', 'v-neck', 'round neckline', 'high neckline', 'sweetheart neckline',
    'crew neck', 'scoop neck', 'halter', 'button-down', 'zip-up', 'lace-up',
    'high-waisted', 'low-rise', 'mid-rise', 'fitted', 'loose', 'flowy', 'tailored',
    'ripped', 'distressed', 'embroidered', 'pleated', 'ruched', 'tiered',
    'maxi', 'midi', 'mini', 'floor-length', 'knee-length', 'ankle-length',
    'oversized', 'slim-fit', 'relaxed fit', 'bodycon', 'a-line',
]


def extract_fashion_sentences(description):
    """
    Extract sentences that contain fashion-related keywords.
    """
    if pd.isna(description) or not description:
        return ""

    sentences = re.split(r'[.!?]', description)
    fashion_sentences = []

    # Keywords to look for (case-insensitive)
    all_keywords = CLOTHING_ITEMS + COLORS + STYLE_DESCRIPTORS
    keyword_pattern = '|'.join([re.escape(kw) for kw in all_keywords])

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if sentence contains "wearing" or fashion keywords
        if re.search(r'\bwearing\b', sentence, re.IGNORECASE):
            fashion_sentences.append(sentence)
        elif re.search(keyword_pattern, sentence, re.IGNORECASE):
            # Additional check: must have clothing item, not just color
            has_clothing = any(item.lower() in sentence.lower() for item in CLOTHING_ITEMS)
            if has_clothing:
                fashion_sentences.append(sentence)

    return '. '.join(fashion_sentences) + '.' if fashion_sentences else ""


def extract_fashion_phrases(description):
    """
    Extract specific fashion phrases using pattern matching.
    More aggressive filtering - only keeps fashion item descriptions.
    """
    if pd.isna(description) or not description:
        return ""

    fashion_items = []
    description_lower = description.lower()

    # Pattern: [color] [style] [item]
    # e.g., "white long-sleeved dress", "black leather jacket"

    for item in CLOTHING_ITEMS:
        # Find all mentions of this item
        pattern = rf'(\b(?:[\w-]+\s+){{0,3}}{re.escape(item)}(?:\s+[\w-]+){{0,2}})'
        matches = re.findall(pattern, description_lower)

        for match in matches:
            # Clean up the match
            cleaned = match.strip()
            if cleaned and len(cleaned) > len(item):  # Has descriptors
                fashion_items.append(cleaned)
            elif cleaned:
                fashion_items.append(cleaned)

    # Remove duplicates while preserving order
    seen = set()
    unique_items = []
    for item in fashion_items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    return ', '.join(unique_items) if unique_items else ""


def extract_fashion_structured(description):
    """
    Extract fashion info in a structured format.
    Returns: "CLOTHING: ... | ACCESSORIES: ... | COLORS: ..."
    """
    if pd.isna(description) or not description:
        return ""

    description_lower = description.lower()

    found_clothing = []
    found_accessories = []
    found_colors = set()
    found_styles = []

    # Accessories list (subset of CLOTHING_ITEMS)
    accessories = ['bag', 'handbag', 'purse', 'tote', 'clutch', 'backpack', 'crossbody',
                   'hat', 'cap', 'beanie', 'scarf', 'belt', 'watch', 'bracelet', 'necklace',
                   'earrings', 'ring', 'sunglasses', 'glasses']

    footwear = ['shoes', 'sneakers', 'boots', 'heels', 'sandals', 'flats', 'loafers',
                'slippers', 'ankle boots', 'high heels', 'stilettos', 'wedges']

    # Extract clothing items with their descriptors
    for item in CLOTHING_ITEMS:
        if item.lower() in description_lower:
            # Try to find color + item pattern
            for color in COLORS:
                pattern = rf'{color}\s+(?:[\w-]+\s+)?{item}'
                match = re.search(pattern, description_lower)
                if match:
                    found_colors.add(color)
                    if item in accessories:
                        found_accessories.append(match.group())
                    elif item in footwear:
                        found_accessories.append(match.group())  # Group footwear with accessories
                    else:
                        found_clothing.append(match.group())
                    break
            else:
                # No color found, just add the item
                if item in accessories or item in footwear:
                    found_accessories.append(item)
                else:
                    found_clothing.append(item)

    # Extract style descriptors
    for style in STYLE_DESCRIPTORS:
        if style.lower() in description_lower:
            found_styles.append(style)

    # Build structured output
    parts = []
    if found_clothing:
        parts.append(f"OUTFIT: {', '.join(set(found_clothing))}")
    if found_accessories:
        parts.append(f"ACCESSORIES: {', '.join(set(found_accessories))}")
    if found_styles:
        parts.append(f"STYLE: {', '.join(set(found_styles))}")

    return ' | '.join(parts) if parts else ""


def process_image_descriptions(input_file, output_file, method='sentences'):
    """
    Process image descriptions and save fashion-only version.

    Args:
        input_file: Path to original image_descriptions.csv
        output_file: Path to save filtered descriptions
        method: 'sentences', 'phrases', or 'structured'
    """
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Processing {len(df)} descriptions using method: {method}")

    if method == 'sentences':
        df['fashion_description'] = df['description'].apply(extract_fashion_sentences)
    elif method == 'phrases':
        df['fashion_description'] = df['description'].apply(extract_fashion_phrases)
    elif method == 'structured':
        df['fashion_description'] = df['description'].apply(extract_fashion_structured)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Statistics
    original_avg_len = df['description'].str.len().mean()
    fashion_avg_len = df['fashion_description'].str.len().mean()
    empty_count = (df['fashion_description'] == '').sum()

    print(f"\n=== Statistics ===")
    print(f"Original avg length: {original_avg_len:.0f} chars")
    print(f"Fashion avg length: {fashion_avg_len:.0f} chars")
    print(f"Reduction: {(1 - fashion_avg_len/original_avg_len)*100:.1f}%")
    print(f"Empty descriptions: {empty_count}/{len(df)}")

    # Save
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Show samples
    print("\n=== Sample Comparisons ===")
    for i in [0, 5, 10]:
        if i < len(df):
            print(f"\n[{df.iloc[i]['image_id']}] Original:")
            print(f"  {df.iloc[i]['description'][:150]}...")
            print(f"[{df.iloc[i]['image_id']}] Fashion:")
            print(f"  {df.iloc[i]['fashion_description']}")

    return df


if __name__ == "__main__":
    input_file = "input/image_descriptions.csv"

    # Method 1: Extract full sentences containing fashion info
    print("\n" + "="*60)
    print("METHOD 1: SENTENCE EXTRACTION")
    print("="*60)
    df1 = process_image_descriptions(
        input_file,
        "input/image_descriptions_fashion_sentences.csv",
        method='sentences'
    )

    # Method 2: Extract only fashion phrases
    print("\n" + "="*60)
    print("METHOD 2: PHRASE EXTRACTION")
    print("="*60)
    df2 = process_image_descriptions(
        input_file,
        "input/image_descriptions_fashion_phrases.csv",
        method='phrases'
    )

    # Method 3: Structured extraction
    print("\n" + "="*60)
    print("METHOD 3: STRUCTURED EXTRACTION")
    print("="*60)
    df3 = process_image_descriptions(
        input_file,
        "input/image_descriptions_fashion_structured.csv",
        method='structured'
    )

    print("\n" + "="*60)
    print("DONE! Created 3 filtered versions:")
    print("  1. image_descriptions_fashion_sentences.csv - Full sentences with fashion info")
    print("  2. image_descriptions_fashion_phrases.csv - Only fashion phrases")
    print("  3. image_descriptions_fashion_structured.csv - Structured format (OUTFIT | ACCESSORIES | STYLE)")
    print("="*60)
