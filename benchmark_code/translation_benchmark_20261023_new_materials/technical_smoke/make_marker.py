from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


out = Path(__file__).with_name("technical_marker.png")
image = Image.new("RGB", (1200, 500), (247, 247, 247))
draw = ImageDraw.Draw(image)
draw.rectangle((20, 20, 1180, 480), outline=(26, 51, 89), width=8)
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 64)
draw.text((120, 205), "TECHNICAL MARKER 20261023", font=font, fill=(20, 31, 51))
image.save(out, format="PNG")
print(out)
