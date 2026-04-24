#from PIL import Image
#print(hasattr(Image, "Resampling"))

from transformers import Wav2Vec2Model

model=Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h",cache_dir="./models")
